#!/usr/bin/env bash
# End-to-end smoke test for sensevoice_server: REST transcription formats plus
# realtime WebSocket streaming, compared against the frozen golden transcript.
# SKIPs (exit 0) when the binary or a model is unavailable; FAILs non-zero
# otherwise. Contributes its own PASS/SKIP/FAIL lines.
#
#   BIN_DIR=/path/to/build/bin MODEL_GGUF=/path/to/sensevoice.gguf VAD_GGUF=/path/to/fsmn-vad.gguf \
#       ./run_server_smoke.sh
set -u
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RT=$(cd "$DIR/.." && pwd)
BIN="${BIN_DIR:-$RT/build/bin}"
SERVER="$BIN/sensevoice_server"
GOLDEN="$DIR/golden/sensevoice.txt"
SAMPLE="$DIR/sample.wav"
HOST=127.0.0.1
PORT="${PORT:-8041}"

# Locate the ASR model: explicit override, then progressively broader defaults.
MODEL="${MODEL_GGUF:-}"
if [ -z "$MODEL" ]; then
  for c in "$RT/../../model/sensevoice-small-q8.gguf" \
           "$RT/../model/sensevoice-small-q8.gguf" \
           "$DIR/models/sensevoice-small-q8.gguf" \
           "$DIR/models/sensevoice-small-f16.gguf"; do
    [ -f "$c" ] && { MODEL="$c"; break; }
  done
fi
VAD="${VAD_GGUF:-$DIR/models/fsmn-vad.gguf}"
[ -f "$VAD" ] || VAD="$RT/../../model/fsmn-vad.gguf"

[ -x "$SERVER" ] || { echo "  SKIP  server-smoke (no binary: $SERVER)"; exit 0; }
[ -f "$MODEL" ]  || { echo "  SKIP  server-smoke (model missing: set MODEL_GGUF)"; exit 0; }
[ -f "$VAD" ]    || { echo "  SKIP  server-smoke (vad missing: $VAD)"; exit 0; }
[ -f "$GOLDEN" ] || { echo "  SKIP  server-smoke (no golden)"; exit 0; }

"$SERVER" -m "$MODEL" -vad "$VAD" "$HOST" "$PORT" > "$DIR/server.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null' EXIT

up=0
for _ in $(seq 1 100); do
  if curl -s -o /dev/null "http://$HOST:$PORT/health"; then up=1; break; fi
  sleep 0.1
done
if [ "$up" = 0 ]; then
  echo "  FAIL  server-smoke (server did not start; see tests/server.log)"; exit 1
fi

pass=0; fail=0
check(){ local name="$1" expected="$2" got="$3"
  if [ "$got" = "$expected" ]; then echo "  PASS  $name"; pass=$((pass+1))
  else echo "  FAIL  $name"; echo "    expected: $expected"; echo "    got:      $got"; fail=$((fail+1)); fi
}

EXP=$(tr -d '\n' < "$GOLDEN")
norm(){ # strip whitespace and trailing/leading CJK punctuation (VAD-segmented ASR adds '。')
  printf '%s' "$1" | python3 -c "import sys,re;p=re.compile(r'^[.\s\u3002\uff01\uff1f\uff0c\u3001\u300b\u300a\u201d\u201c]+|[.\s\u3002\uff01\uff1f\uff0c\u3001\u300b\u300a\u201d\u201c]+$');sys.stdout.write(p.sub('',sys.stdin.read()))"
}
textof(){ # extract the "text" field from a JSON response body
  printf '%s' "$1" | python3 -c "import sys,json;sys.stdout.write(json.load(sys.stdin).get('text',''))"
}

GOT=$(curl -s -F file=@"$SAMPLE" "http://$HOST:$PORT/v1/audio/transcriptions")
check "server rest-json" "$(norm "$EXP")" "$(norm "$(textof "$GOT")")"

GOT=$(curl -s -F file=@"$SAMPLE" -F response_format=text "http://$HOST:$PORT/v1/audio/transcriptions")
check "server rest-text" "$(norm "$EXP")" "$(norm "$GOT")"

GOT=$(curl -s -F file=@"$SAMPLE" -F response_format=verbose_json "http://$HOST:$PORT/v1/audio/transcriptions")
echo "$GOT" | grep -q "segments" && echo "  PASS  server rest-verbose_json" && pass=$((pass+1)) \
  || { echo "  FAIL  server rest-verbose_json"; echo "    got: $GOT"; fail=$((fail+1)); }

GOT=$(curl -s -N -F file=@"$SAMPLE" -F stream=true "http://$HOST:$PORT/v1/audio/transcriptions")
echo "$GOT" | grep -q "transcript.text.done" && echo "  PASS  server rest-sse" && pass=$((pass+1)) \
  || { echo "  FAIL  server rest-sse"; echo "    got: $GOT"; fail=$((fail+1)); }

GOT=$(python3 "$DIR/stream_client.py" "$HOST" "$PORT" "$SAMPLE" 200)
check "server ws-stream" "$(norm "$EXP")" "$(norm "$GOT")"

echo "  server-smoke: $pass passed, $fail failed"
[ "$fail" = 0 ]