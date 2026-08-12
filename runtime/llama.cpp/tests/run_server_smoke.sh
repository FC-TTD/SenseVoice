#!/usr/bin/env bash
# End-to-end smoke test for sensevoice-server: REST transcription formats plus
# realtime WebSocket streaming, compared against the frozen golden transcript.
# FAILs when the requested binary is unavailable; model/golden prerequisites
# still SKIP. Contributes its own PASS/SKIP/FAIL lines.
#
#   BIN_DIR=/path/to/build/bin MODEL_GGUF=/path/to/sensevoice.gguf VAD_GGUF=/path/to/fsmn-vad.gguf \
#       ./run_server_smoke.sh
set -u
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RT=$(cd "$DIR/.." && pwd)
BIN="${BIN_DIR:-$RT/build/bin}"
SERVER="$BIN/sensevoice-server"
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

[ -x "$SERVER" ] || { echo "  FAIL  server-smoke (no binary: $SERVER)"; exit 1; }
[ -f "$MODEL" ]  || { echo "  SKIP  server-smoke (model missing: set MODEL_GGUF)"; exit 0; }
[ -f "$VAD" ]    || { echo "  SKIP  server-smoke (vad missing: $VAD)"; exit 0; }
[ -f "$GOLDEN" ] || { echo "  SKIP  server-smoke (no golden)"; exit 0; }

"$SERVER" -m "$MODEL" -vad "$VAD" --max-connections 4 --max-audio-seconds 10 \
  "$HOST" "$PORT" > "$DIR/server.log" 2>&1 &
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

STATUS=$(curl -s -o /dev/null -w '%{http_code}' -F file=@"$SAMPLE" -F response_format=invalid \
  "http://$HOST:$PORT/v1/audio/transcriptions")
check "server rejects response-format" "400" "$STATUS"

LONG_WAV=$(mktemp "${TMPDIR:-/tmp}/sensevoice-long.XXXXXX.wav")
python3 - "$LONG_WAV" <<'PY'
import sys
import wave

with wave.open(sys.argv[1], "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b"\0\0" * 16000 * 11)
PY
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -F file=@"$LONG_WAV" \
  "http://$HOST:$PORT/v1/audio/transcriptions")
rm -f "$LONG_WAV"
check "server rest-audio-limit" "413" "$STATUS"

GOT=$(curl -s -N -F file=@"$SAMPLE" -F stream=true "http://$HOST:$PORT/v1/audio/transcriptions")
echo "$GOT" | grep -q "transcript.text.done" && echo "  PASS  server rest-sse" && pass=$((pass+1)) \
  || { echo "  FAIL  server rest-sse"; echo "    got: $GOT"; fail=$((fail+1)); }

GOT=$(curl -s -F file=@"$SAMPLE" -F response_format=srt "http://$HOST:$PORT/v1/audio/transcriptions")
if echo "$GOT" | grep -qx '1' && echo "$GOT" | grep -Eq '^[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} --> [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}$'; then
  echo "  PASS  server rest-srt"; pass=$((pass+1))
else
  echo "  FAIL  server rest-srt"; echo "    got: $GOT"; fail=$((fail+1))
fi

GOT=$(curl -s -F file=@"$SAMPLE" -F response_format=vtt "http://$HOST:$PORT/v1/audio/transcriptions")
if echo "$GOT" | grep -qx 'WEBVTT' && echo "$GOT" | grep -Eq '^[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} --> [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}$'; then
  echo "  PASS  server rest-vtt"; pass=$((pass+1))
else
  echo "  FAIL  server rest-vtt"; echo "    got: $GOT"; fail=$((fail+1))
fi

if PYTHONPATH="$DIR" python3 - "$HOST" "$PORT" <<'PY'
import sys
import time
from stream_client import WSClient

ws = WSClient(sys.argv[1], int(sys.argv[2]), "/v1/realtime?intent=transcription")
ws.next_event(2.0)
ws.send_text('{"type":')
time.sleep(0.1)
ws.send_text('{"type":"input_audio_buffer.clear"}')
time.sleep(0.1)
ws.close()
PY
then
  curl -fsS "http://$HOST:$PORT/health" >/dev/null \
    && { echo "  PASS  server malformed-ws-survives"; pass=$((pass+1)); } \
    || { echo "  FAIL  server malformed-ws-survives"; fail=$((fail+1)); }
else
  echo "  FAIL  server malformed-ws-survives"; fail=$((fail+1))
fi

if PYTHONPATH="$DIR" python3 - "$HOST" "$PORT" <<'PY'
import sys
import time
from stream_client import OP_TEXT, WSClient

clients = [WSClient(sys.argv[1], int(sys.argv[2]), "/v1/realtime?intent=transcription") for _ in range(5)]
for client in clients[:4]:
    event = client.next_event(2.0)
    assert event and event[0] == OP_TEXT and b'"session.created"' in event[1]
event = clients[4].next_event(2.0)
assert event and event[0] == OP_TEXT and b'connection limit exceeded' in event[1]
for client in clients:
    client.close()
time.sleep(0.2)
replacement = WSClient(sys.argv[1], int(sys.argv[2]), "/v1/realtime?intent=transcription")
event = replacement.next_event(2.0)
assert event and event[0] == OP_TEXT and b'"session.created"' in event[1]
replacement.close()
PY
then
  echo "  PASS  server ws-connection-limit"; pass=$((pass+1))
else
  echo "  FAIL  server ws-connection-limit"; fail=$((fail+1))
fi

if PYTHONPATH="$DIR" python3 - "$HOST" "$PORT" <<'PY'
import sys
from stream_client import OP_TEXT, WSClient

ws = WSClient(sys.argv[1], int(sys.argv[2]), "/v1/realtime?intent=transcription")
ws.next_event(2.0)
ws.send_binary(b"\0" * (16000 * 2 * 11))
event = ws.next_event(2.0)
assert event and event[0] == OP_TEXT and b'session audio limit exceeded' in event[1]
ws.close()
PY
then
  echo "  PASS  server ws-session-audio-limit"; pass=$((pass+1))
else
  echo "  FAIL  server ws-session-audio-limit"; fail=$((fail+1))
fi

GOT=$(python3 "$DIR/stream_client.py" "$HOST" "$PORT" "$SAMPLE" 200)
check "server ws-stream" "$(norm "$EXP")" "$(norm "$GOT")"

echo "  server-smoke: $pass passed, $fail failed"
[ "$fail" = 0 ]
