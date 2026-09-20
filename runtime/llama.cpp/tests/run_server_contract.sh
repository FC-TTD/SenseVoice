#!/usr/bin/env bash
# Model-free contract checks for the SenseVoice HTTP/WebSocket server.
set -u

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RT=$(cd "$DIR/.." && pwd)
BIN="${BIN_DIR:-$RT/build/bin}"
SERVER="$BIN/sensevoice-server"

pass=0
fail=0

check() {
  local name="$1"
  shift
  if "$@"; then
    echo "  PASS  $name"
    pass=$((pass + 1))
  else
    echo "  FAIL  $name"
    fail=$((fail + 1))
  fi
}

has_server() { [ -x "$SERVER" ]; }
help_has_connection_limit() {
  "$SERVER" --help 2>&1 | grep -q -- '--max-connections <N>'
}
connection_limit_is_parsed() {
  local output
  output=$("$SERVER" --max-connections 2 2>&1 || true)
  ! grep -q 'unknown option' <<<"$output"
}
connection_limit_is_bounded() {
  "$SERVER" --max-connections 129 2>&1 | grep -q 'between 1 and 128'
}
audio_limit_is_documented() {
  "$SERVER" --help 2>&1 | grep -q -- '--max-audio-seconds <N>'
}
audio_limit_is_bounded() {
  "$SERVER" --max-audio-seconds 3601 2>&1 | grep -q 'between 1 and 3600'
}

check "server binary name" has_server
if has_server; then
  check "server help documents connection limit" help_has_connection_limit
  check "server parses connection limit" connection_limit_is_parsed
  check "server bounds connection limit" connection_limit_is_bounded
  check "server documents audio limit" audio_limit_is_documented
  check "server bounds audio limit" audio_limit_is_bounded
fi

echo "  server-contract: $pass passed, $fail failed"
[ "$fail" = 0 ]
