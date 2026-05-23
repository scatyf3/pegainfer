#!/bin/bash
# Issue #78 root-cause check: does the streaming response carry a correct usage chunk?
# Compares non-streaming usage vs the final streaming usage chunk for one request.
# Usage: bash scripts/issue78_token_count/usage_probe.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

start_server /tmp/pega_issue78_usage_server.log || exit 1

echo "=== NON-STREAMING (ignore_eos, max_tokens=256) ==="
curl -s "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" -d "{
  \"model\":\"$MODEL_PATH\",\"prompt\":\"Tell me a long story.\",\"max_tokens\":256,
  \"temperature\":0,\"ignore_eos\":true}" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("usage:", d.get("usage"))'

echo "=== STREAMING (stream_options.include_usage=true) ==="
curl -s -N "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" -d "{
  \"model\":\"$MODEL_PATH\",\"prompt\":\"Tell me a long story.\",\"max_tokens\":256,
  \"temperature\":0,\"ignore_eos\":true,\"stream\":true,
  \"stream_options\":{\"include_usage\":true}}" | python3 - <<'PY'
import json, sys
n = 0
usage = None
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"):
        continue
    body = line[5:].strip()
    if body == "[DONE]":
        continue
    obj = json.loads(body)
    if obj.get("usage"):
        usage = obj["usage"]
    for c in obj.get("choices", []):
        if c.get("text"):
            n += 1
print("text_deltas_streamed:", n)
print("final usage in stream:", usage)
PY

stop_server
echo "DONE"
