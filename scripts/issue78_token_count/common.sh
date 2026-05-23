# Shared setup for the issue #78 (streaming token-count) repro scripts.
# Source this from the other scripts in this directory.
#
# Overridable via environment:
#   PEGAINFER_ENV  path to a shell file that sets up cargo + CUDA + (optional) vllm
#                  (e.g. activates the conda toolchain env). Sourced if it exists.
#   VLLM           vllm executable for `vllm bench serve` (default: vllm on PATH)
#   MODEL_PATH     model dir relative to repo root (default: models/Qwen3-4B)
#   PORT           server port (default: 8000)
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The frontend<->engine-core Unix socket path must stay under SUN_LEN (~108 chars);
# long $TMPDIR values (e.g. CI scratch paths) break server startup.
export TMPDIR="${TMPDIR_SHORT:-/tmp}"

: "${PEGAINFER_ENV:=$REPO_ROOT/../pegainfer-env.sh}"
if [ -f "$PEGAINFER_ENV" ]; then
  # conda activation scripts are not `set -u` safe — keep -u off.
  source "$PEGAINFER_ENV"
fi

: "${VLLM:=vllm}"
: "${MODEL_PATH:=models/Qwen3-4B}"   # relative to repo root; also the served model id
: "${PORT:=8000}"

cd "$REPO_ROOT"

# start_server BG: launches pegainfer release server and waits until /v1/models is up.
# Sets $SERVER_PID. Exits non-zero if the server dies before becoming ready.
start_server() {
  local log="${1:-/tmp/pega_issue78_server.log}"
  RUST_LOG=warn ./target/release/pegainfer --model-path "$MODEL_PATH" --port "$PORT" > "$log" 2>&1 &
  SERVER_PID=$!
  local i
  for i in $(seq 1 180); do
    curl -sf -o /dev/null "http://localhost:$PORT/v1/models" && { echo "server READY after ${i}s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED"; tail -20 "$log"; return 1; }
    sleep 1
  done
  echo "server NOT READY"; kill "$SERVER_PID" 2>/dev/null; return 1
}

stop_server() { kill "${SERVER_PID:-}" 2>/dev/null; }
