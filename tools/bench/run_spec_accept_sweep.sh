#!/usr/bin/env bash
# Spec-decode acceptance sweep driven by the `/metrics` counters.
#
# Reproduces the acceptance tables in docs/models/qwen3/dspark-integration.md,
# but reads acceptance from `vllm:spec_decode_*` instead of parsing the server
# log. Each cell is bracketed by two `/metrics` scrapes, so per-case boundaries
# are exact rather than recovered from log timestamps.
#
# Speculative decoding is gated to the single-GPU path (load_dflash_draft_model
# rejects extra ranks), so this pins one card via CUDA_VISIBLE_DEVICES.
#
# Run once per drafter, into the same RESULT_DIR, then report:
#   MODEL=/data/Qwen3-4B DRAFT_MODEL=/data/dspark_qwen3_4b_block7 \
#     CONFIG=DSpark GPU=7 tools/bench/run_spec_accept_sweep.sh
#   MODEL=/data/Qwen3-4B DRAFT_MODEL=/data/dflash_qwen3_4b_block7 \
#     CONFIG=DFlash GPU=7 tools/bench/run_spec_accept_sweep.sh
#   tools/bench/spec_accept_metrics.py report ./spec-accept-results/cell-*.json
#
# Optional env:
#   MODEL            target model path (required)
#   DRAFT_MODEL      DFlash/DSpark drafter path (required)
#   CONFIG           label for this drafter in the report [default: basename]
#   GPU              CUDA device ordinal [default: 0]
#   PORT             server port [default: 8000]
#   RESULT_DIR       output directory [default: ./spec-accept-results]
#   DATASETS         vllm-bench datasets [default: "sharegpt sonnet random speed-bench"]
#   CONCURRENCY_LIST [default: "1 4 8"]
#   INPUT_LEN        random-dataset input length [default: 1024]
#   OUTPUT_LEN       random-dataset output length [default: 128]
#   SEED             base seed; each cell derives its own [default: 42]
#   SECONDS_PER_RUN  prompts per cell = concurrency * this [default: 60]
#   BENCH            vllm-bench binary [default: vllm-bench on PATH]
#   ACCEPT_LOG_CHECK 1 = also run the engine at debug level and cross-check the
#                    counters against dflash_lane.rs's cumulative_accept_rate
#                    (issue #604's validation). Verbose: one line per request
#                    per verify round. [default: 1]
#   SKIP_BUILD       1 = reuse target/release/openinfer as-is [default: 0]
set -euo pipefail

MODEL=${MODEL:?MODEL (target model path) is required}
DRAFT_MODEL=${DRAFT_MODEL:?DRAFT_MODEL (drafter path) is required}
CONFIG=${CONFIG:-$(basename "$DRAFT_MODEL")}
GPU=${GPU:-0}
PORT=${PORT:-8000}
RESULT_DIR=${RESULT_DIR:-./spec-accept-results}
DATASETS=${DATASETS:-"sharegpt sonnet random speed-bench"}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 4 8"}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SEED=${SEED:-42}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
BENCH=${BENCH:-vllm-bench}
ACCEPT_LOG_CHECK=${ACCEPT_LOG_CHECK:-1}
SKIP_BUILD=${SKIP_BUILD:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METRICS_TOOL="$SCRIPT_DIR/spec_accept_metrics.py"
METRICS_URL="http://localhost:$PORT/metrics"
SERVER_LOG="$RESULT_DIR/server-${CONFIG}.log"

mkdir -p "$RESULT_DIR"

# The derivation is pure arithmetic over the counters; validate it before
# spending GPU time, so a reporting bug can't be mistaken for an engine bug.
echo "=== validating the histogram derivation ==="
"$METRICS_TOOL" selftest

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "=== building openinfer (SKIP_BUILD=1 to skip) ==="
  (cd "$REPO_ROOT" && CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
    cargo build --release -p openinfer-server)
fi

# The counters only exist on the spec-decode-acceptance-metrics work; without it
# every scrape reads empty and the sweep wastes a full GPU run before failing.
if ! grep -q "spec_decode" "$REPO_ROOT/openinfer-engine/src/engine.rs" 2>/dev/null; then
  echo "FATAL: openinfer-engine has no spec_decode counters — this sweep needs" >&2
  echo "       the spec-decode acceptance metrics branch." >&2
  exit 1
fi

RUST_LOG_SETTING=""
if [[ "$ACCEPT_LOG_CHECK" == "1" ]]; then
  RUST_LOG_SETTING="openinfer_qwen3=debug"
fi

echo "=== launching openinfer: model=$MODEL draft=$DRAFT_MODEL gpu=$GPU port=$PORT ==="
CUDA_VISIBLE_DEVICES=$GPU RUST_LOG="$RUST_LOG_SETTING" \
  "$REPO_ROOT/target/release/openinfer" \
  --model-path "$MODEL" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --dflash-draft-model-path "$DRAFT_MODEL" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== shutting down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

READY_TIMEOUT=${READY_TIMEOUT:-180}
echo "=== waiting for readiness (timeout ${READY_TIMEOUT}s) ==="
for _ in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: server died during startup. Log:" >&2
    tail -40 "$SERVER_LOG" >&2
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "=== server ready ==="
    break
  fi
  sleep 1
done
if ! curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "FATAL: server not ready after ${READY_TIMEOUT}s. Log:" >&2
  tail -40 "$SERVER_LOG" >&2
  exit 1
fi

# Counters are absent until the first draft, so the run baseline tolerates an
# empty scrape; per-cell scrapes after the first case do not.
"$METRICS_TOOL" snapshot --url "$METRICS_URL" \
  --out "$RESULT_DIR/run-start-${CONFIG}.json" --allow-missing

CELLS=()
for DATASET in $DATASETS; do
  DATASET_ARGS=(--dataset-name "$DATASET")
  if [[ "$DATASET" == "random" ]]; then
    DATASET_ARGS+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
  fi
  for C in $CONCURRENCY_LIST; do
    NUM_PROMPTS=$(python3 -c "print(int($C * $SECONDS_PER_RUN))")
    # Derive from axis+value, not draw order, so a cell replays the same
    # prompts regardless of which datasets are enabled.
    POINT_SEED=$(( SEED + $(printf '%s' "$DATASET=$C" | cksum | cut -d' ' -f1) % 100000 ))
    TAG="${CONFIG}-${DATASET}-c${C}"
    echo ""
    echo "--- $TAG num_prompts=$NUM_PROMPTS seed=$POINT_SEED ---"

    "$METRICS_TOOL" snapshot --url "$METRICS_URL" \
      --out "$RESULT_DIR/before-${TAG}.json" --allow-missing

    # --temperature 0 is load-bearing, not a default: should_speculative_decode
    # is all-or-nothing, so one non-greedy request drops the whole batch to
    # plain decode and every counter stays flat.
    "$BENCH" \
      --backend openai --model "$MODEL" --port "$PORT" \
      --base-url "http://localhost:$PORT" \
      "${DATASET_ARGS[@]}" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$C" \
      --seed "$POINT_SEED" \
      --ignore-eos --temperature 0 \
      --tokenizer "$MODEL" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$RESULT_DIR" \
      --result-filename "bench-${TAG}.json"

    "$METRICS_TOOL" snapshot --url "$METRICS_URL" \
      --out "$RESULT_DIR/after-${TAG}.json"

    "$METRICS_TOOL" diff \
      --before "$RESULT_DIR/before-${TAG}.json" \
      --after "$RESULT_DIR/after-${TAG}.json" \
      --out "$RESULT_DIR/cell-${TAG}.json" \
      --config "$CONFIG" --dataset "$DATASET" --concurrency "$C"
    CELLS+=("$RESULT_DIR/cell-${TAG}.json")
  done
done

# ---- whole-run totals + issue #604 cross-check ------------------------------
"$METRICS_TOOL" snapshot --url "$METRICS_URL" --out "$RESULT_DIR/run-end-${CONFIG}.json"
"$METRICS_TOOL" diff \
  --before "$RESULT_DIR/run-start-${CONFIG}.json" \
  --after "$RESULT_DIR/run-end-${CONFIG}.json" \
  --out "$RESULT_DIR/run-total-${CONFIG}.json" \
  --config "$CONFIG" --dataset "ALL" --concurrency 0

if [[ "$ACCEPT_LOG_CHECK" == "1" ]]; then
  echo ""
  echo "=== cross-check: /metrics vs dflash_lane.rs cumulative_accept_rate ==="
  # Both sides count the same matched_draft_tokens over the same server
  # lifetime, so they must agree to log-rounding (the trace prints 3 decimals),
  # not merely be close. Drift means the counter plumbing lost or double-counted
  # something. This is issue #604's acceptance criterion.
  LOG_RATE=$(grep -o 'cumulative_accept_rate=[0-9.]*' "$SERVER_LOG" | tail -1 | cut -d= -f2 || true)
  METRICS_RATE=$(python3 -c "
import json,sys
s=json.load(open('$RESULT_DIR/run-total-${CONFIG}.json'))['stats']
print(f\"{s['accept_rate']:.3f}\")
")
  if [[ -z "$LOG_RATE" ]]; then
    echo "WARN: no cumulative_accept_rate lines in $SERVER_LOG."
    echo "      The trace is debug-level; RUST_LOG must include openinfer_qwen3=debug."
  else
    echo "  server log : $LOG_RATE"
    echo "  /metrics   : $METRICS_RATE"
    if [[ "$LOG_RATE" == "$METRICS_RATE" ]]; then
      echo "  MATCH"
    else
      echo "  MISMATCH — the counters disagree with the engine's own tally." >&2
      echo "  Investigate before quoting any number from this run." >&2
      exit 1
    fi
  fi
fi

echo ""
echo "=== $CONFIG sweep complete ==="
"$METRICS_TOOL" report "${CELLS[@]}"
echo ""
echo "cells written to $RESULT_DIR/cell-*.json"
echo "run both drafters into this directory, then:"
echo "  $METRICS_TOOL report $RESULT_DIR/cell-*.json"
