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
#   DATASETS         [default: depends on backend, see BENCH_BACKEND]
#   CONCURRENCY_LIST [default: "1 4 8"]
#   INPUT_LEN        random-dataset input length [default: 1024]
#   OUTPUT_LEN       output tokens per request [default: 128]
#   SEED             base seed; each cell derives its own [default: 42]
#   SECONDS_PER_RUN  prompts per cell = concurrency * this [default: 60]
#   BENCH_BACKEND    vllm-bench | http | auto [default: auto]
#                    `auto` picks vllm-bench when it is on PATH, else `http`.
#                    `http` uses scripts/bench_http_serving.py — stdlib only, no
#                    install — but it has no sonnet/speed-bench datasets, so it
#                    covers `sharegpt` and `synthetic` only.
#   BENCH            vllm-bench binary [default: vllm-bench on PATH]
#
#   PROMPT_FILE      ShareGPT-style JSON. Required by the sharegpt dataset on
#                    BOTH backends — sonnet and speed-bench ship their corpora,
#                    sharegpt does not.
#   SPEED_BENCH_CATEGORY  vllm-bench speed-bench split [default: coding, the
#                    prior study's "code" row]
#
# http-backend only:
#   PROMPT_COUNT     prompts sampled from it [default: 30, the documented protocol]
#   PROMPT_SEED      sampling seed [default: 512, the documented protocol]
#   PROMPT_WORDS     synthetic prompt length [default: 512]
#   WARMUP           warmup requests per cell [default: 0 — warmup lands inside
#                    the scrape bracket and would be counted as measured rounds]
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
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 4 8"}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SEED=${SEED:-42}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
BENCH=${BENCH:-vllm-bench}
BENCH_BACKEND=${BENCH_BACKEND:-auto}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_COUNT=${PROMPT_COUNT:-30}
PROMPT_SEED=${PROMPT_SEED:-512}
PROMPT_WORDS=${PROMPT_WORDS:-512}
SPEED_BENCH_CATEGORY=${SPEED_BENCH_CATEGORY:-coding}
WARMUP=${WARMUP:-0}
ACCEPT_LOG_CHECK=${ACCEPT_LOG_CHECK:-1}
SKIP_BUILD=${SKIP_BUILD:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METRICS_TOOL="$SCRIPT_DIR/spec_accept_metrics.py"
HTTP_BENCH="$REPO_ROOT/scripts/bench_http_serving.py"
METRICS_URL="http://127.0.0.1:$PORT/metrics"
SERVER_LOG="$RESULT_DIR/server-${CONFIG}.log"

if [[ "$BENCH_BACKEND" == "auto" ]]; then
  if command -v "$BENCH" > /dev/null 2>&1; then
    BENCH_BACKEND=vllm-bench
  else
    BENCH_BACKEND=http
    echo "note: $BENCH not on PATH — using the http backend ($HTTP_BENCH)"
  fi
fi

case "$BENCH_BACKEND" in
  vllm-bench)
    DATASETS=${DATASETS:-"sharegpt sonnet random speed-bench"}
    command -v "$BENCH" > /dev/null 2>&1 || {
      echo "FATAL: BENCH_BACKEND=vllm-bench but '$BENCH' is not on PATH." >&2
      echo "       Use BENCH_BACKEND=http for the in-repo Python harness." >&2
      exit 1
    }
    # sonnet and speed-bench carry their own corpora; sharegpt needs a file.
    if [[ " $DATASETS " == *" sharegpt "* ]]; then
      [[ -r "${PROMPT_FILE:-}" ]] || {
        echo "FATAL: dataset 'sharegpt' needs PROMPT_FILE (a ShareGPT-style JSON)." >&2
        exit 1
      }
    fi
    ;;
  http)
    DATASETS=${DATASETS:-"sharegpt"}
    [[ -x "$HTTP_BENCH" ]] || { echo "FATAL: $HTTP_BENCH not found" >&2; exit 1; }
    for DATASET in $DATASETS; do
      case "$DATASET" in
        sharegpt)
          [[ -n "$PROMPT_FILE" ]] || {
            echo "FATAL: dataset 'sharegpt' on the http backend needs PROMPT_FILE" >&2
            echo "       (a ShareGPT-style JSON of {conversations:[{from,value}]})." >&2
            exit 1
          }
          [[ -r "$PROMPT_FILE" ]] || { echo "FATAL: cannot read PROMPT_FILE=$PROMPT_FILE" >&2; exit 1; }
          ;;
        synthetic) ;;
        *)
          # Better to stop than to silently swap in a different prompt
          # distribution: the resulting accept numbers would be quoted against
          # tables built from a dataset this backend never ran.
          echo "FATAL: the http backend has no '$DATASET' dataset (it supports" >&2
          echo "       sharegpt and synthetic). Install vllm-bench for sonnet /" >&2
          echo "       speed-bench, or drop '$DATASET' from DATASETS." >&2
          exit 1
          ;;
      esac
    done
    ;;
  *)
    echo "FATAL: BENCH_BACKEND must be vllm-bench, http, or auto (got '$BENCH_BACKEND')" >&2
    exit 1
    ;;
esac

mkdir -p "$RESULT_DIR"
echo "=== bench backend: $BENCH_BACKEND | datasets: $DATASETS ==="

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
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "=== server ready ==="
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
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
  DATASET_ARGS=()
  if [[ "$BENCH_BACKEND" == "vllm-bench" ]]; then
    DATASET_ARGS=(--dataset-name "$DATASET")
    case "$DATASET" in
      random)
        DATASET_ARGS+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN") ;;
      sharegpt)
        # sonnet and speed-bench ship their corpora; sharegpt does not.
        DATASET_ARGS+=(--dataset-path "$PROMPT_FILE") ;;
      speed-bench)
        # The prior study's "code" row is the coding split specifically.
        DATASET_ARGS+=(--speed-bench-category "$SPEED_BENCH_CATEGORY") ;;
    esac
  elif [[ "$DATASET" == "sharegpt" ]]; then
    # The documented pool protocol: first human turn of each conversation,
    # length-filtered, then seed-sampled. Reproducible via PROMPT_SEED, not by
    # taking a prefix — ShareGPT's own order is not random.
    DATASET_ARGS=(--prompt-file "$PROMPT_FILE"
                  --prompt-count "$PROMPT_COUNT"
                  --prompt-seed "$PROMPT_SEED")
  else
    # Synthetic prompts draft too well and overstate acceptance; useful as a
    # stress case, not as a number to quote (scripts/bench_http_serving.py).
    DATASET_ARGS=(--prompt-words "$PROMPT_WORDS")
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
    if [[ "$BENCH_BACKEND" == "vllm-bench" ]]; then
      "$BENCH" \
        --backend openai --model "$MODEL" --port "$PORT" \
        --base-url "http://127.0.0.1:$PORT" \
        "${DATASET_ARGS[@]}" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$C" \
        --seed "$POINT_SEED" \
        --ignore-eos --temperature 0 \
        --tokenizer "$MODEL" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "bench-${TAG}.json"
    else
      # WARMUP defaults to 0: warmup requests land between the two scrapes and
      # would otherwise be counted as measured rounds.
      python3 "$HTTP_BENCH" \
        --base-url "http://127.0.0.1:$PORT" \
        --model "$MODEL" \
        "${DATASET_ARGS[@]}" \
        --num-requests "$NUM_PROMPTS" \
        --concurrency "$C" \
        --warmup "$WARMUP" \
        --max-tokens "$OUTPUT_LEN" \
        --temperature 0 --ignore-eos \
        --out "$RESULT_DIR/bench-${TAG}.json"
    fi

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
