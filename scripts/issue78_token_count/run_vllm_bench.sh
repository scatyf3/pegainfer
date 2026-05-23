#!/bin/bash
# Issue #78 repro (end-to-end): drive pegainfer with `vllm bench serve --save-detailed`
# and print per-request output_lens. Under --ignore-eos the true total is exactly
# num_prompts * output_len; any deviation is the issue #78 token-count error.
#
# Usage: bash scripts/issue78_token_count/run_vllm_bench.sh [IN] [OUT] [N] [SEED]
#   defaults: IN=1024 OUT=256 N=20 SEED=42
# Requires `vllm` (set VLLM=/path/to/vllm if not on PATH).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

IN=${1:-1024}; OUT=${2:-256}; N=${3:-20}; SEED=${4:-42}
RESULTS="$REPO_ROOT/bench_results/issue78_in${IN}_out${OUT}_n${N}_seed${SEED}"
mkdir -p "$RESULTS"

start_server "$RESULTS/server.log" || exit 1

"$VLLM" bench serve --backend openai --model "$MODEL_PATH" --port "$PORT" \
  --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" \
  --num-prompts "$N" --request-rate inf --max-concurrency 1 \
  --ignore-eos --temperature 0 --tokenizer "$REPO_ROOT/$MODEL_PATH" --seed "$SEED" \
  --save-result --save-detailed --result-dir "$RESULTS" --result-filename detailed.json \
  2>&1 | grep -iE "Successful|Failed|Output token throughput"

python3 - "$RESULTS/detailed.json" "$OUT" "$N" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); O = int(sys.argv[2]); N = int(sys.argv[3])
ol = d.get("output_lens", [])
bad = [(i, x) for i, x in enumerate(ol) if x != O]
print("per-request output_lens:", ol)
print(f"requests != {O}: {len(bad)} -> {bad}")
print(f"sum={sum(ol)}  true_total={O*N}  error={sum(ol) - O*N}")
PY

stop_server
echo "DONE -> $RESULTS/detailed.json"
