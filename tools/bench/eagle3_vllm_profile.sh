#!/usr/bin/env bash
# EAGLE-3 pegainfer vs vLLM: γ-aligned single-stream A/B plus nsys gap breakdown.
#
#   MODEL=models/Qwen3-4B DRAFT=models/Qwen3-4B_eagle3 VLLM_PY=~/venvs/vllm/bin/python \
#     tools/bench/eagle3_vllm_profile.sh
#
# Writes everything to $OUT (default bench_out/eagle3-vllm-<timestamp>/); the
# final table is $OUT/report.md. γ comes from pegainfer's EAGLE3_CHAIN_LENGTH
# (via ab_config.json) and is passed to vLLM as num_speculative_tokens.
#
# Env: MODEL, DRAFT (required) · VLLM_PY (python with vllm, default python3)
#      GSM8K_DIR (default models/gsm8k, fetched if missing) · OUT
#      N_AB (A/B prompts, default 20) · N_PROF (profiled prompts, default 5)
#      SKIP_AB=1 / SKIP_PROF=1 to run only one half
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${MODEL:?set MODEL to the Qwen3-4B checkpoint}"
: "${DRAFT:?set DRAFT to the EAGLE-3 drafter (e.g. AngelSlim/Qwen3-4B_eagle3)}"
VLLM_PY=${VLLM_PY:-python3}
GSM8K_DIR=${GSM8K_DIR:-models/gsm8k}
OUT=${OUT:-bench_out/eagle3-vllm-$(date +%Y%m%d-%H%M%S)}
N_AB=${N_AB:-20}
N_PROF=${N_PROF:-5}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")

if [[ ! -f $GSM8K_DIR/test.jsonl || ! -f $GSM8K_DIR/train.jsonl ]]; then
  mkdir -p "$GSM8K_DIR"
  for f in train test; do
    curl -sSL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$f.jsonl" \
      -o "$GSM8K_DIR/$f.jsonl"
  done
fi

{
  echo "date: $(date -Is)"
  echo "head: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
  echo "vllm: $("$VLLM_PY" -c 'import vllm; print(vllm.__version__)')"
  echo "nsys: $(nsys --version 2>/dev/null || echo missing)"
} | tee "$OUT/env.txt"

export PEGAINFER_TEST_MODEL_PATH=$MODEL PEGAINFER_EAGLE3_TEST_MODEL_PATH=$DRAFT \
  PEGAINFER_GSM8K_DIR=$GSM8K_DIR PEGAINFER_AB_OUT=$OUT

TEST_BIN=$(cargo test --release -p pegainfer-qwen3 --test eagle3_gsm8k_ab --no-run --message-format=json 2>/dev/null |
  "$VLLM_PY" -c 'import json,sys
for l in sys.stdin:
    m=json.loads(l)
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]=="eagle3_gsm8k_ab": print(m["executable"])' | tail -1)
[[ -x $TEST_BIN ]] || { echo "failed to build eagle3_gsm8k_ab"; exit 1; }

if [[ ${SKIP_AB:-0} != 1 ]]; then
  echo "== A/B: pegainfer (spec off + on, $N_AB prompts)"
  PEGAINFER_TEST_LOG=1 PEGAINFER_AB_NUM_PROMPTS=$N_AB "$TEST_BIN" --nocapture --test-threads=1 2>&1 | tee "$OUT/ours_ab.log"
  echo "== A/B: vLLM (separate processes for off / on)"
  for m in off on; do
    "$VLLM_PY" tools/bench/eagle3_vllm_ab.py --model "$MODEL" --draft "$DRAFT" --ab-dir "$OUT" --mode $m \
      --num-prompts "$N_AB" 2>&1 | tee "$OUT/vllm_ab_$m.log"
  done
fi

if [[ ${SKIP_PROF:-0} != 1 ]]; then
  NSYS=(nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi
    --capture-range-end=stop --force-overwrite=true)
  echo "== nsys: pegainfer spec on ($N_PROF prompts)"
  PEGAINFER_AB_MODE=on PEGAINFER_AB_NUM_PROMPTS=$N_PROF PEGAINFER_AB_PROFILE_RANGE=1 \
    "${NSYS[@]}" -o "$OUT/ours_on" "$TEST_BIN" --nocapture --test-threads=1 2>&1 | tail -20
  echo "== nsys: vLLM spec on ($N_PROF prompts)"
  "${NSYS[@]}" -o "$OUT/vllm_on" "$VLLM_PY" tools/bench/eagle3_vllm_ab.py --model "$MODEL" --draft "$DRAFT" \
    --ab-dir "$OUT" --mode on --num-prompts "$N_PROF" --profile-range 2>&1 | tail -20
  for t in ours_on vllm_on; do
    nsys export --type sqlite --force-overwrite=true -o "$OUT/$t.sqlite" "$OUT/$t.nsys-rep" >/dev/null
  done
  GAMMA=$("$VLLM_PY" -c "import json;print(json.load(open('$OUT/ab_config.json'))['gamma'])")
  "$VLLM_PY" tools/bench/eagle3_gap_report.py --gamma "$GAMMA" \
    --trace "pegainfer=$OUT/ours_on.sqlite:$OUT/pegainfer_on_profiled.json" \
    --trace "vllm=$OUT/vllm_on.sqlite:$OUT/vllm_on_profiled.json" \
    --ab "pegainfer=$OUT/pegainfer_on.json" --ab "vllm=$OUT/vllm_on.json" \
    $([[ -f $OUT/ours_ab.log ]] && echo --ours-log "$OUT/ours_ab.log") >"$OUT/report.md"
fi

echo
echo "== A/B summary (γ from ab_config.json)"
"$VLLM_PY" - "$OUT" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
for name in ("pegainfer_off", "pegainfer_on", "vllm_off", "vllm_on"):
    p = out / f"{name}.json"
    if p.exists():
        r = json.loads(p.read_text())
        tau = r.get("acceptance", {}).get("tau")
        print(f"  {name:<14} decode {r['decode_tok_s']:7.1f} tok/s  TPOT {r['tpot_ms']:6.2f} ms  TTFT {r['mean_ttft_ms']:6.1f} ms"
              + (f"  τ {tau:.2f}" if tau else ""))
EOF
echo "report: $OUT/report.md"
