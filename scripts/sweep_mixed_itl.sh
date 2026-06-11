#!/usr/bin/env bash
# Mixed-load ITL sweep cube: qps × warm_frac × prompt-length (#244).
# Background held constant at 4-way / 512-prompt / 1024-out so one decode-only
# baseline covers every cell and all three prompts fit the 16 GB KV budget.
# (16k excluded — prefill activation OOMs on 16 GB; 12k is the feasible ceiling.)
set -u
export CUDA_HOME=/opt/cuda
export LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64
export RUST_LOG=error
BIN=./target/release/bench_serving
MODEL=models/Qwen3-4B
OUT=/tmp/sweep; rm -rf "$OUT"; mkdir -p "$OUT"
BG="--bg-prompt-len 512 --bg-concurrency 4 --bg-output-len 1024"

PROMPTS="4096 8192 12288"
QPS="0.25 0.5 1.0"
WARM="0.0 0.5 1.0"

COOLDOWN=${COOLDOWN:-30}   # seconds idle between cells so each starts thermally comparable

ninj() { case "$1" in 0.25) echo 3;; 0.5) echo 5;; 1.0) echo 8;; *) echo 5;; esac; }

run() { # prompt qps warm extra...
  local p=$1 q=$2 w=$3; shift 3
  local tag="p${p}_q${q}_w${w}"
  sleep "$COOLDOWN"   # let the GPU clocks recover before measuring (avoids throttle bias)
  local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  echo ">> $tag (start temp ${t}C)"
  $BIN --model-path $MODEL --format json --out "$OUT/$tag.json" \
    mixed $BG --inj-prompt-len "$p" --inj-output-len 1 \
    --qps "$q" --num-injections "$(ninj "$q")" --warmup 5 \
    --inj-warm-frac "$w" "$@" >/dev/null 2>>"$OUT/sweep.err" \
    && echo "   ok" || echo "   FAILED (see $OUT/sweep.err)"
}

# One cell WITH baseline (decode-only control); reused for all (same bg config).
run 4096 0.5 0.0
for p in $PROMPTS; do for q in $QPS; do for w in $WARM; do
  [ "$p" = "4096" ] && [ "$q" = "0.5" ] && [ "$w" = "0.0" ] && continue   # already ran
  run "$p" "$q" "$w" --skip-baseline
done; done; done

echo "=== AGGREGATE ==="
python3 - "$OUT" <<'PY'
import json, glob, sys
d=sys.argv[1]; cells={}; pfill={}; base=None
for f in glob.glob(d+'/*.json'):
    j=json.load(open(f)); c=j['config']; m=j['mixed_itl']
    if j.get('baseline_itl'): base=j['baseline_itl']
    sat = len(j['warnings'])>0
    cells[(c['inj_prompt_len'], c['qps'], c['inj_warm_frac'])] = (m['all']['p99_ms'], sat)
    cold=sorted(r['prefill_ms'] for r in j['injections'] if not r['warm'])
    if c['inj_warm_frac']==0.0 and cold:
        pfill[(c['inj_prompt_len'], c['qps'])] = cold[len(cold)//2]
if base: print(f"baseline (decode-only, 4-way): p50={base['p50_ms']:.1f}  p99={base['p99_ms']:.1f}\n")
prompts=[4096,8192,12288]; qpss=[0.25,0.5,1.0]; warms=[0.0,0.5,1.0]
plabel={4096:'4k',8192:'8k',12288:'12k'}
for q in qpss:
    print(f"=== qps={q}  (ITL p99 ms; * = saturated) ===")
    print(f"{'prompt':>7} | " + " | ".join(f"warm={w}".rjust(11) for w in warms))
    for p in prompts:
        cs=[]
        for w in warms:
            v=cells.get((p,q,w))
            cs.append((f"{v[0]:.1f}{'*' if v[1] else ''}").rjust(11) if v else "n/a".rjust(11))
        print(f"{plabel[p]:>7} | " + " | ".join(cs))
    print()
print("=== cold prefill median (ms) — throttle check (12k ~2235 when cool) ===")
print(f"{'prompt':>7} | " + " | ".join(f"qps={q}".rjust(9) for q in qpss))
for p in prompts:
    print(f"{plabel[p]:>7} | " + " | ".join((f"{pfill[(p,q)]:.0f}".rjust(9) if (p,q) in pfill else "n/a".rjust(9)) for q in qpss))
PY
echo "=== DONE ==="
