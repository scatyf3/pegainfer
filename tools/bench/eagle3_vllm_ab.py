#!/usr/bin/env python3
"""vLLM side of the EAGLE-3 GSM8K single-stream A/B.

Mirrors ``pegainfer-qwen3/tests/eagle3_gsm8k_ab.rs`` exactly:

* prompts: the token ids that harness dumped (``prompts.jsonl``), fed as
  ``prompt_token_ids`` so tokenization can't differ between engines;
* gamma: read from the harness's ``ab_config.json`` (``EAGLE3_CHAIN_LENGTH``) and
  passed as ``num_speculative_tokens``; a mismatch with ``--gamma`` aborts;
* greedy, ``ignore_eos``, fixed 256 tokens, bs=1, requests back to back, two
  discarded warm-ups, prefix caching off.

Decode time is isolated the same way as the Rust side reports it: per prompt we
time a ``max_tokens=1`` request (prefill + first token = TTFT) and the full
request; ``tpot = (t_full - t_first) / (n - 1)``.

Acceptance comes from vLLM's own spec-decode counters (``LLM.get_metrics``),
diffed around the measured loop.

``--profile-range`` brackets the measured loop with cudaProfilerStart/Stop for
``nsys --capture-range=cudaProfilerApi``. That needs the engine core in this
process, so it sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0``; take the A/B numbers
from a run without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--ab-dir", required=True, type=Path, help="PEGAINFER_AB_OUT of the Rust harness")
    p.add_argument("--mode", choices=["both", "off", "on"], default="both")
    p.add_argument("--gamma", type=int, default=None, help="must match ab_config.json if given")
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--profile-range", action="store_true")
    return p.parse_args()


def spec_counters(llm) -> dict[str, object]:
    """Snapshot vLLM's spec-decode counters (names as of the V1 metrics API)."""
    out: dict[str, object] = {}
    try:
        metrics = llm.get_metrics()
    except Exception as exc:  # older vLLM: no offline metrics API
        print(f"[warn] LLM.get_metrics unavailable: {exc}", file=sys.stderr)
        return out
    for m in metrics:
        if not m.name.startswith("vllm:spec_decode"):
            continue
        if hasattr(m, "values"):  # Vector (per-position)
            out[m.name] = list(m.values)
        elif hasattr(m, "value"):
            out[m.name] = m.value
    return out


def diff_counters(before: dict, after: dict) -> dict:
    d = {}
    for k, v in after.items():
        b = before.get(k)
        if isinstance(v, list):
            b = b or [0] * len(v)
            d[k] = [x - y for x, y in zip(v, b)]
        else:
            d[k] = v - (b or 0)
    return d


def acceptance_summary(d: dict, gamma: int) -> dict:
    drafts = d.get("vllm:spec_decode_num_drafts")
    accepted = d.get("vllm:spec_decode_num_accepted_tokens")
    per_pos = d.get("vllm:spec_decode_num_accepted_tokens_per_pos")
    if not drafts:
        return {"raw": d}
    return {
        "num_rounds": drafts,
        "tau": 1.0 + accepted / drafts,  # accepted drafts + the bonus token
        "accept_rate": accepted / (drafts * gamma),
        "accept_per_pos": [x / drafts for x in per_pos] if per_pos else None,
        "raw": d,
    }


def run_mode(args, spec: bool, gamma: int, prompts: list[list[int]], cfg: dict) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        max_model_len=cfg["max_model_len"] + cfg["generated_tokens"] + 64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        max_num_seqs=1,
        seed=0,
        disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": args.draft,
            "num_speculative_tokens": gamma,
        }
    llm = LLM(**kwargs)

    n = cfg["generated_tokens"]
    full = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=n)
    first = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1)

    def gen(tokens, params):
        t0 = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=tokens)], params, use_tqdm=False)
        return time.perf_counter() - t0, len(out[0].outputs[0].token_ids)

    for p in prompts[:2]:  # warm-up: cudagraph capture, torch.compile caches
        gen(p, full)

    ttfts = [gen(p, first)[0] for p in prompts]  # outside the profile range
    before = spec_counters(llm)
    if args.profile_range:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    totals, counts = [], []
    for p in prompts:
        dt, k = gen(p, full)
        totals.append(dt)
        counts.append(k)
    if args.profile_range:
        torch.cuda.cudart().cudaProfilerStop()
    after = spec_counters(llm)

    decode_s = sum(max(t - f, 0.0) for t, f in zip(totals, ttfts))
    decode_tokens = sum(k - 1 for k in counts)
    result = {
        "engine": "vllm",
        "vllm_version": __import__("vllm").__version__,
        "spec": spec,
        "gamma": gamma if spec else 0,
        "num_prompts": len(prompts),
        "generated_tokens": sum(counts),
        "e2e_tok_s": sum(counts) / sum(totals),
        "decode_tok_s": decode_tokens / decode_s,
        "tpot_ms": decode_s * 1e3 / decode_tokens,
        "mean_ttft_ms": 1e3 * sum(ttfts) / len(ttfts),
        "profiled": args.profile_range,
    }
    if spec:
        result["acceptance"] = acceptance_summary(diff_counters(before, after), gamma)
    del llm
    return result


def main() -> int:
    args = parse_args()
    if args.profile_range:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    cfg = json.loads((args.ab_dir / "ab_config.json").read_text())
    gamma = cfg["gamma"]
    if args.gamma is not None and args.gamma != gamma:
        sys.exit(f"--gamma {args.gamma} != pegainfer EAGLE3_CHAIN_LENGTH {gamma}; refusing an unaligned A/B")
    prompts = [json.loads(l)["prompt_token_ids"] for l in (args.ab_dir / "prompts.jsonl").read_text().splitlines() if l]
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]
    print(f"vLLM GSM8K A/B: {len(prompts)} prompts, γ={gamma}, {cfg['generated_tokens']} tokens/request, bs=1")

    modes = {"both": [False, True], "off": [False], "on": [True]}[args.mode]
    for spec in modes:
        r = run_mode(args, spec, gamma, prompts, cfg)
        name = f"vllm_{'on' if spec else 'off'}{'_profiled' if args.profile_range else ''}.json"
        (args.ab_dir / name).write_text(json.dumps(r, indent=2))
        acc = r.get("acceptance", {})
        tau = f"  τ {acc['tau']:.2f}" if "tau" in acc else ""
        print(
            f"  spec {'ON ' if spec else 'OFF'}  decode {r['decode_tok_s']:7.1f} tok/s  "
            f"TPOT {r['tpot_ms']:6.2f} ms  TTFT {r['mean_ttft_ms']:6.1f} ms{tau}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
