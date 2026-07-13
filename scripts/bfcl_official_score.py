#!/usr/bin/env python3
"""Re-score already-saved toolcall traces with the OFFICIAL BFCL AST checker.

No model, no server, no regeneration: reads bfcl_data/results/<cond>.jsonl
(scenario_id + tool_name + arguments_raw), rebuilds each call, and runs
bfcl_eval's ast_checker against the official ground_truth. Prints official
AST accuracy per condition next to our harness's own accuracy.
"""
import json
import sys
from pathlib import Path

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

ROOT = Path("bfcl_data")
MODEL_NAME = "gpt-4o-2024-11-20-FC"  # FC handler: underscore_to_dot=True, matches our
# native-tools setup where BFCL's `math.factorial` was sanitized to `math_factorial`.

# category -> (prompt file, possible_answer file, checker test_category)
CATS = {
    "simple_python": ("BFCL_v4_simple_python.json", "simple"),
    "multiple": ("BFCL_v4_multiple.json", "multiple"),
}


def load_jsonl_by_id(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["id"]] = d
    return out


def score(cat, cond):
    prompt_file, test_category = CATS[cat]
    prompts = load_jsonl_by_id(ROOT / prompt_file)
    answers = load_jsonl_by_id(ROOT / "possible_answer" / prompt_file)

    n = ok = 0
    fails = []
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    for line in open(ROOT / "results" / f"{cat}_{cond}{suffix}.jsonl"):
        r = json.loads(line)
        sid = r["scenario_id"]
        n += 1
        funcs = prompts[sid]["function"]
        gt = answers[sid]["ground_truth"]

        # rebuild the model call from our saved trace
        raw = r.get("arguments_raw")
        tool = r.get("tool_name")
        if not tool or raw is None:
            fails.append((sid, "no_tool_call"))
            continue
        try:
            args = json.loads(raw)
        except Exception:
            fails.append((sid, "malformed_json"))
            continue
        model_output = [{tool: args}]

        try:
            res = ast_checker(funcs, model_output, gt, Language.PYTHON, test_category, MODEL_NAME)
        except Exception as e:
            fails.append((sid, f"checker_err:{type(e).__name__}"))
            continue
        if res.get("valid"):
            ok += 1
        else:
            et = (res.get("error_type") or "invalid")
            fails.append((sid, et))
    return n, ok, fails


def throughput(cat, cond):
    d = json.load(open(ROOT / "results" / f"{cat}_{cond}.json"))
    t = d["throughput"]
    return t["requests_per_s"], t["completion_tokens_per_s"]


def main():
    hdr = f"{'category':<16}{'cond':<10}{'official_acc':>13}{'ok/total':>11}{'req/s':>9}{'tok/s':>10}"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for cat in CATS:
        for cond in ("off", "required"):
            n, ok, fails = score(cat, cond)
            acc = ok / n if n else 0.0
            rps, tps = throughput(cat, cond)
            summary[(cat, cond)] = (acc, ok, n, fails)
            print(f"{cat:<16}{cond:<10}{acc*100:>12.2f}%{f'{ok}/{n}':>11}{rps:>9.1f}{tps:>10.0f}")
    # dump failure-type histogram to stderr for sanity
    for (cat, cond), (_, _, _, fails) in summary.items():
        from collections import Counter
        h = Counter(t for _, t in fails)
        print(f"[{cat}/{cond}] fail types: {dict(h)}", file=sys.stderr)


if __name__ == "__main__":
    main()
