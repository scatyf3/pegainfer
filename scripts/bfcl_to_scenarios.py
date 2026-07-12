#!/usr/bin/env python3
"""Convert BFCL (Berkeley Function-Calling Leaderboard) non-executable entries
into the scenario JSONL that `scripts/toolcall_trace.py` consumes (issue #655).

BFCL ships two parallel files per category:
  * a prompt file, e.g. BFCL_v3_simple.json  — one JSON record per line:
      {"id": "simple_0",
       "question": [[{"role": "user", "content": "..."}]],
       "function": [{"name": ..., "description": ..., "parameters": {...}}]}
  * a possible-answer file, e.g. possible_answer/BFCL_v3_simple.json:
      {"id": "simple_0",
       "ground_truth": [{"func_name": {"param": [acceptable, values], ...}}]}

This merges them by id and emits, per line:
  {"id", "messages", "tools" (OpenAI function format), "expect_tool",
   "ground_truth": {func: {param: [values]}}, "tool_choice": "auto"}

BFCL's schema dialect differs from JSON Schema (types `dict`/`float`/`tuple`,
etc.); parameter types are normalized so guided-decoding backends and the
harness's validator accept them.

Best suited to the single-call categories `simple` and `multiple` (the harness
scores the first tool call). `parallel` / `parallel_multiple` need multi-call
scoring — convert them if you like, but treat their accuracy as a lower bound.

Usage:
    # get BFCL data:  git clone https://github.com/ShishirPatil/gorilla
    #   data lives in gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/
    python3 scripts/bfcl_to_scenarios.py \
        --prompt   gorilla/.../data/BFCL_v3_simple.json \
        --answers  gorilla/.../data/possible_answer/BFCL_v3_simple.json \
        --out data/bfcl_simple.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

# BFCL type dialect -> JSON Schema types.
TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "double": "number",
    "integer": "integer",
    "int": "integer",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "tuple": "array",
    "number": "number",
    "object": "object",
    "any": None,  # drop — accept anything
}


def sanitize_name(name: str) -> str:
    """Make a BFCL function name a valid OpenAI tool name (`^[A-Za-z0-9_-]+$`).
    BFCL uses dotted names like `triangle_properties.get`, which vLLM/SGLang
    reject. Replace every invalid char with `_`. Applied identically to tool
    names AND ground-truth keys so scoring stays aligned (this is what BFCL's own
    harness does before inference)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def normalize_schema(node: Any) -> Any:
    """Recursively rewrite BFCL parameter schemas into JSON Schema."""
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            mapped = TYPE_MAP.get(value.lower(), value)
            if mapped is not None:
                out["type"] = mapped  # `any` maps to None -> drop the type entirely
            continue
        if key == "properties" and isinstance(value, dict):
            # each value under `properties` is itself a schema keyed by param name
            out["properties"] = {name: normalize_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out["items"] = normalize_schema(value)
        else:
            out[key] = value
    return out


def load_jsonl(path: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            records[record["id"]] = record
    return records


def to_openai_tool(func: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": sanitize_name(func["name"]),
            "description": func.get("description", ""),
            "parameters": normalize_schema(func.get("parameters", {"type": "object", "properties": {}})),
        },
    }


def merge_ground_truth(gt_list: list[dict[str, Any]] | None) -> dict[str, dict[str, list[Any]]] | None:
    """BFCL ground_truth is a list (one dict per expected call). Merge into
    {func: {param: [values]}}; later calls to the same func win on conflict,
    which is fine for the single-call categories this converter targets."""
    if not gt_list:
        return None
    merged: dict[str, dict[str, list[Any]]] = {}
    for entry in gt_list:
        for func_name, params in entry.items():
            # sanitize keys identically to tool names so match_ground_truth aligns
            merged.setdefault(sanitize_name(func_name), {}).update(params)
    return merged or None


def messages_from_question(question: Any) -> list[dict[str, Any]]:
    """BFCL `question` is a list of turns; each turn is a list of messages. The
    single-turn categories carry one turn — take it. Multi-turn categories are
    out of scope; take the first turn and let the harness score that."""
    if isinstance(question, list) and question and isinstance(question[0], list):
        return question[0]
    if isinstance(question, list):
        return question
    raise ValueError(f"unexpected question shape: {type(question)}")


def convert(prompt_path: str, answers_path: str | None, limit: int | None) -> list[dict[str, Any]]:
    prompts = load_jsonl(prompt_path)
    answers = load_jsonl(answers_path) if answers_path else {}
    scenarios: list[dict[str, Any]] = []
    for entry_id, record in prompts.items():
        funcs = record.get("function", [])
        if not funcs:
            continue
        tools = [to_openai_tool(f) for f in funcs]
        ground_truth = merge_ground_truth((answers.get(entry_id) or {}).get("ground_truth"))
        # expect_tool: the ground-truth function when unambiguous, else the sole
        # candidate, else leave unset (multi-candidate with no answer).
        expect_tool: str | None = None
        if ground_truth and len(ground_truth) == 1:
            expect_tool = next(iter(ground_truth))  # already sanitized in merge_ground_truth
        elif len(funcs) == 1:
            expect_tool = sanitize_name(funcs[0]["name"])
        scenario = {
            "id": entry_id,
            "messages": messages_from_question(record["question"]),
            "tools": tools,
            "tool_choice": "auto",
        }
        if expect_tool:
            scenario["expect_tool"] = expect_tool
        if ground_truth:
            scenario["ground_truth"] = ground_truth
        scenarios.append(scenario)
        if limit and len(scenarios) >= limit:
            break
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="BFCL prompt file (BFCL_v3_<category>.json)")
    parser.add_argument("--answers", help="BFCL possible_answer file (enables ground-truth accuracy)")
    parser.add_argument("--out", required=True, help="output scenario JSONL")
    parser.add_argument("--limit", type=int, help="cap number of scenarios")
    args = parser.parse_args()

    scenarios = convert(args.prompt, args.answers, args.limit)
    if not scenarios:
        raise SystemExit(f"{args.prompt}: produced no scenarios")
    with open(args.out, "w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario) + "\n")
    with_gt = sum(1 for s in scenarios if "ground_truth" in s)
    print(f"wrote {len(scenarios)} scenarios to {args.out} ({with_gt} with ground truth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
