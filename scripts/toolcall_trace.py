#!/usr/bin/env python3
"""A/B a tool-calling model with vs. without constrained (guided) decoding —
throughput and accuracy — over an OpenAI-compatible endpoint (issue #655).

The empirical question behind #655: grammar-constrained decoding can make a
small model emit valid tool-call JSON, but it is widely reported to cost
throughput. Is that trade worth it for our target (Qwen3-4B)? Answer it by
measuring, on an engine that *already* implements constrained decoding
(vLLM or SGLang), the two axes at once:

  * accuracy   — did the model call the right function with well-formed,
                 schema-valid, ground-truth-matching arguments?
  * throughput — requests/sec and completion-tokens/sec under load.

Run the harness twice against the same server — once `--constrained off`, once
`--constrained required` — and diff the two JSON summaries. That delta is the
input to the "should openinfer build constrained decoding" decision.

Recommended dataset: BFCL (Berkeley Function-Calling Leaderboard) non-executable
categories — `simple`, `multiple`, `parallel`. Convert to this harness's scenario
JSONL with `scripts/bfcl_to_scenarios.py` (it carries BFCL ground truth so the
accuracy check is an AST-style value match, not just schema validity). The bundled
`test_data/toolcall_scenarios.jsonl` is a 12-case smoke set for wiring checks.

Constrained-decoding toggle (engine-agnostic — both vLLM and SGLang gate guided
decoding on `tool_choice`):
  --constrained off        tool_choice="auto"      free generation; parser extracts
  --constrained required   tool_choice="required"  engine forces *some* valid call
  --constrained named      tool_choice=<expect>    engine forces the expected function
Guided decoding is the difference between `off` and `required`/`named`; tools,
parser, and prompt are otherwise identical, so the throughput/accuracy delta is
attributable to the constraint.

Outcome buckets (worst→best): error, no_tool_call, leaked_tool_call,
malformed_json, schema_violation, wrong_tool, value_mismatch, ok.
  value_mismatch — right function, schema-valid args, but a value disagrees with
                   BFCL ground truth (only reachable when the scenario carries it).
accuracy = ok / total.

Usage:
    # vLLM:  vllm serve Qwen/Qwen3-4B --enable-auto-tool-choice \
    #            --tool-call-parser hermes --port 8000
    # SGLang: python -m sglang.launch_server --model-path Qwen/Qwen3-4B \
    #            --tool-call-parser qwen25 --port 8000
    python3 scripts/toolcall_trace.py --base-url http://127.0.0.1:8000 \
        --model Qwen/Qwen3-4B --dataset data/bfcl_simple.jsonl \
        --concurrency 32 --constrained off      --trace-out off.jsonl
    python3 scripts/toolcall_trace.py --base-url http://127.0.0.1:8000 \
        --model Qwen/Qwen3-4B --dataset data/bfcl_simple.jsonl \
        --concurrency 32 --constrained required --trace-out on.jsonl

The JSON-schema validator is intentionally small (type / required / properties /
enum / array items) and errs toward NOT flagging constructs it doesn't
understand — enough to separate malformed from schema-violating, not a
spec-complete validator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# Outcome buckets, ordered worst-to-best for stable report rendering.
OUTCOMES = [
    "error",
    "no_tool_call",
    "leaked_tool_call",
    "malformed_json",
    "schema_violation",
    "wrong_tool",
    "value_mismatch",
    "ok",
]

# Hermes / Qwen-style inline tool call: <tool_call>{...}</tool_call>
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class Scenario:
    id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    expect_tool: str | None = None
    tool_choice: Any = "auto"
    system: str | None = None
    # BFCL ground truth: {func_name: {param: [allowed values, ...], ...}}.
    ground_truth: dict[str, dict[str, list[Any]]] | None = None

    def request_messages(self) -> list[dict[str, Any]]:
        if self.system:
            return [{"role": "system", "content": self.system}, *self.messages]
        return list(self.messages)

    def schema_for(self, tool_name: str) -> dict[str, Any] | None:
        for tool in self.tools:
            fn = tool.get("function", tool)
            if fn.get("name") == tool_name:
                return fn.get("parameters")
        return None


@dataclass
class Sample:
    scenario_id: str
    outcome: str
    detail: str
    tool_name: str | None = None
    arguments_raw: str | None = None
    source: str | None = None  # "parsed" | "content" | None
    raw_content: str | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    completion_tokens: int | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def load_dataset(path: str) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            scenarios.append(
                Scenario(
                    id=record.get("id", f"line{line_no}"),
                    messages=record["messages"],
                    tools=record["tools"],
                    expect_tool=record.get("expect_tool"),
                    tool_choice=record.get("tool_choice", "auto"),
                    system=record.get("system"),
                    ground_truth=record.get("ground_truth"),
                )
            )
    if not scenarios:
        raise SystemExit(f"{path}: no scenarios found")
    return scenarios


def resolve_tool_choice(scenario: Scenario, constrained: str) -> Any:
    """Map the --constrained knob to an OpenAI `tool_choice`. Both vLLM and
    SGLang engage guided decoding for `required` / named choices and skip it for
    `auto`, so this is the engine-agnostic on/off switch."""
    if constrained == "off":
        return "auto"
    if constrained == "required":
        return "required"
    if constrained == "named":
        if not scenario.expect_tool:
            return "required"  # nothing to name; fall back to the weaker constraint
        return {"type": "function", "function": {"name": scenario.expect_tool}}
    return scenario.tool_choice


def post_chat(
    base_url: str,
    model: str,
    scenario: Scenario,
    tool_choice: Any,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": scenario.request_messages(),
        "tools": scenario.tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        body.update(extra_body)
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return payload, latency_ms


# ---------------------------------------------------------------------------
# Minimal JSON-schema validation. Returns a list of human-readable violations;
# empty means "no violation found by the checks we implement".
# ---------------------------------------------------------------------------
def validate_schema(instance: Any, schema: dict[str, Any] | None, path: str = "") -> list[str]:
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    expected = schema.get("type")

    if expected and not _type_matches(instance, expected):
        errors.append(f"{path or '<root>'}: expected {expected}, got {_json_type(instance)}")
        return errors  # further checks assume the type held

    if expected == "object" or isinstance(instance, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if not isinstance(instance, dict) or req not in instance:
                errors.append(f"{path or '<root>'}: missing required field '{req}'")
        if isinstance(instance, dict):
            for key, sub_schema in props.items():
                if key in instance:
                    errors.extend(validate_schema(instance[key], sub_schema, f"{path}.{key}" if path else key))

    if expected == "array" and isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(instance):
                errors.extend(validate_schema(item, item_schema, f"{path}[{idx}]"))

    enum = schema.get("enum")
    if enum is not None and instance not in enum:
        errors.append(f"{path or '<root>'}: value {instance!r} not in enum {enum}")

    return errors


def _type_matches(instance: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    return any(_one_type_matches(instance, t) for t in types)


def _one_type_matches(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    return True  # unknown type keyword: don't flag


def _json_type(instance: Any) -> str:
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, dict):
        return "object"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if instance is None:
        return "null"
    return type(instance).__name__


# ---------------------------------------------------------------------------
# BFCL-style ground-truth matching. BFCL records, per function, the set of
# acceptable values for each param: {func: {param: [v1, v2, ...]}}. A call is
# correct when the function matches and every ground-truth param's value is in
# its acceptable set (a param whose acceptable set contains "" is optional).
# ---------------------------------------------------------------------------
def match_ground_truth(
    tool_name: str, args: Any, ground_truth: dict[str, dict[str, list[Any]]]
) -> list[str]:
    if tool_name not in ground_truth:
        return [f"function '{tool_name}' not in ground truth {list(ground_truth)}"]
    if not isinstance(args, dict):
        return [f"arguments are {_json_type(args)}, expected object"]
    mismatches: list[str] = []
    for param, allowed in ground_truth[tool_name].items():
        optional = isinstance(allowed, list) and "" in allowed
        if param not in args:
            if not optional:
                mismatches.append(f"missing param '{param}'")
            continue
        if not _value_in(args[param], allowed):
            mismatches.append(f"{param}={args[param]!r} not in {allowed}")
    return mismatches


def _value_in(value: Any, allowed: list[Any]) -> bool:
    for candidate in allowed:
        if value == candidate:
            return True
        # light normalization: BFCL stores some numerics/bools as strings
        if isinstance(candidate, str) and str(value).strip().lower() == candidate.strip().lower():
            return True
        try:
            if isinstance(value, (int, float)) and float(value) == float(candidate):
                return True
        except (TypeError, ValueError):
            pass
    return False


# ---------------------------------------------------------------------------
# Tool-call extraction: structured first, then content fallback.
# ---------------------------------------------------------------------------
def extract_from_content(content: str | None) -> tuple[str, str] | None:
    """Best-effort recovery of a tool call the parser missed. Returns
    (tool_name, arguments_raw) or None. Conservative on purpose."""
    if not content:
        return None
    candidates: list[str] = [m.group(1) for m in TOOL_CALL_TAG_RE.finditer(content)]
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters"))
        if isinstance(name, str):
            if isinstance(args, (dict, list)):
                return name, json.dumps(args)
            if isinstance(args, str):
                return name, args
            return name, "{}"
    return None


def classify(scenario: Scenario, payload: dict[str, Any], latency_ms: float | None) -> Sample:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason")
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    completion_tokens = (payload.get("usage") or {}).get("completion_tokens")

    name: str | None = None
    args_raw: str | None = None
    source: str | None = None

    if tool_calls:
        fn = tool_calls[0].get("function", {})
        name = fn.get("name")
        args_raw = fn.get("arguments")
        source = "parsed"
    else:
        recovered = extract_from_content(content)
        if recovered is not None:
            name, args_raw = recovered
            source = "content"

    base = dict(
        scenario_id=scenario.id,
        tool_name=name,
        arguments_raw=args_raw,
        source=source,
        raw_content=content,
        finish_reason=finish,
        latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
        completion_tokens=completion_tokens,
    )

    if name is None:
        return Sample(outcome="no_tool_call", detail="model produced no tool call", **base)

    if source == "content":
        return Sample(
            outcome="leaked_tool_call",
            detail="tool call present in content but not parsed into tool_calls",
            **base,
        )

    if scenario.expect_tool and name != scenario.expect_tool:
        return Sample(
            outcome="wrong_tool",
            detail=f"called '{name}', expected '{scenario.expect_tool}'",
            **base,
        )

    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except json.JSONDecodeError as exc:
        return Sample(outcome="malformed_json", detail=f"arguments not valid JSON: {exc}", **base)

    schema = scenario.schema_for(name)
    violations = validate_schema(args, schema)
    if violations:
        return Sample(outcome="schema_violation", detail="; ".join(violations[:4]), **base)

    if scenario.ground_truth:
        mismatches = match_ground_truth(name, args, scenario.ground_truth)
        if mismatches:
            return Sample(outcome="value_mismatch", detail="; ".join(mismatches[:4]), **base)

    return Sample(outcome="ok", detail="well-formed tool call", **base)


@dataclass
class Report:
    counts: dict[str, int] = field(default_factory=lambda: {o: 0 for o in OUTCOMES})
    per_scenario: dict[str, dict[str, int]] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    completion_tokens: int = 0
    total: int = 0
    wall_s: float = 0.0

    def add(self, sample: Sample) -> None:
        self.counts[sample.outcome] += 1
        self.per_scenario.setdefault(sample.scenario_id, {o: 0 for o in OUTCOMES})
        self.per_scenario[sample.scenario_id][sample.outcome] += 1
        self.total += 1
        if sample.latency_ms is not None:
            self.latencies_ms.append(sample.latency_ms)
        if sample.completion_tokens:
            self.completion_tokens += sample.completion_tokens

    def accuracy(self) -> float:
        return self.counts["ok"] / self.total if self.total else 0.0

    def _pct(self, q: float) -> float | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return round(ordered[idx], 2)

    def throughput(self) -> dict[str, Any]:
        rps = self.total / self.wall_s if self.wall_s else 0.0
        tps = self.completion_tokens / self.wall_s if self.wall_s else 0.0
        mean = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
        return {
            "wall_s": round(self.wall_s, 3),
            "requests_per_s": round(rps, 3),
            "completion_tokens_per_s": round(tps, 2),
            "latency_ms": {
                "mean": round(mean, 2) if mean is not None else None,
                "p50": self._pct(0.50),
                "p95": self._pct(0.95),
                "p99": self._pct(0.99),
            },
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "total_samples": self.total,
            "accuracy": round(self.accuracy(), 4),
            "error_rate": round(1.0 - self.accuracy(), 4),
            "counts": self.counts,
            "throughput": self.throughput(),
            "per_scenario": self.per_scenario,
        }


def render(report: Report, constrained: str) -> str:
    tp = report.throughput()
    lines = [
        "",
        f"tool-call A/B summary  (constrained={constrained}, {report.total} samples)",
        "=" * 56,
    ]
    for outcome in reversed(OUTCOMES):
        n = report.counts[outcome]
        pct = 100.0 * n / report.total if report.total else 0.0
        lines.append(f"  {outcome:<18} {n:>5}  {pct:5.1f}%")
    lines.append("-" * 56)
    lines.append(f"  accuracy: {report.accuracy():.1%}   error-rate: {1 - report.accuracy():.1%}")
    lines.append(
        f"  throughput: {tp['requests_per_s']} req/s, {tp['completion_tokens_per_s']} tok/s"
        f"   latency p50/p95 = {tp['latency_ms']['p50']}/{tp['latency_ms']['p95']} ms"
    )
    worst = sorted(
        report.per_scenario.items(),
        key=lambda kv: kv[1]["ok"] / max(1, sum(kv[1].values())),
    )[:5]
    if worst:
        lines.append("")
        lines.append("  weakest scenarios (by accuracy):")
        for sid, counts in worst:
            n = sum(counts.values())
            lines.append(f"    {sid:<26} {counts['ok']}/{n} ok")
    return "\n".join(lines)


def run_one(args, scenario: Scenario) -> Sample:
    tool_choice = resolve_tool_choice(scenario, args.constrained)
    try:
        payload, latency_ms = post_chat(
            args.base_url, args.model, scenario, tool_choice,
            args.temperature, args.max_tokens, args.timeout, args._extra_body,
        )
        return classify(scenario, payload, latency_ms)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        return Sample(scenario.id, "error", f"HTTP {exc.code}", error=detail)
    except Exception as exc:  # noqa: BLE001 - transport failures are data
        return Sample(scenario.id, "error", str(exc), error=str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="test_data/toolcall_scenarios.jsonl")
    parser.add_argument("--repeat", type=int, default=1, help="samples per scenario")
    parser.add_argument(
        "--constrained",
        choices=["off", "required", "named"],
        default="off",
        help="guided-decoding toggle via tool_choice (off=auto)",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="in-flight requests (for throughput)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--trace-out", help="write full per-sample traces as JSONL")
    parser.add_argument("--fail-under", type=float, help="exit 1 if accuracy is below this (0..1)")
    parser.add_argument(
        "--extra-body",
        help='JSON merged into each request body, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'',
    )
    parser.add_argument("--quiet", action="store_true", help="only print the JSON summary")
    args = parser.parse_args()
    args._extra_body = json.loads(args.extra_body) if args.extra_body else None

    scenarios = load_dataset(args.dataset)
    jobs = [scenario for scenario in scenarios for _ in range(args.repeat)]
    report = Report()
    trace_handle = open(args.trace_out, "w", encoding="utf-8") if args.trace_out else None

    start = time.perf_counter()
    try:
        if args.concurrency <= 1:
            results = (run_one(args, scenario) for scenario in jobs)
            for sample in results:
                _record(report, trace_handle, sample, args.quiet)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for sample in pool.map(lambda s: run_one(args, s), jobs):
                    _record(report, trace_handle, sample, args.quiet)
    finally:
        report.wall_s = time.perf_counter() - start
        if trace_handle:
            trace_handle.close()

    if not args.quiet:
        print(render(report, args.constrained))
    summary = report.to_json()
    summary["constrained"] = args.constrained
    summary["concurrency"] = args.concurrency
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.fail_under is not None and report.accuracy() < args.fail_under:
        print(f"accuracy {report.accuracy():.3f} < --fail-under {args.fail_under}", file=sys.stderr)
        return 1
    return 0


def _record(report: Report, trace_handle, sample: Sample, quiet: bool) -> None:
    report.add(sample)
    if trace_handle:
        trace_handle.write(json.dumps(sample.to_json()) + "\n")
    if not quiet:
        marker = "OK " if sample.outcome == "ok" else "!! "
        print(f"{marker}{sample.scenario_id:<26} {sample.outcome:<18} {sample.detail}")


if __name__ == "__main__":
    sys.exit(main())
