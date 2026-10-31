"""Generate a synthetic but realistic agent regression dataset.

Two buckets of regressions, because a model swap fails in two different ways:

1. **structural** (6 modes) — missing tool, wrong tool, missing fact, token
   explosion, latency blowup, error span. Deterministic checks can catch these.
2. **semantic** (1 mode) — the output still contains the expected fact, but the
   answer is hedged/unreliable ("source unavailable, may be stale"). Only a
   task-aware judge can catch this; a substring check cannot.

Ground truth is constructed, so recall and false-alarm rate are measurable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

TASKS = 120
STRUCTURAL_REGRESSIONS = 30
SEMANTIC_REGRESSIONS = 10
SEED = 42

STRUCTURAL_MODES = [
    "missing_tool",
    "wrong_tool",
    "missing_fact",
    "token_explosion",
    "latency",
    "error_span",
]
SEMANTIC_MODE = "plausible_wrong"


def any_value(value):
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    return {"stringValue": str(value)}


def span(trace_id, span_id, name, start_ms, end_ms, attributes, parent=None):
    item = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": str(int(start_ms * 1_000_000)),
        "endTimeUnixNano": str(int(end_ms * 1_000_000)),
        "attributes": [
            {"key": key, "value": any_value(value)} for key, value in attributes.items()
        ],
    }
    if parent:
        item["parentSpanId"] = parent
    return item


def build_trace(task_id, fact, failure, duration_ms, input_tokens, output_tokens):
    trace_id = f"trace-{task_id}"
    spans = [
        span(
            trace_id, "s-1", "plan", 0, 100,
            {
                "aeh.task.id": task_id,
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
                "gen_ai.completion": "Planning the lookup.",
            },
        )
    ]

    if failure == "missing_tool":
        pass
    elif failure == "wrong_tool":
        spans.append(span(trace_id, "s-2", "search_web", 100, 200,
                          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_web"}, "s-1"))
    elif failure == "error_span":
        spans.append(span(trace_id, "s-2", "search_docs", 100, 200,
                          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_docs",
                           "error": True, "error.type": "TimeoutError"}, "s-1"))
    else:
        spans.append(span(trace_id, "s-2", "search_docs", 100, 200,
                          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_docs"}, "s-1"))

    if failure == "missing_fact":
        output = "I could not find the result."
    elif failure == "plausible_wrong":
        output = f"The result is {fact}, but the source was unavailable so this may be stale."
    else:
        output = f"The result is {fact}."

    spans.append(span(trace_id, "s-3", "answer", 200, duration_ms,
                      {"gen_ai.operation.name": "chat", "gen_ai.completion": output}, "s-1"))
    return spans


def wrap(resource_spans):
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "bench-agent"}}]},
                "scopeSpans": [{"scope": {"name": "bench"}, "spans": spans}],
            }
            for spans in resource_spans
        ]
    }


def main():
    random.seed(SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    structural_idx = set(random.sample(range(TASKS), STRUCTURAL_REGRESSIONS))
    remaining = [i for i in range(TASKS) if i not in structural_idx]
    semantic_idx = set(random.sample(remaining, SEMANTIC_REGRESSIONS))

    cases, baseline, degraded, truth = [], [], [], []
    mode_cursor = 0

    for index in range(TASKS):
        task_id = f"bench-{index:03d}"
        fact = f"fact{index:03d}"
        cases.append({
            "id": task_id,
            "task": f"Look up {fact} and report it",
            "rubric": {
                "expect_tools": ["search_docs"],
                "output_contains": [fact],
                "max_total_tokens": 800,
                "max_duration_ms": 4000,
            },
            "judge_threshold": 0.0,
        })
        baseline.append(build_trace(task_id, fact, None, 1000, 120, 40))

        if index in structural_idx:
            failure = STRUCTURAL_MODES[mode_cursor % len(STRUCTURAL_MODES)]
            mode_cursor += 1
            if failure == "latency":
                spans = build_trace(task_id, fact, None, 6000, 120, 40)
            elif failure == "token_explosion":
                spans = build_trace(task_id, fact, None, 3000, 900, 400)
            else:
                spans = build_trace(task_id, fact, failure, 1000, 120, 40)
            degraded.append(spans)
            truth.append({"task_id": task_id, "bucket": "structural", "failure": failure})
        elif index in semantic_idx:
            degraded.append(build_trace(task_id, fact, SEMANTIC_MODE, 1000, 120, 40))
            truth.append({"task_id": task_id, "bucket": "semantic", "failure": SEMANTIC_MODE})
        else:
            degraded.append(build_trace(task_id, fact, None, 1000, 120, 40))

    (DATA / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    (DATA / "baseline.json").write_text(json.dumps(wrap(baseline), indent=1), encoding="utf-8")
    (DATA / "degraded.json").write_text(json.dumps(wrap(degraded), indent=1), encoding="utf-8")
    (DATA / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    print(f"tasks={TASKS} regressions={len(truth)} "
          f"(structural={sum(1 for t in truth if t['bucket']=='structural')}, "
          f"semantic={sum(1 for t in truth if t['bucket']=='semantic')})")
    for mode in STRUCTURAL_MODES + [SEMANTIC_MODE]:
        print(f"  {mode:16s} {sum(1 for t in truth if t['failure'] == mode)}")


if __name__ == "__main__":
    main()
