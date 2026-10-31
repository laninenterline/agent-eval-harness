"""Zero-configuration demo.

`aeh demo` must work immediately after `pip install aehcli`: no files, no
network, no API keys. The dataset is small and intentionally includes one
failing case so the output shows what a real regression looks like.
"""

from __future__ import annotations

from typing import Any

from .judge import RuleJudge
from .otel import parse_otlp_json
from .runner import EvalCase, RunReport, run


def _span(
    trace_id: str,
    span_id: str,
    name: str,
    start_ms: float,
    end_ms: float,
    attributes: dict[str, Any],
    parent: str | None = None,
) -> dict[str, Any]:
    def any_value(value: Any) -> dict[str, Any]:
        if isinstance(value, bool):
            return {"boolValue": value}
        if isinstance(value, int):
            return {"intValue": str(value)}
        return {"stringValue": str(value)}

    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": str(int(start_ms * 1_000_000)),
        "endTimeUnixNano": str(int(end_ms * 1_000_000)),
        "attributes": [{"key": key, "value": any_value(value)} for key, value in attributes.items()],
    }
    if parent:
        span["parentSpanId"] = parent
    return span


CASES: list[dict[str, Any]] = [
    {
        "id": "demo-weather",
        "task": "Find the weather in Shanghai and summarize it for the user",
        "rubric": {
            "expect_tools": ["get_weather"],
            "output_contains": ["Shanghai"],
            "max_total_tokens": 2000,
        },
    },
    {
        "id": "demo-review",
        "task": "Review the pull request and list at least two issues",
        "rubric": {
            "expect_tools": ["read_file", "post_comment"],
            "output_contains": ["issues"],
            "max_duration_ms": 5000,
            "forbid_errors": False,
        },
    },
    {
        # Deliberately failing: a model swap doubled the token cost.
        "id": "demo-budget",
        "task": "Summarize the incident timeline and list the root cause",
        "rubric": {
            "expect_tools": ["read_logs"],
            "output_contains": ["root cause"],
            "max_total_tokens": 500,
        },
    },
]

_SPANS: list[dict[str, Any]] = [
    # demo-weather -> passes
    _span("t-weather", "w1", "chat", 0, 400,
          {"aeh.task.id": "demo-weather", "gen_ai.operation.name": "chat",
           "gen_ai.usage.input_tokens": 120, "gen_ai.usage.output_tokens": 60,
           "gen_ai.completion": "Let me check the weather."}),
    _span("t-weather", "w2", "get_weather", 400, 900,
          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather"}, "w1"),
    _span("t-weather", "w3", "chat", 900, 1400,
          {"gen_ai.operation.name": "chat",
           "gen_ai.completion": "The weather in Shanghai is 22C and sunny. Summarized for the user: warm and clear."}, "w1"),
    # demo-review -> passes
    _span("t-review", "r1", "chat", 0, 500,
          {"aeh.task.id": "demo-review", "gen_ai.operation.name": "chat",
           "gen_ai.usage.input_tokens": 80, "gen_ai.usage.output_tokens": 40,
           "gen_ai.completion": "I will read the diff first."}),
    _span("t-review", "r2", "read_file", 500, 900,
          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "read_file"}, "r1"),
    _span("t-review", "r3", "post_comment", 900, 1400,
          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "post_comment"}, "r1"),
    _span("t-review", "r4", "chat", 1400, 3200,
          {"gen_ai.operation.name": "chat",
           "gen_ai.completion": "Reviewed the pull request and listed two issues: missing tests, unclear error handling."}, "r1"),
    # demo-budget -> fails the token budget
    _span("t-budget", "b1", "chat", 0, 900,
          {"aeh.task.id": "demo-budget", "gen_ai.operation.name": "chat",
           "gen_ai.usage.input_tokens": 1400, "gen_ai.usage.output_tokens": 400,
           "gen_ai.completion": "Reading the incident log."}),
    _span("t-budget", "b2", "read_logs", 900, 1600,
          {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "read_logs"}, "b1"),
    _span("t-budget", "b3", "chat", 1600, 2400,
          {"gen_ai.operation.name": "chat",
           "gen_ai.completion": "The timeline shows a deploy at 14:02. The root cause was a stale config cache."}, "b1"),
]


def build_report() -> RunReport:
    """Run the built-in demo dataset deterministically (no network)."""
    cases = [EvalCase.from_dict(case) for case in CASES]
    traces = {trace.task_id: trace for trace in parse_otlp_json({"resourceSpans": [
        {"scopeSpans": [{"scope": {"name": "aeh-demo"}, "spans": _SPANS}]}
    ]})}
    return run(cases, traces, RuleJudge())


def intro() -> str:
    return (
        "This is a built-in demo: 3 tasks, 2 healthy runs, 1 token-budget regression.\n"
        "No files, no network, no API keys. Try `aeh demo --judge-model <model>`\n"
        "with a provider key from `aeh judges` to see an LLM judge in action.\n"
    )
