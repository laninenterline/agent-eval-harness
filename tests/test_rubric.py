import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.models import Span, Trace  # noqa: E402
from agent_eval_harness.rubric import Rubric  # noqa: E402


def make_trace(tools=("get_weather",), output="Shanghai is warm.", errored=False):
    spans = [
        Span(
            span_id="1",
            trace_id="t1",
            name="chat",
            start_ns=0,
            end_ns=1_000_000,
            attributes={"gen_ai.operation.name": "chat", "gen_ai.completion": output},
        )
    ]
    for index, tool in enumerate(tools):
        spans.append(
            Span(
                span_id=f"tool-{index}",
                trace_id="t1",
                name=f"execute_tool {tool}",
                start_ns=index,
                end_ns=index + 1,
                attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool, "error": errored},
            )
        )
    spans[0].attributes["error"] = errored
    return Trace(trace_id="t1", spans=spans)


class TestRubric(unittest.TestCase):
    def test_all_checks_pass(self):
        rubric = Rubric(expect_tools=["get_weather"], output_contains=["Shanghai"], max_duration_ms=10)
        results = rubric.evaluate(make_trace())
        self.assertTrue(all(r.passed for r in results), [r.name for r in results if not r.passed])

    def test_missing_tool_fails(self):
        rubric = Rubric(expect_tools=["get_forecast"])
        results = rubric.evaluate(make_trace())
        self.assertFalse(results[0].passed)

    def test_missing_output_fails(self):
        rubric = Rubric(output_contains=["Beijing"])
        results = rubric.evaluate(make_trace())
        self.assertFalse(results[0].passed)

    def test_error_span_fails_when_forbidden(self):
        rubric = Rubric(forbid_errors=True)
        results = rubric.evaluate(make_trace(errored=True))
        self.assertFalse(results[-1].passed)

    def test_error_span_allowed_when_configured(self):
        rubric = Rubric(forbid_errors=False)
        results = rubric.evaluate(make_trace(errored=True))
        self.assertTrue(all(r.passed for r in results))


if __name__ == "__main__":
    unittest.main()
