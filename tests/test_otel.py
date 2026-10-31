import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.otel import load_otlp_json  # noqa: E402


class TestOtelIngest(unittest.TestCase):
    def setUp(self):
        self.traces = load_otlp_json(ROOT / "examples" / "traces" / "otel_sample.json")
        self.by_task = {t.task_id: t for t in self.traces}

    def test_two_traces(self):
        self.assertEqual(len(self.traces), 2)

    def test_task_ids_extracted(self):
        self.assertIn("weather-agent-001", self.by_task)
        self.assertIn("review-002", self.by_task)

    def test_tool_calls(self):
        weather = self.by_task["weather-agent-001"]
        self.assertEqual(weather.tool_calls(), ["get_weather"])

    def test_final_output_is_last_chat(self):
        weather = self.by_task["weather-agent-001"]
        self.assertIn("Shanghai", weather.final_output() or "")

    def test_tokens_and_duration(self):
        weather = self.by_task["weather-agent-001"]
        self.assertEqual(weather.total_tokens, 200)
        self.assertEqual(weather.input_tokens, 120)
        self.assertEqual(weather.output_tokens, 80)
        self.assertAlmostEqual(weather.duration_ms, 1000.0, places=1)


if __name__ == "__main__":
    unittest.main()
