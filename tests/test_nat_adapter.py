import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from adapters.nvidia_nat import load  # noqa: E402


class TestNvidiaNatAdapter(unittest.TestCase):
    def setUp(self):
        self.traces = load(ROOT / "examples" / "traces" / "nat_sample.json")
        self.trace = next(t for t in self.traces if t.task_id == "nat-weather-001")

    def test_openinference_kind_becomes_operation(self):
        operations = [s.operation for s in self.trace.spans]
        self.assertIn("invoke_agent", operations)
        self.assertIn("chat", operations)
        self.assertIn("execute_tool", operations)

    def test_tool_name_from_span_name(self):
        self.assertEqual(self.trace.tool_calls(), ["get_weather"])

    def test_token_counts_are_normalized(self):
        self.assertEqual(self.trace.input_tokens, 350)
        self.assertEqual(self.trace.output_tokens, 140)
        self.assertEqual(self.trace.total_tokens, 490)

    def test_output_value_becomes_completion(self):
        self.assertIn("Shanghai", self.trace.final_output() or "")


if __name__ == "__main__":
    unittest.main()
