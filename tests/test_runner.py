import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.cli import load_cases, load_traces  # noqa: E402
from agent_eval_harness.judge import RuleJudge  # noqa: E402
from agent_eval_harness.runner import run  # noqa: E402


class TestRunnerEndToEnd(unittest.TestCase):
    def test_example_suite_passes(self):
        cases = load_cases(ROOT / "examples" / "cases" / "cases.jsonl")
        traces = load_traces(ROOT / "examples" / "traces")
        report = run(cases, traces, RuleJudge())
        self.assertEqual(report.total, 3)
        self.assertEqual(report.passed, 3, [(c.case_id, [k.name for k in c.checks if not k.passed]) for c in report.cases])
        self.assertTrue(report.all_passed)


if __name__ == "__main__":
    unittest.main()
