import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.demo import CASES, build_report  # noqa: E402


class TestDemo(unittest.TestCase):
    def setUp(self):
        self.report = build_report()

    def test_has_three_cases(self):
        self.assertEqual(self.report.total, 3)
        self.assertEqual(len(CASES), 3)

    def test_two_pass_one_fails(self):
        self.assertEqual(self.report.passed, 2)
        failed = [case.case_id for case in self.report.cases if not case.passed]
        self.assertEqual(failed, ["demo-budget"])

    def test_budget_case_fails_on_tokens(self):
        budget = next(c for c in self.report.cases if c.case_id == "demo-budget")
        failed_checks = [check.name for check in budget.checks if not check.passed]
        self.assertIn("tokens", failed_checks)

    def test_healthy_cases_have_no_failing_checks(self):
        for case in self.report.cases:
            if case.case_id == "demo-budget":
                continue
            self.assertTrue(all(check.passed for check in case.checks), case.case_id)


if __name__ == "__main__":
    unittest.main()
