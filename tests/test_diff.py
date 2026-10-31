import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.diff import diff_reports, has_regressions  # noqa: E402


def report(cases):
    return {"cases": [{"case_id": cid, "passed": ok, "judge": {"score": score}} for cid, ok, score in cases]}


class TestDiff(unittest.TestCase):
    def test_regression_detected(self):
        before = report([("a", True, 0.9), ("b", True, 0.8)])
        after = report([("a", True, 0.9), ("b", False, 0.3)])
        entries = diff_reports(before, after)
        self.assertTrue(has_regressions(entries))
        kinds = {e.case_id: e.kind for e in entries}
        self.assertEqual(kinds["b"], "regression")

    def test_fix_detected(self):
        before = report([("a", False, 0.2)])
        after = report([("a", True, 0.9)])
        entries = diff_reports(before, after)
        self.assertFalse(has_regressions(entries))
        self.assertEqual(entries[0].kind, "fix")

    def test_new_case_counted(self):
        before = report([("a", True, 0.9)])
        after = report([("a", True, 0.9), ("b", True, 0.7)])
        entries = diff_reports(before, after)
        self.assertEqual(len(entries), 2)


if __name__ == "__main__":
    unittest.main()
