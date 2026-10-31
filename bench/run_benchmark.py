"""Measure regression-detection quality of agent-eval-harness vs naive detectors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.cli import load_cases, load_traces  # noqa: E402
from agent_eval_harness.judge import RuleJudge  # noqa: E402
from agent_eval_harness.runner import run  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def metrics(flagged: set[str], truth: set[str]) -> dict[str, float]:
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"detected": tp, "false_alarms": fp, "missed": fn, "recall": recall, "precision": precision}


def main() -> int:
    cases = load_cases(DATA / "cases.jsonl")
    base_traces = load_traces(DATA / "baseline.json")
    deg_traces = load_traces(DATA / "degraded.json")

    truth = {item["task_id"] for item in json.loads((DATA / "ground_truth.json").read_text())}

    base_report = run(cases, base_traces, RuleJudge())
    deg_report = run(cases, deg_traces, RuleJudge())

    base_failures = {c.case_id for c in base_report.cases if not c.passed}
    aeh_rubric = {c.case_id for c in deg_report.cases if any(not k.passed for k in c.checks)}
    aeh_full = {c.case_id for c in deg_report.cases if not c.passed}

    outputs_base = {t.task_id: (t.final_output() or "") for t in base_traces.values()}
    outputs_deg = {t.task_id: (t.final_output() or "") for t in deg_traces.values()}

    naive_empty = {cid for cid, out in outputs_deg.items() if not out.strip()}
    naive_length = {
        cid
        for cid, out in outputs_deg.items()
        if len(out) < 0.6 * len(outputs_base.get(cid, "") or "x")
    }

    detectors = {
        "aeh (deterministic rubric)": aeh_rubric,
        "aeh (rubric + judge)": aeh_full,
        "naive: output non-empty": naive_empty,
        "naive: output length drop": naive_length,
    }

    print(f"tasks={len(cases)}  baseline_failures={len(base_failures)}  true_regressions={len(truth)}\n")
    header = f"{'detector':30s} {'detected':>9s} {'false_alarms':>13s} {'missed':>7s} {'recall':>8s} {'precision':>10s}"
    print(header)
    print("-" * len(header))
    for name, flagged in detectors.items():
        m = metrics(flagged, truth)
        print(
            f"{name:30s} {m['detected']:9d} {m['false_alarms']:13d} {m['missed']:7d} "
            f"{m['recall']:8.0%} {m['precision']:10.0%}"
        )

    result = {
        "tasks": len(cases),
        "true_regressions": len(truth),
        "detectors": {
            name: {"flagged": sorted(flagged), **metrics(flagged, truth)}
            for name, flagged in detectors.items()
        },
    }
    (DATA / "benchmark_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {DATA / 'benchmark_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
