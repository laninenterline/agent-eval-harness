"""Run the regression benchmark with a real LLM judge.

Deterministic checks are evaluated first. Only cases that pass deterministically
are sent to the judge, so the judge is measured on the cases it can actually
contribute to (and on the healthy baseline, to measure false alarms).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.cli import load_cases, load_traces  # noqa: E402
from agent_eval_harness.judge import LLMJudge  # noqa: E402
from agent_eval_harness.rubric import Rubric  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
THRESHOLD = 0.5
WORKERS = 3


def metrics(flagged: set[str], truth: set[str]) -> dict:
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    return {
        "detected": tp,
        "false_alarms": fp,
        "missed": fn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }


def judge_cases(cases, traces, judge, workers=WORKERS):
    def work(case):
        return case.id, judge.evaluate(case.task, traces[case.id], case.rubric.output_contains)

    scores = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for case_id, verdict in pool.map(work, cases):
            scores[case_id] = verdict
    return scores


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; source .env first", file=sys.stderr)
        return 2

    cases = load_cases(DATA / "cases.jsonl")
    base_traces = load_traces(DATA / "baseline.json")
    deg_traces = load_traces(DATA / "degraded.json")
    truth = {item["task_id"] for item in json.loads((DATA / "ground_truth.json").read_text())}

    judge = LLMJudge()
    print(f"judge model: {judge.model} @ {judge.base_url}")

    # Deterministic layer
    rubric_only = {}
    check_pass = {}
    for case in cases:
        checks = case.rubric.evaluate(deg_traces[case.id])
        check_pass[case.id] = all(c.passed for c in checks)
        rubric_only[case.id] = not check_pass[case.id]

    base_checks_fail = {
        case.id for case in cases if not all(c.passed for c in case.rubric.evaluate(base_traces[case.id]))
    }

    # Judge layer: baseline (false alarms) + degraded cases that passed checks
    base_judged = judge_cases(cases, base_traces, judge)
    deg_pending = [c for c in cases if check_pass[c.id]]
    deg_judged = judge_cases(deg_pending, deg_traces, judge)

    base_flagged = base_checks_fail | {cid for cid, v in base_judged.items() if v.score < THRESHOLD}
    deg_flagged = ({cid for cid, ok in rubric_only.items() if ok}) | {
        cid for cid, v in deg_judged.items() if v.score < THRESHOLD
    }

    naive_empty = {c.id for c in cases if not (deg_traces[c.id].final_output() or "").strip()}

    detectors = {
        "aeh (deterministic rubric)": set(cid for cid, ok in rubric_only.items() if ok),
        "aeh (rubric + LLM judge)": deg_flagged,
        "naive: output non-empty": naive_empty,
    }

    print(f"\ntasks={len(cases)}  true_regressions={len(truth)}\n")
    header = f"{'detector':30s} {'detected':>9s} {'false_alarms':>13s} {'missed':>7s} {'recall':>8s} {'precision':>10s}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, flagged in detectors.items():
        m = metrics(flagged, truth)
        results[name] = m
        print(f"{name:30s} {m['detected']:9d} {m['false_alarms']:13d} {m['missed']:7d} "
              f"{m['recall']:8.0%} {m['precision']:10.0%}")

    print(f"\nbaseline false alarms (healthy runs flagged): {len(base_flagged)}/{len(cases)}")
    print(f"judge calls: baseline={len(base_judged)} degraded={len(deg_judged)}")

    semantic = {t["task_id"] for t in json.loads((DATA / "ground_truth.json").read_text())
                if t["bucket"] == "semantic"}
    caught = sorted(semantic & deg_flagged)
    print(f"\nsemantic regressions caught by judge: {len(caught)}/{len(semantic)}")
    for cid in caught:
        print(f"  {cid}: {deg_judged[cid].score:.2f}  {deg_judged[cid].rationale[:90]}")

    out = {
        "tasks": len(cases),
        "true_regressions": len(truth),
        "judge_model": judge.model,
        "results": results,
        "baseline_false_alarms": len(base_flagged),
        "semantic_caught": len(caught),
        "semantic_total": len(semantic),
    }
    (DATA / "judge_benchmark_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
