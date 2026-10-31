"""Compare judge models on the same regression benchmark."""

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

DATA = Path(__file__).resolve().parent / "data"
THRESHOLD = 0.5
WORKERS = 3

# model -> (base_url, key env var)
PROVIDERS = {
    "deepseek-v4-pro": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "deepseek-flash": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "glm-5.3": ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY"),
    "glm-5.3-flash": ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY"),
    "glm-4.6": ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY"),
}


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


def judge_cases(cases, traces, judge):
    def work(case):
        return case.id, judge.evaluate(case.task, traces[case.id], case.rubric.output_contains)

    scores = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for case_id, verdict in pool.map(work, cases):
            scores[case_id] = verdict
    return scores


def main() -> int:
    models = sys.argv[1:] or ["deepseek-v4-pro", "deepseek-flash"]

    cases = load_cases(DATA / "cases.jsonl")
    base_traces = load_traces(DATA / "baseline.json")
    deg_traces = load_traces(DATA / "degraded.json")
    truth_doc = json.loads((DATA / "ground_truth.json").read_text())
    truth = {item["task_id"] for item in truth_doc}
    semantic = {item["task_id"] for item in truth_doc if item["bucket"] == "semantic"}

    deterministic = {}
    for case in cases:
        checks = case.rubric.evaluate(deg_traces[case.id])
        deterministic[case.id] = not all(c.passed for c in checks)
    det_flagged = {cid for cid, failed in deterministic.items() if failed}

    print(f"tasks={len(cases)}  true_regressions={len(truth)}  "
          f"(structural={len(truth) - len(semantic)}, semantic={len(semantic)})")
    det = metrics(det_flagged, truth)
    print(f"\nbaseline detector  deterministic only: recall {det['recall']:.0%}  "
          f"precision {det['precision']:.0%}  false_alarms {det['false_alarms']}")

    rows = []
    for model in models:
        base_url, key_env = PROVIDERS.get(model, (None, "OPENAI_API_KEY"))
        api_key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(f"skipping {model}: {key_env} not set", file=sys.stderr)
            continue
        judge = LLMJudge(model=model, base_url=base_url, api_key=api_key)
        base_judged = judge_cases(cases, base_traces, judge)
        pending = [c for c in cases if not deterministic[c.id]]
        deg_judged = judge_cases(pending, deg_traces, judge)

        flagged = det_flagged | {cid for cid, v in deg_judged.items() if v.score < THRESHOLD}
        m = metrics(flagged, truth)
        base_flags = {cid for cid, v in base_judged.items() if v.score < THRESHOLD}
        caught = semantic & flagged
        stats = judge.summary()
        row = {
            "model": model,
            **m,
            "baseline_false_alarms": len(base_flags),
            "semantic_caught": len(caught),
            "semantic_total": len(semantic),
            **stats,
        }
        rows.append(row)
        print(f"\n[{model}] judged {stats['calls']} calls, errors={stats['errors']}")
        print(f"  recall={m['recall']:.0%} precision={m['precision']:.0%} "
              f"false_alarms={m['false_alarms']} baseline_false_alarms={len(base_flags)}")
        print(f"  tokens={stats['total_tokens']} p50={stats['latency_p50_ms']}ms p95={stats['latency_p95_ms']}ms")

    print("\n" + "=" * 100)
    header = (f"{'model':18s} {'recall':>7s} {'precision':>10s} {'FA':>4s} {'baseFA':>7s} "
              f"{'semantic':>9s} {'tokens':>8s} {'p50ms':>7s} {'p95ms':>7s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:18s} {r['recall']:7.0%} {r['precision']:10.0%} {r['false_alarms']:4d} "
              f"{r['baseline_false_alarms']:7d} {r['semantic_caught']:4d}/{r['semantic_total']:<4d} "
              f"{r['total_tokens']:8d} {r['latency_p50_ms']:7.0f} {r['latency_p95_ms']:7.0f}")

    result_path = DATA / "judge_compare_result.json"
    existing = json.loads(result_path.read_text()) if result_path.exists() else []
    replaced = {row["model"] for row in rows}
    merged = [row for row in existing if row["model"] not in replaced] + rows
    result_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nwritten to {DATA / 'judge_compare_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
