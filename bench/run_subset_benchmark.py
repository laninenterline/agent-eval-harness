"""Small-subset judge benchmark for rate-limited providers (e.g. Kimi, 3 RPM).

Runs 10 healthy baseline runs (false-alarm check) and the 10 known semantic
regressions (recall check), sequentially, honouring the provider's rate limit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_eval_harness.cli import load_cases, load_traces  # noqa: E402
from agent_eval_harness.judge import LLMJudge  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
THRESHOLD = 0.5
SPACING_SECONDS = 2  # calls take ~22s, which already satisfies 3 RPM


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "kimi-k2.6"
    cases = {c.id: c for c in load_cases(DATA / "cases.jsonl")}
    base = load_traces(DATA / "baseline.json")
    deg = load_traces(DATA / "degraded.json")
    truth = json.loads((DATA / "ground_truth.json").read_text())
    semantic_ids = [t["task_id"] for t in truth if t["bucket"] == "semantic"]
    healthy_ids = [f"bench-{i:03d}" for i in range(10)]

    judge = LLMJudge(model=model, base_url=os.environ.get("KIMI_BASE_URL"),
                     api_key=os.environ.get("KIMI_API_KEY"), temperature=1.0)

    results = {"healthy": [], "semantic": []}

    def run(case_id, traces, bucket):
        case = cases[case_id]
        verdict = judge.evaluate(case.task, traces[case_id], case.rubric.output_contains)
        results[bucket].append({"case_id": case_id, "score": verdict.score,
                                "rationale": verdict.rationale[:100]})
        print(f"  {case_id} {bucket:8s} score={verdict.score:.2f} {verdict.rationale[:70]}")
        time.sleep(SPACING_SECONDS)

    print(f"model={model} healthy={len(healthy_ids)} semantic={len(semantic_ids)} "
          f"spacing={SPACING_SECONDS}s")
    print("healthy baseline (expect score >= threshold):")
    for cid in healthy_ids:
        run(cid, base, "healthy")
    print("semantic regressions (expect score < threshold):")
    for cid in semantic_ids:
        run(cid, deg, "semantic")

    false_alarms = sum(1 for r in results["healthy"] if r["score"] < THRESHOLD)
    caught = sum(1 for r in results["semantic"] if r["score"] < THRESHOLD)
    stats = judge.summary()
    out = {
        "model": model,
        "subset": {"healthy": len(healthy_ids), "semantic": len(semantic_ids)},
        "false_alarms": false_alarms,
        "semantic_caught": caught,
        "semantic_total": len(semantic_ids),
        **stats,
    }
    print(f"\nhealthy false alarms: {false_alarms}/{len(healthy_ids)}")
    print(f"semantic caught: {caught}/{len(semantic_ids)}")
    print(f"calls={stats['calls']} errors={stats['errors']} tokens={stats['total_tokens']} "
          f"p50={stats['latency_p50_ms']}ms p95={stats['latency_p95_ms']}ms")
    (DATA / "subset_benchmark_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
