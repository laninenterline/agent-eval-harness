import json
from pathlib import Path

rows = json.loads((Path(__file__).resolve().parent / "data" / "judge_compare_result.json").read_text())
header = f"{'model':18s} {'recall':>7s} {'precision':>9s} {'FA':>3s} {'baseFA':>6s} {'semantic':>8s} {'tokens':>8s} {'p50ms':>7s} {'p95ms':>7s}"
print(header)
print("-" * len(header))
for r in rows:
    sem = f"{r['semantic_caught']}/{r['semantic_total']}"
    print(f"{r['model']:18s} {r['recall']:7.0%} {r['precision']:9.0%} {r['false_alarms']:3d} "
          f"{r['baseline_false_alarms']:6d} {sem:>8s} {r['total_tokens']:8d} "
          f"{r['latency_p50_ms']:7.0f} {r['latency_p95_ms']:7.0f}")
