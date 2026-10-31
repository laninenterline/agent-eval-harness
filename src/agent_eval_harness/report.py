"""Render run reports as JSON (diffable) and Markdown (readable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import RunReport


def to_dict(report: RunReport) -> dict[str, Any]:
    return {
        "summary": {
            "passed": report.passed,
            "total": report.total,
            "pass_rate": round(report.pass_rate, 4),
            "missing_traces": report.meta.get("missing_traces", []),
        },
        "cases": [
            {
                "case_id": case.case_id,
                "trace_id": case.trace_id,
                "passed": case.passed,
                "duration_ms": round(case.duration_ms, 2),
                "total_tokens": case.total_tokens,
                "judge": {
                    "score": round(case.judge.score, 3),
                    "rationale": case.judge.rationale,
                },
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in case.checks
                ],
            }
            for case in report.cases
        ],
    }


def write_json(report: RunReport, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(to_dict(report), indent=2) + "\n", encoding="utf-8")
    return target


def to_markdown(report: RunReport) -> str:
    rows: list[list[str]] = [["case", "result", "ms", "tokens", "judge", "first failing check"]]
    for case in report.cases:
        failed = next((check.name for check in case.checks if not check.passed), "")
        rows.append([
            case.case_id,
            "PASS" if case.passed else "FAIL",
            f"{case.duration_ms:.0f}",
            str(case.total_tokens),
            f"{case.judge.score:.2f}",
            failed,
        ])

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    lines = [
        "# agent-eval-harness report",
        "",
        f"**Pass rate:** {report.passed}/{report.total} ({report.pass_rate:.0%})",
        "",
        render_row(rows[0]),
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    lines.extend(render_row(row) for row in rows[1:])

    missing = report.meta.get("missing_traces") or []
    if missing:
        lines.extend(["", f"**Missing traces:** {', '.join(missing)}"])
    return "\n".join(lines) + "\n"


def to_plain(report: RunReport) -> str:
    """Plain-text report for terminals (no markdown syntax)."""
    rows: list[list[str]] = [["case", "result", "ms", "tokens", "judge", "first failing check"]]
    for case in report.cases:
        failed = next((check.name for check in case.checks if not check.passed), "")
        rows.append([
            case.case_id,
            "PASS" if case.passed else "FAIL",
            f"{case.duration_ms:.0f}",
            str(case.total_tokens),
            f"{case.judge.score:.2f}",
            failed,
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def row_text(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()

    lines = [
        "agent-eval-harness report",
        f"pass rate: {report.passed}/{report.total} ({report.pass_rate:.0%})",
        "",
        row_text(rows[0]),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(row_text(row) for row in rows[1:])
    missing = report.meta.get("missing_traces") or []
    if missing:
        lines.extend(["", f"missing traces: {', '.join(missing)}"])
    return "\n".join(lines) + "\n"
