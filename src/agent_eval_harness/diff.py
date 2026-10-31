"""Compare two reports and surface regressions and fixes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DiffEntry:
    case_id: str
    before: bool
    after: bool
    judge_before: float
    judge_after: float

    @property
    def kind(self) -> str:
        if self.before and not self.after:
            return "regression"
        if not self.before and self.after:
            return "fix"
        return "unchanged"


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diff_reports(before: dict[str, Any], after: dict[str, Any]) -> list[DiffEntry]:
    before_cases = {case["case_id"]: case for case in before.get("cases", [])}
    after_cases = {case["case_id"]: case for case in after.get("cases", [])}
    entries: list[DiffEntry] = []
    for case_id in sorted(set(before_cases) | set(after_cases)):
        old = before_cases.get(case_id, {})
        new = after_cases.get(case_id, {})
        entries.append(
            DiffEntry(
                case_id=case_id,
                before=bool(old.get("passed", False)),
                after=bool(new.get("passed", False)),
                judge_before=float(old.get("judge", {}).get("score", 0.0)),
                judge_after=float(new.get("judge", {}).get("score", 0.0)),
            )
        )
    return entries


def has_regressions(entries: list[DiffEntry]) -> bool:
    return any(entry.kind == "regression" for entry in entries)


def render(entries: list[DiffEntry]) -> str:
    regressions = [e for e in entries if e.kind == "regression"]
    fixes = [e for e in entries if e.kind == "fix"]
    lines = [
        f"regressions: {len(regressions)}",
        f"fixes: {len(fixes)}",
        "",
    ]
    for entry in entries:
        if entry.kind == "unchanged":
            continue
        lines.append(
            f"{entry.kind.upper():10s} {entry.case_id}: "
            f"{entry.before} -> {entry.after} "
            f"(judge {entry.judge_before:.2f} -> {entry.judge_after:.2f})"
        )
    return "\n".join(lines) + "\n"
