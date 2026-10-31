"""Run evaluation cases against ingested traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .judge import Judge, RuleJudge, Verdict
from .models import Trace
from .rubric import CheckResult, Rubric


@dataclass
class EvalCase:
    """One expected agent behaviour, keyed by the task id inside the trace."""

    id: str
    task: str = ""
    rubric: Rubric = field(default_factory=Rubric)
    judge_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(data["id"]),
            task=str(data.get("task", "")),
            rubric=Rubric.from_dict(data.get("rubric", {})),
            judge_threshold=float(data.get("judge_threshold", 0.5)),
        )


@dataclass
class CaseResult:
    case_id: str
    trace_id: str
    passed: bool
    checks: list[CheckResult]
    judge: Verdict
    duration_ms: float
    total_tokens: int


@dataclass
class RunReport:
    cases: list[CaseResult]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total and not self.meta.get("missing_traces")


def run_case(case: EvalCase, trace: Trace, judge: Judge) -> CaseResult:
    checks = case.rubric.evaluate(trace)
    verdict = judge.evaluate(case.task, trace, case.rubric.output_contains)
    passed = all(check.passed for check in checks) and verdict.score >= case.judge_threshold
    return CaseResult(
        case_id=case.id,
        trace_id=trace.trace_id,
        passed=passed,
        checks=checks,
        judge=verdict,
        duration_ms=trace.duration_ms,
        total_tokens=trace.total_tokens,
    )


def run(
    cases: list[EvalCase],
    traces_by_task: dict[str, Trace],
    judge: Judge | None = None,
) -> RunReport:
    active_judge = judge or RuleJudge()
    results: list[CaseResult] = []
    missing: list[str] = []

    for case in cases:
        trace = traces_by_task.get(case.id)
        if trace is None:
            missing.append(case.id)
            continue
        results.append(run_case(case, trace, active_judge))

    return RunReport(cases=results, meta={"missing_traces": missing})
