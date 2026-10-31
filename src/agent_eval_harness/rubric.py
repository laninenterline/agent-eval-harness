"""Deterministic rubric checks over a trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Trace


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Rubric:
    """Deterministic, model-free acceptance rules for one agent run."""

    expect_tools: list[str] = field(default_factory=list)
    output_contains: list[str] = field(default_factory=list)
    max_duration_ms: float | None = None
    max_total_tokens: int | None = None
    forbid_errors: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        return cls(
            expect_tools=list(data.get("expect_tools", [])),
            output_contains=list(data.get("output_contains", [])),
            max_duration_ms=data.get("max_duration_ms"),
            max_total_tokens=data.get("max_total_tokens"),
            forbid_errors=bool(data.get("forbid_errors", True)),
        )

    def evaluate(self, trace: Trace) -> list[CheckResult]:
        results: list[CheckResult] = []
        called = trace.tool_calls()

        for tool in self.expect_tools:
            results.append(CheckResult(f"tool:{tool}", tool in called, f"called={called}"))

        output = trace.final_output() or ""
        for needle in self.output_contains:
            results.append(CheckResult(f"output:{needle}", needle in output, output[:200]))

        if self.max_duration_ms is not None:
            ok = trace.duration_ms <= self.max_duration_ms
            results.append(CheckResult("duration", ok, f"{trace.duration_ms:.0f}ms <= {self.max_duration_ms}ms"))

        if self.max_total_tokens is not None:
            ok = trace.total_tokens <= self.max_total_tokens
            results.append(CheckResult("tokens", ok, f"{trace.total_tokens} <= {self.max_total_tokens}"))

        if self.forbid_errors:
            errors = trace.errors()
            results.append(CheckResult("no_errors", not errors, f"{len(errors)} error span(s)"))

        return results
