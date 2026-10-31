"""Normalized data models for agent traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Span:
    """A single span from an agent run, normalized from OTel GenAI."""

    span_id: str
    trace_id: str
    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1_000_000)

    @property
    def operation(self) -> str | None:
        value = self.attributes.get("gen_ai.operation.name")
        return value if isinstance(value, str) else None

    @property
    def tool_name(self) -> str | None:
        value = self.attributes.get("gen_ai.tool.name")
        return value if isinstance(value, str) else None

    def attr_str(self, key: str) -> str | None:
        value = self.attributes.get(key)
        return value if isinstance(value, str) else None

    def attr_int(self, key: str) -> int:
        value = self.attributes.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


@dataclass
class Trace:
    """All spans sharing one trace id, plus the task id used to match a case."""

    trace_id: str
    spans: list[Span] = field(default_factory=list)
    task_id: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(s.attr_int("gen_ai.usage.input_tokens") for s in self.spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.attr_int("gen_ai.usage.output_tokens") for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_ns for s in self.spans)
        end = max(s.end_ns for s in self.spans)
        return max(0.0, (end - start) / 1_000_000)

    def tool_calls(self) -> list[str]:
        return [s.tool_name for s in self.spans if s.operation == "execute_tool" and s.tool_name]

    def errors(self) -> list[Span]:
        return [s for s in self.spans if s.attributes.get("error") is True]

    def final_output(self) -> str | None:
        chats = [s for s in self.spans if s.operation == "chat"]
        if not chats:
            return None
        last = max(chats, key=lambda s: s.end_ns)
        return last.attr_str("gen_ai.completion") or last.attr_str("output.value")
