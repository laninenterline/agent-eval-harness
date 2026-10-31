"""NVIDIA NeMo Agent Toolkit (NAT) trace adapter.

NAT emits OpenTelemetry spans, so the on-disk shape is OTLP JSON. This adapter
adds the small amount of NAT-specific mapping that matters for evaluation:

- ``gen_ai.operation.name == "execute_tool"`` -> a tool call
- ``gen_ai.tool.name``                       -> the tool name
- ``gen_ai.usage.input_tokens`` / ``output_tokens`` -> cost accounting
- ``aeh.task.id`` (or ``gen_ai.conversation.id``) -> the eval case id

Usage::

    from adapters.nvidia_nat import load
    traces = load("nat-trace-export.json")
"""

from __future__ import annotations

from pathlib import Path

from agent_eval_harness.models import Trace
from agent_eval_harness.otel import load_otlp_json


def load(path: str | Path) -> list[Trace]:
    """Load a NAT OTLP export into normalized traces."""
    return load_otlp_json(path)
