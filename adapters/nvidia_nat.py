"""NVIDIA NeMo Agent Toolkit (NAT) trace adapter.

NAT emits OpenTelemetry spans through ``nat.plugins.opentelemetry``. Its spans
carry **OpenInference** semantic attributes, not the OTel ``gen_ai.*`` family.
The harness normalizes them in ``agent_eval_harness.semconv``; this adapter just
documents and wraps that path.

Mapping used (verified against ``nat/plugins/opentelemetry/span_converter.py``
and the installed ``nvidia-nat`` 1.9.0):

| NAT / OpenInference attribute   | normalized attribute            |
|---------------------------------|---------------------------------|
| ``openinference.span.kind=LLM`` | ``gen_ai.operation.name=chat``         |
| ``openinference.span.kind=TOOL``| ``gen_ai.operation.name=execute_tool`` |
| ``openinference.span.kind=AGENT``| ``gen_ai.operation.name=invoke_agent``|
| ``llm.token_count.prompt``      | ``gen_ai.usage.input_tokens``          |
| ``llm.token_count.completion``  | ``gen_ai.usage.output_tokens``         |
| ``llm.model_name``              | ``gen_ai.request.model``               |
| ``output.value``                | ``gen_ai.completion``                  |
| ``input.value``                 | ``gen_ai.prompt``                      |
| TOOL span ``name``              | ``gen_ai.tool.name``                   |

Task correlation: inject ``aeh.task.id`` into the workflow/agent span (or set
``gen_ai.conversation.id``) so each trace maps to one eval case.

Usage::

    from adapters.nvidia_nat import load
    traces = load("nat-otel-export.json")
"""

from __future__ import annotations

from pathlib import Path

from agent_eval_harness.models import Trace
from agent_eval_harness.otel import load_otlp_json


def load(path: str | Path) -> list[Trace]:
    """Load a NAT OTLP export into normalized traces."""
    return load_otlp_json(path)
