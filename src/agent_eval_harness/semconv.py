"""Normalize multiple trace semantic conventions into the gen_ai.* namespace.

Different agents emit different tracing conventions:

- OpenTelemetry GenAI semantic conventions -> ``gen_ai.*``
- OpenInference (used by NVIDIA NeMo Agent Toolkit) -> ``openinference.span.kind``,
  ``llm.token_count.*``, ``output.value``

The harness evaluates one normalized view. This module maps the second family
onto the first so the rest of the code only ever sees ``gen_ai.*``.
"""

from __future__ import annotations

from typing import Any

_KIND_TO_OPERATION = {
    "LLM": "chat",
    "TOOL": "execute_tool",
    "CHAIN": "invoke_agent",
    "AGENT": "invoke_agent",
    "RETRIEVER": "retrieval",
    "EMBEDDING": "embeddings",
    "RERANKER": "rerank",
    "GUARDRAIL": "guardrail",
    "EVALUATOR": "evaluation",
}


def _kind(attributes: dict[str, Any]) -> str:
    value = attributes.get("openinference.span.kind", "")
    return str(value).upper()


def normalize(attributes: dict[str, Any], span_name: str = "") -> dict[str, Any]:
    """Return a copy of ``attributes`` enriched with ``gen_ai.*`` aliases."""
    out = dict(attributes)
    kind = _kind(attributes)

    if kind and "gen_ai.operation.name" not in out:
        operation = _KIND_TO_OPERATION.get(kind)
        if operation:
            out["gen_ai.operation.name"] = operation

    if "gen_ai.tool.name" not in out:
        tool = attributes.get("tool.name") or attributes.get("openinference.tool.name")
        if tool:
            out["gen_ai.tool.name"] = tool
        elif kind == "TOOL" and span_name:
            # NAT does not always set a tool attribute; the span name is the tool.
            out["gen_ai.tool.name"] = span_name

    input_tokens = attributes.get("llm.token_count.prompt")
    if input_tokens is not None and "gen_ai.usage.input_tokens" not in out:
        out["gen_ai.usage.input_tokens"] = input_tokens

    output_tokens = attributes.get("llm.token_count.completion")
    if output_tokens is not None and "gen_ai.usage.output_tokens" not in out:
        out["gen_ai.usage.output_tokens"] = output_tokens

    model = attributes.get("llm.model_name")
    if model and "gen_ai.request.model" not in out:
        out["gen_ai.request.model"] = model

    if "gen_ai.completion" not in out and attributes.get("output.value") is not None:
        out["gen_ai.completion"] = attributes["output.value"]

    if "gen_ai.prompt" not in out and attributes.get("input.value") is not None:
        out["gen_ai.prompt"] = attributes["input.value"]

    return out
