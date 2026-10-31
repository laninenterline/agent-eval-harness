"""Ingest OpenTelemetry (OTLP JSON) exports into normalized traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Span, Trace
from .semconv import normalize

TASK_ID_ATTRS = ("aeh.task.id", "gen_ai.conversation.id")

# OTLP StatusCode: 0 = UNSET, 1 = OK, 2 = ERROR
_STATUS_ERROR = 2


def decode_any_value(value: dict[str, Any]) -> Any:
    """Decode one OTLP AnyValue into a plain Python value."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [decode_any_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: decode_any_value(kv["value"])
            for kv in value["kvlistValue"].get("values", [])
        }
    return None


def decode_attributes(raw: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {item["key"]: decode_any_value(item["value"]) for item in (raw or [])}


def _to_ns(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_spans(raw_spans: list[dict[str, Any]]) -> list[Trace]:
    """Parse a flat list of OTLP span dicts into traces."""
    traces: dict[str, Trace] = {}
    for raw in raw_spans:
        name = raw.get("name", "")
        attributes = normalize(decode_attributes(raw.get("attributes")), name)

        status = raw.get("status") or {}
        if status.get("code") == _STATUS_ERROR and attributes.get("error") is None:
            attributes["error"] = True

        span = Span(
            span_id=raw.get("spanId", ""),
            trace_id=raw.get("traceId", ""),
            name=name,
            start_ns=_to_ns(raw.get("startTimeUnixNano")),
            end_ns=_to_ns(raw.get("endTimeUnixNano")),
            parent_span_id=raw.get("parentSpanId") or None,
            attributes=attributes,
        )
        trace = traces.setdefault(span.trace_id, Trace(trace_id=span.trace_id))
        trace.spans.append(span)

    for trace in traces.values():
        for span in trace.spans:
            for attr in TASK_ID_ATTRS:
                task_id = span.attributes.get(attr)
                if task_id:
                    trace.task_id = str(task_id)
                    break
            if trace.task_id:
                break
    return list(traces.values())


def parse_otlp_json(payload: dict[str, Any]) -> list[Trace]:
    """Parse an OTLP JSON export (resourceSpans wrapper) into traces."""
    flat: list[dict[str, Any]] = []
    for resource_span in payload.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            flat.extend(scope_span.get("spans", []))
    return parse_spans(flat)


def load_otlp_json(path: str | Path) -> list[Trace]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "resourceSpans" in data:
        return parse_otlp_json(data)
    if isinstance(data, list):
        return parse_spans(data)
    raise ValueError(f"unrecognized trace format in {path}")
