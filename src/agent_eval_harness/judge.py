"""Pluggable judges for the non-deterministic part of agent acceptance."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import Trace


@dataclass
class Verdict:
    score: float
    rationale: str = ""


class Judge(Protocol):
    def evaluate(self, task: str, trace: Trace, expected: list[str] | None = None) -> Verdict:
        ...


class RuleJudge:
    """Zero-dependency judge: scores keyword overlap between task and output."""

    def evaluate(self, task: str, trace: Trace, expected: list[str] | None = None) -> Verdict:
        output = (trace.final_output() or "").strip()
        words = [w.strip(".,!?;:").lower() for w in task.split() if len(w) > 2]
        if not output or not words:
            return Verdict(0.0, "empty output or empty task")
        haystack = output.lower()
        hits = sum(1 for word in words if word in haystack)
        score = min(1.0, hits / len(words))
        return Verdict(score, f"keyword overlap {hits}/{len(words)}")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating fences and prose."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"no JSON object in judge response: {text[:200]!r}")


class LLMJudge:
    """Judge backed by any OpenAI-compatible chat completions endpoint.

    Configured entirely by environment variables so keys never live in code:

    - ``OPENAI_BASE_URL`` (e.g. ``https://api.deepseek.com/v1``)
    - ``OPENAI_API_KEY``
    - ``AEH_JUDGE_MODEL`` (e.g. ``deepseek-v4-pro``)
    """

    PROMPT = (
        "You are grading one AI agent run against its task.\n"
        "Decision rule:\n"
        "- COMPLETED: the final output states every required value directly, and the\n"
        "  required tools were called. A terse answer such as 'fact042 = v042' counts\n"
        "  as completed. Do not ask for extra explanation.\n"
        "- NOT COMPLETED: a required value is missing, OR the output explicitly hedges\n"
        "  or disclaims (e.g. 'source was unavailable', 'may be stale', 'could not\n"
        "  verify', 'I am not confident').\n"
        "- Use intermediate scores only for genuinely partial answers.\n"
        'Return strict JSON: {{"score": <0..1>, "rationale": "<one sentence>"}}\n\n'
        "Task: {task}\n"
        "Required in the answer: {expected}\n"
        "Tool calls: {tools}\n"
        "Final output: {output}\n"
    )

    def __init__(self, model: str | None = None, timeout: int = 60) -> None:
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("AEH_JUDGE_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("LLMJudge requires OPENAI_API_KEY (or DEEPSEEK_API_KEY exported as OPENAI_API_KEY)")

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def evaluate(self, task: str, trace: Trace, expected: list[str] | None = None) -> Verdict:
        prompt = self.PROMPT.format(
            task=task,
            expected=expected or [],
            tools=trace.tool_calls(),
            output=(trace.final_output() or "")[:4000],
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(4):
            if attempt:
                time.sleep(0.3 * (2 ** (attempt - 1)))
            try:
                payload = self._post(body)
                content = (payload["choices"][0]["message"].get("content") or "").strip()
                if not content:
                    raise ValueError("empty judge response")
                parsed = extract_json(content)
                return Verdict(float(parsed.get("score", 0.0)), str(parsed.get("rationale", "")))
            except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        return Verdict(0.0, f"judge error: {last_error}")
