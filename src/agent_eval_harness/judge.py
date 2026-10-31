"""Pluggable judges for the non-deterministic part of agent acceptance."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import Trace


@dataclass
class Verdict:
    score: float
    rationale: str = ""


class Judge(Protocol):
    def evaluate(self, task: str, trace: Trace) -> Verdict:
        ...


class RuleJudge:
    """Zero-dependency judge: scores keyword overlap between task and output."""

    def __init__(self) -> None:
        pass

    def evaluate(self, task: str, trace: Trace) -> Verdict:
        output = (trace.final_output() or "").strip()
        words = [w.strip(".,!?;:").lower() for w in task.split() if len(w) > 2]
        if not output or not words:
            return Verdict(0.0, "empty output or empty task")
        haystack = output.lower()
        hits = sum(1 for word in words if word in haystack)
        score = min(1.0, hits / len(words))
        return Verdict(score, f"keyword overlap {hits}/{len(words)}")


class LLMJudge:
    """Judge backed by any OpenAI-compatible chat completions endpoint."""

    PROMPT = (
        "You are grading one AI agent run against its task.\n"
        "Return strict JSON: {{\"score\": <0..1>, \"rationale\": \"<one sentence>\"}}\n\n"
        "Task: {task}\n"
        "Tool calls: {tools}\n"
        "Final output: {output}\n"
    )

    def __init__(self, model: str | None = None) -> None:
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("AEH_JUDGE_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("LLMJudge requires OPENAI_API_KEY")

    def evaluate(self, task: str, trace: Trace) -> Verdict:
        prompt = self.PROMPT.format(
            task=task,
            tools=trace.tool_calls(),
            output=(trace.final_output() or "")[:4000],
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return Verdict(float(parsed.get("score", 0.0)), str(parsed.get("rationale", "")))
