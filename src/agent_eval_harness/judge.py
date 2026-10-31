"""Pluggable judges for the non-deterministic part of agent acceptance."""

from __future__ import annotations

import json
import os
import re
import threading
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
    error: bool = False


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

    def __init__(self, model: str | None = None, timeout: int = 120,
                 base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("AEH_JUDGE_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.environ.get("AEH_JUDGE_MAX_TOKENS", "1500"))
        self.temperature = temperature
        self.timeout = timeout
        self._lock = threading.Lock()
        self.calls = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latencies_ms: list[float] = []
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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(6):
            if attempt:
                time.sleep(0.3 * (2 ** (attempt - 1)))
            started = time.perf_counter()
            try:
                payload = self._post(body)
                elapsed_ms = (time.perf_counter() - started) * 1000
                choice = payload["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("judge response truncated by max_tokens (reasoning budget)")
                content = (choice["message"].get("content") or "").strip()
                if not content:
                    raise ValueError("empty judge response")
                parsed = extract_json(content)
                usage = payload.get("usage") or {}
                with self._lock:
                    self.calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                    self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                    self.latencies_ms.append(elapsed_ms)
                return Verdict(float(parsed.get("score", 0.0)), str(parsed.get("rationale", "")))
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code == 429:
                    retry_after = error.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else 2.0
                    time.sleep(min(wait, 8.0))
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        with self._lock:
            self.errors += 1
        return Verdict(0.0, f"judge error: {last_error}", error=True)

    def summary(self) -> dict:
        with self._lock:
            latencies = sorted(self.latencies_ms)
            count = len(latencies)
            p50 = latencies[count // 2] if count else 0.0
            p95 = latencies[int(count * 0.95)] if count else 0.0
            return {
                "calls": self.calls,
                "errors": self.errors,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "latency_p50_ms": round(p50, 1),
                "latency_p95_ms": round(p95, 1),
            }
