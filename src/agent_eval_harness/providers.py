"""Judge provider registry.

Maps a model name to the endpoint, credential environment variable and
generation settings it needs. This is the seam that keeps provider quirks out
of the core harness:

- Moonshot rejects ``temperature != 1``
- Gemini's OpenAI-compatible endpoint requires the ``models/`` prefix
- reasoning models need a generous ``max_tokens`` (thinking tokens are billed
  against it, and an exhausted budget returns empty content)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .judge import LLMJudge


@dataclass(frozen=True)
class Provider:
    """Everything needed to build a judge for one model."""

    model: str
    base_url: str
    key_env: str
    temperature: float = 0.0
    max_tokens: int = 1500


PROVIDERS: dict[str, Provider] = {
    "deepseek-v4-pro": Provider(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        key_env="DEEPSEEK_API_KEY",
    ),
    "deepseek-flash": Provider(
        model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        key_env="DEEPSEEK_API_KEY",
    ),
    "glm-5.3": Provider(
        model="glm-5.3",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        key_env="GLM_API_KEY",
    ),
    "glm-5.3-flash": Provider(
        model="glm-5.3-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        key_env="GLM_API_KEY",
    ),
    "models/gemini-3.1-flash-lite": Provider(
        model="models/gemini-3.1-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
    ),
    "nvidia/nemotron-3.5-bolt-e4b": Provider(
        model="nvidia/nemotron-3.5-bolt-e4b",
        base_url="https://integrate.api.nvidia.com/v1",
        key_env="NVIDIA_API_KEY",
    ),
    # Same underlying model as deepseek-flash, but served by NVIDIA NIM.
    # Useful as a controlled experiment on hosting effects.
    "nvidia/deepseek-v4.1-flash": Provider(
        model="deepseek-ai/deepseek-v4.1-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        key_env="NVIDIA_API_KEY",
        max_tokens=2000,
    ),
    "kimi-k2.6": Provider(
        model="kimi-k2.6",
        base_url="https://api.moonshot.cn/v1",
        key_env="KIMI_API_KEY",
        temperature=1.0,
    ),
}


def resolve(model: str) -> Provider:
    """Look up a known provider, falling back to an OpenAI-compatible default."""
    if model in PROVIDERS:
        return PROVIDERS[model]
    return Provider(
        model=model,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        key_env="OPENAI_API_KEY",
    )


def make_judge(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMJudge:
    """Build an :class:`LLMJudge` for ``model`` using the provider registry."""
    provider = resolve(model)
    resolved_key = (
        api_key
        or os.environ.get(provider.key_env)
        or os.environ.get("OPENAI_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError(
            f"no API key for {model}: set {provider.key_env} or OPENAI_API_KEY"
        )
    return LLMJudge(
        model=provider.model,
        base_url=base_url or provider.base_url,
        api_key=resolved_key,
        temperature=provider.temperature if temperature is None else temperature,
        max_tokens=max_tokens or provider.max_tokens,
    )
