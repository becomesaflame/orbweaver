"""Earth Runtime / OpenRouter open-weight models (OpenAI-compatible chat API)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_OPENROUTER_BASE = "https://api.earthruntime.com/v1"


@dataclass(frozen=True)
class OpenModel:
    id: str
    label: str
    context_window: int
    reasoning: bool = False
    max_tokens: int = 4096
    description: str = ""


OPEN_MODELS: dict[str, OpenModel] = {
    "qwen3.6-35b": OpenModel(
        id="qwen3.6-35b",
        label="Qwen 3.6 35B",
        context_window=262_144,
        description="Fastest, good for most tasks (262K context)",
    ),
    "qwen3.8-27b": OpenModel(
        id="qwen3.8-27b",
        label="Qwen 3.8 27B",
        context_window=262_144,
        description="Dense 27B model (262K context)",
    ),
    "gpt-oss-120b": OpenModel(
        id="gpt-oss-120b",
        label="GPT-OSS 120B",
        context_window=131_072,
        max_tokens=8192,
        description="Largest open-weight option (128K context)",
    ),
    "deepseek-v4-flash-0731": OpenModel(
        id="deepseek-v4-flash-0731",
        label="DeepSeek V4 Flash",
        context_window=262_144,
        reasoning=True,
        max_tokens=16_384,
        description="Reasoning-capable, good for complex problems (262K context)",
    ),
}


def is_open_model(model: str) -> bool:
    return (model or "").strip() in OPEN_MODELS


def is_claude_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("claude")


def context_window_for(model: str, default: int) -> int:
    spec = OPEN_MODELS.get((model or "").strip())
    if spec is None:
        return default
    return spec.context_window


def max_tokens_for(model: str, requested: int) -> int:
    spec = OPEN_MODELS.get((model or "").strip())
    if spec is None:
        return requested
    return max(requested, spec.max_tokens)
