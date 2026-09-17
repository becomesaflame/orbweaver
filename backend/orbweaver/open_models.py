"""Earth Runtime / OpenRouter open-weight models (OpenAI-compatible chat API)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_OPENROUTER_BASE = "https://api.earthruntime.com/v1"

# Stand-in for models whose real context length we have not confirmed; matches
# the smallest window in the catalog so compaction math errs on the safe side.
PROVISIONAL_CONTEXT_WINDOW = 131_072


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
    # Live on Earth Runtime but the provider's /models gives no context length,
    # so these carry the conservative 128K default until issue #155 confirms them.
    # An id missing from this table routes nowhere (see llm.select_provider), which
    # is how ORBWEAVER_TELEGRAM_MODEL=glm-5.3-flash 404'd against Anthropic.
    "glm-5.3": OpenModel(
        id="glm-5.3",
        label="GLM 5.3",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        description="GLM 5.3 (context window unconfirmed)",
    ),
    "glm-5.3-flash": OpenModel(
        id="glm-5.3-flash",
        label="GLM 5.3 Flash",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        description="Faster GLM 5.3 (context window unconfirmed)",
    ),
    "deepseek-v4.1-flash": OpenModel(
        id="deepseek-v4.1-flash",
        label="DeepSeek V4.1 Flash",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        reasoning=True,
        max_tokens=16_384,
        description="Newer DeepSeek V4 Flash (context window unconfirmed)",
    ),
    "kimi-k3": OpenModel(
        id="kimi-k3",
        label="Kimi K3",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        description="Kimi K3 (context window unconfirmed)",
    ),
    "minimax-m3": OpenModel(
        id="minimax-m3",
        label="MiniMax M3",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        description="MiniMax M3 (context window unconfirmed)",
    ),
    "nemotron-3-ultra": OpenModel(
        id="nemotron-3-ultra",
        label="Nemotron 3 Ultra",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        description="Nemotron 3 Ultra (context window unconfirmed)",
    ),
    "hy4": OpenModel(
        id="hy4",
        label="HY4",
        context_window=PROVISIONAL_CONTEXT_WINDOW,
        description="HY4 (context window unconfirmed)",
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
