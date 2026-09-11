"""Per-channel agent model defaults and per-turn overrides (#137).

Resolution order for one turn: explicit override → model stored on the
session → channel default (``ORBWEAVER_WEB_MODEL`` / ``ORBWEAVER_VSCODE_MODEL``
/ ``ORBWEAVER_TELEGRAM_MODEL``) → ``ORBWEAVER_MODEL``. Provider routing is
unchanged: Claude ids go to Anthropic, catalog names to Earth Runtime.

The resolved model is published through a ``ContextVar`` for the duration of
the turn so context-window maths (``settings.event_budget``) follow the model
actually in use instead of the process-wide default.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from orbweaver.config import settings
from orbweaver.open_models import OPEN_MODELS, context_window_for, is_claude_model, is_open_model

MODEL_ID_MAX_LEN = 120

# Current Claude ids offered in pickers (Sept 2026). Configured defaults are
# merged in ahead of these, so an operator id never disappears from the list.
KNOWN_CLAUDE_MODELS: tuple[str, ...] = (
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)

WEB_CHANNEL = "web"
VSCODE_CHANNEL = "vscode"
TELEGRAM_CHANNEL = "telegram"

_current_model: ContextVar[str] = ContextVar("orbweaver_current_model", default="")


def current_model() -> str:
    """Model of the running turn, or ``ORBWEAVER_MODEL`` outside a turn."""
    return _current_model.get() or settings.orbweaver_model.strip()


def set_current_model(model: str) -> Token[str]:
    return _current_model.set((model or "").strip())


def reset_current_model(token: Token[str]) -> None:
    _current_model.reset(token)


def _norm_channel(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("_", "-").replace(" ", "")
    if raw in {"vscode", "vs-code", "visualstudiocode"}:
        return VSCODE_CHANNEL
    return raw


def channel_default_model(channel: str | None) -> str:
    """Env default for a channel; ``ORBWEAVER_MODEL`` for cron, subagents and unknown."""
    fallback = settings.orbweaver_model.strip()
    ch = _norm_channel(channel)
    if ch == WEB_CHANNEL:
        return settings.orbweaver_web_model.strip() or fallback
    if ch == VSCODE_CHANNEL:
        return settings.orbweaver_vscode_model.strip() or fallback
    if ch == TELEGRAM_CHANNEL:
        return settings.orbweaver_telegram_model.strip() or fallback
    return fallback


def model_defaults() -> dict[str, str]:
    return {
        WEB_CHANNEL: channel_default_model(WEB_CHANNEL),
        VSCODE_CHANNEL: channel_default_model(VSCODE_CHANNEL),
        TELEGRAM_CHANNEL: channel_default_model(TELEGRAM_CHANNEL),
        "fallback": settings.orbweaver_model.strip(),
    }


def is_supported_model(model: str | None) -> bool:
    """Claude ids, Earth Runtime catalog names, or the configured Ollama model."""
    m = (model or "").strip()
    if not m or len(m) > MODEL_ID_MAX_LEN or any(c.isspace() for c in m):
        return False
    if is_open_model(m) or is_claude_model(m):
        return True
    ollama = settings.ollama_model.strip()
    return bool(ollama) and m == ollama


def resolve_turn_model(
    *,
    override: str | None = None,
    session: dict[str, Any] | None = None,
    channel: str | None = None,
) -> str:
    """override → session ``model`` → channel default → ``ORBWEAVER_MODEL``."""
    for candidate in (override, (session or {}).get("model")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return channel_default_model(channel)


def supported_models() -> list[dict[str, Any]]:
    """Picker catalog: configured defaults, known Claude ids, Earth Runtime, Ollama."""
    from orbweaver.llm import select_provider

    ordered: list[str] = []

    def add(mid: str) -> None:
        mid = (mid or "").strip()
        if mid and mid not in ordered:
            ordered.append(mid)

    add(settings.orbweaver_model)
    for ch in (WEB_CHANNEL, VSCODE_CHANNEL, TELEGRAM_CHANNEL):
        add(channel_default_model(ch))
    for mid in KNOWN_CLAUDE_MODELS:
        add(mid)
    for mid in OPEN_MODELS:
        add(mid)
    add(settings.ollama_model)

    out: list[dict[str, Any]] = []
    for mid in ordered:
        provider = select_provider(mid)
        out.append(
            {
                "id": mid,
                "provider": provider,
                "available": provider != "none",
                "context_window": context_window_for(mid, settings.context_window),
            }
        )
    return out
