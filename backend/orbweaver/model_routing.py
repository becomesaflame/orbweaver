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

CLAUDE_MODEL_LABELS: dict[str, str] = {
    "claude-fable-5-1": "Claude Fable",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}

# Telegram ``/model default`` (and friends) clear the session override.
CLEAR_MODEL_ALIASES = frozenset({"default", "clear", "reset", "none", "-"})

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


def model_label(model: str) -> str:
    """Short picker name; falls back to the id."""
    mid = (model or "").strip()
    if mid in CLAUDE_MODEL_LABELS:
        return CLAUDE_MODEL_LABELS[mid]
    spec = OPEN_MODELS.get(mid)
    if spec is not None:
        return spec.label
    return mid


def format_model_id(model: str) -> str:
    """``Claude Opus 5 (claude-opus-5)`` when the label differs from the id."""
    mid = (model or "").strip()
    if not mid:
        return ""
    lab = model_label(mid)
    return f"{lab} ({mid})" if lab != mid else mid


def resolve_model_pick(
    query: str, ids: list[str] | None = None
) -> tuple[str | None, list[str]]:
    """Match a user-typed model id, prefix, or label.

    Returns ``(chosen, matches)``. ``chosen`` is ``""`` for clear-aliases,
    an id when the match is unique, or ``None`` when nothing unique matched.
    ``matches`` is the candidate list (empty means unknown).
    """
    q = (query or "").strip()
    if not q:
        return None, []
    if q.lower() in CLEAR_MODEL_ALIASES:
        return "", []
    catalog = list(ids) if ids is not None else [row["id"] for row in supported_models()]
    lowered = {mid.lower(): mid for mid in catalog}
    if q.lower() in lowered:
        return lowered[q.lower()], [lowered[q.lower()]]
    label_exact = [mid for mid in catalog if model_label(mid).lower() == q.lower()]
    if len(label_exact) == 1:
        return label_exact[0], label_exact
    needle = q.lower()

    def _starts(mid: str) -> bool:
        return mid.lower().startswith(needle) or model_label(mid).lower().startswith(needle)

    def _contains(mid: str) -> bool:
        return needle in mid.lower() or needle in model_label(mid).lower()

    starts = [mid for mid in catalog if _starts(mid)]
    if len(starts) == 1:
        return starts[0], starts
    if len(starts) > 1:
        return None, starts
    contains = [mid for mid in catalog if _contains(mid)]
    if len(contains) == 1:
        return contains[0], contains
    return None, contains


async def store_session_model(store: Any, sess: Any, model: str | None) -> None:
    """Persist a per-session model override (empty string removes it)."""
    if model is None:
        return
    jsonld = sess.jsonld
    if model:
        if jsonld.get("model") == model:
            return
        jsonld["model"] = model
    elif "model" in jsonld:
        del jsonld["model"]
    else:
        return
    await store.put_entity(sess)


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
                "label": model_label(mid),
                "provider": provider,
                "available": provider != "none",
                "context_window": context_window_for(mid, settings.context_window),
            }
        )
    return out
