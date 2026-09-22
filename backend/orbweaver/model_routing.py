"""Per-channel agent model defaults, Auto routing, and per-turn overrides (#137).

Resolution order for one turn: explicit override → model stored on the
session → channel default (``ORBWEAVER_WEB_MODEL`` / ``ORBWEAVER_VSCODE_MODEL``
/ ``ORBWEAVER_TELEGRAM_MODEL``) → ``ORBWEAVER_MODEL``. Web and Telegram default
to the virtual id ``auto``, which the router replaces with a concrete model
that currently has a provider. Provider routing is otherwise unchanged: Claude
ids go to Anthropic, catalog names to Earth Runtime. Auto runs the user prompt
through a lightweight hosted model (``ORBWEAVER_ROUTER_MODEL``, Haiku by
default) that replies with one catalog id; a failure falls back to the ranked
default.

The resolved *concrete* model is published through a ``ContextVar`` for the
duration of the turn so context-window maths (``settings.event_budget``) follow
the model actually in use instead of the process-wide default. HTTP 503
(upstream model unavailable) switches to a different model, preferring a
different provider, rather than retrying the same id.
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

AUTO_MODEL_ID = "auto"
AUTO_MODEL_LABEL = "Auto"
# Extra concrete models to try after an upstream-unavailable failure (503/529).
MAX_UNAVAILABLE_FALLBACKS = 3

# Preferred Auto / 503-fallback order after ``ORBWEAVER_MODEL``. Mix providers so
# a down Anthropic model does not land on another Anthropic id first.
AUTO_RANKING: tuple[str, ...] = (
    "claude-sonnet-4-6",
    "glm-5.3-flash",
    "claude-sonnet-5",
    "qwen3.6-35b",
    "gpt-oss-120b",
    "claude-haiku-4-5",
    "deepseek-v4-flash-0731",
    "glm-5.3",
    "claude-opus-5",
    "qwen3.8-27b",
    "kimi-k3",
    "deepseek-v4.1-flash",
    "claude-fable-5-1",
    "minimax-m3",
    "nemotron-3-ultra",
    "hy4",
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


def is_auto_model(model: str | None) -> bool:
    return (model or "").strip().lower() == AUTO_MODEL_ID


def channel_default_model(channel: str | None) -> str:
    """Env default for a channel.

    Web and Telegram fall back to Auto when their env is empty. VS Code, cron,
    subagents and unknown channels use ``ORBWEAVER_MODEL``.
    """
    fallback = settings.orbweaver_model.strip()
    ch = _norm_channel(channel)
    if ch == WEB_CHANNEL:
        return settings.orbweaver_web_model.strip() or AUTO_MODEL_ID
    if ch == VSCODE_CHANNEL:
        return settings.orbweaver_vscode_model.strip() or fallback
    if ch == TELEGRAM_CHANNEL:
        return settings.orbweaver_telegram_model.strip() or AUTO_MODEL_ID
    return fallback


def model_defaults() -> dict[str, str]:
    return {
        WEB_CHANNEL: channel_default_model(WEB_CHANNEL),
        VSCODE_CHANNEL: channel_default_model(VSCODE_CHANNEL),
        TELEGRAM_CHANNEL: channel_default_model(TELEGRAM_CHANNEL),
        "fallback": settings.orbweaver_model.strip(),
    }


def is_supported_model(model: str | None) -> bool:
    """Auto, Claude ids, Earth Runtime catalog names, or the configured Ollama model."""
    m = (model or "").strip()
    if not m or len(m) > MODEL_ID_MAX_LEN or any(c.isspace() for c in m):
        return False
    if is_auto_model(m) or is_open_model(m) or is_claude_model(m):
        return True
    ollama = settings.ollama_model.strip()
    return bool(ollama) and m == ollama


# Every model id an operator can set. Checked at startup so a typo is visible
# before a turn fails on it.
CONFIGURED_MODEL_ENV: tuple[tuple[str, str], ...] = (
    ("ORBWEAVER_MODEL", "orbweaver_model"),
    ("ORBWEAVER_WEB_MODEL", "orbweaver_web_model"),
    ("ORBWEAVER_VSCODE_MODEL", "orbweaver_vscode_model"),
    ("ORBWEAVER_TELEGRAM_MODEL", "orbweaver_telegram_model"),
    ("ORBWEAVER_CLASSIFIER_MODEL", "orbweaver_classifier_model"),
    ("ORBWEAVER_INJECTION_PROBE_MODEL", "orbweaver_injection_probe_model"),
    ("ORBWEAVER_ROUTER_MODEL", "orbweaver_router_model"),
    ("ORBWEAVER_COMPACT_MODEL", "orbweaver_compact_model"),
)


def unsupported_configured_models() -> list[tuple[str, str]]:
    """``(env name, id)`` for configured models that route nowhere."""
    return [
        (env_name, mid)
        for env_name, attr in CONFIGURED_MODEL_ENV
        if (mid := str(getattr(settings, attr, "") or "").strip()) and not is_supported_model(mid)
    ]


def model_label(model: str) -> str:
    """Short picker name; falls back to the id."""
    mid = (model or "").strip()
    if is_auto_model(mid):
        return AUTO_MODEL_LABEL
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
    if is_auto_model(mid):
        return AUTO_MODEL_LABEL
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
    """override → session ``model`` → channel default → ``ORBWEAVER_MODEL``.

    May return the virtual id ``auto``; call ``realize_turn_model`` before
    talking to a provider.
    """
    for candidate in (override, (session or {}).get("model")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return channel_default_model(channel)


def routed_models(*, skip: set[str] | None = None) -> list[str]:
    """Concrete models that currently have a provider, ranked for Auto / 503 fallback."""
    from orbweaver.llm import select_provider

    ignore = {m.strip() for m in (skip or set()) if m and not is_auto_model(m)}
    ordered: list[str] = []

    def add(mid: str) -> None:
        mid = (mid or "").strip()
        if not mid or is_auto_model(mid) or mid in ordered or mid in ignore:
            return
        provider = select_provider(mid)
        if provider in {"none", "auto"}:
            return
        ordered.append(mid)

    add(settings.orbweaver_model)
    for mid in AUTO_RANKING:
        add(mid)
    for mid in KNOWN_CLAUDE_MODELS:
        add(mid)
    for mid in OPEN_MODELS:
        add(mid)
    add(settings.ollama_model)
    return ordered


def pick_auto_model(*, skip: set[str] | None = None) -> str:
    """First Auto candidate with a live provider, or empty when none are configured."""
    ranked = routed_models(skip=skip)
    return ranked[0] if ranked else ""


def realize_turn_model(model: str | None, *, skip: set[str] | None = None) -> str:
    """Replace ``auto`` with the ranked default; pass other ids through.

    Agent turns with a user prompt should call ``realize_turn_model_async`` so
    the lightweight router LLM can pick. This sync path is the ranked fallback
    (catalog, classifier, empty prompt).
    """
    mid = (model or "").strip()
    if is_auto_model(mid):
        return pick_auto_model(skip=skip)
    return mid


async def realize_turn_model_async(
    model: str | None,
    *,
    prompt: str = "",
    skip: set[str] | None = None,
    extra: str = "",
) -> str:
    """Replace ``auto`` using the prompt router; pass other ids through."""
    mid = (model or "").strip()
    if not is_auto_model(mid):
        return mid
    if not (prompt or "").strip() and not extra.strip():
        return pick_auto_model(skip=skip)
    from orbweaver.auto_router import route_auto_model

    return await route_auto_model(prompt, skip=skip, extra=extra)


def fallback_model(failed: str, *, tried: set[str] | None = None) -> str:
    """Next model after an unavailable failure, preferring a different provider."""
    from orbweaver.llm import select_provider

    skip = set(tried or ())
    skip.add((failed or "").strip())
    failed_provider = select_provider(failed)
    candidates = routed_models(skip=skip)
    other = [m for m in candidates if select_provider(m) != failed_provider]
    same = [m for m in candidates if select_provider(m) == failed_provider]
    ordered = other + same
    return ordered[0] if ordered else ""


def supported_models() -> list[dict[str, Any]]:
    """Picker catalog: Auto, configured defaults, known Claude ids, Earth Runtime, Ollama."""
    from orbweaver.llm import select_provider

    ordered: list[str] = []

    def add(mid: str) -> None:
        mid = (mid or "").strip()
        if mid and mid not in ordered:
            ordered.append(mid)

    add(AUTO_MODEL_ID)
    add(settings.orbweaver_model)
    for ch in (WEB_CHANNEL, VSCODE_CHANNEL, TELEGRAM_CHANNEL):
        add(channel_default_model(ch))
    for mid in KNOWN_CLAUDE_MODELS:
        add(mid)
    for mid in OPEN_MODELS:
        add(mid)
    add(settings.ollama_model)

    auto_pick = pick_auto_model()
    out: list[dict[str, Any]] = []
    for mid in ordered:
        provider = select_provider(mid)
        window = context_window_for(mid, settings.context_window)
        if is_auto_model(mid) and auto_pick:
            window = context_window_for(auto_pick, settings.context_window)
        out.append(
            {
                "id": mid,
                "label": model_label(mid),
                "provider": provider,
                "available": provider != "none",
                "context_window": window,
            }
        )
    return out
