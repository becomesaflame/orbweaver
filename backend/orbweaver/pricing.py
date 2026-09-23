"""Anthropic list prices in USD per million tokens.

The Messages API reports tokens, not dollars. Daily spend caps convert usage
with this table. Cache writes are the 5-minute ephemeral multiplier (1.25×
input); cache reads are 0.1× input. Unknown Claude ids use Opus 5 so the cap
trips early rather than under-counting.
"""

from __future__ import annotations

from typing import Any

# (input, output, cache_write_5m, cache_read) USD / million tokens.
_RATES: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
    "claude-sonnet-4-6": (3.0, 15.0, 3.75, 0.30),
    "claude-sonnet-4-5": (3.0, 15.0, 3.75, 0.30),
    "claude-sonnet-4": (3.0, 15.0, 3.75, 0.30),
    "claude-sonnet-5": (3.0, 15.0, 3.75, 0.30),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4": (15.0, 75.0, 18.75, 1.50),
    "claude-fable-5-1": (5.0, 25.0, 6.25, 0.50),
    "claude-fable-5": (5.0, 25.0, 6.25, 1.00),
}

# Conservative upper bound for an unrecognized claude-* id.
_UNKNOWN = (5.0, 25.0, 6.25, 0.50)

_PREFIXES = tuple(sorted(_RATES, key=len, reverse=True))


def rates_for(model: str) -> tuple[float, float, float, float]:
    mid = (model or "").strip().lower()
    for prefix in _PREFIXES:
        if mid == prefix or mid.startswith(prefix):
            return _RATES[prefix]
    return _UNKNOWN


def _tok(usage: object | None, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        if isinstance(usage, dict):
            raw = usage.get(name)
        else:
            raw = getattr(usage, name, None)
        if raw is None:
            continue
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    return 0


def _cost_field(usage: object | None) -> float | None:
    if usage is None:
        return None
    for name in ("cost", "total_cost", "cost_usd"):
        if isinstance(usage, dict):
            raw = usage.get(name)
        else:
            raw = getattr(usage, name, None)
        if raw is None:
            continue
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None
    return None


def anthropic_usd(model: str, usage: object | None) -> float:
    """USD for one Anthropic-shaped usage object (tokens × list prices)."""
    inp = _tok(usage, "input_tokens")
    out = _tok(usage, "output_tokens", "completion_tokens")
    cache_read = _tok(usage, "cache_read_input_tokens")
    cache_write = _tok(usage, "cache_creation_input_tokens")
    return usd_from_tokens(model, inp, out, cache_read, cache_write)


def usd_from_tokens(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    inp_r, out_r, write_r, read_r = rates_for(model)
    million = 1_000_000.0
    return (
        max(0, input_tokens) / million * inp_r
        + max(0, output_tokens) / million * out_r
        + max(0, cache_read_input_tokens) / million * read_r
        + max(0, cache_creation_input_tokens) / million * write_r
    )


def estimate_prompt_usd(model: str, prompt_tokens: int) -> float:
    """Pre-flight: uncached input tokens only (the self-heal spike was prompt-sized)."""
    return usd_from_tokens(model, max(0, int(prompt_tokens or 0)), 0, 0, 0)


def usage_cost_usd(provider: str, model: str, usage: object | None) -> float:
    """Best USD figure for a completed call: provider cost field, else token table."""
    billed = _cost_field(usage)
    if billed is not None:
        return billed
    if provider == "openrouter":
        return 0.0
    if provider == "anthropic":
        return anthropic_usd(model, usage)
    return 0.0


def prompt_tokens_of(kwargs: dict[str, Any]) -> int:
    from orbweaver.compact.usage import json_tokens, static_prompt_tokens

    return static_prompt_tokens(kwargs.get("system"), kwargs.get("tools")) + json_tokens(
        kwargs.get("messages")
    )
