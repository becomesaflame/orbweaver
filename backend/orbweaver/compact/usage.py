"""Prompt-size estimates from last API usage plus a local delta."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from orbweaver.store import Event
from orbweaver.tokens import estimate_tokens

log = logging.getLogger(__name__)


@dataclass
class UsageAnchor:
    input_tokens: int
    at_seq: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


_anchors: dict[UUID, UsageAnchor] = {}
_failures: dict[UUID, int] = {}


def reset_compact_state() -> None:
    _anchors.clear()
    _failures.clear()


def record_usage(
    session_id: UUID,
    input_tokens: int,
    at_seq: int,
    *,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> None:
    """Anchor the prompt-size estimate; ``input_tokens`` is the full prompt incl. cache."""
    _anchors[session_id] = UsageAnchor(
        input_tokens=max(0, input_tokens),
        at_seq=at_seq,
        cache_read_input_tokens=max(0, cache_read_input_tokens),
        cache_creation_input_tokens=max(0, cache_creation_input_tokens),
    )


def record_response_usage(session_id: UUID, usage: object | None, at_seq: int) -> UsageAnchor:
    """Record a messages.create ``usage`` and log the cache split per round."""
    cache_read = _usage_int(usage, "cache_read_input_tokens")
    cache_creation = _usage_int(usage, "cache_creation_input_tokens")
    total = usage_input_tokens(usage)
    record_usage(
        session_id,
        total,
        at_seq,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    log.info(
        "llm usage session=%s input=%d cache_read=%d cache_creation=%d uncached=%d",
        session_id,
        total,
        cache_read,
        cache_creation,
        _usage_int(usage, "input_tokens"),
    )
    return _anchors[session_id]


def last_usage(session_id: UUID) -> UsageAnchor | None:
    return _anchors.get(session_id)


def clear_usage(session_id: UUID) -> None:
    _anchors.pop(session_id, None)


def compact_failures(session_id: UUID) -> int:
    return _failures.get(session_id, 0)


def record_compact_success(session_id: UUID) -> None:
    _failures[session_id] = 0


def record_compact_failure(session_id: UUID) -> int:
    _failures[session_id] = compact_failures(session_id) + 1
    return _failures[session_id]


def _usage_int(usage: object | None, field: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field, 0)
    else:
        value = getattr(usage, field, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_input_tokens(usage: object | None) -> int:
    if usage is None:
        return 0
    total = _usage_int(usage, "input_tokens")
    total += _usage_int(usage, "cache_creation_input_tokens")
    total += _usage_int(usage, "cache_read_input_tokens")
    return total


def event_token_count(events: list[Event]) -> int:
    if not events:
        return 0
    blob = []
    for e in events:
        blob.append(e.kind)
        blob.append(str(e.payload))
    return estimate_tokens("\n".join(blob))


def estimate_prompt_tokens(session_id: UUID, events: list[Event]) -> int:
    """Prefer last API usage plus tokens added after that seq; else payload estimate.

    ``maybe_compact`` uses this as the over-budget trigger so system+tools,
    images, and MCP schemas already counted in ``usage.input_tokens`` are not
    dropped from the decision.
    """
    anchor = _anchors.get(session_id)
    if anchor is None:
        return event_token_count(events)
    newer = [e for e in events if e.seq > anchor.at_seq]
    return anchor.input_tokens + event_token_count(newer)
