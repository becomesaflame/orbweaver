"""Prompt-size estimates from last API usage plus a local delta."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from orbweaver.store import Event
from orbweaver.tokens import estimate_tokens


@dataclass
class UsageAnchor:
    input_tokens: int
    at_seq: int


_anchors: dict[UUID, UsageAnchor] = {}
_failures: dict[UUID, int] = {}


def reset_compact_state() -> None:
    _anchors.clear()
    _failures.clear()


def record_usage(session_id: UUID, input_tokens: int, at_seq: int) -> None:
    _anchors[session_id] = UsageAnchor(input_tokens=max(0, input_tokens), at_seq=at_seq)


def clear_usage(session_id: UUID) -> None:
    _anchors.pop(session_id, None)


def compact_failures(session_id: UUID) -> int:
    return _failures.get(session_id, 0)


def record_compact_success(session_id: UUID) -> None:
    _failures[session_id] = 0


def record_compact_failure(session_id: UUID) -> int:
    _failures[session_id] = compact_failures(session_id) + 1
    return _failures[session_id]


def usage_input_tokens(usage: object | None) -> int:
    if usage is None:
        return 0
    total = int(getattr(usage, "input_tokens", 0) or 0)
    total += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    total += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
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
    """Prefer last API usage plus tokens added after that seq; else payload estimate."""
    anchor = _anchors.get(session_id)
    if anchor is None:
        return event_token_count(events)
    newer = [e for e in events if e.seq > anchor.at_seq]
    return anchor.input_tokens + event_token_count(newer)
