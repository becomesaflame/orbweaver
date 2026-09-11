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


@dataclass
class ProbeStats:
    """Cumulative injection-probe overhead for one session (see ``record_probe_usage``)."""

    calls: int = 0
    results: int = 0
    flagged: int = 0
    late: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


_anchors: dict[UUID, UsageAnchor] = {}
_failures: dict[UUID, int] = {}
_probes: dict[UUID, ProbeStats] = {}


def reset_compact_state() -> None:
    _anchors.clear()
    _failures.clear()
    _probes.clear()


def record_usage(session_id: UUID, input_tokens: int, at_seq: int) -> None:
    _anchors[session_id] = UsageAnchor(input_tokens=max(0, input_tokens), at_seq=at_seq)


def record_probe_usage(
    session_id: UUID,
    *,
    calls: int = 1,
    results: int = 1,
    flagged: int = 0,
    late: int = 0,
    latency_s: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ProbeStats:
    """Accumulate injection-probe calls, latency, and tokens so the overhead is visible."""
    stats = _probes.setdefault(session_id, ProbeStats())
    stats.calls += max(0, calls)
    stats.results += max(0, results)
    stats.flagged += max(0, flagged)
    stats.late += max(0, late)
    stats.latency_s += max(0.0, latency_s)
    stats.input_tokens += max(0, input_tokens)
    stats.output_tokens += max(0, output_tokens)
    return stats


def probe_stats(session_id: UUID) -> ProbeStats:
    return _probes.get(session_id, ProbeStats())


def clear_usage(session_id: UUID) -> None:
    _anchors.pop(session_id, None)
    _probes.pop(session_id, None)


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
