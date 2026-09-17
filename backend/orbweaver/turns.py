"""Per-session turn registry shared by web, Telegram, cron, and subagents.

Only one ``agent_turn`` may run on a session at a time. Two concurrent turns
interleave ``tool_call``/``tool_result`` events by ``seq`` and the next prompt
pairs them wrongly (the tool_use/tool_result mismatch class). Every channel
must go through :func:`acquire` (or :func:`running_turn`) before calling
``agent_turn``; the registry holds the cancel flag and inject queue that the
web UI's Stop and inject endpoints use.

:func:`begin_drain` stops new user-facing turns so a deploy can wait until
:func:`active` is empty, then restart. In-flight turns (and subagents they
spawn) keep running.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RunningTurn:
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    inject: asyncio.Event = field(default_factory=asyncio.Event)
    discard: bool = False
    user_seq: int = 0
    channel: str = ""
    session_id: UUID | None = None


class TurnBusy(Exception):
    """A turn is already running on this session."""

    def __init__(self, session_id: UUID, channel: str = "") -> None:
        self.session_id = session_id
        self.channel = channel
        via = f" via {channel}" if channel else ""
        super().__init__(f"turn already running on session {session_id}{via}")


class GatewayDraining(Exception):
    """New user-facing turns are refused because the gateway is about to restart."""

    def __init__(self) -> None:
        super().__init__("gateway is draining for deploy")


# Subagents belong to an in-flight parent turn; blocking them would stall drain.
_DRAIN_EXEMPT = frozenset({"subagent"})

_running: dict[UUID, RunningTurn] = {}
_draining = False


def begin_drain() -> None:
    """Refuse new user-facing turns. Idempotent."""
    global _draining
    _draining = True


def is_draining() -> bool:
    return _draining


def acquire(session_id: UUID, channel: str = "") -> RunningTurn | None:
    """Register a turn on ``session_id``. Return None when one is already running.

    The check-and-set is atomic under asyncio because there is no await here;
    callers must :func:`release` the returned state in a ``finally`` block.
    Raises :class:`GatewayDraining` when a deploy is waiting for idle and this
    is not a subagent of an already-running turn.
    """
    if session_id in _running:
        return None
    if _draining and channel not in _DRAIN_EXEMPT:
        raise GatewayDraining()
    state = RunningTurn(channel=channel, session_id=session_id)
    _running[session_id] = state
    return state


def release(session_id: UUID, state: RunningTurn) -> None:
    """Drop ``state`` from the registry if it is still the one registered."""
    if _running.get(session_id) is state:
        _running.pop(session_id, None)


def get(session_id: UUID) -> RunningTurn | None:
    return _running.get(session_id)


def is_running(session_id: UUID) -> bool:
    return session_id in _running


def active() -> dict[UUID, RunningTurn]:
    """Snapshot of the sessions with a running turn."""
    return dict(_running)


def snapshot() -> list[dict[str, str]]:
    """JSON-safe view of :func:`active` for the deploy drain endpoint."""
    return [
        {"session_id": str(sid), "channel": state.channel or "unknown"}
        for sid, state in _running.items()
    ]


@asynccontextmanager
async def running_turn(session_id: UUID, channel: str = "") -> AsyncIterator[RunningTurn]:
    """``async with running_turn(sid) as state:`` — raises :class:`TurnBusy` if taken."""
    state = acquire(session_id, channel)
    if state is None:
        current = _running.get(session_id)
        raise TurnBusy(session_id, current.channel if current else "")
    try:
        yield state
    finally:
        release(session_id, state)


def reset_for_tests() -> None:
    global _draining
    _running.clear()
    _draining = False
