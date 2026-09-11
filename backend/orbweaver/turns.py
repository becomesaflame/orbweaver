"""Per-session turn registry shared by web, Telegram, cron, and subagents.

Only one ``agent_turn`` may run on a session at a time. Two concurrent turns
interleave ``tool_call``/``tool_result`` events by ``seq`` and the next prompt
pairs them wrongly (the tool_use/tool_result mismatch class). Every channel
must go through :func:`acquire` (or :func:`running_turn`) before calling
``agent_turn``; the registry holds the cancel flag and inject queue that the
web UI's Stop and inject endpoints use.
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


_running: dict[UUID, RunningTurn] = {}


def acquire(session_id: UUID, channel: str = "") -> RunningTurn | None:
    """Register a turn on ``session_id``. Return None when one is already running.

    The check-and-set is atomic under asyncio because there is no await here;
    callers must :func:`release` the returned state in a ``finally`` block.
    """
    if session_id in _running:
        return None
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
    _running.clear()
