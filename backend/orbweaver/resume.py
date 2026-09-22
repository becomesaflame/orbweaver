"""Resume turns that a deploy drain preempted, after the new process starts.

The asyncio task is gone. What survives is the event log (and the workspace on
disk). A ``turn_interrupted`` with ``reason=drain`` plus Continue/``resume=True``
is the pickup. Cron jobs keep their row and the sweeper resumes them; web and
Telegram sessions are started here on boot.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from orbweaver.store import SESSION_TYPE, Entity, Event, Store, get_store, is_deleted_session
from orbweaver.subagent import is_subagent_session
from orbweaver.turns import DRAIN_INTERRUPT
from orbweaver.turnstate import BOOKKEEPING_KINDS, STATUS_STOPPED, session_turn_status

log = logging.getLogger(__name__)

# Cron keeps the Job row and the sweeper resumes it. Do not start a second turn.
_CRON_CHANNEL = "cron"


def is_drain_paused(events: list[Event]) -> bool:
    """True when the last turn was Stopped because drain cancelled it."""
    if session_turn_status(events) != STATUS_STOPPED:
        return False
    for ev in reversed(events):
        if ev.kind in BOOKKEEPING_KINDS:
            continue
        if ev.kind == "turn_interrupted":
            return str((ev.payload or {}).get("reason") or "") == DRAIN_INTERRUPT
        return False
    return False


async def drain_paused_sessions(store: Store) -> list[Entity]:
    """Sessions the new process should Continue (not cron — those have jobs)."""
    out: list[Entity] = []
    for ent in await store.list_entities(SESSION_TYPE):
        if is_subagent_session(ent) or is_deleted_session(ent):
            continue
        channel = str((ent.jsonld or {}).get("channel") or "")
        if channel == _CRON_CHANNEL:
            continue
        events = await store.list_events(ent.id)
        if is_drain_paused(events):
            out.append(ent)
    return out


async def kickoff_drain_resumes() -> list[UUID]:
    """Spawn resume turns for drain-paused interactive sessions. Used on boot."""
    from orbweaver.app import _run_turn

    store = get_store()
    started: list[UUID] = []
    for sess in await drain_paused_sessions(store):
        log.info("drain-resume session %s", sess.id)
        started.append(sess.id)
        asyncio.create_task(
            _run_turn(store, sess, sess.id, "", resume=True),
            name=f"drain-resume-{sess.id}",
        )
    return started
