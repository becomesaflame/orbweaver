from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from orbweaver.agent import TurnCancelled, agent_turn
from orbweaver.config import settings
from orbweaver.store import Job, get_store
from orbweaver.turns import TurnBusy, running_turn
from orbweaver.workspace import bind_workspace

log = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None

# How many cron turns may run at once across all sessions.
MAX_CONCURRENT_JOBS = 3
# A job whose session is mid-turn is retried after this delay.
SKIP_RETRY = timedelta(minutes=1)
_job_slots: asyncio.Semaphore | None = None
_job_slots_loop: asyncio.AbstractEventLoop | None = None
_in_flight: set[UUID] = set()
_tasks: set[asyncio.Task[None]] = set()

# Aliases accepted after stripping a leading "every " and a trailing "s".
_RECURRENCE_ALIASES = {"minute": 60, "hour": 3600, "day": 86400}


def _alias_seconds(rec: str) -> int | None:
    alias = rec.strip().lower().removeprefix("every ").removesuffix("s")
    return _RECURRENCE_ALIASES.get(alias)


def _cron_trigger(rec: str) -> CronTrigger | None:
    parts = rec.split()
    if len(parts) != 5:
        return None
    try:
        return CronTrigger.from_crontab(rec, timezone=UTC)
    except ValueError:
        return None


def _parse_recurrence(rec: str | None, due: datetime, now: datetime | None = None) -> datetime | None:
    """Next fire after ``due``.

    ``rec`` is ``minute`` / ``hour`` / ``day`` (also ``every hour``) or a 5-field
    cron expression ``minute hour day-of-month month day-of-week``
    (names like ``mon`` work; 0 is Monday). One-shots and unknown strings
    return None. When ``now`` is set, skip occurrences that are already past.
    """
    if not rec:
        return None
    raw = rec.strip()
    if not raw:
        return None
    seconds = _alias_seconds(raw)
    if seconds is not None:
        step = timedelta(seconds=seconds)
        nxt = due + step
        if now is not None:
            while nxt <= now:
                nxt += step
        return nxt
    trigger = _cron_trigger(raw)
    if trigger is None:
        return None
    ref = now if now is not None and now > due else due
    return trigger.get_next_fire_time(None, ref)


def _result_text(events, status: str) -> str:
    from orbweaver.channels.telegram import texts_for_reply

    text = texts_for_reply(events)
    if text:
        return text
    if status == "aborted":
        return "Turn aborted."
    if status == "stopped":
        return "Turn stopped."
    return "Scheduled task finished."


async def _notify_originating_channel(store, sess, session_id, job, events) -> None:
    from orbweaver.channels.telegram import notify_telegram_chat

    kinds = {e.kind for e in events}
    status = "ok"
    if "turn_aborted" in kinds:
        status = "aborted"
    elif "turn_interrupted" in kinds:
        status = "stopped"
    text = _result_text(events, status)
    await store.append_event(
        session_id,
        "cron_result",
        {
            "job_id": str(job.id),
            "text": text,
            "status": status,
        },
    )
    chat_id = sess.jsonld.get("telegram_chat_id")
    if chat_id:
        await notify_telegram_chat(int(chat_id), text)


def _slots() -> asyncio.Semaphore:
    """Semaphore bound to the current loop (tests run each case on a fresh loop)."""
    global _job_slots, _job_slots_loop
    loop = asyncio.get_running_loop()
    if _job_slots is None or _job_slots_loop is not loop:
        _job_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        _job_slots_loop = loop
    return _job_slots


async def _run_job_turn(store, sess, session_id: UUID, job: Job, message: str) -> None:
    """One cron turn under the per-session lock. Raises TurnBusy if the session is mid-turn."""
    async with running_turn(session_id, channel="cron") as state:
        ws, kind, changed = bind_workspace(
            sess.jsonld, settings.workspace_root, session_key=str(session_id)
        )
        if changed:
            await store.put_entity(sess)
        await store.append_event(session_id, "cron", {"job_id": str(job.id)})
        try:
            events = await agent_turn(
                store,
                session_id,
                message,
                ws,
                workspace_kind=kind,
                headless=True,
                channel="cron",
                cancel=state.cancel,
                turn_state=state,
            )
        except TurnCancelled as e:
            # Stop from the web UI now reaches cron turns through the shared registry.
            marker = await store.append_event(session_id, "turn_interrupted", {"reason": "stop"})
            events = [*e.produced, marker]
        await _notify_originating_channel(store, sess, session_id, job, events)


async def _finish_job(store, job: Job) -> None:
    nxt = _parse_recurrence(job.recurrence, job.due_at, now=datetime.now(UTC))
    if nxt:
        job.due_at = nxt
        await store.reschedule_job(job)
    else:
        await store.delete_job(job.id)


async def _run_job(store, job: Job) -> None:
    """One due job as its own task, so a long turn does not hold up the tick."""
    try:
        async with _slots():
            session_id = job.session_id
            sess = await store.get_entity(session_id) if session_id else None
            if session_id and sess is not None:
                message = str(job.payload.get("message") or "scheduled task")
                try:
                    await _run_job_turn(store, sess, session_id, job, message)
                except TurnBusy as busy:
                    log.info(
                        "cron: job %s skipped, session %s has a running turn (%s); retry in 1 min",
                        job.id,
                        session_id,
                        busy.channel or "unknown channel",
                    )
                    job.due_at = datetime.now(UTC) + SKIP_RETRY
                    await store.reschedule_job(job)
                    return
            await _finish_job(store, job)
    except Exception:
        log.exception("cron: job %s failed", job.id)
    finally:
        _in_flight.discard(job.id)


async def sweep() -> list[asyncio.Task[None]]:
    """Start one task per due job. Returns the tasks so callers (tests) can await them.

    Jobs already in flight from an earlier tick are not started twice; a job whose
    session has a running turn is rescheduled ``SKIP_RETRY`` ahead instead.
    """
    store = get_store()
    now = datetime.now(UTC)
    tasks: list[asyncio.Task[None]] = []
    for job in await store.due_jobs(now):
        if job.id in _in_flight:
            continue
        _in_flight.add(job.id)
        task = asyncio.create_task(_run_job(store, job), name=f"cron-job-{job.id}")
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        tasks.append(task)
    return tasks


async def sweep_and_wait() -> None:
    """Run one tick and wait for every job it started (tests, CLI)."""
    tasks = await sweep()
    if tasks:
        await asyncio.gather(*tasks)


def start_cron() -> None:
    global _scheduler
    if _scheduler:
        return
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(sweep, "interval", seconds=30, id="orbweaver-cron")
    _scheduler.start()
    log.info("cron sweeper started")
