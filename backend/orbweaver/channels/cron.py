from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from orbweaver.agent import agent_turn
from orbweaver.config import settings
from orbweaver.store import get_store
from orbweaver.workspace import bind_workspace

log = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None

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


def _result_text(events, aborted: bool) -> str:
    from orbweaver.channels.telegram import texts_for_reply

    text = texts_for_reply(events)
    if text:
        return text
    return "Turn aborted." if aborted else "Scheduled task finished."


async def _notify_originating_channel(store, sess, session_id, job, events) -> None:
    from orbweaver.channels.telegram import notify_telegram_chat

    aborted = any(e.kind == "turn_aborted" for e in events)
    text = _result_text(events, aborted)
    await store.append_event(
        session_id,
        "cron_result",
        {
            "job_id": str(job.id),
            "text": text,
            "status": "aborted" if aborted else "ok",
        },
    )
    chat_id = sess.jsonld.get("telegram_chat_id")
    if chat_id:
        await notify_telegram_chat(int(chat_id), text)


async def sweep() -> None:
    store = get_store()
    now = datetime.now(UTC)
    for job in await store.due_jobs(now):
        message = str(job.payload.get("message") or "scheduled task")
        session_id = job.session_id
        if session_id:
            sess = await store.get_entity(session_id)
            if sess:
                ws, kind, changed = bind_workspace(sess.jsonld, settings.workspace_root)
                if changed:
                    await store.put_entity(sess)
                await store.append_event(session_id, "cron", {"job_id": str(job.id)})
                events = await agent_turn(
                    store,
                    session_id,
                    message,
                    ws,
                    workspace_kind=kind,
                    headless=True,
                    channel="cron",
                )
                await _notify_originating_channel(store, sess, session_id, job, events)
        nxt = _parse_recurrence(job.recurrence, job.due_at, now=now)
        if nxt:
            job.due_at = nxt
            await store.reschedule_job(job)
        else:
            await store.delete_job(job.id)


def start_cron() -> None:
    global _scheduler
    if _scheduler:
        return
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(sweep, "interval", seconds=30, id="orbweaver-cron")
    _scheduler.start()
    log.info("cron sweeper started")
