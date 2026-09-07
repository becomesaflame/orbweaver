from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from orbweaver.agent import agent_turn
from orbweaver.config import settings
from orbweaver.store import get_store
from orbweaver.workspace import bind_workspace

log = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


def _parse_recurrence(rec: str | None, due: datetime) -> datetime | None:
    if not rec:
        return None
    rec = rec.strip().lower()
    rec = rec.removeprefix("every ")
    rec = rec.removesuffix("s")
    mapping = {"minute": 60, "hour": 3600, "day": 86400}
    if rec in mapping:
        return due + timedelta(seconds=mapping[rec])
    return None


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
                    store, session_id, message, ws, workspace_kind=kind, headless=True
                )
                aborted = next((e for e in events if e.kind == "turn_aborted"), None)
                chat_id = sess.jsonld.get("telegram_chat_id")
                if aborted and chat_id:
                    from orbweaver.channels.telegram import notify_telegram_chat, texts_for_reply

                    await notify_telegram_chat(int(chat_id), texts_for_reply(events))
        nxt = _parse_recurrence(job.recurrence, job.due_at)
        if nxt:
            job.due_at = nxt
            await store.reschedule_job(job)
        else:
            job.due_at = datetime.now(UTC) + timedelta(days=36500)
            await store.reschedule_job(job)


def start_cron() -> None:
    global _scheduler
    if _scheduler:
        return
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(sweep, "interval", seconds=30, id="orbweaver-cron")
    _scheduler.start()
    log.info("cron sweeper started")
