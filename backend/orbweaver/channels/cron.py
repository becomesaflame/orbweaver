from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from orbweaver.agent import agent_turn
from orbweaver.config import settings
from orbweaver.store import get_store
from orbweaver.workspace import make_workspace

log = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


def _parse_recurrence(rec: str | None, due: datetime) -> datetime | None:
    if not rec:
        return None
    rec = rec.strip().lower()
    if rec.startswith("every "):
        rec = rec[6:]
    if rec.endswith("s"):
        rec = rec[:-1]
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
                kind = str(sess.jsonld.get("workspace_kind") or "docker")
                uri = str(sess.jsonld.get("workspace_uri") or "workspace:default")
                ws = make_workspace(kind, uri, settings.workspace_root)
                await store.append_event(session_id, "cron", {"job_id": str(job.id)})
                await agent_turn(store, session_id, message, ws, workspace_kind=kind)
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
