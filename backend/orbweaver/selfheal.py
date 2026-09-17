"""Gateway intake for self-heal turns (issue #69).

Fingerprints process exceptions and some turn_aborted reasons, stores a ledger
entity per fingerprint, and enqueues a one-shot cron job. Opt-in via
``ORBWEAVER_SELFHEAL``. The agent still does the coding; this module only
decides whether to spend a turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from orbweaver.config import settings
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Event,
    Job,
    Store,
    get_store,
    new_uuid,
    session_at_id,
)
from orbweaver.uris import resolve_workspace_uri, validate_workspace_uri

log = logging.getLogger(__name__)

AT_TYPE = "SelfHealAttempt"
HANDLER_NAME = "orbweaver-selfheal"
REPO_SSH = "git@github.com:becomesaflame/orbweaver.git"
NOOP_MARK = "SELFHEAL_NOOP"
_PR_URL = re.compile(r"https://github.com/[^/\s]+/[^/\s]+/pull/\d+")
_SKIP_EXC_NAMES = frozenset({"CancelledError", "TurnCancelled", "KeyboardInterrupt", "SystemExit"})
# Expected headless/user-loop aborts — not source bugs to patch.
_SKIP_ABORT_REASONS = frozenset(
    {
        "ask_required_headless",
        "ask_user_headless",
        "hook_cancel",
        "stuck",
        "stop",
        "cancelled",
        "parent_cancelled",
        "parent_turn_ended",
    }
)

_loop: asyncio.AbstractEventLoop | None = None
_gate = asyncio.Lock()


def bind_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    global _loop
    _loop = loop


def attempt_at_id(fingerprint: str) -> str:
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", fingerprint).strip("-")[:120]
    return f"urn:orbweaver:selfheal:{slug or 'unknown'}-{digest}"


def _is_orbweaver_frame(filename: str) -> bool:
    parts = Path(filename).parts
    if "tests" in parts:
        return False
    try:
        idx = parts.index("orbweaver")
    except ValueError:
        return False
    return idx + 1 < len(parts)  # backend/orbweaver/<module>


def fingerprint_exc(exc: BaseException, tb: Any | None = None) -> str:
    """Stable id: ``ExcType:file.py:function`` from the innermost orbweaver frame."""
    stack = tb if tb is not None else exc.__traceback__
    frames = traceback.extract_tb(stack) if stack is not None else []
    chosen = None
    for frame in reversed(frames):
        if _is_orbweaver_frame(frame.filename):
            chosen = frame
            break
    if chosen is None and frames:
        chosen = frames[-1]
    name = type(exc).__name__
    if chosen is None:
        return name
    return f"{name}:{Path(chosen.filename).name}:{chosen.name}"


def fingerprint_abort(reason: str) -> str:
    code = (reason or "unknown").strip() or "unknown"
    return f"turn_aborted:{code}"


def _parse_dt(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


def _traceback_text(exc: BaseException, tb: Any | None = None) -> str:
    stack = tb if tb is not None else exc.__traceback__
    return "".join(traceback.format_exception(type(exc), exc, stack))[-4000:]


def build_prompt(
    fingerprint: str,
    *,
    traceback_text: str = "",
    abort_payload: dict[str, Any] | None = None,
    logger_name: str = "",
    count: int = 1,
) -> str:
    abort_payload = abort_payload or {}
    lines = [
        "You are Orbweaver's self-heal session. Fix ONE production failure in this checkout.",
        f"Fingerprint: {fingerprint}",
        f"Occurrences so far: {count}",
    ]
    if logger_name:
        lines.append(f"Logger: {logger_name}")
    if abort_payload:
        lines.append(f"turn_aborted reason: {abort_payload.get('reason')}")
        if abort_payload.get("text"):
            lines.append(str(abort_payload["text"])[:1500])
        if abort_payload.get("last_tool"):
            lines.append(
                f"Last tool: {abort_payload.get('last_tool')} "
                f"{abort_payload.get('last_input') or ''}"
            )
    if traceback_text.strip():
        lines.append("Traceback:")
        lines.append(traceback_text.strip()[-3500:])
    lines.extend(
        [
            "",
            "Do this:",
            f"1. This workspace should be a clone of {REPO_SSH}. If `.git` is missing, clone it here.",
            "2. Load the Skill named self-heal if it is listed, then follow it.",
            (
                "3. Skip if git already has this fingerprint as a selfheal branch or PR; "
                f"reply {NOOP_MARK} and stop."
            ),
            (
                "4. git fetch origin && git checkout -B selfheal/<slug> origin/main "
                "(slug = fingerprint with unsafe chars as -)."
            ),
            "5. Write a failing test that reproduces this error class on the same code path, then fix it.",
            "6. python -m pytest tests -q, bump PATCH in backend/orbweaver/__init__.py.",
            (
                "7. Push the feature branch. Open a draft PR with fingerprint, traceback, diagnosis, "
                "and tests. You may gh pr ready and gh pr merge --squash --auto. Never push main."
            ),
            "8. Never edit `/home/orbweaver/orbweaver`, systemd, or deploy.",
            f"9. If there is nothing to fix, reply {NOOP_MARK}.",
            "Reply with the PR URL or SELFHEAL_NOOP.",
        ]
    )
    return "\n".join(lines)


def extract_pr_url(text: str) -> str | None:
    match = _PR_URL.search(text or "")
    return match.group(0) if match else None


def _is_selfheal_session(sess: Entity | None) -> bool:
    if sess is None:
        return False
    want = (settings.orbweaver_selfheal_workspace or "").strip()
    got = str(sess.jsonld.get("workspace_uri") or "").strip()
    return bool(want) and got == want


async def _enqueued_today(store: Store, now: datetime) -> int:
    day = now.date().isoformat()
    n = 0
    for ent in await store.list_entities(AT_TYPE):
        raw = str(ent.jsonld.get("last_enqueued_at") or "")
        if raw.startswith(day):
            n += 1
    return n


async def _ensure_workspace() -> None:
    uri = validate_workspace_uri(settings.orbweaver_selfheal_workspace)
    path = resolve_workspace_uri(uri, settings.workspace_root)
    path.mkdir(parents=True, exist_ok=True)


async def _session_for(store: Store, fingerprint: str, prior: Entity | None) -> Entity:
    raw_sid = (prior.jsonld.get("session_id") if prior else None) or ""
    if raw_sid:
        try:
            existing = await store.get_entity(UUID(str(raw_sid)))
        except ValueError:
            existing = None
        if existing is not None:
            return existing
    await _ensure_workspace()
    uid = new_uuid()
    uri = validate_workspace_uri(settings.orbweaver_selfheal_workspace)
    jsonld: dict[str, Any] = {
        "@id": session_at_id(uid),
        "@type": SESSION_TYPE,
        "workspace_uri": uri,
        "workspace_kind": "local",
        "title": f"selfheal {fingerprint}"[:120],
        "status": "active",
        "channel": "cron",
        "created_at": datetime.now(UTC).isoformat(),
    }
    chat = int(settings.orbweaver_selfheal_telegram_chat_id or 0)
    if chat:
        jsonld["telegram_chat_id"] = chat
    sess = Entity(id=uid, at_id=session_at_id(uid), at_type=SESSION_TYPE, jsonld=jsonld)
    await store.put_entity(sess)
    return sess


async def _queued_job(store: Store, fingerprint: str) -> Job | None:
    now = datetime.now(UTC)
    for job in await store.due_jobs(now + timedelta(days=3650)):
        if job.payload.get("fingerprint") == fingerprint:
            return job
    return None


def _should_skip(ent: Entity | None, now: datetime, cooldown: timedelta) -> str | None:
    if ent is None:
        return None
    state = str(ent.jsonld.get("state") or "")
    last = _parse_dt(ent.jsonld.get("last_enqueued_at"))
    if state == "rejected":
        return "rejected"
    if state == "open":
        return "open"
    if last is not None and now - last < cooldown:
        if state in {"attempting", "cooling-down", "merged"}:
            return state
        return "cooldown"
    return None


async def maybe_enqueue(
    fingerprint: str,
    *,
    traceback_text: str = "",
    abort_payload: dict[str, Any] | None = None,
    logger_name: str = "",
    source_session_id: UUID | None = None,
    store: Store | None = None,
) -> Job | None:
    """Create or skip a self-heal job. Returns the new job, or None if skipped."""
    if not settings.orbweaver_selfheal:
        return None
    fp = (fingerprint or "").strip()
    if not fp:
        return None
    db = store or get_store()
    async with _gate:
        if source_session_id is not None:
            src = await db.get_entity(source_session_id)
            if _is_selfheal_session(src):
                return None
        now = datetime.now(UTC)
        cooldown = timedelta(seconds=max(0.0, float(settings.orbweaver_selfheal_cooldown_s)))
        at_id = attempt_at_id(fp)
        ent = await db.get_entity_by_at_id(at_id)
        if await _queued_job(db, fp):
            if ent is not None:
                ent.jsonld["count"] = int(ent.jsonld.get("count") or 1) + 1
                ent.jsonld["last_seen"] = now.isoformat()
                await db.put_entity(ent)
            return None
        why = _should_skip(ent, now, cooldown)
        if why:
            if ent is not None:
                ent.jsonld["count"] = int(ent.jsonld.get("count") or 1) + 1
                ent.jsonld["last_seen"] = now.isoformat()
                await db.put_entity(ent)
            log.info("selfheal skip %s (%s)", fp, why)
            return None
        cap = max(0, int(settings.orbweaver_selfheal_daily_cap))
        if cap and await _enqueued_today(db, now) >= cap:
            log.info("selfheal skip %s (daily cap %s)", fp, cap)
            return None
        sess = await _session_for(db, fp, ent)
        count = int((ent.jsonld.get("count") if ent else 0) or 0) + 1
        message = build_prompt(
            fp,
            traceback_text=traceback_text,
            abort_payload=abort_payload,
            logger_name=logger_name,
            count=count,
        )
        job = Job(
            id=new_uuid(),
            due_at=now,
            payload={"message": message, "fingerprint": fp, "selfheal": True},
            session_id=sess.id,
        )
        await db.put_job(job)
        body = {
            "@id": at_id,
            "@type": AT_TYPE,
            "fingerprint": fp,
            "state": "attempting",
            "count": count,
            "attempts": int((ent.jsonld.get("attempts") if ent else 0) or 0) + 1,
            "first_seen": (ent.jsonld.get("first_seen") if ent else None) or now.isoformat(),
            "last_seen": now.isoformat(),
            "last_enqueued_at": now.isoformat(),
            "session_id": str(sess.id),
            "pr_url": (ent.jsonld.get("pr_url") if ent else None),
        }
        if ent is None:
            ent = Entity(id=new_uuid(), at_id=at_id, at_type=AT_TYPE, jsonld=body)
        else:
            ent.jsonld.update(body)
        await db.put_entity(ent)
        log.info("selfheal enqueue %s job=%s session=%s", fp, job.id, sess.id)
        return job


async def note_exception(
    exc: BaseException,
    tb: Any | None = None,
    *,
    message: str = "",
    logger_name: str = "",
    source_session_id: UUID | None = None,
) -> Job | None:
    if type(exc).__name__ in _SKIP_EXC_NAMES:
        return None
    try:
        fp = fingerprint_exc(exc, tb)
        text = _traceback_text(exc, tb)
        if message and message not in text:
            text = f"{message}\n{text}"
        return await maybe_enqueue(
            fp,
            traceback_text=text,
            logger_name=logger_name,
            source_session_id=source_session_id,
        )
    except Exception:
        log.exception("selfheal note_exception failed")
        return None


async def note_abort(session_id: UUID, payload: dict[str, Any]) -> Job | None:
    reason = str(payload.get("reason") or "")
    if reason in _SKIP_ABORT_REASONS:
        return None
    try:
        return await maybe_enqueue(
            fingerprint_abort(reason),
            abort_payload=dict(payload),
            source_session_id=session_id,
        )
    except Exception:
        log.exception("selfheal note_abort failed")
        return None


def _event_text(events: list[Event]) -> str:
    parts: list[str] = []
    for ev in events:
        if ev.kind in {"assistant", "cron_result", "turn_aborted"}:
            text = ev.payload.get("text") or ev.payload.get("content") or ""
            if text:
                parts.append(str(text))
    return "\n".join(parts)


async def finish_attempt(job: Job, events: list[Event], store: Store | None = None) -> None:
    fp = str(job.payload.get("fingerprint") or "")
    if not fp or not job.payload.get("selfheal"):
        return
    db = store or get_store()
    ent = await db.get_entity_by_at_id(attempt_at_id(fp))
    if ent is None:
        return
    blob = _event_text(events)
    pr = extract_pr_url(blob)
    if pr:
        ent.jsonld["state"] = "open"
        ent.jsonld["pr_url"] = pr
    else:
        ent.jsonld["state"] = "cooling-down"
    ent.jsonld["last_seen"] = datetime.now(UTC).isoformat()
    await db.put_entity(ent)


class SelfHealLogHandler(logging.Handler):
    """Enqueue from ``log.exception`` on ``orbweaver.*`` (not this module)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not settings.orbweaver_selfheal:
                return
            if record.name == "orbweaver.selfheal" or record.name.startswith("orbweaver.selfheal."):
                return
            info = record.exc_info
            if not info or info[1] is None:
                return
            exc = info[1]
            tb = info[2]
            loop = _loop
            if loop is None or not loop.is_running():
                return
            coro = note_exception(exc, tb, message=record.getMessage(), logger_name=record.name)
            loop.call_soon_threadsafe(loop.create_task, coro)
        except Exception:
            self.handleError(record)


def attach_log_handler() -> None:
    logger = logging.getLogger("orbweaver")
    if any(getattr(h, "name", None) == HANDLER_NAME for h in logger.handlers):
        return
    handler = SelfHealLogHandler()
    handler.name = HANDLER_NAME
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)


def detach_log_handler() -> None:
    logger = logging.getLogger("orbweaver")
    for handler in list(logger.handlers):
        if getattr(handler, "name", None) == HANDLER_NAME:
            logger.removeHandler(handler)
            handler.close()


def reset_for_tests() -> None:
    bind_loop(None)
    detach_log_handler()
