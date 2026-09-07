from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from orbweaver.agent import run_tools
from orbweaver.channels.cron import _parse_recurrence, sweep
from orbweaver.config import settings
from orbweaver.store import SESSION_TYPE, Entity, Job, reset_store_for_tests, session_at_id
from orbweaver.workspace import LocalWorkspace


def test_recurrence_hour():
    due = datetime.now(UTC)
    nxt = _parse_recurrence("every hour", due)
    assert nxt is not None
    assert nxt - due == timedelta(hours=1)


@pytest.mark.asyncio
async def test_job_roundtrip():
    store = reset_store_for_tests()
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "ping"},
    )
    await store.put_job(job)
    due = await store.due_jobs(datetime.now(UTC))
    assert any(j.id == job.id for j in due)


@pytest.mark.asyncio
async def test_sweep_runs_due_job(tmp_path: Path, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    sid = uuid4()
    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:default",
                "workspace_kind": "local",
            },
        )
    )
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "do it"},
        session_id=sid,
    )
    await store.put_job(job)
    await sweep()
    events = await store.list_events(sid)
    assert any(e.kind == "cron" for e in events)
    assert any(e.kind == "assistant" for e in events)


@pytest.mark.asyncio
async def test_sweep_migrates_telegram_docker_workspace(tmp_path: Path, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    sid = uuid4()
    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:default",
                "workspace_kind": "docker",
                "telegram_user_id": 42,
                "telegram_chat_id": 42,
            },
        )
    )
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "do it"},
        session_id=sid,
    )
    await store.put_job(job)
    await sweep()
    sess = await store.get_entity(sid)
    assert sess is not None
    assert sess.jsonld["workspace_kind"] == "local"


@pytest.mark.asyncio
async def test_schedule_tool_persists_job(tmp_path: Path):
    store = reset_store_for_tests()
    sid = uuid4()
    due = datetime.now(UTC) + timedelta(hours=1)
    result = await run_tools(
        "ScheduleTask",
        {"due_at": due.isoformat(), "message": "ping later"},
        {
            "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
            "store": store,
            "session_id": sid,
        },
    )
    assert "job_id" in result
    jobs = await store.due_jobs(datetime.now(UTC) + timedelta(days=2))
    assert any(j.payload.get("message") == "ping later" for j in jobs)
