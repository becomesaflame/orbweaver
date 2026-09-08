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


def test_recurrence_aliases():
    due = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    assert _parse_recurrence("minute", due) == due + timedelta(minutes=1)
    assert _parse_recurrence("hours", due) == due + timedelta(hours=1)
    assert _parse_recurrence("every day", due) == due + timedelta(days=1)
    assert _parse_recurrence(None, due) is None
    assert _parse_recurrence("", due) is None
    assert _parse_recurrence("whenever", due) is None


def test_recurrence_cron_weekday_morning():
    due = datetime(2026, 9, 7, 10, 0, tzinfo=UTC)  # Monday
    nxt = _parse_recurrence("0 9 * * tue", due)
    assert nxt is not None
    assert nxt == datetime(2026, 9, 8, 9, 0, tzinfo=UTC)


def test_recurrence_cron_hourly_and_invalid():
    due = datetime(2026, 9, 7, 10, 30, tzinfo=UTC)
    nxt = _parse_recurrence("0 * * * *", due)
    assert nxt == datetime(2026, 9, 7, 11, 0, tzinfo=UTC)
    assert _parse_recurrence("0 9 *", due) is None
    assert _parse_recurrence("* * * * * *", due) is None
    assert _parse_recurrence("99 9 * * *", due) is None


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


@pytest.mark.asyncio
async def test_schedule_tool_rejects_bad_recurrence(tmp_path: Path):
    store = reset_store_for_tests()
    sid = uuid4()
    due = datetime.now(UTC) + timedelta(hours=1)
    result = await run_tools(
        "ScheduleTask",
        {"due_at": due.isoformat(), "message": "nope", "recurrence": "whenever"},
        {
            "workspace": LocalWorkspace("workspace:default", str(tmp_path)),
            "store": store,
            "session_id": sid,
        },
    )
    assert "invalid recurrence" in result
    assert await store.due_jobs(datetime.now(UTC) + timedelta(days=2)) == []


@pytest.mark.asyncio
async def test_sweep_notifies_telegram_on_success(tmp_path: Path, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    sent: list[tuple[int, str]] = []

    async def fake_notify(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr("orbweaver.channels.telegram.notify_telegram_chat", fake_notify)
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
                "telegram_chat_id": 99,
            },
        )
    )
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "check CI"},
        session_id=sid,
    )
    await store.put_job(job)
    await sweep()
    assert sent
    assert sent[0][0] == 99
    assert "check CI" in sent[0][1]
    events = await store.list_events(sid)
    result = next(e for e in events if e.kind == "cron_result")
    assert result.payload["status"] == "ok"
    assert "check CI" in result.payload["text"]
    assert job.id not in store.jobs


@pytest.mark.asyncio
async def test_sweep_deletes_finished_oneshot():
    store = reset_store_for_tests()
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "once"},
    )
    await store.put_job(job)
    await sweep()
    assert job.id not in store.jobs
    assert await store.due_jobs(datetime.now(UTC) + timedelta(days=40000)) == []


@pytest.mark.asyncio
async def test_sweep_reschedules_daily(tmp_path: Path, monkeypatch):
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
    due = datetime.now(UTC) - timedelta(seconds=1)
    job = Job(
        id=uuid4(),
        due_at=due,
        payload={"message": "daily ping"},
        recurrence="day",
        session_id=sid,
    )
    await store.put_job(job)
    await sweep()
    kept = store.jobs[job.id]
    assert kept.due_at > datetime.now(UTC)
    assert kept.due_at >= due + timedelta(days=1) - timedelta(seconds=2)
