"""One agent_turn per session at a time, across web, Telegram, cron and subagents (#100)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import anthropic
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver import turns
from orbweaver.agent import agent_turn
from orbweaver.app import app
from orbweaver.channels import cron
from orbweaver.channels import telegram as tg
from orbweaver.compact import ensure_tool_use_results, events_to_messages
from orbweaver.config import settings
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Job,
    reset_store_for_tests,
    session_at_id,
)
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


def _ws(tmp_path):
    return LocalWorkspace("workspace:default", str(tmp_path))


def _session(sid, **extra):
    jsonld = {
        "@id": session_at_id(sid),
        "@type": SESSION_TYPE,
        "workspace_uri": "workspace:default",
        "workspace_kind": "local",
        "status": "active",
    }
    jsonld.update(extra)
    return Entity(id=sid, at_id=session_at_id(sid), at_type=SESSION_TYPE, jsonld=jsonld)


class _ToolUse:
    def __init__(self, uid, name, inp):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _SlowAnthropic:
    """Each messages.create waits ``delay`` then returns the next scripted response."""

    def __init__(self, responses, delay=0.05):
        self._responses = list(responses)
        self.delay = delay
        self.calls = 0
        self.messages = self

    async def create(self, **_kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if not self._responses:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


async def _wait_until(pred, tries=200):
    for _ in range(tries):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def _assert_pairable(events):
    """Every tool_use in the prompt already has its tool_result; no stubs needed."""
    messages = events_to_messages(events)
    assert messages
    assert ensure_tool_use_results(messages) == messages
    tool_use_ids = [
        b["id"]
        for m in messages
        if m["role"] == "assistant" and isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    result_ids = [
        b["tool_use_id"]
        for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_use_ids
    assert sorted(tool_use_ids) == sorted(result_ids)


# --- registry ---------------------------------------------------------------


def test_acquire_is_exclusive_per_session():
    a, b = uuid4(), uuid4()
    first = turns.acquire(a, channel="web")
    assert first is not None
    assert turns.acquire(a, channel="telegram") is None
    assert turns.is_running(a)
    other = turns.acquire(b)
    assert other is not None
    turns.release(a, first)
    assert not turns.is_running(a)
    assert turns.is_running(b)
    # releasing with a stale handle does not evict the current holder
    again = turns.acquire(a)
    turns.release(a, first)
    assert turns.get(a) is again


@pytest.mark.asyncio
async def test_running_turn_raises_busy_with_holder_channel():
    sid = uuid4()
    async with turns.running_turn(sid, channel="cron"):
        with pytest.raises(turns.TurnBusy) as ei:
            async with turns.running_turn(sid, channel="telegram"):
                pass
        assert ei.value.channel == "cron"
        assert ei.value.session_id == sid
    assert not turns.is_running(sid)


@pytest.mark.asyncio
async def test_second_agent_turn_on_session_is_refused_and_log_stays_pairable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    (tmp_path / "notes.txt").write_text("hello\n")
    client = _SlowAnthropic(
        [
            SimpleNamespace(content=[_ToolUse("tu-1", "Read", {"path": "notes.txt"})]),
            SimpleNamespace(content=[_ToolUse("tu-2", "Read", {"path": "notes.txt"})]),
        ]
    )
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_session(sid))
    ws = _ws(tmp_path)

    async def locked_turn(text):
        async with turns.running_turn(sid, channel="test"):
            return await agent_turn(store, sid, text, ws)

    first = asyncio.create_task(locked_turn("read the notes"))
    await _wait_until(lambda: turns.is_running(sid) and client.calls >= 1)
    before = len(await store.list_events(sid))

    started = asyncio.get_running_loop().time()
    with pytest.raises(turns.TurnBusy):
        await locked_turn("and again")
    assert asyncio.get_running_loop().time() - started < client.delay
    # the refused turn wrote nothing, so the running turn's tool rounds stay contiguous
    assert len(await store.list_events(sid)) == before

    await first
    assert not turns.is_running(sid)
    events = await store.list_events(sid)
    assert [e.kind for e in events if e.kind == "user"] == ["user"]
    assert sum(1 for e in events if e.kind == "tool_call") == 2
    assert sum(1 for e in events if e.kind == "tool_result") == 2
    _assert_pairable(events)

    # once the lock is free the same session accepts a new turn again
    client._responses.append(
        SimpleNamespace(content=[SimpleNamespace(type="text", text="second turn")])
    )
    await locked_turn("follow up")
    assert [e.payload["text"] for e in await store.list_events(sid) if e.kind == "user"] == [
        "read the notes",
        "follow up",
    ]


# --- web ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_turn_refused_while_another_channel_holds_session(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        made = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        sid = made.json()["id"]
        state = turns.acquire(UUID(sid), channel="telegram")
        assert state is not None
        r = await client.post(
            f"/v1/sessions/{sid}/turns", json={"text": "hi"}, headers=auth_header
        )
        assert r.status_code == 409
        rewound = await client.post(
            f"/v1/sessions/{sid}/rewind", json={"from_seq": 1}, headers=auth_header
        )
        assert rewound.status_code == 409
        # inject reaches the Telegram-held turn through the same registry entry
        injected = await client.post(
            f"/v1/sessions/{sid}/turns/inject", json={"text": "also this"}, headers=auth_header
        )
        assert injected.status_code == 200
        assert state.inject.is_set()
        cancelled = await client.post(
            f"/v1/sessions/{sid}/turns/cancel", json={"discard": False}, headers=auth_header
        )
        assert cancelled.json()["status"] == "cancelling"
        assert state.cancel.is_set()


# --- telegram ----------------------------------------------------------------


def _telegram_update():
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


@pytest.mark.asyncio
async def test_telegram_handler_injects_instead_of_second_turn(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_session(sid, channel="telegram", telegram_user_id=1))
    fake_turn = AsyncMock(return_value=[])
    monkeypatch.setattr(tg, "agent_turn", fake_turn)

    async def fake_workspace(_update, _context):
        return store, sid, _ws(tmp_path), "local"

    monkeypatch.setattr(tg, "_session_workspace", fake_workspace)
    running = turns.acquire(sid, channel="web")
    assert running is not None

    update = _telegram_update()
    await tg._run_turn(update, MagicMock(), "second message")

    fake_turn.assert_not_awaited()
    assert turns.get(sid) is running  # the handler did not steal or drop the lock
    assert running.inject.is_set()
    events = await store.list_events(sid)
    assert [(e.kind, e.payload.get("text"), e.payload.get("injected")) for e in events] == [
        ("user", "second message", True)
    ]
    reply = update.message.reply_text.await_args.args[0]
    assert "already running" in reply
    assert "web" in reply


@pytest.mark.asyncio
async def test_telegram_handler_holds_lock_while_turn_runs(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    seen: dict = {}

    async def fake_turn(*_a, **kwargs):
        seen["running"] = turns.is_running(sid)
        seen["state"] = kwargs.get("turn_state")
        seen["cancel"] = kwargs.get("cancel")
        return []

    monkeypatch.setattr(tg, "agent_turn", fake_turn)

    async def fake_workspace(_update, _context):
        return store, sid, _ws(tmp_path), "local"

    monkeypatch.setattr(tg, "_session_workspace", fake_workspace)
    await tg._run_turn(_telegram_update(), MagicMock(), "hello")
    assert seen["running"] is True
    assert seen["state"] is not None
    assert seen["state"].channel == "telegram"
    assert seen["cancel"] is seen["state"].cancel
    assert not turns.is_running(sid)


@pytest.mark.asyncio
async def test_telegram_releases_lock_when_turn_raises(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()

    async def boom(*_a, **_k):
        raise RuntimeError("nope")

    monkeypatch.setattr(tg, "agent_turn", boom)

    async def fake_workspace(_update, _context):
        return store, sid, _ws(tmp_path), "local"

    monkeypatch.setattr(tg, "_session_workspace", fake_workspace)
    update = _telegram_update()
    await tg._run_turn(update, MagicMock(), "hello")
    assert not turns.is_running(sid)
    assert "Turn failed" in update.message.reply_text.await_args.args[0]


# --- cron --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_sweep_skips_session_with_running_turn(tmp_path, monkeypatch, caplog):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    fake_turn = AsyncMock(return_value=[])
    monkeypatch.setattr(cron, "agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_session(sid))
    job = Job(
        id=uuid4(),
        due_at=datetime.now(UTC) - timedelta(seconds=1),
        payload={"message": "nightly"},
        recurrence="day",
        session_id=sid,
    )
    await store.put_job(job)
    holder = turns.acquire(sid, channel="web")
    assert holder is not None

    before = datetime.now(UTC)
    with caplog.at_level(logging.INFO, logger="orbweaver.channels.cron"):
        await cron.sweep_and_wait()

    fake_turn.assert_not_awaited()
    assert await store.list_events(sid) == []
    kept = store.jobs[job.id]
    assert before + cron.SKIP_RETRY - timedelta(seconds=2) <= kept.due_at
    assert kept.due_at <= before + cron.SKIP_RETRY + timedelta(seconds=5)
    assert kept.recurrence == "day"
    assert any("skipped" in r.getMessage() and str(sid) in r.getMessage() for r in caplog.records)
    assert turns.get(sid) is holder

    # once the web turn ends, the retried job runs
    turns.release(sid, holder)
    kept.due_at = datetime.now(UTC) - timedelta(seconds=1)
    await store.put_job(kept)
    await cron.sweep_and_wait()
    fake_turn.assert_awaited_once()
    assert fake_turn.await_args.kwargs["turn_state"].channel == "cron"
    assert [e.kind for e in await store.list_events(sid)] == ["cron", "cron_result"]
    assert store.jobs[job.id].due_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_cron_due_jobs_run_concurrently_not_sequentially(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    active = {"now": 0, "peak": 0}

    async def slow_turn(*_a, **_k):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.05)
        active["now"] -= 1
        return []

    monkeypatch.setattr(cron, "agent_turn", slow_turn)
    sids = [uuid4() for _ in range(3)]
    for sid in sids:
        await store.put_entity(_session(sid))
        await store.put_job(
            Job(
                id=uuid4(),
                due_at=datetime.now(UTC) - timedelta(seconds=1),
                payload={"message": "go"},
                session_id=sid,
            )
        )
    tasks = await cron.sweep()
    assert len(tasks) == 3
    # the tick itself returned before any job finished
    assert active["now"] > 0 or active["peak"] == 0
    await asyncio.gather(*tasks)
    assert active["peak"] > 1
    assert active["peak"] <= cron.MAX_CONCURRENT_JOBS
    assert store.jobs == {}
    for sid in sids:
        assert not turns.is_running(sid)


@pytest.mark.asyncio
async def test_cron_does_not_start_in_flight_job_twice(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    gate = asyncio.Event()
    calls = {"n": 0}

    async def blocking_turn(*_a, **_k):
        calls["n"] += 1
        await gate.wait()
        return []

    monkeypatch.setattr(cron, "agent_turn", blocking_turn)
    sid = uuid4()
    await store.put_entity(_session(sid))
    await store.put_job(
        Job(
            id=uuid4(),
            due_at=datetime.now(UTC) - timedelta(seconds=1),
            payload={"message": "long"},
            session_id=sid,
        )
    )
    first = await cron.sweep()
    await _wait_until(lambda: calls["n"] == 1)
    second = await cron.sweep()  # same job still due while its task runs
    assert second == []
    gate.set()
    await asyncio.gather(*first)
    assert calls["n"] == 1


# --- subagent ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_child_session_is_registered_while_running(tmp_path, monkeypatch):
    from orbweaver.subagent import run_subagent

    store = reset_store_for_tests()
    parent = uuid4()
    await store.put_entity(_session(parent))
    seen: dict = {}

    async def fake_turn(store_, session_id, *_a, **_k):
        seen["child"] = session_id
        seen["registered"] = turns.get(session_id)
        await store_.append_event(session_id, "assistant", {"text": "ok"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", None)
    await run_subagent(
        {"task": "look around", "type": "explore"},
        {
            "workspace": _ws(tmp_path),
            "store": store,
            "session_id": parent,
            "workspace_kind": "local",
            "subagent_depth": 0,
        },
    )
    assert seen["child"] != parent
    assert seen["registered"] is not None
    assert seen["registered"].channel == "subagent"
    assert not turns.is_running(seen["child"])
    assert not turns.is_running(parent)


# --- postgres seq ------------------------------------------------------------


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.in_tx = False

    @asynccontextmanager
    async def transaction(self):
        self.in_tx = True
        try:
            yield
        finally:
            self.in_tx = False

    async def execute(self, sql, *args):
        self.calls.append(("execute" if self.in_tx else "execute-outside-tx", sql, args))

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval" if self.in_tx else "fetchval-outside-tx", sql, args))
        return 7


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


def test_pg_append_event_sql_computes_seq_in_the_insert():
    from orbweaver.pgstore import APPEND_EVENT_LOCK_SQL, APPEND_EVENT_SQL

    sql = " ".join(APPEND_EVENT_SQL.split())
    assert sql.startswith("INSERT INTO events(id, session_id, seq, kind, payload) SELECT")
    assert "COALESCE(MAX(seq), 0) + 1" in sql
    assert "FROM events WHERE session_id = $2" in sql
    assert sql.endswith("RETURNING seq")
    assert "pg_advisory_xact_lock" in APPEND_EVENT_LOCK_SQL


@pytest.mark.asyncio
async def test_pg_append_event_locks_session_and_inserts_once():
    from orbweaver.pgstore import APPEND_EVENT_LOCK_SQL, APPEND_EVENT_SQL, PostgresStore

    store = PostgresStore()
    conn = _FakeConn()
    store._pool = _FakePool(conn)  # type: ignore[assignment]
    sid = uuid4()
    ev = await store.append_event(sid, "tool_result", {"tool_use_id": "tu-1", "content": "x"})
    assert ev.seq == 7
    assert ev.kind == "tool_result"
    assert ev.session_id == sid
    assert [c[0] for c in conn.calls] == ["execute", "fetchval"]
    lock, insert = conn.calls
    assert lock[1] == APPEND_EVENT_LOCK_SQL
    assert lock[2] == (str(sid),)
    assert insert[1] == APPEND_EVENT_SQL
    assert insert[2][0] == ev.id
    assert insert[2][1] == sid
    assert insert[2][2] == "tool_result"
    assert '"tool_use_id": "tu-1"' in insert[2][3]
