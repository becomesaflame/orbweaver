"""Self-heal gateway intake (issue #69)."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from orbweaver.selfheal import (
    NOOP_MARK,
    attempt_at_id,
    extract_pr_url,
    fingerprint_abort,
    fingerprint_exc,
    finish_attempt,
    maybe_enqueue,
    note_abort,
    note_exception,
    reset_for_tests,
)
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.uris import WorkspaceURIError, validate_workspace_uri


@pytest.fixture(autouse=True)
def _selfheal(monkeypatch, tmp_path):
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_selfheal", True)
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "orbweaver_selfheal_cooldown_s", 86_400.0)
    monkeypatch.setattr(settings, "orbweaver_selfheal_daily_cap", 3)
    reset_store_for_tests()
    reset_for_tests()
    yield
    reset_for_tests()


def test_fingerprint_prefers_innermost_orbweaver_frame():
    try:
        validate_workspace_uri("/not/allowed")
    except WorkspaceURIError as exc:
        fp = fingerprint_exc(exc)
    assert fp.startswith("WorkspaceURIError:uris.py:")
    assert "validate_workspace_uri" in fp


def test_fingerprint_abort_reason():
    assert fingerprint_abort("classifier_denial_limit") == "turn_aborted:classifier_denial_limit"


def test_extract_pr_url():
    text = "opened https://github.com/becomesaflame/orbweaver/pull/165 for review"
    assert extract_pr_url(text) == "https://github.com/becomesaflame/orbweaver/pull/165"


@pytest.mark.asyncio
async def test_disabled_does_not_enqueue(monkeypatch):
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_selfheal", False)
    store = reset_store_for_tests()
    job = await maybe_enqueue("KeyError:agent.py:run_tools", store=store)
    assert job is None
    assert await store.due_jobs(datetime.now(UTC) + timedelta(days=1)) == []


@pytest.mark.asyncio
async def test_enqueue_creates_session_job_and_ledger(tmp_path):
    store = reset_store_for_tests()
    fp = "KeyError:agent.py:run_tools"
    job = await maybe_enqueue(fp, traceback_text="Traceback: KeyError", store=store)
    assert job is not None
    assert job.payload["selfheal"] is True
    assert job.payload["fingerprint"] == fp
    assert "KeyError:agent.py:run_tools" in job.payload["message"]
    assert "Traceback: KeyError" in job.payload["message"]
    sess = await store.get_entity(job.session_id)
    assert sess is not None
    assert sess.jsonld["workspace_uri"] == "workspace:orbweaver-selfheal"
    assert (tmp_path / "orbweaver-selfheal").is_dir()
    ent = await store.get_entity_by_at_id(attempt_at_id(fp))
    assert ent is not None
    assert ent.jsonld["state"] == "attempting"
    assert ent.jsonld["count"] == 1


@pytest.mark.asyncio
async def test_duplicate_fingerprint_bumps_count_without_second_job():
    store = reset_store_for_tests()
    fp = "RuntimeError:app.py:turn"
    first = await maybe_enqueue(fp, store=store)
    second = await maybe_enqueue(fp, store=store)
    assert first is not None
    assert second is None
    jobs = await store.due_jobs(datetime.now(UTC) + timedelta(days=1))
    assert len([j for j in jobs if j.payload.get("fingerprint") == fp]) == 1
    ent = await store.get_entity_by_at_id(attempt_at_id(fp))
    assert ent is not None
    assert ent.jsonld["count"] == 2
    assert ent.jsonld["attempts"] == 1


@pytest.mark.asyncio
async def test_daily_cap_blocks_new_fingerprints():
    store = reset_store_for_tests()
    from orbweaver.config import settings

    settings.orbweaver_selfheal_daily_cap = 1
    assert await maybe_enqueue("ValueError:a.py:f", store=store) is not None
    assert await maybe_enqueue("TypeError:b.py:g", store=store) is None


@pytest.mark.asyncio
async def test_skips_errors_from_selfheal_session():
    store = reset_store_for_tests()
    first = await maybe_enqueue("OSError:sandbox.py:run", store=store)
    assert first is not None
    again = await maybe_enqueue(
        "PermissionError:other.py:x",
        source_session_id=first.session_id,
        store=store,
    )
    assert again is None


@pytest.mark.asyncio
async def test_rejected_never_retries():
    store = reset_store_for_tests()
    fp = "KeyError:done.py:f"
    job = await maybe_enqueue(fp, store=store)
    assert job is not None
    await store.delete_job(job.id)
    ent = await store.get_entity_by_at_id(attempt_at_id(fp))
    assert ent is not None
    ent.jsonld["state"] = "rejected"
    await store.put_entity(ent)
    assert await maybe_enqueue(fp, store=store) is None


@pytest.mark.asyncio
async def test_note_abort_skips_headless_ask():
    store = reset_store_for_tests()
    sid = uuid4()
    await note_abort(sid, {"reason": "ask_required_headless", "text": "need a human"})
    assert await store.due_jobs(datetime.now(UTC) + timedelta(days=1)) == []


@pytest.mark.asyncio
async def test_note_abort_enqueues_unexpected_reason():
    store = reset_store_for_tests()
    from orbweaver.store import SESSION_TYPE, Entity, session_at_id

    sid = uuid4()
    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:other",
            },
        )
    )
    job = await note_abort(sid, {"reason": "classifier_denial_limit", "text": "too many denials"})
    assert job is not None
    assert job.payload["fingerprint"] == "turn_aborted:classifier_denial_limit"


@pytest.mark.asyncio
async def test_note_exception_from_gateway_frame():
    reset_store_for_tests()
    try:
        validate_workspace_uri("/etc/shadow")
    except WorkspaceURIError as exc:
        job = await note_exception(exc, message="websocket turn failed", logger_name="orbweaver.app")
    assert job is not None
    assert "WorkspaceURIError:uris.py:" in job.payload["fingerprint"]
    assert "websocket turn failed" in job.payload["message"]


@pytest.mark.asyncio
async def test_finish_attempt_records_pr_url():
    store = reset_store_for_tests()
    fp = "KeyError:agent.py:run_tools"
    job = await maybe_enqueue(fp, store=store)
    assert job is not None
    events = [
        Event(
            id=uuid4(),
            session_id=job.session_id,
            seq=1,
            kind="assistant",
            payload={"text": "https://github.com/becomesaflame/orbweaver/pull/99"},
        )
    ]
    await finish_attempt(job, events, store)
    ent = await store.get_entity_by_at_id(attempt_at_id(fp))
    assert ent is not None
    assert ent.jsonld["state"] == "open"
    assert ent.jsonld["pr_url"].endswith("/pull/99")


@pytest.mark.asyncio
async def test_finish_attempt_noop_cools_down():
    store = reset_store_for_tests()
    fp = "KeyError:agent.py:run_tools"
    job = await maybe_enqueue(fp, store=store)
    assert job is not None
    events = [
        Event(
            id=uuid4(),
            session_id=job.session_id,
            seq=1,
            kind="assistant",
            payload={"text": NOOP_MARK},
        )
    ]
    await finish_attempt(job, events, store)
    ent = await store.get_entity_by_at_id(attempt_at_id(fp))
    assert ent.jsonld["state"] == "cooling-down"


@pytest.mark.asyncio
async def test_log_exception_enqueues():
    from orbweaver.selfheal import attach_log_handler, bind_loop

    store = reset_store_for_tests()
    bind_loop(asyncio.get_running_loop())
    attach_log_handler()
    try:
        validate_workspace_uri("/nope")
    except WorkspaceURIError:
        logging.getLogger("orbweaver.app").exception("websocket turn failed")
    deadline = asyncio.get_running_loop().time() + 2
    jobs: list = []
    while asyncio.get_running_loop().time() < deadline:
        jobs = await store.due_jobs(datetime.now(UTC) + timedelta(days=1))
        if jobs:
            break
        await asyncio.sleep(0)
    assert jobs
    assert jobs[0].payload.get("selfheal") is True


@pytest.mark.asyncio
async def test_cron_finish_updates_ledger(monkeypatch):
    from orbweaver.channels.cron import _run_job_turn

    store = reset_store_for_tests()
    fp = "KeyError:agent.py:run_tools"
    job = await maybe_enqueue(fp, store=store)
    assert job is not None
    sess = await store.get_entity(job.session_id)
    assert sess is not None

    async def fake_turn(*_a, **_k):
        return [
            Event(
                id=uuid4(),
                session_id=job.session_id,
                seq=2,
                kind="assistant",
                payload={"text": "done https://github.com/becomesaflame/orbweaver/pull/7"},
            )
        ]

    monkeypatch.setattr("orbweaver.channels.cron.agent_turn", fake_turn)
    cron_pings: list = []

    async def _cron_notify(*_a, **_k):
        cron_pings.append(1)

    monkeypatch.setattr("orbweaver.channels.cron._notify_originating_channel", _cron_notify)
    pings: list[str] = []

    async def _ping(text: str) -> None:
        pings.append(text)

    monkeypatch.setattr("orbweaver.selfheal.notify_selfheal", _ping)
    await _run_job_turn(store, sess, job.session_id, job, job.payload["message"])
    ent = await store.get_entity_by_at_id(attempt_at_id(fp))
    assert ent.jsonld["pr_url"].endswith("/pull/7")
    assert cron_pings == []
    assert any("pull/7" in t and "Self-heal opened a PR" in t for t in pings)


@pytest.mark.asyncio
async def test_enqueue_notifies_telegram(monkeypatch):
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_selfheal_telegram_chat_id", 99)
    pings: list[str] = []

    async def _ping(text: str) -> None:
        pings.append(text)

    monkeypatch.setattr("orbweaver.selfheal.notify_selfheal", _ping)
    store = reset_store_for_tests()
    job = await maybe_enqueue(
        "KeyError:agent.py:run_tools",
        traceback_text="Traceback: KeyError: x",
        logger_name="orbweaver.app",
        store=store,
    )
    assert job is not None
    assert len(pings) == 1
    assert "Self-heal triggered: KeyError:agent.py:run_tools" in pings[0]
    assert "logger=orbweaver.app" in pings[0]
    assert "KeyError: x" in pings[0]
    assert await maybe_enqueue("KeyError:agent.py:run_tools", store=store) is None
    assert len(pings) == 1


@pytest.mark.asyncio
async def test_finish_noop_does_not_notify_pr(monkeypatch):
    pings: list[str] = []

    async def _ping(text: str) -> None:
        pings.append(text)

    monkeypatch.setattr("orbweaver.selfheal.notify_selfheal", _ping)
    store = reset_store_for_tests()
    fp = "KeyError:agent.py:run_tools"
    job = await maybe_enqueue(fp, store=store)
    assert job is not None
    pings.clear()
    events = [
        Event(
            id=uuid4(),
            session_id=job.session_id,
            seq=1,
            kind="assistant",
            payload={"text": NOOP_MARK},
        )
    ]
    await finish_attempt(job, events, store)
    assert pings == []
