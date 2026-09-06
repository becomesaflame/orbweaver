import json
from uuid import uuid4

import pytest

from orbweaver.agent import maybe_compact
from orbweaver.compact import (
    events_to_messages,
    live_events,
    persist_tool_result,
    prompt_events,
    record_usage,
    rehydrate_messages,
    reset_compact_state,
)
from orbweaver.compact.usage import compact_failures, estimate_prompt_tokens, record_compact_failure
from orbweaver.config import settings
from orbweaver.store import SESSION_TYPE, Entity, new_uuid, reset_store_for_tests, session_at_id
from orbweaver.workspace import LocalWorkspace


def _session(store, sid):
    return store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:default",
            },
        )
    )


@pytest.mark.asyncio
async def test_compact_keeps_events_and_projects_prompt(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    await _session(store, sid)
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    ev = await maybe_compact(store, sid)
    assert ev is not None
    assert ev.kind == "compact_boundary"
    events = await store.list_events(sid)
    assert len(events) == 41
    assert events[-1].kind == "compact_boundary"
    live = live_events(events)
    assert live[0].kind == "compact_boundary"
    assert len(live) < 40
    kinds = [e.kind for e in events]
    assert kinds.count("user") == 40


@pytest.mark.asyncio
async def test_microcompact_stubs_old_tool_results_not_store():
    store = reset_store_for_tests()
    sid = new_uuid()
    await _session(store, sid)
    for i in range(8):
        tid = f"t{i}"
        await store.append_event(sid, "tool_call", {"id": tid, "name": "Bash", "input": {"command": "x"}})
        await store.append_event(
            sid,
            "tool_result",
            {"tool_use_id": tid, "name": "Bash", "content": f"output-{i}-" + "x" * 50},
        )
    events = await store.list_events(sid)
    projected = prompt_events(events)
    stubs = [
        e
        for e in projected
        if e.kind == "tool_result" and "cleared" in str(e.payload.get("content"))
    ]
    assert stubs
    stored = await store.list_events(sid)
    assert all("cleared" not in str(e.payload.get("content")) for e in stored if e.kind == "tool_result")


def test_persist_writes_preview_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "compact_tool_result_chars", 100)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    big = "a" * 500
    stored, rel = persist_tool_result(ws, "toolu_1", "Bash", big)
    assert rel == ".orbweaver/tool-results/toolu_1.txt"
    assert "persisted-output" in stored
    assert ws.read(rel) == big
    skipped, rel2 = persist_tool_result(ws, "toolu_2", "Read", big)
    assert rel2 is None
    assert skipped == big


@pytest.mark.asyncio
async def test_session_notes_used_as_summary(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    ent = Entity(
        id=sid,
        at_id=session_at_id(sid),
        at_type=SESSION_TYPE,
        jsonld={
            "@id": session_at_id(sid),
            "@type": SESSION_TYPE,
            "workspace_uri": "workspace:default",
            "session_notes": "# Current task\nShip compaction\n# Files\nagent.py\n",
        },
    )
    await store.put_entity(ent)
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    ev = await maybe_compact(store, sid)
    assert ev is not None
    assert ev.payload["trigger"] == "session_notes"
    assert "Ship compaction" in ev.payload["text"]


@pytest.mark.asyncio
async def test_llm_failure_falls_back_extractive(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    sid = new_uuid()
    await _session(store, sid)
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})

    class Boom:
        async def messages_create(self, **_k):
            raise RuntimeError("nope")

        @property
        def messages(self):
            return self

        async def create(self, **_k):
            raise RuntimeError("nope")

    ev = await maybe_compact(store, sid, client=Boom(), system=[{"type": "text", "text": "x"}])
    assert ev is not None
    assert ev.payload["trigger"] == "extractive"
    assert compact_failures(sid) == 1


@pytest.mark.asyncio
async def test_circuit_breaker_skips_llm_after_failures(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "compact_max_failures", 3)
    sid = new_uuid()
    await _session(store, sid)
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    for _ in range(3):
        record_compact_failure(sid)
    called = {"n": 0}

    class Probe:
        async def create(self, **_k):
            called["n"] += 1
            raise AssertionError("llm should not run")

        @property
        def messages(self):
            return self

    ev = await maybe_compact(store, sid, client=Probe(), system=[{"type": "text", "text": "x"}])
    assert ev is not None
    assert called["n"] == 0
    assert ev.payload["trigger"] == "extractive"


def test_usage_anchor_plus_delta():
    reset_compact_state()
    sid = uuid4()
    events = []
    for i in range(3):
        from orbweaver.store import Event

        events.append(
            Event(id=uuid4(), session_id=sid, seq=i + 1, kind="user", payload={"text": "aa"})
        )
    record_usage(sid, 1000, at_seq=2)
    total = estimate_prompt_tokens(sid, events)
    assert total > 1000
    assert total < 1000 + 50


def test_rehydrate_injects_recent_read(tmp_path):
    from orbweaver.store import Event

    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "print('hi')\n")
    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={"id": "t1", "name": "Read", "input": {"path": "src/a.py"}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_result",
            payload={"tool_use_id": "t1", "name": "Read", "content": "print('hi')"},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="compact_boundary",
            payload={"text": "did some reading", "keep_from_seq": 4},
        ),
        Event(id=uuid4(), session_id=sid, seq=4, kind="user", payload={"text": "continue"}),
    ]
    messages = events_to_messages(live_events(events))
    out = rehydrate_messages(messages, events, ws)
    blob = json.dumps(out)
    assert "src/a.py" in blob
    assert "print('hi')" in blob
