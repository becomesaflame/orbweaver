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
from orbweaver.compact.project import ensure_tool_use_results, unpaired_tool_use_ids
from orbweaver.compact.usage import (
    compact_failures,
    estimate_prompt_tokens,
    event_token_count,
    record_compact_failure,
)
from orbweaver.config import settings
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    new_uuid,
    reset_store_for_tests,
    session_at_id,
)
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
    web, rel3 = persist_tool_result(ws, "toolu_3", "WebFetch", big)
    assert rel3 is None
    assert web == big
    search, rel4 = persist_tool_result(ws, "toolu_4", "WebSearch", big)
    assert rel4 is None
    assert search == big


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


@pytest.mark.asyncio
async def test_maybe_compact_uses_recorded_usage_not_payload_estimate(monkeypatch):
    """usage.input_tokens above threshold must compact even when the payload estimate is low.

    Production: system+tools, images, and MCP schemas are in the API usage
    figure but missing from event_token_count, so a session can hit
    prompt_too_long without ever compacting.
    """
    store = reset_store_for_tests()
    reset_compact_state()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    await _session(store, sid)
    for i in range(5):
        await store.append_event(sid, "user", {"text": f"hi {i}"})
    events = await store.list_events(sid)
    projected = prompt_events(events)
    budget = int(settings.event_budget * settings.compact_ratio)
    assert event_token_count(projected) <= budget
    record_usage(sid, 100_000, at_seq=events[-1].seq)
    ev = await maybe_compact(store, sid)
    assert ev is not None
    assert ev.kind == "compact_boundary"
    stored = await store.list_events(sid)
    assert stored[-1].kind == "compact_boundary"
    assert [e.kind for e in stored].count("user") == 5


@pytest.mark.asyncio
async def test_maybe_compact_skips_when_usage_below_threshold(monkeypatch):
    """Low recorded usage must not compact even if the payload estimate is high."""
    store = reset_store_for_tests()
    reset_compact_state()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    await _session(store, sid)
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    events = await store.list_events(sid)
    projected = prompt_events(events)
    budget = int(settings.event_budget * settings.compact_ratio)
    assert event_token_count(projected) > budget
    record_usage(sid, 10, at_seq=events[-1].seq)
    ev = await maybe_compact(store, sid)
    assert ev is None
    stored = await store.list_events(sid)
    assert all(e.kind != "compact_boundary" for e in stored)
    assert len(stored) == 40


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


def test_events_to_messages_stubs_orphan_tool_use():
    from orbweaver.store import Event

    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={
                "id": "toolu_01V85rqU59D9VfMou3GjerMm",
                "name": "Bash",
                "input": {"command": "git push"},
            },
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="turn_aborted",
            payload={"text": "Stopped this turn"},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="assistant",
            payload={"text": "Stopped this turn"},
        ),
        Event(id=uuid4(), session_id=sid, seq=4, kind="user", payload={"text": "Go ahead"}),
    ]
    messages = events_to_messages(events)
    results = [
        b
        for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert any(b.get("tool_use_id") == "toolu_01V85rqU59D9VfMou3GjerMm" for b in results)
    assert messages[-1]["role"] == "user"
    assert "Go ahead" in str(messages[-1]["content"])


def _tool_result_ids(messages: list[dict]) -> list[str]:
    ids: list[str] = []
    for m in messages:
        if m["role"] != "user" or not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                ids.append(str(b.get("tool_use_id")))
    return ids


def _tool_use_ids(messages: list[dict]) -> list[str]:
    ids: list[str] = []
    for m in messages:
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                ids.append(str(b.get("id")))
    return ids


def test_events_to_messages_stubs_interrupted_tool_before_later_result():
    """Stop mid-Read, then Continue a Bash: the Read must not share the Bash result."""
    from orbweaver.store import Event

    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={
                "id": "toolu_read",
                "name": "Read",
                "input": {"path": "backend/orbweaver/agent.py", "offset": 601},
            },
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="turn_interrupted",
            payload={"reason": "stop"},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_call",
            payload={
                "id": "toolu_bash",
                "name": "Bash",
                "input": {"command": "grep turn_state agent.py"},
            },
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=4,
            kind="tool_result",
            payload={
                "tool_use_id": "toolu_bash",
                "name": "Bash",
                "content": "956:    turn_state: Any | None = None,",
            },
        ),
        Event(id=uuid4(), session_id=sid, seq=5, kind="user", payload={"text": "what caused the error?"}),
    ]
    messages = events_to_messages(events)
    uses = _tool_use_ids(messages)
    results = _tool_result_ids(messages)
    assert uses == ["toolu_read", "toolu_bash"]
    assert results == ["toolu_read", "toolu_bash"]
    assert uses == results
    read_stub = [
        b
        for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("tool_use_id") == "toolu_read"
    ]
    assert read_stub and read_stub[0].get("is_error") is True
    assert "interrupted" in str(read_stub[0].get("content") or "").lower()
    assert unpaired_tool_use_ids(messages) == []


def test_ensure_tool_use_results_fills_gap_before_user_text():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {"path": "a.py"}},
            ],
        },
        {"role": "user", "content": "continue please"},
    ]
    assert unpaired_tool_use_ids(messages) == ["toolu_a"]
    fixed = ensure_tool_use_results(messages)
    assert unpaired_tool_use_ids(fixed) == []
    follow = fixed[2]
    assert follow["role"] == "user"
    blocks = follow["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "toolu_a"
    assert blocks[0]["is_error"] is True
    assert any(b.get("type") == "text" and "continue please" in b.get("text", "") for b in blocks)


def test_ensure_tool_use_results_stubs_unmatched_use_beside_a_real_result():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {}},
                {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_bash", "content": "ok"}],
        },
    ]
    assert unpaired_tool_use_ids(messages) == ["toolu_read"]
    fixed = ensure_tool_use_results(messages)
    assert unpaired_tool_use_ids(fixed) == []
    ids = [
        b["tool_use_id"]
        for b in fixed[1]["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert ids == ["toolu_read", "toolu_bash"]
    assert ensure_tool_use_results(fixed) == fixed


def test_events_to_messages_pairs_when_only_the_later_tool_has_a_result():
    from orbweaver.store import Event

    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="tool_call",
            payload={"id": "toolu_read", "name": "Read", "input": {"path": "a.py"}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_call",
            payload={"id": "toolu_bash", "name": "Bash", "input": {"command": "true"}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_result",
            payload={"tool_use_id": "toolu_bash", "name": "Bash", "content": "ok"},
        ),
    ]
    messages = events_to_messages(events)
    assert unpaired_tool_use_ids(messages) == []
    assert "toolu_read" in _tool_result_ids(messages)
    assert "toolu_bash" in _tool_result_ids(messages)


@pytest.mark.asyncio
async def test_overflow_force_compacts_under_budget_without_llm(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    sid = new_uuid()
    await _session(store, sid)
    for i in range(6):
        tid = f"t{i}"
        await store.append_event(sid, "user", {"text": f"round {i}"})
        await store.append_event(
            sid, "tool_call", {"id": tid, "name": "Read", "input": {"path": f"{i}.py"}}
        )
        await store.append_event(
            sid,
            "tool_result",
            {"tool_use_id": tid, "name": "Read", "content": f"body-{i}"},
        )

    class Probe:
        async def create(self, **_k):
            raise AssertionError("overflow compact must not call the LLM")

        @property
        def messages(self):
            return self

    ev = await maybe_compact(
        store,
        sid,
        client=Probe(),
        system=[{"type": "text", "text": "x"}],
        source="overflow",
        force=True,
        keep_recent_rounds=2,
    )
    assert ev is not None
    assert ev.kind == "compact_boundary"
    assert ev.payload["trigger"] == "overflow"
    keep_from = int(ev.payload["keep_from_seq"])
    stored = await store.list_events(sid)
    dropped = [e for e in stored if e.kind == "tool_call" and e.seq < keep_from]
    kept = [e for e in stored if e.kind == "tool_call" and e.seq >= keep_from]
    assert dropped
    assert len(kept) == 2
    live = live_events(stored)
    assert live[0].kind == "compact_boundary"
    assert all(e.kind != "tool_result" or e.seq >= keep_from for e in live[1:])
