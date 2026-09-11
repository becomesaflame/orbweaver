"""Background / parallel subagents (issue #110): fan-out, wait, cap, timeout, cancel."""

import asyncio
import json
import time
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import TOOL_SPEC, TurnCancelled, agent_turn, run_tools
from orbweaver.compact import events_to_messages
from orbweaver.config import settings
from orbweaver.store import SESSION_TYPE, Entity, Event, reset_store_for_tests, session_at_id
from orbweaver.subagent import (
    CHILD_BLOCKED_TOOLS,
    child_tool_spec,
    reset_subagent_semaphore_for_tests,
)
from orbweaver.workspace import LocalWorkspace


class _ToolUse:
    def __init__(self, name, inp, uid):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


def _text(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _system_text(kwargs) -> str:
    system = kwargs.get("system") or []
    if isinstance(system, str):
        return system
    return " ".join(str(b.get("text") or "") for b in system if isinstance(b, dict))


def _first_user_text(kwargs) -> str:
    for msg in kwargs.get("messages") or []:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text") or "")
    return ""


class _ScriptedLLM:
    """Parent responses are scripted; child calls sleep then answer with their task."""

    def __init__(self, parent_responses, child_sleep=0.5, before_parent_call=None):
        self._parent = list(parent_responses)
        self.child_sleep = child_sleep
        self.before_parent_call = before_parent_call
        self.parent_calls = []
        self.child_calls = []
        self.messages = self

    async def create(self, **kwargs):
        if "Orbweaver subagent" in _system_text(kwargs):
            self.child_calls.append(kwargs)
            await asyncio.sleep(self.child_sleep)
            return _text(f"child result: {_first_user_text(kwargs)}")
        idx = len(self.parent_calls)
        self.parent_calls.append(kwargs)
        if self.before_parent_call is not None:
            await self.before_parent_call(idx)
        if not self._parent:
            return _text("done")
        return self._parent.pop(0)


async def _allow(*_a, **_k):
    return {
        "verdict": "allow",
        "should_block": False,
        "should_ask": False,
        "reason": "test",
        "stage": "test",
    }


async def _passthrough_review(_parent, _calls, payload, **_k):
    return {"status": "ok", "reason": "ok", "payload": payload}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    reset_store_for_tests()
    reset_subagent_semaphore_for_tests()
    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", _allow)
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)
    monkeypatch.setattr(settings, "orbweaver_subagent_grace_s", 2.0)
    yield
    reset_subagent_semaphore_for_tests()


def _install_llm(monkeypatch, client):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)


def _ws(tmp_path):
    return LocalWorkspace("workspace:default", str(tmp_path))


def _parent_entity(sid, **extra):
    jsonld = {
        "@id": session_at_id(sid),
        "@type": SESSION_TYPE,
        "workspace_uri": "workspace:default",
        "workspace_kind": "local",
        "title": "Parent",
        "status": "active",
    }
    jsonld.update(extra)
    return Entity(id=sid, at_id=session_at_id(sid), at_type=SESSION_TYPE, jsonld=jsonld)


def _ctx(store, ws, sid, **extra):
    ctx = {
        "workspace": ws,
        "store": store,
        "session_id": sid,
        "workspace_kind": "local",
        "subagent_depth": 0,
    }
    ctx.update(extra)
    return ctx


async def _finished_count(store, sid) -> int:
    return sum(1 for e in await store.list_events(sid) if e.kind == "subagent_finished")


def test_tool_spec_has_background_and_wait():
    spawn = next(t for t in TOOL_SPEC if t["name"] == "SpawnSubagent")
    props = spawn["input_schema"]["properties"]
    assert props["background"]["type"] == "boolean"
    assert "timeout_s" in props
    assert "max_rounds" in props
    wait = next(t for t in TOOL_SPEC if t["name"] == "SubagentWait")
    assert wait["input_schema"]["properties"]["ids"]["type"] == "array"
    assert "SubagentWait" in CHILD_BLOCKED_TOOLS
    child_names = {t["name"] for t in child_tool_spec(TOOL_SPEC)}
    assert "SubagentWait" not in child_names
    assert "SpawnSubagent" not in child_names


async def test_background_children_run_in_parallel_and_results_are_injected(
    tmp_path, monkeypatch
):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))

    async def wait_for_children(idx):
        # The parent's second LLM call "takes a while": it returns once both
        # background children have reported in, like a real slow completion.
        if idx == 1:
            for _ in range(400):
                if await _finished_count(store, sid) >= 2:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("children never finished")

    llm = _ScriptedLLM(
        [
            SimpleNamespace(
                content=[
                    _ToolUse(
                        "SpawnSubagent",
                        {"task": "alpha task", "label": "alpha", "background": True},
                        "tu-a",
                    ),
                    _ToolUse(
                        "SpawnSubagent",
                        {"task": "beta task", "label": "beta", "background": True},
                        "tu-b",
                    ),
                ]
            ),
            SimpleNamespace(content=[_ToolUse("Glob", {"pattern": "*"}, "tu-glob")]),
            _text("all collected"),
        ],
        before_parent_call=wait_for_children,
    )
    _install_llm(monkeypatch, llm)

    t0 = time.monotonic()
    produced = await agent_turn(store, sid, "fan out", _ws(tmp_path))
    wall = time.monotonic() - t0
    assert 0.45 <= wall < 0.9, wall

    # Spawn returned immediately with a running status.
    spawn_results = [
        json.loads(e.payload["content"])
        for e in produced
        if e.kind == "tool_result" and e.payload.get("name") == "SpawnSubagent"
    ]
    assert [r["status"] for r in spawn_results] == ["running", "running"]
    ids = {r["subagent_id"] for r in spawn_results}
    assert len(ids) == 2

    started = [e for e in produced if e.kind == "subagent_started"]
    assert [e.payload["background"] for e in started] == [True, True]
    assert {e.payload["subagent_id"] for e in started} == ids
    assert {e.payload["name"] for e in started} == {"alpha", "beta"}
    finished = [e for e in produced if e.kind == "subagent_finished"]
    assert len(finished) == 2
    for ev in finished:
        assert ev.payload["subagent_id"] in ids
        assert ev.payload["status"] == "ok"
        assert 0.4 <= ev.payload["elapsed_s"] < 0.9

    # Both children ran concurrently (two child LLM calls, ~0.5 s total wall).
    assert len(llm.child_calls) == 2

    # Results were injected as user-side text in the parent's next prompt.
    injected = [e for e in produced if e.kind == "subagent_result"]
    assert {e.payload["subagent_id"] for e in injected} == ids
    assert {e.payload["status"] for e in injected} == {"ok"}
    third = llm.parent_calls[2]["messages"]
    last_user = third[-1]
    assert last_user["role"] == "user"
    text_blocks = [
        b.get("text", "")
        for b in last_user["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    blob = "\n".join(text_blocks)
    assert "child result: alpha task" in blob
    assert "child result: beta task" in blob
    assert "Background subagent" in blob
    # Not delivered as tool_result blocks: only Glob's result is a tool_result there.
    tool_results = [
        b for b in last_user["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in tool_results] == ["tu-glob"]
    # No SubagentWait was needed and no child is left behind.
    assert not any(e.kind == "subagent_cancelled" for e in produced)


async def test_subagent_wait_returns_only_requested_child(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    gates = {}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        gate = gates.setdefault(user_text, asyncio.Event())
        await gate.wait()
        await store.append_event(session_id, "assistant", {"text": f"done {user_text}"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    ctx = _ctx(store, _ws(tmp_path), sid, cancel=asyncio.Event())
    a = json.loads(await run_tools("SpawnSubagent", {"task": "one", "background": True}, ctx))
    b = json.loads(await run_tools("SpawnSubagent", {"task": "two", "background": True}, ctx))
    assert a["status"] == b["status"] == "running"
    assert set(ctx["children"]) == {a["subagent_id"], b["subagent_id"]}

    waiter = asyncio.create_task(run_tools("SubagentWait", {"ids": [a["subagent_id"]]}, ctx))
    await asyncio.sleep(0.05)
    assert not waiter.done()
    gates.setdefault("one", asyncio.Event()).set()
    result = json.loads(await asyncio.wait_for(waiter, 2))
    assert [r["subagent_id"] for r in result["results"]] == [a["subagent_id"]]
    assert result["results"][0]["status"] == "ok"
    assert result["results"][0]["text"] == "done one"
    assert ctx["children"][a["subagent_id"]].delivered is True
    assert ctx["children"][b["subagent_id"]].delivered is False
    assert not ctx["children"][b["subagent_id"]].finished

    unknown = json.loads(await run_tools("SubagentWait", {"ids": ["nope"]}, ctx))
    assert unknown["results"][0]["status"] == "unknown"

    gates["two"].set()
    rest = json.loads(await asyncio.wait_for(run_tools("SubagentWait", {}, ctx), 2))
    assert [r["subagent_id"] for r in rest["results"]] == [b["subagent_id"]]
    assert rest["results"][0]["text"] == "done two"
    again = json.loads(await run_tools("SubagentWait", {}, ctx))
    assert again["results"] == []


async def test_semaphore_of_one_serializes_children(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    running = {"now": 0, "peak": 0}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        running["now"] += 1
        running["peak"] = max(running["peak"], running["now"])
        try:
            await asyncio.sleep(0.3)
        finally:
            running["now"] -= 1
        await store.append_event(session_id, "assistant", {"text": user_text})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)

    async def spawn_two_and_wait(limit):
        monkeypatch.setattr(settings, "orbweaver_max_concurrent_subagents", limit)
        reset_subagent_semaphore_for_tests()
        running["peak"] = 0
        ctx = _ctx(store, _ws(tmp_path), sid, cancel=asyncio.Event())
        t0 = time.monotonic()
        await run_tools("SpawnSubagent", {"task": "x", "background": True}, ctx)
        await run_tools("SpawnSubagent", {"task": "y", "background": True}, ctx)
        out = json.loads(await run_tools("SubagentWait", {}, ctx))
        assert sorted(r["text"] for r in out["results"]) == ["x", "y"]
        return time.monotonic() - t0

    serial = await spawn_two_and_wait(1)
    assert serial >= 0.55, serial
    assert running["peak"] == 1
    parallel = await spawn_two_and_wait(4)
    assert parallel < 0.5, parallel
    assert running["peak"] == 2


async def test_child_timeout_cancels_task_and_reports_error(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    seen = {"cancelled": False}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        await store.append_event(session_id, "assistant", {"text": "halfway"})
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    ctx = _ctx(store, _ws(tmp_path), sid, cancel=asyncio.Event())
    t0 = time.monotonic()
    inline = json.loads(
        await run_tools("SpawnSubagent", {"task": "slow", "timeout_s": 0.2}, ctx)
    )
    assert time.monotonic() - t0 < 1.5
    assert seen["cancelled"] is True
    assert inline["status"] == "timeout"
    assert inline["text"].startswith("error:")
    assert "timed out" in inline["text"]
    assert "halfway" in inline["text"]
    finished = [e for e in await store.list_events(sid) if e.kind == "subagent_finished"]
    assert finished[-1].payload["status"] == "timeout"

    seen["cancelled"] = False
    bg = json.loads(
        await run_tools(
            "SpawnSubagent", {"task": "slow bg", "background": True, "timeout_s": 0.2}, ctx
        )
    )
    out = json.loads(await asyncio.wait_for(run_tools("SubagentWait", {}, ctx), 3))
    assert out["results"][0]["subagent_id"] == bg["subagent_id"]
    assert out["results"][0]["status"] == "timeout"
    assert out["results"][0]["text"].startswith("error:")
    assert seen["cancelled"] is True
    assert ctx["children"][bg["subagent_id"]].task.done()


async def test_parent_cancel_cancels_children(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    cancel = asyncio.Event()
    child_state = {"started": 0, "cancelled": 0}

    async def fake_child_turn(store, session_id, user_text, workspace, **kwargs):
        child_state["started"] += 1
        child_cancel = kwargs.get("cancel")
        try:
            await child_cancel.wait()
        except asyncio.CancelledError:
            child_state["cancelled"] += 1
            raise
        child_state["cancelled"] += 1
        raise TurnCancelled()

    async def cancel_parent(idx):
        if idx == 1:
            # Give the children a tick to start, then stop the parent turn.
            await asyncio.sleep(0.05)
            cancel.set()

    llm = _ScriptedLLM(
        [
            SimpleNamespace(
                content=[
                    _ToolUse("SpawnSubagent", {"task": "p", "background": True}, "tu-p"),
                    _ToolUse("SpawnSubagent", {"task": "q", "background": True}, "tu-q"),
                ]
            ),
            _text("still thinking"),
        ],
        before_parent_call=cancel_parent,
    )
    _install_llm(monkeypatch, llm)
    # Children use a fake loop (they must not consume the parent's scripted responses).
    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_child_turn)

    t0 = time.monotonic()
    with pytest.raises(TurnCancelled):
        await agent_turn(store, sid, "spawn then stop", _ws(tmp_path), cancel=cancel)
    assert time.monotonic() - t0 < 2.0
    assert child_state["started"] == 2
    assert child_state["cancelled"] == 2
    events = await store.list_events(sid)
    cancelled = [e for e in events if e.kind == "subagent_cancelled"]
    assert len(cancelled) == 2
    assert {e.payload["reason"] for e in cancelled} == {"parent_cancelled"}
    started_ids = {e.payload["subagent_id"] for e in events if e.kind == "subagent_started"}
    assert {e.payload["subagent_id"] for e in cancelled} == started_ids
    finished = [e for e in events if e.kind == "subagent_finished"]
    assert {e.payload["status"] for e in finished} == {"cancelled"}
    # Nothing is silently lost: the partial status is recorded on the parent.
    results = [e for e in events if e.kind == "subagent_result"]
    assert {e.payload["status"] for e in results} == {"cancelled"}


async def test_uncollected_children_are_waited_then_recorded(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))

    async def fake_child_turn(store, session_id, user_text, workspace, **kwargs):
        await asyncio.sleep(0.2)
        await store.append_event(session_id, "assistant", {"text": f"late {user_text}"})
        return []

    llm = _ScriptedLLM(
        [
            SimpleNamespace(
                content=[_ToolUse("SpawnSubagent", {"task": "z", "background": True}, "tu-z")]
            ),
            _text("bye without waiting"),
        ]
    )
    _install_llm(monkeypatch, llm)
    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_child_turn)
    produced = await agent_turn(store, sid, "spawn and leave", _ws(tmp_path))
    kinds = [e.kind for e in produced]
    assert "subagent_cancelled" not in kinds
    result = next(e for e in produced if e.kind == "subagent_result")
    assert result.payload["status"] == "ok"
    assert result.payload["text"] == "late z"
    # The next turn's prompt shows the late result as a user-side message.
    messages = events_to_messages(await store.list_events(sid))
    assert messages[-1]["role"] == "user"
    assert "late z" in str(messages[-1]["content"])


async def test_grace_period_expiry_cancels_and_records(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    monkeypatch.setattr(settings, "orbweaver_subagent_grace_s", 0.1)

    async def fake_child_turn(store, session_id, user_text, workspace, **kwargs):
        await store.append_event(session_id, "assistant", {"text": "so far"})
        await asyncio.sleep(30)
        return []

    llm = _ScriptedLLM(
        [
            SimpleNamespace(
                content=[_ToolUse("SpawnSubagent", {"task": "w", "background": True}, "tu-w")]
            ),
            _text("bye"),
        ]
    )
    _install_llm(monkeypatch, llm)
    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_child_turn)
    t0 = time.monotonic()
    produced = await agent_turn(store, sid, "spawn and leave", _ws(tmp_path))
    assert time.monotonic() - t0 < 2.0
    cancelled = next(e for e in produced if e.kind == "subagent_cancelled")
    assert cancelled.payload["reason"] == "parent_turn_ended"
    result = next(e for e in produced if e.kind == "subagent_result")
    assert result.payload["status"] == "cancelled"
    assert "so far" in result.payload["text"]


async def test_inline_spawn_unchanged(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    captured = {}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        captured["kwargs"] = kwargs
        await store.append_event(session_id, "assistant", {"text": "inline done"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    parent_cancel = asyncio.Event()
    ctx = _ctx(store, _ws(tmp_path), sid, cancel=parent_cancel)
    body = json.loads(await run_tools("SpawnSubagent", {"task": "sync"}, ctx))
    assert body["status"] == "ok"
    assert body["text"] == "inline done"
    assert body["subagent_id"] == body["child_session_id"]
    # Inline children share the parent's cancel flag, as before.
    assert captured["kwargs"]["cancel"] is parent_cancel
    assert captured["kwargs"]["max_rounds"] == settings.orbweaver_subagent_max_rounds
    kinds = [e.kind for e in await store.list_events(sid)]
    assert kinds.count("subagent_started") == 1
    assert kinds.count("subagent_finished") == 1
    assert "subagent_result" not in kinds
    run = ctx["children"][body["subagent_id"]]
    assert run.delivered is True
    assert run.background is False
    # Nothing pending for SubagentWait.
    assert json.loads(await run_tools("SubagentWait", {}, ctx))["results"] == []


async def test_child_events_stream_to_parent_emit(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(sid))
    streamed = []

    async def fake_turn(store, session_id, user_text, workspace, emit=None, **kwargs):
        ev = await store.append_event(session_id, "assistant", {"text": "hi"})
        if emit:
            emit({"kind": ev.kind, "payload": ev.payload, "id": str(ev.id), "seq": ev.seq})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    ctx = _ctx(store, _ws(tmp_path), sid, emit=streamed.append)
    body = json.loads(await run_tools("SpawnSubagent", {"task": "stream"}, ctx))
    tree = [m for m in streamed if m["kind"] == "subagent_event"]
    assert len(tree) == 1
    assert tree[0]["payload"]["subagent_id"] == body["subagent_id"]
    assert tree[0]["payload"]["event"]["kind"] == "assistant"


def test_subagent_result_projects_as_user_message():
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_call",
            payload={"id": "tu-1", "name": "Glob", "input": {"pattern": "*"}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_result",
            payload={"tool_use_id": "tu-1", "name": "Glob", "content": "a.py"},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=4,
            kind="subagent_result",
            payload={"subagent_id": "abc", "name": "alpha", "status": "ok", "text": "found it"},
        ),
    ]
    messages = events_to_messages(events)
    assert messages[-1]["role"] == "user"
    blocks = messages[-1]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[-1]["type"] == "text"
    assert "found it" in blocks[-1]["text"]
    assert "alpha" in blocks[-1]["text"]
