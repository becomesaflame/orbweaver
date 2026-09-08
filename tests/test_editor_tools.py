import json
from uuid import uuid4

import pytest
from orbweaver.agent import run_tools
from orbweaver.compact import events_to_messages, maybe_compact, prompt_events
from orbweaver.config import settings
from orbweaver.store import SESSION_TYPE, Entity, new_uuid, reset_store_for_tests, session_at_id
from orbweaver.todos import inject_session_todos, persist_todos
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


def _ctx(store, ws, sid, events=None):
    return {
        "workspace": ws,
        "store": store,
        "session_id": sid,
        "workspace_kind": "local",
        "events": events or [],
    }


@pytest.mark.asyncio
async def test_delete_tool_removes_file(tmp_path):
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "x")
    result = await run_tools("Delete", {"path": "src/a.py"}, _ctx(store, ws, sid))
    assert result.startswith("deleted")
    assert not (tmp_path / "src" / "a.py").exists()


@pytest.mark.asyncio
async def test_delete_tool_blocks_env(tmp_path):
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    result = await run_tools("Delete", {"path": ".env"}, _ctx(store, ws, sid))
    assert "error deleting" in result
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1"


@pytest.mark.asyncio
async def test_todos_persist_across_compact_boundary(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    await _session(store, sid)
    todos = await persist_todos(
        store,
        sid,
        {
            "todos": [
                {"id": "1", "content": "ship delete tool", "status": "in_progress"},
                {"id": "2", "content": "add readlints stub", "status": "pending"},
            ]
        },
    )
    assert todos[0]["content"] == "ship delete tool"
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    ev = await maybe_compact(store, sid)
    assert ev is not None
    assert ev.kind == "compact_boundary"
    events = await store.list_events(sid)
    keep_from = int(ev.payload["keep_from_seq"])
    todo_ev = next(e for e in events if e.kind == "todo_state")
    assert todo_ev.seq < keep_from
    projected = prompt_events(events)
    assert not any(e.kind == "todo_state" for e in projected)
    messages = inject_session_todos(events_to_messages(projected), events)
    blob = json.dumps(messages)
    assert "ship delete tool" in blob
    assert "add readlints stub" in blob
    sess = await store.get_entity(sid)
    assert sess.jsonld["todos"][0]["content"] == "ship delete tool"


@pytest.mark.asyncio
async def test_todowrite_merge_and_session_entity(tmp_path):
    store = reset_store_for_tests()
    sid = new_uuid()
    await _session(store, sid)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    first = await run_tools(
        "TodoWrite",
        {"todos": [{"id": "1", "content": "one", "status": "pending"}]},
        _ctx(store, ws, sid),
    )
    assert "one" in first
    second = await run_tools(
        "TodoWrite",
        {
            "merge": True,
            "todos": [{"id": "1", "content": "one", "status": "completed"}, {"id": "2", "content": "two"}],
        },
        _ctx(store, ws, sid),
    )
    body = json.loads(second)
    assert body["todos"][0]["status"] == "completed"
    assert body["todos"][1]["content"] == "two"
    sess = await store.get_entity(sid)
    assert [t["id"] for t in sess.jsonld["todos"]] == ["1", "2"]


@pytest.mark.asyncio
async def test_readlints_python_ast_stub(tmp_path):
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/bad.py", "def oops(\n")
    result = await run_tools("ReadLints", {"paths": ["src/bad.py"]}, _ctx(store, ws, sid))
    body = json.loads(result)
    assert body["diagnostics"]
    assert body["diagnostics"][0]["path"] == "src/bad.py"
    assert body["diagnostics"][0]["source"] == "python-ast"
    assert body["diagnostics"][0]["severity"] == "error"


@pytest.mark.asyncio
async def test_readlints_configured_linter(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/ok.py", "x = 1\n")
    monkeypatch.setattr(settings, "orbweaver_linter", "echo {paths}:2:1: stub lint")
    result = await run_tools("ReadLints", {"paths": ["src/ok.py"]}, _ctx(store, ws, sid))
    body = json.loads(result)
    assert body["diagnostics"]
    assert body["diagnostics"][0]["message"] == "stub lint"
    assert body["diagnostics"][0]["line"] == 2
    assert "echo" in body["command"]


@pytest.mark.asyncio
async def test_readlints_configured_linter_survives_bwrap_eperm(tmp_path, monkeypatch):
    """CI runners reject bwrap loopback setup; stdout must still be parsed."""
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/ok.py", "x = 1\n")
    monkeypatch.setattr(settings, "orbweaver_linter", "echo {paths}:2:1: stub lint")
    orig = ws.bash

    def bash(command, timeout=30, sandbox=True, unsandboxed=False, permissions=None):
        if sandbox and not unsandboxed:
            return (
                "sandbox_unavailable: bwrap: loopback: Failed RTM_NEWADDR: "
                "Operation not permitted"
            )
        return orig(
            command,
            timeout=timeout,
            sandbox=False,
            unsandboxed=unsandboxed,
            permissions=permissions,
        )

    ws.bash = bash
    result = await run_tools("ReadLints", {"paths": ["src/ok.py"]}, _ctx(store, ws, sid))
    body = json.loads(result)
    assert body["diagnostics"]
    assert body["diagnostics"][0]["message"] == "stub lint"
    assert body["diagnostics"][0]["line"] == 2


@pytest.mark.asyncio
async def test_readlints_defaults_to_recent_edits(tmp_path):
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/edited.py", "def oops(\n")
    events = [
        type(
            "E",
            (),
            {
                "kind": "tool_call",
                "payload": {"name": "Write", "input": {"path": "src/edited.py"}},
            },
        )()
    ]
    result = await run_tools("ReadLints", {}, _ctx(store, ws, sid, events=events))
    body = json.loads(result)
    assert body["diagnostics"]
    assert body["diagnostics"][0]["path"] == "src/edited.py"
