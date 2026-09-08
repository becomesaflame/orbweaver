import asyncio
import json
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import TOOL_SPEC, TurnCancelled, run_tools
from orbweaver.app import app
from orbweaver.store import SESSION_TYPE, Entity, reset_store_for_tests, session_at_id
from orbweaver.subagent import (
    CHILD_BLOCKED_TOOLS,
    SUBAGENT_MAX_ROUNDS,
    child_tool_spec,
    is_subagent_session,
    normalize_subagent_type,
)
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


@pytest.fixture
def ws(tmp_path):
    return LocalWorkspace("workspace:default", str(tmp_path))


def _parent_entity(store, sid, **extra):
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


async def _passthrough_review(_parent, _calls, payload, **_k):
    return {"status": "ok", "reason": "ok", "payload": payload}


@pytest.mark.asyncio
async def test_spawn_creates_hidden_child_session(tmp_path, ws, monkeypatch, auth_header):
    store = reset_store_for_tests()
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)
    captured = {}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        captured["session_id"] = session_id
        captured["user_text"] = user_text
        captured["kwargs"] = kwargs
        await store.append_event(session_id, "assistant", {"text": "found 3 files"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    result = await run_tools(
        "SpawnSubagent",
        {"task": "list the files", "label": "explore"},
        _ctx(store, ws, sid),
    )
    body = json.loads(result)
    assert body["status"] == "ok"
    assert body["text"] == "found 3 files"
    child_id = body["child_session_id"]
    child = await store.get_entity(UUID(child_id))
    assert child is not None
    assert is_subagent_session(child)
    assert child.jsonld["parent_session"] == session_at_id(sid)
    assert child.jsonld["title"] == "explore"
    assert captured["user_text"] == "list the files"
    assert captured["kwargs"]["subagent_depth"] == 1
    assert captured["kwargs"]["max_rounds"] == SUBAGENT_MAX_ROUNDS
    names = {t["name"] for t in captured["kwargs"]["tools"]}
    assert CHILD_BLOCKED_TOOLS.isdisjoint(names)
    assert "Read" in names
    assert "ProposePatch" not in names
    assert "Write" in names
    assert child.jsonld.get("channel") in {None, ""}
    parent_kinds = [e.kind for e in await store.list_events(sid)]
    assert "subagent_started" in parent_kinds
    assert "subagent_finished" in parent_kinds

    from orbweaver.config import settings

    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/v1/sessions", headers=auth_header)
        assert listed.status_code == 200, listed.text
        ids = {row["id"] for row in listed.json()["sessions"]}
        assert str(sid) in ids
        assert child_id not in ids


@pytest.mark.asyncio
async def test_nested_spawn_refused(ws, monkeypatch):
    store = reset_store_for_tests()
    called = False

    async def fake_turn(*_a, **_k):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    before = await store.list_entities(SESSION_TYPE)
    result = await run_tools(
        "SpawnSubagent",
        {"task": "go deeper"},
        _ctx(store, ws, sid, subagent_depth=1),
    )
    assert "nested" in result.lower()
    after = await store.list_entities(SESSION_TYPE)
    assert len(after) == len(before)
    assert called is False


@pytest.mark.asyncio
async def test_cancel_during_child_returns_cancelled(ws, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        await store.append_event(session_id, "assistant", {"text": "partial"})
        raise TurnCancelled()

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    cancel = asyncio.Event()
    cancel.set()
    result = await run_tools(
        "SpawnSubagent",
        {"task": "long job"},
        _ctx(store, ws, sid, cancel=cancel),
    )
    body = json.loads(result)
    assert body["status"] == "cancelled"
    assert body["text"] == "partial"
    assert "child_session_id" in body


@pytest.mark.asyncio
async def test_child_abort_reports_aborted_status(ws, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        await store.append_event(
            session_id,
            "turn_aborted",
            {"text": "Stopped this turn after 1 blocked actions."},
        )
        await store.append_event(
            session_id,
            "assistant",
            {"text": "Stopped this turn after 1 blocked actions."},
        )
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    result = await run_tools("SpawnSubagent", {"task": "push main"}, _ctx(store, ws, sid))
    body = json.loads(result)
    assert body["status"] == "aborted"
    assert "Stopped this turn" in body["text"]


@pytest.mark.asyncio
async def test_child_patch_proposal_copied_to_parent(ws, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        await store.append_event(
            session_id,
            "patch_proposal",
            {"tool_use_id": "tu1", "content": json.dumps({"ok": True, "path": "a.py"})},
        )
        await store.append_event(session_id, "assistant", {"text": "patched"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    result = await run_tools("SpawnSubagent", {"task": "patch a.py"}, _ctx(store, ws, sid))
    body = json.loads(result)
    parent_events = await store.list_events(sid)
    patches = [e for e in parent_events if e.kind == "patch_proposal"]
    assert len(patches) == 1
    assert patches[0].payload["child_session_id"] == body["child_session_id"]
    assert "a.py" in patches[0].payload["content"]


@pytest.mark.asyncio
async def test_empty_task_does_not_create_session(ws):
    store = reset_store_for_tests()
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    result = await run_tools("SpawnSubagent", {"task": "  "}, _ctx(store, ws, sid))
    assert "task is required" in result
    kids = [e for e in await store.list_entities(SESSION_TYPE) if is_subagent_session(e)]
    assert kids == []


def test_child_tool_spec_by_type():
    names = {t["name"] for t in TOOL_SPEC}
    explore = {t["name"] for t in child_tool_spec(TOOL_SPEC, "explore")}
    implement = {t["name"] for t in child_tool_spec(TOOL_SPEC, "implement")}
    shell = {t["name"] for t in child_tool_spec(TOOL_SPEC, "shell")}
    assert CHILD_BLOCKED_TOOLS.isdisjoint(explore | implement | shell)
    assert "Read" in explore
    assert "MemoryGraph" in explore
    assert "Write" not in explore
    assert "Bash" not in explore
    assert "NotebookEdit" not in explore
    assert "Write" in implement
    assert "Bash" in implement
    assert "NotebookEdit" in implement
    assert "Bash" in shell
    assert "Read" in shell
    assert "Write" not in shell
    assert "NotebookEdit" not in shell
    assert names >= {"Read", "Write", "Bash"}
    assert normalize_subagent_type("") == "implement"
    assert normalize_subagent_type("research") == "explore"
    assert normalize_subagent_type("nope") is None


@pytest.mark.asyncio
async def test_typed_subagent_explore_and_unknown(ws, monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)
    captured = {}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        captured["kwargs"] = kwargs
        await store.append_event(session_id, "assistant", {"text": "looked"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid))
    result = await run_tools(
        "SpawnSubagent",
        {"task": "scan the repo", "type": "explore"},
        _ctx(store, ws, sid),
    )
    body = json.loads(result)
    child = await store.get_entity(UUID(body["child_session_id"]))
    assert child.jsonld["subagent_type"] == "explore"
    names = {t["name"] for t in captured["kwargs"]["tools"]}
    assert "Read" in names
    assert "Write" not in names
    assert "Bash" not in names
    assert "explore" in captured["kwargs"]["system_extra"]

    before = len(await store.list_entities(SESSION_TYPE))
    bad = await run_tools(
        "SpawnSubagent",
        {"task": "nope", "role": "overlord"},
        _ctx(store, ws, sid),
    )
    assert "type must be" in bad
    assert len(await store.list_entities(SESSION_TYPE)) == before


def test_parent_tool_spec_still_includes_spawn():
    names = {t["name"] for t in TOOL_SPEC}
    assert "SpawnSubagent" in names
    assert "AskUser" in names
    assert "SendPhoto" in names
    assert "GenerateImage" in names
    assert "WorkspaceSearch" in names
    assert "MemorySearch" in names
    assert "Delete" in names
    assert "TodoWrite" in names
    assert "ReadLints" in names
    assert "ProposePatch" in names
    assert "Browser" in names


@pytest.mark.asyncio
async def test_vscode_parent_subagent_keeps_proposepatch(ws, monkeypatch):
    store = reset_store_for_tests()
    captured = {}

    async def fake_turn(store, session_id, user_text, workspace, **kwargs):
        captured["kwargs"] = kwargs
        await store.append_event(session_id, "assistant", {"text": "patched"})
        return []

    monkeypatch.setattr("orbweaver.agent.agent_turn", fake_turn)
    sid = uuid4()
    await store.put_entity(_parent_entity(store, sid, channel="vscode"))
    result = await run_tools(
        "SpawnSubagent",
        {"task": "edit files", "label": "edit"},
        _ctx(store, ws, sid, channel="vscode"),
    )
    body = json.loads(result)
    child = await store.get_entity(UUID(body["child_session_id"]))
    assert child.jsonld["channel"] == "vscode"
    names = {t["name"] for t in captured["kwargs"]["tools"]}
    assert "ProposePatch" in names
    assert "SpawnSubagent" not in names
    assert captured["kwargs"]["channel"] == "vscode"
