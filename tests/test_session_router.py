"""Session router: Telegram operator attaches to web/VS Code sessions and drives them.

These run the real ``agent_turn`` (scripted Anthropic client) so the cross-channel
paths are exercised end to end: an attached Telegram message answers a web
``AskUser``; a web turn's result is delivered to the attached chat; operator
tools move the attach pointer from inside a turn; ``/stop`` cancels a
background Telegram turn through the shared registry.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import anthropic
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver import turns
from orbweaver.agent import pending_ask_user
from orbweaver.app import app
from orbweaver.channels import router
from orbweaver.channels import telegram as tg
from orbweaver.config import settings
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.store import SESSION_TYPE, Entity, reset_store_for_tests, session_at_id
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    reset_store_for_tests()
    tg.reset_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "telegram_allowlist", "")
    yield
    tg.reset_for_tests()


def _ws(tmp_path):
    return LocalWorkspace("workspace:default", str(tmp_path))


def _session(sid, title="New chat", **extra):
    jsonld = {
        "@id": session_at_id(sid),
        "@type": SESSION_TYPE,
        "workspace_uri": "workspace:default",
        "workspace_kind": "local",
        "title": title,
        "status": "active",
    }
    jsonld.update(extra)
    return Entity(id=sid, at_id=session_at_id(sid), at_type=SESSION_TYPE, jsonld=jsonld)


class _ToolUse:
    def __init__(self, name, inp, uid=None):
        self.type = "tool_use"
        self.id = uid or f"tu-{uuid4().hex[:8]}"
        self.name = name
        self.input = inp


def _text(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


class _ScriptedAnthropic:
    def __init__(self, responses, delay=0.0):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.delay = delay
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self._responses:
            return _text("done")
        return self._responses.pop(0)


def _install_llm(monkeypatch, client):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    return client


def _update(user_id=42, chat_id=42, text="hi"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def _context():
    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


async def _drain_tasks():
    while tg._tasks:
        await asyncio.gather(*list(tg._tasks), return_exceptions=True)


# --------------------------------------------------------------------- router core


@pytest.mark.asyncio
async def test_attach_detach_and_resolve_target():
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    assert op.jsonld["role"] == router.OPERATOR_ROLE
    web = _session(uuid4(), title="Fix the proxy", channel="web")
    await store.put_entity(web)

    assert (await router.resolve_target(store, op)).id == op.id
    target = await router.attach(store, op, web.id)
    assert target.id == web.id
    assert op.jsonld[router.ATTACHED_KEY] == session_at_id(web.id)
    assert router.attached_id(op) == web.id
    assert (await router.resolve_target(store, op)).id == web.id
    # The pointer is an IRI, so it is indexed as a graph edge like parent_session.
    assert any(p == router.ATTACHED_KEY for _s, p, _o in await store.graph(op.at_id))

    assert await router.detach(store, op) == web.id
    assert router.ATTACHED_KEY not in op.jsonld
    assert await router.detach(store, op) is None


@pytest.mark.asyncio
async def test_attach_refuses_self_subagents_and_operators():
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    other_op = await tg.session_for_telegram_user(store, 7, chat_id=7)
    child = _session(uuid4(), role="subagent", parent_session=session_at_id(uuid4()))
    await store.put_entity(child)
    with pytest.raises(router.RouterError):
        await router.attach(store, op, op.id)
    with pytest.raises(router.RouterError):
        await router.attach(store, op, child.id)
    with pytest.raises(router.RouterError):
        await router.attach(store, op, other_op.id)
    with pytest.raises(router.RouterError):
        await router.attach(store, op, uuid4())


@pytest.mark.asyncio
async def test_dangling_attach_pointer_detaches():
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    op.jsonld[router.ATTACHED_KEY] = session_at_id(uuid4())
    await store.put_entity(op)
    assert (await router.resolve_target(store, op)).id == op.id
    assert router.ATTACHED_KEY not in op.jsonld


@pytest.mark.asyncio
async def test_find_session_by_uuid_prefix_and_title():
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    a = _session(UUID("aaaaaaaa-0000-4000-8000-000000000001"), title="Airbed controller")
    b = _session(UUID("bbbbbbbb-0000-4000-8000-000000000002"), title="Airbed docs")
    c = _session(UUID("cccccccc-0000-4000-8000-000000000003"), title="New chat")
    for s in (a, b, c):
        await store.put_entity(s)
    await store.append_event(c.id, "user", {"text": "Rewrite the README intro"})
    ex = {op.id}
    assert (await router.find_session(store, str(a.id), exclude=ex)).id == a.id
    assert (await router.find_session(store, session_at_id(a.id), exclude=ex)).id == a.id
    assert (await router.find_session(store, "bbbbbbbb", exclude=ex)).id == b.id
    assert (await router.find_session(store, "controller", exclude=ex)).id == a.id
    # Generic titles fall back to the first user message, like the web session list.
    assert (await router.find_session(store, "readme", exclude=ex)).id == c.id
    with pytest.raises(router.AmbiguousSessionRef) as ei:
        await router.find_session(store, "airbed", exclude=ex)
    assert {m.id for m in ei.value.matches} == {a.id, b.id}
    assert await router.find_session(store, "nothing-here", exclude=ex) is None
    # Operators are never candidates.
    assert await router.find_session(store, str(op.id)) is None


@pytest.mark.asyncio
async def test_list_and_digest_report_running_and_pending():
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    web = _session(uuid4(), title="Sandbox proxy", channel="vscode", todos=[
        {"id": "1", "content": "wire the proxy", "status": "completed"},
        {"id": "2", "content": "add tests", "status": "in_progress"},
    ])
    await store.put_entity(web)
    await store.append_event(web.id, "user", {"text": "make the proxy honour allowlists"})
    await store.append_event(web.id, "compact_summary", {"text": "Earlier: designed the proxy."})
    await store.append_event(
        web.id, "tool_call", {"id": "tu-ask", "name": "AskUser", "input": {"question": "Port?"}}
    )
    await store.append_event(web.id, "ask_user", {"question": "Port?", "tool_use_id": "tu-ask"})
    state = turns.acquire(web.id, channel="web")
    try:
        rows = await router.list_sessions(store, exclude={op.id})
        assert [r["id"] for r in rows] == [str(web.id)]
        row = rows[0]
        assert row["running"] and row["running_channel"] == "web"
        assert row["pending_question"] == "Port?"
        listing = router.format_session_list(rows, attached=web.id)
        assert "attached" in listing and "running via web" in listing
        digest = await router.session_digest(store, web.id)
        assert digest["open_todo_count"] == 1
        assert digest["compact_summary"].startswith("Earlier")
        text = router.format_digest(digest)
        assert "waiting for your answer: Port?" in text
        assert "add tests" in text and "wire the proxy" not in text
    finally:
        turns.release(web.id, state)


def test_emit_fans_out_to_global_and_session_sinks_and_survives_errors():
    sid, other = uuid4(), uuid4()
    seen: list[tuple[str, UUID, str]] = []
    router.add_global_sink(lambda s, m: seen.append(("global", s, m["kind"])))

    def boom(_msg):
        raise RuntimeError("sink broke")

    router.add_sink(sid, "bad", boom)
    router.add_sink(sid, "good", lambda m: seen.append(("session", sid, m["kind"])))
    router.emit(sid, {"kind": "assistant"})
    router.emit(other, {"kind": "user"})
    assert ("global", sid, "assistant") in seen
    assert ("session", sid, "assistant") in seen
    assert ("global", other, "user") in seen
    assert not any(s == "session" and u == other for s, u, _k in seen)
    router.remove_sink(sid, "good")
    router.remove_sink(sid, "bad")
    assert router.sink_keys(sid) == []


# ---------------------------------------------------------------- chat sinks


@pytest.mark.asyncio
async def test_sync_chat_sinks_follows_attach_pointer():
    store = reset_store_for_tests()
    tg.install_router_hooks()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = _session(uuid4(), title="web work", channel="web")
    await store.put_entity(web)
    tg.sync_chat_sinks(op)
    assert router.sink_keys(op.id) == ["telegram:900"]
    assert router.sink_keys(web.id) == []

    await router.attach(store, op, web.id)  # hook re-syncs
    assert router.sink_keys(web.id) == ["telegram:900"]
    assert tg.chats_for_session(web.id) == [900]
    assert tg._telegram_chat_id(web) == 900  # SendPhoto on the attached session reaches the chat

    await router.detach(store, op)
    assert router.sink_keys(web.id) == []
    assert router.sink_keys(op.id) == ["telegram:900"]


@pytest.mark.asyncio
async def test_rehydrate_chat_sinks_after_restart():
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = _session(uuid4(), channel="web")
    await store.put_entity(web)
    op.jsonld[router.ATTACHED_KEY] = session_at_id(web.id)
    await store.put_entity(op)
    tg.reset_for_tests()
    router.reset_for_tests()
    assert await tg.rehydrate_chat_sinks(store) == 1
    assert router.sink_keys(web.id) == ["telegram:900"]
    assert router.sink_keys(op.id) == ["telegram:900"]


# ------------------------------------------------ attached turns (real agent loop)


@pytest.mark.asyncio
async def test_attached_telegram_message_answers_web_ask_user(tmp_path, monkeypatch, auth_header):
    """Web turn parks on AskUser; the reply typed on Telegram lands on that session."""
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([
        SimpleNamespace(content=[_ToolUse("AskUser", {"question": "Port?"}, "tu-ask")]),
    ]))
    store = reset_store_for_tests()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "web", "title": "Pick a port"},
            headers=auth_header,
        )
        sid = UUID(created.json()["id"])
        paused = await client.post(
            f"/v1/sessions/{sid}/turns", json={"text": "pick a port"}, headers=auth_header
        )
        assert paused.json()["status"] == "waiting_ask"

    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    await tg.attach_text(bound, "pick a port")
    llm._responses.append(_text("8080 it is"))
    await tg._run_turn(update, context, "8080")

    events = await store.list_events(sid)
    user_events = [e for e in events if e.kind == "user"]
    assert user_events[-1].payload == {"text": "8080", "ask_answer": True, "via": "telegram"}
    results = [e for e in events if e.kind == "tool_result" and e.payload.get("name") == "AskUser"]
    assert results and results[0].payload["content"] == "8080"
    assert pending_ask_user(events) is None
    assert update.message.reply_text.await_args.args[0] == "8080 it is"
    # Nothing was written to the operator's own session.
    assert await store.list_events(bound.operator.id) == []
    # The Telegram-driven turn told WebSocket clients it finished (via the router).
    from orbweaver.app import _last_turn_done

    assert _last_turn_done[sid]["status"] == "ok"
    await _drain_tasks()


@pytest.mark.asyncio
async def test_web_turn_result_is_delivered_to_attached_chat(tmp_path, monkeypatch, auth_header):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("Tests are green, PR is up.")]))
    store = reset_store_for_tests()
    sent: list[tuple[int, str]] = []

    async def fake_notify(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(tg, "notify_telegram_chat", fake_notify)
    web = _session(uuid4(), title="Ship the router", channel="web")
    await store.put_entity(web)
    bound = await tg._session_workspace(_update(chat_id=900), _context())
    await router.attach(store, bound.operator, web.id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            f"/v1/sessions/{web.id}/turns", json={"text": "run the tests"}, headers=auth_header
        )
        assert r.status_code == 200
    await _drain_tasks()
    assert sent == [(900, "[Ship the router]\nTests are green, PR is up.")]


@pytest.mark.asyncio
async def test_telegram_driven_turn_does_not_echo_through_its_own_sink(tmp_path, monkeypatch):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("hello from the target")]))
    store = reset_store_for_tests()
    sent: list[tuple[int, str]] = []

    async def fake_notify(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(tg, "notify_telegram_chat", fake_notify)
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(chat_id=900), _context()
    bound = await tg._session_workspace(update, context)
    await router.attach(store, bound.operator, web.id)
    await tg._run_turn(update, context, "hi")
    await _drain_tasks()
    update.message.reply_text.assert_awaited_once_with("hello from the target")
    assert sent == []


@pytest.mark.asyncio
async def test_attached_turn_uses_channel_tools_and_hint(tmp_path, monkeypatch):
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([_text("ok")]))
    store = reset_store_for_tests()
    vscode = _session(uuid4(), title="Editor session", channel="vscode")
    await store.put_entity(vscode)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    await router.attach(store, bound.operator, vscode.id)
    await tg._run_turn(update, context, "continue")
    call = llm.calls[0]
    names = {t["name"] for t in call["tools"]}
    assert "ProposePatch" not in names  # Telegram cannot render a diff overlay
    assert not names & router.OPERATOR_TOOLS  # operator tools stay on the operator
    system_text = json.dumps(call["system"])
    assert "driving this session from Telegram" in system_text


# ------------------------------------------------------------ operator tools


@pytest.mark.asyncio
async def test_operator_tools_are_allowlisted(tmp_path):
    ws = _ws(tmp_path)
    for name in router.OPERATOR_TOOLS:
        decision = await can_use_tool(
            name, {"session": "x"}, {"workspace": ws, "headless": True, "session_id": uuid4()}
        )
        assert decision.behavior == "allow", (name, decision)


@pytest.mark.asyncio
async def test_operator_turn_lists_and_attaches(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Airbed controller", channel="web")
    await store.put_entity(web)
    await store.append_event(web.id, "user", {"text": "tune the PID loop"})
    await store.append_event(web.id, "assistant", {"text": "Gains adjusted; tests pass."})
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([
        SimpleNamespace(content=[_ToolUse("ListSessions", {}, "tu-list")]),
        SimpleNamespace(content=[_ToolUse("SessionDigest", {"session": "airbed"}, "tu-digest")]),
        SimpleNamespace(content=[_ToolUse("AttachSession", {"session": "airbed"}, "tu-attach")]),
        _text("You're on the airbed session now."),
    ]))
    update, context = _update(chat_id=900), _context()
    await tg._run_turn(update, context, "what was I doing with the airbed?")

    op = (await tg._session_workspace(update, context)).operator
    assert router.attached_id(op) == web.id
    assert router.sink_keys(web.id) == ["telegram:900"]
    results = {
        e.payload["tool_use_id"]: json.loads(e.payload["content"])
        for e in await store.list_events(op.id)
        if e.kind == "tool_result"
    }
    assert results["tu-list"]["sessions"][0]["title"] == "Airbed controller"
    assert results["tu-digest"]["last_assistant_text"] == "Gains adjusted; tests pass."
    assert results["tu-attach"]["attached"] == str(web.id)
    assert update.message.reply_text.await_args.args[0] == "You're on the airbed session now."
    names = {t["name"] for t in llm.calls[0]["tools"]}
    assert router.OPERATOR_TOOLS <= names
    assert "operator for this chat" in json.dumps(llm.calls[0]["system"])

    # The next plain message runs on the attached session, not the operator.
    llm._responses.append(_text("continuing"))
    await tg._run_turn(update, context, "keep going")
    assert [e.payload["text"] for e in await store.list_events(web.id) if e.kind == "user"][-1] == (
        "keep going"
    )


@pytest.mark.asyncio
async def test_operator_tool_errors_are_reported_not_raised(tmp_path):
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    a = _session(uuid4(), title="Airbed one")
    b = _session(uuid4(), title="Airbed two")
    await store.put_entity(a)
    await store.put_entity(b)
    ctx = {"store": store, "session_id": op.id}
    out = json.loads(await router.run_operator_tool("AttachSession", {"session": "airbed"}, ctx))
    assert "matches" in out and len(out["matches"]) == 2
    assert router.attached_id(await store.get_entity(op.id)) is None
    assert (await router.run_operator_tool("AttachSession", {"session": "zzz"}, ctx)).startswith("error")
    assert (await router.run_operator_tool("SessionDigest", {}, ctx)).startswith("error")
    detached = json.loads(await router.run_operator_tool("DetachSession", {}, ctx))
    assert detached == {"detached": None}


# ----------------------------------------------------------- commands / phase 3


@pytest.mark.asyncio
async def test_command_texts(tmp_path):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Ship it", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    status = await tg.status_text(bound)
    assert "not attached" in status and "/sessions" in status
    listing = await tg.sessions_text(bound)
    assert "Ship it" in listing and "/attach" in listing
    attached = await tg.attach_text(bound, "ship")
    assert attached.startswith("Attached.") and "Ship it" in attached
    bound = await tg._session_workspace(update, context)
    assert bound.attached
    status = await tg.status_text(bound)
    assert "attached to: Ship it" in status and "/detach" in status
    assert (await tg.attach_text(bound, "")).startswith(listing.split("\n", 1)[0])
    assert "No session matches" in await tg.attach_text(bound, "nope")
    assert (await tg.detach_text(bound)).startswith("Detached from")
    assert (await tg.detach_text(bound)).startswith("Not attached")


@pytest.mark.asyncio
async def test_start_turn_runs_in_background_and_stop_cancels(tmp_path, monkeypatch):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("slow answer")], delay=5.0))
    store = reset_store_for_tests()
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    task = tg.start_turn(update, context, "think hard")
    for _ in range(200):
        if turns.is_running(bound.session_id):
            break
        await asyncio.sleep(0.005)
    assert turns.is_running(bound.session_id)  # the handler returned while the turn runs
    assert not task.done()
    state = turns.get(bound.session_id)
    assert state is not None and state.channel == "telegram"
    state.cancel.set()  # what /stop (and the web Stop button) does
    await asyncio.wait_for(task, 5)
    assert not turns.is_running(bound.session_id)
    kinds = [e.kind for e in await store.list_events(bound.session_id)]
    assert "turn_interrupted" in kinds
    assert update.message.reply_text.await_args.args[0] == "Turn stopped."


@pytest.mark.asyncio
async def test_operator_only_turn_bypasses_attachment(tmp_path, monkeypatch):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("operator here")]))
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    await router.attach(store, bound.operator, web.id)
    await tg._run_turn(update, context, "what sessions do I have?", operator_only=True)
    assert await store.list_events(web.id) == []
    assert [e.kind for e in await store.list_events(bound.operator.id)][:2] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_approval_on_attached_session_resolves_from_callback(tmp_path, monkeypatch):
    """The keyboard callback carries only the tool_use id; the adapter finds the session."""
    from orbweaver.agent import PendingApproval, _register_approval, _unregister_approval

    store = reset_store_for_tests()
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    await router.attach(store, bound.operator, web.id)
    loop = asyncio.get_running_loop()
    pend = PendingApproval(
        session_id=web.id,
        tool_use_id="tu-9",
        name="Bash",
        input={"command": "rm -rf build"},
        reason="ask",
        future=loop.create_future(),
    )
    _register_approval(pend)
    try:
        assert tg._approval_session([bound.operator.id, web.id], "tu-9") == web.id
        assert tg._approval_session([bound.operator.id], "tu-9") is None
    finally:
        _unregister_approval(pend)


@pytest.mark.asyncio
async def test_via_tag_on_injected_message(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    await router.attach(store, bound.operator, web.id)
    running = turns.acquire(web.id, channel="web")
    try:
        await tg._run_turn(update, context, "also do X")
    finally:
        turns.release(web.id, running)
    ev = (await store.list_events(web.id))[-1]
    assert ev.kind == "user"
    assert ev.payload == {"text": "also do X", "injected": True, "via": "telegram"}
    assert "/stop" in update.message.reply_text.await_args.args[0]
