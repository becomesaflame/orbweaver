"""Session router: Telegram operator dispatches to other sessions without attaching.

These run the real ``agent_turn`` (scripted Anthropic client) so the cross-channel
paths are exercised end to end: PromptSession answers a web ``AskUser``; a watched
web turn's result is delivered to the chat; unwatched completions stay quiet;
WatchSession subscribes without a polling cron that would hold the operator lock;
reply-to a tagged report injects; leftover ``attached_session`` is ignored.
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
    tg.install_telegram_sink()
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


def _update(user_id=42, chat_id=42, text="hi", reply_text=None):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.is_bot = False
    update.effective_chat.id = chat_id
    update.message.text = text
    update.message.reply_text = AsyncMock()
    if reply_text is None:
        update.message.reply_to_message = None
    else:
        reply = MagicMock()
        reply.text = reply_text
        reply.caption = None
        reply.from_user = MagicMock()
        reply.from_user.is_bot = True
        update.message.reply_to_message = reply
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
async def test_deleted_sessions_are_not_prompt_candidates():
    """A chat deleted in the web UI must not be reachable from another channel."""
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    live = _session(UUID("aaaaaaaa-0000-4000-8000-000000000011"), title="Airbed controller")
    gone = _session(
        UUID("bbbbbbbb-0000-4000-8000-000000000012"),
        title="Airbed deleted",
        status="deleted",
    )
    for s in (live, gone):
        await store.put_entity(s)
    ex = {op.id}
    assert [e.id for e in await router.candidate_sessions(store, exclude=ex)] == [live.id]
    assert await router.find_session(store, str(gone.id), exclude=ex) is None
    assert (await router.find_session(store, "airbed", exclude=ex)).id == live.id


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
        listing = router.format_session_list(rows, watching=web.id)
        assert "watching" in listing and "running via web" in listing
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

    def capture(s, m):
        seen.append(("global", s, m["kind"]))

    router.add_global_sink(capture)

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
    router.remove_global_sink(capture)
    assert router.sink_keys(sid) == []


def test_session_tag_round_trip():
    sid = UUID("aaaaaaaa-0000-4000-8000-000000000001")
    tag = tg.session_tag("sidebar context menu", sid)
    assert tag == "[sidebar context menu · aaaaaaaa]"
    assert tg.parse_session_tag(f"{tag}\nTests are green.") == "aaaaaaaa"
    assert tg.parse_session_tag("plain text") is None


@pytest.mark.asyncio
async def test_legacy_attach_is_cleared_on_normalize(tmp_path, monkeypatch):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("still the operator")]))
    store = reset_store_for_tests()
    web = _session(uuid4(), title="sidebar context menu", channel="web")
    await store.put_entity(web)
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    op.jsonld[router.ATTACHED_KEY] = session_at_id(web.id)
    await store.put_entity(op)
    update, context = _update(), _context()
    await tg._run_turn(update, context, "what's up")
    op = await store.get_entity(op.id)
    assert router.ATTACHED_KEY not in (op.jsonld or {})
    assert [e.payload.get("text") for e in await store.list_events(web.id) if e.kind == "user"] == []
    assert [e.kind for e in await store.list_events(op.id)][:2] == ["user", "assistant"]


# ---------------------------------------------------------------- dispatcher sink


@pytest.mark.asyncio
async def test_approval_from_unwatched_session_reaches_telegram(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sent: list[tuple[int, str]] = []

    async def fake_approval(chat_id, payload, *, tag=""):
        sent.append((chat_id, tag, payload.get("tool_use_id")))

    monkeypatch.setattr(tg, "notify_telegram_approval", fake_approval)
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = _session(uuid4(), title="Ship the router", channel="web")
    await store.put_entity(web)
    _remember = op
    del _remember
    router.emit(
        web.id,
        {
            "kind": "permission_request",
            "payload": {"tool_use_id": "tu-9", "name": "Bash", "summary": "rm -rf build", "reason": "ask"},
        },
    )
    await _drain_tasks()
    assert sent and sent[0][0] == 900 and sent[0][2] == "tu-9"
    assert "Ship the router" in sent[0][1] and str(web.id)[:8] in sent[0][1]


@pytest.mark.asyncio
async def test_unwatched_ok_is_silent_watched_ok_and_abort_report(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sent: list[tuple[int, str]] = []

    async def fake_notify(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(tg, "notify_telegram_chat", fake_notify)
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = _session(uuid4(), title="Ship the router", channel="web")
    await store.put_entity(web)
    await store.append_event(web.id, "assistant", {"text": "desk-only reply"})
    router.emit(web.id, {"kind": "turn_done", "status": "ok", "user_seq": 0, "channel": "web"})
    await _drain_tasks()
    assert sent == []

    await router.mark_prompted(store, op, web.id)
    tg._watched_chats[web.id] = 900
    await store.append_event(web.id, "assistant", {"text": "prompted reply"})
    router.emit(web.id, {"kind": "turn_done", "status": "ok", "user_seq": 0, "channel": "web"})
    await _drain_tasks()
    assert len(sent) == 1
    assert "Ship the router" in sent[0][1] and "prompted reply" in sent[0][1]
    op = await store.get_entity(op.id)
    assert web.id not in router.prompted_ids(op)

    other = _session(uuid4(), title="Unwatched fail", channel="web")
    await store.put_entity(other)
    await store.append_event(other.id, "turn_aborted", {"text": "classifier denied"})
    await store.append_event(other.id, "assistant", {"text": "Turn aborted: classifier denied"})
    router.emit(other.id, {"kind": "turn_done", "status": "ok", "user_seq": 0, "channel": "web"})
    await _drain_tasks()
    assert any("Unwatched fail" in t and "classifier denied" in t for _c, t in sent)


@pytest.mark.asyncio
async def test_web_turn_result_is_delivered_when_watched(tmp_path, monkeypatch, auth_header):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("Tests are green, PR is up.")]))
    store = reset_store_for_tests()
    sent: list[tuple[int, str]] = []

    async def fake_notify(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(tg, "notify_telegram_chat", fake_notify)
    web = _session(uuid4(), title="Ship the router", channel="web")
    await store.put_entity(web)
    bound = await tg._session_workspace(_update(chat_id=900), _context())
    await router.mark_prompted(store, bound.operator, web.id)
    tg._watched_chats[web.id] = 900

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            f"/v1/sessions/{web.id}/turns", json={"text": "run the tests"}, headers=auth_header
        )
        assert r.status_code == 200
    await _drain_tasks()
    assert sent
    assert sent[-1][0] == 900
    assert "Ship the router" in sent[-1][1]
    assert "Tests are green, PR is up." in sent[-1][1]


@pytest.mark.asyncio
async def test_watch_session_notifies_without_cron_holding_operator_lock(
    tmp_path, monkeypatch, auth_header
):
    """Watching a web session must not poll via cron on the operator.

    Tonight a 1-minute progress cron on the Telegram operator held the
    per-session turn lock, so later messages were refused. WatchSession
    plus the existing ``turn_done`` sink reports completion without
    occupying the operator.
    """
    from datetime import UTC, datetime, timedelta

    llm = _install_llm(
        monkeypatch,
        _ScriptedAnthropic(
            [
                SimpleNamespace(
                    content=[_ToolUse("WatchSession", {"session": "Ship the router"}, "tu-watch")]
                ),
                _text("Watching Ship the router. I'll get a report when it finishes."),
            ]
        ),
    )
    store = reset_store_for_tests()
    sent: list[tuple[int, str]] = []

    async def fake_notify(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(tg, "notify_telegram_chat", fake_notify)
    web = _session(uuid4(), title="Ship the router", channel="web")
    await store.put_entity(web)
    update, context = _update(chat_id=900), _context()
    await tg._run_turn(update, context, "watch the ship session")

    op = (await tg._session_workspace(update, context)).operator
    assert turns.get(op.id) is None
    assert await store.due_jobs(datetime.now(UTC) + timedelta(days=3650)) == []
    results = {
        e.payload["tool_use_id"]: json.loads(e.payload["content"])
        for e in await store.list_events(op.id)
        if e.kind == "tool_result"
    }
    assert results["tu-watch"]["action"] == "watching"
    assert results["tu-watch"]["session"] == str(web.id)
    assert web.id in router.prompted_ids(await store.get_entity(op.id))

    llm._responses.append(_text("Tests are green, PR is up."))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            f"/v1/sessions/{web.id}/turns", json={"text": "run the tests"}, headers=auth_header
        )
        assert r.status_code == 200
    await _drain_tasks()
    assert sent
    assert sent[-1][0] == 900
    assert "Ship the router" in sent[-1][1]
    assert "Tests are green, PR is up." in sent[-1][1]
    assert turns.get(op.id) is None

    llm._responses.append(_text("got the ping"))
    await tg._run_turn(update, context, "anything new?")
    assert [e.payload.get("text") for e in await store.list_events(op.id) if e.kind == "user"][-1] == (
        "anything new?"
    )
    assert turns.get(op.id) is None
    assert await store.due_jobs(datetime.now(UTC) + timedelta(days=3650)) == []


@pytest.mark.asyncio
async def test_minute_cron_on_operator_blocks_telegram_message(tmp_path, monkeypatch):
    """The production failure: a recurring progress cron occupies the operator lock."""
    from datetime import UTC, datetime, timedelta

    from orbweaver.channels.cron import sweep_and_wait
    from orbweaver.store import Job

    _install_llm(
        monkeypatch,
        _ScriptedAnthropic([_text("still running"), _text("operator after cron")], delay=0.35),
    )
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    await store.put_job(
        Job(
            id=uuid4(),
            due_at=datetime.now(UTC) - timedelta(seconds=1),
            payload={"message": "check progress on the three web sessions"},
            recurrence="minute",
            session_id=op.id,
        )
    )
    update, context = _update(chat_id=900), _context()
    context.user_data["session_id"] = str(op.id)

    sweep_task = asyncio.create_task(sweep_and_wait())
    running = None
    for _ in range(80):
        running = turns.get(op.id)
        if running is not None:
            break
        await asyncio.sleep(0.02)
    assert running is not None
    assert running.channel == "cron"
    assert turns.acquire(op.id, channel="telegram") is None

    # #202 queues the human turn behind the cron holder; it must not start now.
    queued = asyncio.create_task(tg._run_turn(update, context, "are they done?"))
    for _ in range(50):
        if update.message.reply_text.await_count:
            break
        await asyncio.sleep(0.02)
    reply = update.message.reply_text.await_args.args[0]
    assert "A turn is already running on this session" in reply
    assert "cron" in reply
    assert "queued" in reply
    assert turns.pending_waiters(op.id) >= 1
    assert turns.get(op.id) is not None and turns.get(op.id).channel == "cron"

    await sweep_task
    await queued
    assert turns.get(op.id) is None


@pytest.mark.asyncio
async def test_watch_and_unwatch_operator_tools(tmp_path):
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = _session(uuid4(), title="Ship the router", channel="web")
    await store.put_entity(web)
    ctx = {"store": store, "session_id": op.id}

    watched = json.loads(await router.run_operator_tool("WatchSession", {"session": "ship"}, ctx))
    assert watched["action"] == "watching"
    assert watched["session"] == str(web.id)
    assert web.id in router.prompted_ids(await store.get_entity(op.id))
    assert tg._watched_chats[web.id] == 900

    gone = json.loads(await router.run_operator_tool("UnwatchSession", {"session": str(web.id)}, ctx))
    assert gone["action"] == "unwatched"
    assert web.id not in router.prompted_ids(await store.get_entity(op.id))
    assert web.id not in tg._watched_chats


@pytest.mark.asyncio
async def test_send_photo_on_prompted_session_uses_operator_chat(tmp_path, monkeypatch):
    import httpx

    class _Resp:
        status_code = 200
        text = "ok"

    class _FakeAsyncClient:
        seen: dict | None = None

        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data=None, files=None, json=None):
            _FakeAsyncClient.seen = {"url": url, "data": data}
            return _Resp()

    monkeypatch.setattr(settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    await router.mark_prompted(store, op, web.id)
    tg._watched_chats[web.id] = 900
    ws = _ws(tmp_path)
    ws.write_bytes("shot.jpg", b"\xff\xd8\xff")
    result = await tg.send_session_photo(
        {"workspace": ws, "store": store, "session_id": web.id},
        {"path": "shot.jpg", "caption": "here"},
    )
    assert '"sent": true' in result
    assert _FakeAsyncClient.seen["data"]["chat_id"] == "900"


# ------------------------------------------------ prompted turns (real agent loop)


@pytest.mark.asyncio
async def test_prompt_session_answers_web_ask_user(tmp_path, monkeypatch, auth_header):
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

    update, context = _update(chat_id=900), _context()
    bound = await tg._session_workspace(update, context)
    llm._responses.append(_text("8080 it is"))
    monkeypatch.setattr(tg, "notify_telegram_chat", AsyncMock())
    result = json.loads(await tg.prompt_session(bound.store, bound.operator, "pick a port", "8080"))
    assert result["action"] == "started"
    await _drain_tasks()

    events = await store.list_events(sid)
    user_events = [e for e in events if e.kind == "user"]
    assert user_events[-1].payload["text"] == "8080"
    assert user_events[-1].payload.get("ask_answer") is True
    assert user_events[-1].payload.get("via") == "telegram"
    results = [e for e in events if e.kind == "tool_result" and e.payload.get("name") == "AskUser"]
    assert results and results[0].payload["content"] == "8080"
    assert pending_ask_user(events) is None
    op_users = [e for e in await store.list_events(bound.operator.id) if e.kind == "user"]
    assert op_users == []


@pytest.mark.asyncio
async def test_unreplied_text_stays_on_the_operator(tmp_path, monkeypatch):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("operator here")]))
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    await tg._run_turn(update, context, "what's going on")
    assert await store.list_events(web.id) == []
    assert [e.kind for e in await store.list_events((await tg._session_workspace(update, context)).operator.id)][:2] == [
        "user",
        "assistant",
    ]


async def _web_waiting_ask(store, title="Merge hook"):
    web = _session(uuid4(), title=title, channel="web")
    await store.put_entity(web)
    await store.append_event(
        web.id,
        "tool_call",
        {"id": "tu-ask", "name": "AskUser", "input": {"question": "Install ripgrep?"}},
    )
    await store.append_event(
        web.id,
        "ask_user",
        {"question": "Install ripgrep?", "tool_use_id": "tu-ask", "name": "AskUser"},
    )
    return web


@pytest.mark.asyncio
async def test_ask_user_ping_marks_the_next_message(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    sent: list[tuple] = []

    async def fake_notify(chat_id, text, *, force_reply=False):
        sent.append((chat_id, text, force_reply))

    monkeypatch.setattr(tg, "notify_telegram_chat", fake_notify)
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = await _web_waiting_ask(store)
    router.emit(
        web.id,
        {"kind": "ask_user", "payload": {"question": "Install ripgrep?", "tool_use_id": "tu-ask"}},
    )
    await _drain_tasks()
    op = await store.get_entity(op.id)
    assert op.jsonld[tg.OPEN_ASK_KEY] == str(web.id)
    assert sent and sent[0][0] == 900 and sent[0][2] is True
    assert "Waiting for your answer" in sent[0][1]
    assert "Install ripgrep?" in sent[0][1]
    assert str(web.id)[:8] in sent[0][1]


@pytest.mark.asyncio
async def test_plain_message_answers_forwarded_ask(tmp_path, monkeypatch):
    """Production failure: Telegram showed the question, and a bare 'yes' stayed on the operator."""
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([_text("Installing ripgrep.")]))
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = await _web_waiting_ask(store)
    op.jsonld[tg.OPEN_ASK_KEY] = str(web.id)
    await store.put_entity(op)
    monkeypatch.setattr(tg, "notify_telegram_chat", AsyncMock())
    update, context = _update(chat_id=900, text="yes"), _context()
    await tg.dispatch_inbound_text(update, context, "yes")
    await _drain_tasks()

    events = await store.list_events(web.id)
    users = [e for e in events if e.kind == "user"]
    assert users[-1].payload["text"] == "yes"
    assert users[-1].payload.get("ask_answer") is True
    assert users[-1].payload.get("via") == "telegram"
    answers = [e for e in events if e.kind == "tool_result" and e.payload.get("name") == "AskUser"]
    assert answers and answers[0].payload["content"] == "yes"
    assert pending_ask_user(events) is None
    assert [e for e in await store.list_events(op.id) if e.kind == "user"] == []
    assert llm.calls


@pytest.mark.asyncio
async def test_plain_message_stays_on_operator_after_ask_is_answered(tmp_path, monkeypatch):
    _install_llm(monkeypatch, _ScriptedAnthropic([_text("operator here")]))
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = await _web_waiting_ask(store)
    await store.append_event(
        web.id, "tool_result", {"tool_use_id": "tu-ask", "name": "AskUser", "content": "yes"}
    )
    op.jsonld[tg.OPEN_ASK_KEY] = str(web.id)
    await store.put_entity(op)
    update, context = _update(chat_id=900, text="what's next"), _context()
    await tg.dispatch_inbound_text(update, context, "what's next")
    await _drain_tasks()
    op = await store.get_entity(op.id)
    assert tg.OPEN_ASK_KEY not in op.jsonld
    assert [e.payload.get("text") for e in await store.list_events(web.id) if e.kind == "user"] == []
    assert [e.kind for e in await store.list_events(op.id)][:2] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_running_operator_turn_keeps_a_plain_message(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=900)
    web = await _web_waiting_ask(store)
    op.jsonld[tg.OPEN_ASK_KEY] = str(web.id)
    await store.put_entity(op)
    routed: list[str] = []
    monkeypatch.setattr(tg, "start_prompted", lambda *a, **k: routed.append("ask"))
    monkeypatch.setattr(tg, "start_turn", lambda *a, **k: routed.append("operator"))
    state = turns.acquire(op.id, channel="telegram")
    assert state is not None
    try:
        update, context = _update(chat_id=900, text="yes"), _context()
        await tg.dispatch_inbound_text(update, context, "yes")
    finally:
        turns.release(op.id, state)
    assert routed == ["operator"]


@pytest.mark.asyncio
async def test_reply_to_tagged_report_injects_without_operator_turn(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Airbed controller", channel="web")
    await store.put_entity(web)
    update, context = _update(
        chat_id=900,
        text="bump the gain",
        reply_text=f"{tg.session_tag('Airbed controller', web.id)}\nGains look fine.",
    ), _context()
    assert tg._reply_session_ref(update) == str(web.id)[:8]
    running = turns.acquire(web.id, channel="web")
    try:
        bound = await tg._session_workspace(update, context)
        await tg._prompt_from_reply(update, context, "bump the gain", str(web.id)[:8])
        ev = (await store.list_events(web.id))[-1]
        assert ev.kind == "user"
        assert ev.payload == {"text": "bump the gain", "injected": True, "via": "telegram"}
        assert await store.list_events(bound.operator.id) == []
        assert update.message.reply_text.await_count == 1
        assert "Airbed controller" in update.message.reply_text.await_args.args[0]
    finally:
        turns.release(web.id, running)


@pytest.mark.asyncio
async def test_prompt_session_starts_target_with_home_channel_tools(tmp_path, monkeypatch):
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([_text("ok")]))
    store = reset_store_for_tests()
    vscode = _session(uuid4(), title="Editor session", channel="vscode")
    await store.put_entity(vscode)
    update, context = _update(chat_id=900), _context()
    bound = await tg._session_workspace(update, context)
    monkeypatch.setattr(tg, "notify_telegram_chat", AsyncMock())
    await tg.prompt_session(bound.store, bound.operator, "Editor", "continue")
    await _drain_tasks()
    call = llm.calls[0]
    names = {t["name"] for t in call["tools"]}
    assert "ProposePatch" in names
    assert not names & router.OPERATOR_TOOLS
    events = await store.list_events(vscode.id)
    assert events[0].kind == "user"
    assert events[0].payload.get("via") == "telegram"


@pytest.mark.asyncio
async def test_operator_turn_lists_and_prompts(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Airbed controller", channel="web")
    await store.put_entity(web)
    await store.append_event(web.id, "user", {"text": "tune the PID loop"})
    await store.append_event(web.id, "assistant", {"text": "Gains adjusted; tests pass."})
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([
        SimpleNamespace(content=[_ToolUse("ListSessions", {}, "tu-list")]),
        SimpleNamespace(content=[_ToolUse("SessionDigest", {"session": "airbed"}, "tu-digest")]),
        SimpleNamespace(content=[_ToolUse("PromptSession", {"session": "airbed", "text": "keep going"}, "tu-prompt")]),
        _text("I prompted the airbed session."),
        _text("continuing"),
    ]))
    monkeypatch.setattr(tg, "notify_telegram_chat", AsyncMock())
    update, context = _update(chat_id=900), _context()
    await tg._run_turn(update, context, "what was I doing with the airbed?")

    op = (await tg._session_workspace(update, context)).operator
    assert router.prompted_ids(op) == [web.id]
    results = {
        e.payload["tool_use_id"]: json.loads(e.payload["content"])
        for e in await store.list_events(op.id)
        if e.kind == "tool_result"
    }
    assert results["tu-list"]["sessions"][0]["title"] == "Airbed controller"
    assert results["tu-digest"]["last_assistant_text"] == "Gains adjusted; tests pass."
    assert results["tu-prompt"]["action"] == "started"
    names = {t["name"] for t in llm.calls[0]["tools"]}
    assert router.OPERATOR_TOOLS <= names
    assert "AttachSession" not in names
    system = json.dumps(llm.calls[0]["system"])
    assert "dispatcher" in system
    assert "WatchSession" in system
    assert "recurring minute" in system
    await _drain_tasks()
    assert [e.payload.get("via") for e in await store.list_events(web.id) if e.kind == "user"][-1] == "telegram"

    llm._responses.append(_text("still operator"))
    await tg._run_turn(update, context, "thanks")
    assert [e.payload.get("text") for e in await store.list_events(op.id) if e.kind == "user"][-1] == "thanks"


@pytest.mark.asyncio
async def test_operator_tool_errors_are_reported_not_raised(tmp_path):
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    a = _session(uuid4(), title="Airbed one")
    b = _session(uuid4(), title="Airbed two")
    await store.put_entity(a)
    await store.put_entity(b)
    ctx = {"store": store, "session_id": op.id}
    out = json.loads(
        await router.run_operator_tool("PromptSession", {"session": "airbed", "text": "go"}, ctx)
    )
    assert "matches" in out and len(out["matches"]) == 2
    assert (await router.run_operator_tool("PromptSession", {"session": "zzz", "text": "go"}, ctx)).startswith("error")
    assert (await router.run_operator_tool("SessionDigest", {}, ctx)).startswith("error")
    assert (await router.run_operator_tool("StopSession", {"session": "zzz"}, ctx)).startswith("error")
    assert (await router.run_operator_tool("WatchSession", {"session": "zzz"}, ctx)).startswith("error")
    assert (await router.run_operator_tool("UnwatchSession", {}, ctx)).startswith("error")
    assert (await router.run_operator_tool("CreateSession", {}, ctx)).startswith("error")


@pytest.mark.asyncio
async def test_create_session_idle_prompts_and_rejects_bad_workspace(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    op = await tg.session_for_telegram_user(store, 42, chat_id=42)
    ctx = {"store": store, "session_id": op.id}

    idle = json.loads(await router.run_operator_tool("CreateSession", {"title": "Airbed PID"}, ctx))
    assert idle["title"] == "Airbed PID"
    assert idle["channel"] == "web"
    assert idle["workspace_uri"] == "workspace:default"
    sid = UUID(idle["created"])
    ent = await store.get_entity(sid)
    assert ent is not None and ent.jsonld["channel"] == "web"
    assert not router.is_operator_session(ent)
    assert await store.list_events(sid) == []
    assert "note" in idle

    named = json.loads(
        await router.run_operator_tool("CreateSession", {"title": "Docs", "workspace": "orbweaver"}, ctx)
    )
    assert named["workspace_uri"] == "workspace:orbweaver"
    assert named["created"] != idle["created"]

    bad = await router.run_operator_tool("CreateSession", {"title": "x", "workspace": "/etc"}, ctx)
    assert bad.startswith("error:")

    llm = _install_llm(monkeypatch, _ScriptedAnthropic([_text("started")]))
    monkeypatch.setattr(tg, "notify_telegram_chat", AsyncMock())
    started = json.loads(
        await router.run_operator_tool(
            "CreateSession", {"text": "Tune the PID loop on the airbed"}, ctx
        )
    )
    assert started["title"] == "Tune the PID loop on the airbed"
    assert started["prompt"]["action"] == "started"
    created = UUID(started["created"])
    op_mid = await store.get_entity(op.id)
    assert created in router.prompted_ids(op_mid)
    await _drain_tasks()
    events = await store.list_events(created)
    assert events[0].kind == "user"
    assert events[0].payload.get("via") == "telegram"
    assert events[0].payload.get("text") == "Tune the PID loop on the airbed"
    op = await store.get_entity(op.id)
    assert router.last_prompted_id(op) == created
    assert llm.calls
    names = {t["name"] for t in llm.calls[0]["tools"]}
    assert "ProposePatch" not in names  # home channel is web, not vscode
    assert not names & router.OPERATOR_TOOLS


@pytest.mark.asyncio
async def test_operator_tools_are_allowlisted(tmp_path):
    ws = _ws(tmp_path)
    for name in router.OPERATOR_TOOLS:
        if name in {
            "PromptSession",
            "StopSession",
            "SessionDigest",
            "WatchSession",
            "UnwatchSession",
        }:
            inp = {"session": "x", "text": "hi"}
        elif name == "CreateSession":
            inp = {"title": "x"}
        else:
            inp = {}
        decision = await can_use_tool(
            name, inp, {"workspace": ws, "headless": True, "session_id": uuid4()}
        )
        assert decision.behavior == "allow", (name, decision)


# ----------------------------------------------------------- commands


@pytest.mark.asyncio
async def test_command_texts(tmp_path):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Ship it", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    status = await tg.status_text(bound)
    assert "not watching" in status and "/sessions" in status
    listing = await tg.sessions_text(bound)
    assert "Ship it" in listing and "PromptSession" in listing
    await router.mark_prompted(store, bound.operator, web.id)
    bound = await tg._session_workspace(update, context)
    status = await tg.status_text(bound)
    assert "watching:" in status and "Ship it" in status


def _picker_keys(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "orbweaver_web_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_telegram_model", "qwen3.6-35b")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "earthruntime_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")


@pytest.mark.asyncio
async def test_model_command_lists_sets_and_clears(tmp_path, monkeypatch):
    _picker_keys(monkeypatch)
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Ship it", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)

    listed = await tg.model_text(bound, "")
    assert "Available:" in listed and "qwen3.6-35b" in listed
    assert "/model <id>" in listed
    status = await tg.status_text(bound)
    assert "model: default" in status and "qwen3.6-35b" in status

    set_msg = await tg.model_text(bound, "opus")
    assert "claude-opus-5" in set_msg
    op = await store.get_entity(bound.operator.id)
    assert op.jsonld["model"] == "claude-opus-5"
    assert "claude-opus-5" in await tg.status_text(bound)
    assert "Unknown model" in await tg.model_text(bound, "gpt-4o")
    assert "Several models match" in await tg.model_text(bound, "sonnet")

    cleared = await tg.model_text(bound, "default")
    assert "default" in cleared.lower()
    op = await store.get_entity(bound.operator.id)
    assert not op.jsonld.get("model")
    await tg.model_text(bound, "gpt-oss-120b")
    assert (await store.get_entity(bound.operator.id)).jsonld["model"] == "gpt-oss-120b"
    assert not (await store.get_entity(web.id)).jsonld.get("model")


@pytest.mark.asyncio
async def test_model_command_is_used_on_the_next_telegram_turn(tmp_path, monkeypatch):
    _picker_keys(monkeypatch)
    llm = _install_llm(monkeypatch, _ScriptedAnthropic([_text("ok")]))
    update, context = _update(), _context()
    bound = await tg._session_workspace(update, context)
    await tg.model_text(bound, "claude-opus-5")
    await tg._run_turn(update, context, "hello")
    assert llm.calls and llm.calls[0]["model"] == "claude-opus-5"


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
    assert turns.is_running(bound.session_id)
    assert not task.done()
    state = turns.get(bound.session_id)
    assert state is not None and state.channel == "telegram"
    state.cancel.set()
    await asyncio.wait_for(task, 5)
    assert not turns.is_running(bound.session_id)
    kinds = [e.kind for e in await store.list_events(bound.session_id)]
    assert "turn_interrupted" in kinds
    assert update.message.reply_text.await_args.args[0] == "Turn stopped."


@pytest.mark.asyncio
async def test_approval_resolves_from_callback_across_sessions(tmp_path, monkeypatch):
    from orbweaver.agent import (
        PendingApproval,
        _register_approval,
        _unregister_approval,
        find_pending_approval,
    )

    store = reset_store_for_tests()
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(), _context()
    await tg._session_workspace(update, context)
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
        assert find_pending_approval("tu-9") == web.id
        assert find_pending_approval("missing") is None
    finally:
        _unregister_approval(pend)


@pytest.mark.asyncio
async def test_via_tag_on_injected_prompt(tmp_path, monkeypatch):
    store = reset_store_for_tests()
    web = _session(uuid4(), title="Target", channel="web")
    await store.put_entity(web)
    update, context = _update(chat_id=900), _context()
    bound = await tg._session_workspace(update, context)
    running = turns.acquire(web.id, channel="web")
    try:
        monkeypatch.setattr(tg, "notify_telegram_chat", AsyncMock())
        result = json.loads(await tg.prompt_session(bound.store, bound.operator, str(web.id)[:8], "also do X"))
        assert result["action"] == "injected"
    finally:
        turns.release(web.id, running)
    ev = (await store.list_events(web.id))[-1]
    assert ev.kind == "user"
    assert ev.payload == {"text": "also do X", "injected": True, "via": "telegram"}
