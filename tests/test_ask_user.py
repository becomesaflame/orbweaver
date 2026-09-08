from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import anthropic
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import (
    _ask_user_headless_abort,
    agent_turn,
    can_wait_for_user,
    pending_ask_user,
    run_tools,
)
from orbweaver.app import app
from orbweaver.channels.telegram import texts_for_reply
from orbweaver.config import settings
from orbweaver.permissions.pipeline import TurnAborted
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


class _ToolUse:
    def __init__(self, name, inp, uid="tu-ask"):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _RecordingAnthropic:
    def __init__(self, responses, *a, **k):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="got it")]
            )
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


def _ws(tmp_path):
    return LocalWorkspace("workspace:default", str(tmp_path))


def test_can_wait_telegram_even_when_headless():
    assert can_wait_for_user({"headless": True, "interactive": True}) is True
    assert can_wait_for_user({"headless": True}) is False
    assert can_wait_for_user({"headless": False}) is True


def test_pending_ask_user_finds_unanswered_call():
    sid = uuid4()
    call = Event(
        id=uuid4(),
        session_id=sid,
        seq=1,
        kind="tool_call",
        payload={"id": "tu-ask", "name": "AskUser", "input": {"question": "Name?"}},
    )
    assert pending_ask_user([call]) is call
    result = Event(
        id=uuid4(),
        session_id=sid,
        seq=2,
        kind="tool_result",
        payload={"tool_use_id": "tu-ask", "name": "AskUser", "content": "Ada"},
    )
    assert pending_ask_user([call, result]) is None


def test_texts_for_reply_includes_ask_user():
    sid = uuid4()
    events = [
        Event(
            id=uuid4(),
            session_id=sid,
            seq=1,
            kind="ask_user",
            payload={"question": "What should I name the file?"},
        )
    ]
    assert "What should I name the file?" in texts_for_reply(events)


@pytest.mark.asyncio
async def test_run_tools_askuser_aborts_headless(tmp_path):
    store = reset_store_for_tests()
    with pytest.raises(TurnAborted) as ei:
        await run_tools(
            "AskUser",
            {"question": "Ship it?"},
            {
                "workspace": _ws(tmp_path),
                "store": store,
                "session_id": uuid4(),
                "headless": True,
            },
        )
    assert ei.value.payload["reason"] == "ask_user_headless"
    assert "headless" in ei.value.message.lower()


@pytest.mark.asyncio
async def test_interactive_askuser_pauses_until_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    first = SimpleNamespace(
        content=[_ToolUse("AskUser", {"question": "Which branch?"})]
    )
    client = _RecordingAnthropic([first])

    def factory(*a, **k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = _ws(tmp_path)
    events = await agent_turn(store, sid, "need a name", ws)
    kinds = [e.kind for e in events]
    assert "ask_user" in kinds
    assert "tool_call" in kinds
    assert not any(e.kind == "tool_result" for e in events)
    ask = next(e for e in events if e.kind == "ask_user")
    assert ask.payload["question"] == "Which branch?"
    assert pending_ask_user(await store.list_events(sid)) is not None
    assert len(client.calls) == 1

    client._responses.append(
        SimpleNamespace(content=[SimpleNamespace(type="text", text="using feature/x")])
    )
    resumed = await agent_turn(store, sid, "feature/x", ws)
    results = [e for e in resumed if e.kind == "tool_result"]
    assert results
    assert results[0].payload["content"] == "feature/x"
    assert results[0].payload["name"] == "AskUser"
    texts = [e.payload.get("text") for e in resumed if e.kind == "assistant"]
    assert any("using feature/x" in (t or "") for t in texts)
    assert pending_ask_user(await store.list_events(sid)) is None
    follow = client.calls[1]["messages"]
    blob = str(follow)
    assert "feature/x" in blob
    assert "tool_result" in blob


@pytest.mark.asyncio
async def test_telegram_interactive_waits_despite_headless(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    first = SimpleNamespace(content=[_ToolUse("AskUser", {"question": "OK to push?"})])

    def factory(*a, **k):
        return _RecordingAnthropic([first], *a, **k)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    events = await agent_turn(
        store, sid, "push?", _ws(tmp_path), headless=True, interactive=True
    )
    assert any(e.kind == "ask_user" for e in events)
    assert not any(e.kind == "turn_aborted" for e in events)
    assert "OK to push?" in texts_for_reply(events)


@pytest.mark.asyncio
async def test_headless_askuser_aborts_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    first = SimpleNamespace(content=[_ToolUse("AskUser", {"question": "OK to push?"})])

    def factory(*a, **k):
        return _RecordingAnthropic([first], *a, **k)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    events = await agent_turn(store, sid, "cron job", _ws(tmp_path), headless=True)
    kinds = [e.kind for e in events]
    assert "turn_aborted" in kinds
    abort = next(e for e in events if e.kind == "turn_aborted")
    assert abort.payload["reason"] == "ask_user_headless"
    assert "headless" in abort.payload["text"].lower()
    assert not any(e.kind == "ask_user" for e in events)
    assert texts_for_reply(events)


@pytest.mark.asyncio
async def test_api_wait_and_resume(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    first = SimpleNamespace(content=[_ToolUse("AskUser", {"question": "Port?"})])
    client_llm = _RecordingAnthropic([first])

    def factory(*a, **k):
        return client_llm

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        sid = sess.json()["id"]
        paused = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "pick a port"},
            headers=auth_header,
        )
        assert paused.status_code == 200, paused.text
        body = paused.json()
        assert body["status"] == "waiting_ask"
        assert body["question"] == "Port?"
        kinds = [e["kind"] for e in body["events"]]
        assert "ask_user" in kinds
        cont = await client.post(
            f"/v1/sessions/{sid}/turns/continue",
            headers=auth_header,
        )
        assert cont.status_code == 400
        client_llm._responses.append(
            SimpleNamespace(content=[SimpleNamespace(type="text", text="8080 it is")])
        )
        answered = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "8080"},
            headers=auth_header,
        )
        assert answered.status_code == 200, answered.text
        assert answered.json()["status"] == "ok"
        listed = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        evs = listed.json()["events"]
        results = [e for e in evs if e["kind"] == "tool_result"]
        assert results
        assert results[0]["payload"]["content"] == "8080"
        assert any(
            e["kind"] == "assistant" and "8080" in (e["payload"].get("text") or "")
            for e in answered.json()["events"]
        )


@pytest.mark.asyncio
async def test_telegram_run_turn_marks_interactive(monkeypatch):
    from orbweaver.channels import telegram as tg

    seen: dict = {}

    async def fake_turn(*_a, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tg, "agent_turn", fake_turn)

    async def fake_workspace(_update, _context):
        store = reset_store_for_tests()
        return store, uuid4(), object(), "local"

    monkeypatch.setattr(tg, "_session_workspace", fake_workspace)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await tg._run_turn(update, MagicMock(), "hello")
    assert seen.get("interactive") is True
    assert seen.get("headless") is True
    update.message.reply_text.assert_awaited()


def test_ask_user_abort_message_is_specific():
    exc = _ask_user_headless_abort("Ship to prod?")
    assert exc.payload["reason"] == "ask_user_headless"
    assert "web or Telegram" in exc.message
    assert "Ship to prod?" in exc.payload["last_input"]
