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
from orbweaver.compact.project import (
    INTERRUPTED_TOOL,
    events_to_messages,
    unpaired_tool_use_ids,
)
from orbweaver.config import settings
from orbweaver.permissions.pipeline import TurnAborted
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def orphan_tool_results(messages: list[dict]) -> list[str]:
    """tool_use_ids Anthropic would reject: a tool_result with no tool_use before it.

    Mirrors ``unexpected tool_use_id found in tool_result blocks: ...``.
    """
    orphans: list[str] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            continue
        prev = messages[i - 1] if i else None
        uses: set[str] = set()
        if prev and prev.get("role") == "assistant" and isinstance(prev.get("content"), list):
            uses = {
                str(b.get("id"))
                for b in prev["content"]
                if isinstance(b, dict) and b.get("type") == "tool_use"
            }
        orphans += [
            str(b.get("tool_use_id"))
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "tool_result"
            and str(b.get("tool_use_id")) not in uses
        ]
    return orphans


def _tool_results(messages: list[dict]) -> list[dict]:
    return [
        b
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]


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

    async def fake_workspace(_update, _context, **_kw):
        from orbweaver.channels.telegram import session_for_telegram_user

        store = reset_store_for_tests()
        op = await session_for_telegram_user(store, 42, chat_id=42)
        return tg.Bound(store=store, operator=op, target=op, ws=object(), kind="local", chat_id=42)

    monkeypatch.setattr(tg, "_session_workspace", fake_workspace)
    monkeypatch.setattr(tg.router, "operator_tools", AsyncMock(return_value=[]))
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await tg._run_turn(update, MagicMock(), "hello")
    assert seen.get("interactive") is True
    assert seen.get("headless") is True
    update.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_ask_user_answer_does_not_orphan_its_tool_result(tmp_path, monkeypatch):
    """Production 400 on session b3985a44: the answer was sent twice.

    The answer is stored as a ``user`` event *and* as the AskUser ``tool_result``.
    Projecting both paired the tool_use with an "interrupted" stub and left the real
    result with no tool_use in the message before it, which Anthropic rejects with
    ``unexpected tool_use_id found in tool_result blocks``.
    """
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ask = SimpleNamespace(
        content=[_ToolUse("AskUser", {"question": "Which branch?"}, uid="toolu_ask")]
    )
    client = _RecordingAnthropic([ask])
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = _ws(tmp_path)

    await agent_turn(store, sid, "need a name", ws)
    client._responses.append(
        SimpleNamespace(content=[SimpleNamespace(type="text", text="using feature/x")])
    )
    await agent_turn(store, sid, "feature/x", ws)

    sent = client.calls[-1]["messages"]
    assert orphan_tool_results(sent) == []
    assert unpaired_tool_use_ids(sent) == []
    # The answer still reaches the model, once, as the AskUser result.
    answers = [b for b in _tool_results(sent) if b["tool_use_id"] == "toolu_ask"]
    assert [b["content"] for b in answers] == ["feature/x"]
    assert INTERRUPTED_TOOL not in str(sent)


def test_ask_answer_events_project_without_orphan_result():
    """Event shape recorded in production: tool_call, ask_user, user(ask_answer), tool_result."""
    sid = uuid4()

    def ev(seq: int, kind: str, payload: dict) -> Event:
        return Event(id=uuid4(), session_id=sid, seq=seq, kind=kind, payload=payload)

    events = [
        ev(1, "user", {"text": "sort out gh auth"}),
        ev(2, "assistant", {"text": "No stored credentials anywhere I can find."}),
        ev(3, "tool_call", {"id": "toolu_ask", "name": "AskUser", "input": {"question": "Token?"}}),
        ev(4, "permission_decision", {"name": "AskUser", "behavior": "allow", "tool_use_id": "toolu_ask"}),
        ev(5, "ask_user", {"question": "Token?", "tool_use_id": "toolu_ask", "name": "AskUser"}),
        ev(6, "user", {"text": "drop it, I'll fix gh myself", "ask_answer": True}),
        ev(7, "tool_result", {"tool_use_id": "toolu_ask", "name": "AskUser", "content": "drop it, I'll fix gh myself"}),
        ev(8, "user", {"text": "did the gh auth issue get resolved?"}),
    ]
    messages = events_to_messages(events)
    assert orphan_tool_results(messages) == []
    assert unpaired_tool_use_ids(messages) == []
    answers = [b for b in _tool_results(messages) if b["tool_use_id"] == "toolu_ask"]
    assert [b["content"] for b in answers] == ["drop it, I'll fix gh myself"]
    assert INTERRUPTED_TOOL not in str(messages)
    # The follow-up question after the answer is still there.
    assert "did the gh auth issue get resolved?" in str(messages)


def test_ask_answer_with_image_keeps_the_image_and_pairs_the_result():
    sid = uuid4()

    def ev(seq: int, kind: str, payload: dict) -> Event:
        return Event(id=uuid4(), session_id=sid, seq=seq, kind=kind, payload=payload)

    events = [
        ev(1, "tool_call", {"id": "toolu_ask", "name": "AskUser", "input": {"question": "Which one?"}}),
        ev(2, "ask_user", {"question": "Which one?", "tool_use_id": "toolu_ask", "name": "AskUser"}),
        ev(
            3,
            "user",
            {
                "text": "this one",
                "ask_answer": True,
                "images": [{"media_type": "image/png", "path": "shot.png"}],
            },
        ),
        ev(4, "tool_result", {"tool_use_id": "toolu_ask", "name": "AskUser", "content": "this one"}),
    ]
    messages = events_to_messages(events)
    assert orphan_tool_results(messages) == []
    assert unpaired_tool_use_ids(messages) == []
    assert any(
        isinstance(b, dict) and b.get("type") == "image"
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
    )


def test_ask_user_abort_message_is_specific():
    exc = _ask_user_headless_abort("Ship to prod?")
    assert exc.payload["reason"] == "ask_user_headless"
    assert "web or Telegram" in exc.message
    assert "Ship to prod?" in exc.payload["last_input"]
