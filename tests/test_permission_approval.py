"""Hold-for-approval flow: permission_request, approve/deny/timeout, allow-for-session."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import anthropic
import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import (
    agent_turn,
    pending_approvals,
    reset_pending_approvals_for_tests,
    resolve_approval,
)
from orbweaver.app import app
from orbweaver.channels import telegram as tg
from orbweaver.compact import events_to_messages
from orbweaver.config import settings
from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.permissions.session_rules import (
    approval_subject,
    make_session_rule,
    matching_session_rule,
    rule_allows,
)
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Event,
    reset_store_for_tests,
    session_at_id,
)
from orbweaver.workspace import LocalWorkspace

RM_BUILD = {"command": "rm -rf build"}


class _ToolUse:
    def __init__(self, name, inp, uid="tu-rm"):
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
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _reset():
    reset_store_for_tests()
    reset_pending_approvals_for_tests()
    reset_denial_states()
    yield
    reset_pending_approvals_for_tests()


@pytest.fixture
def ask_classifier(monkeypatch):
    """Classifier says `ask` for every Bash call; counts invocations."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    monkeypatch.setattr(settings, "orbweaver_auto_allow_bash_if_sandboxed", False)
    monkeypatch.setattr(settings, "orbweaver_approval_timeout_s", 5.0)
    calls: list[tuple[str, dict]] = []

    async def classify(_events, name, inp, **_k):
        calls.append((name, dict(inp)))
        if name == "Bash":
            return {"verdict": "ask", "reason": "destructive delete", "stage": "fast"}
        return {"verdict": "allow", "reason": "fine", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    return calls


@pytest.fixture
def fake_run_tools(monkeypatch):
    ran: list[tuple[str, dict]] = []

    async def run(name, inp, _ctx):
        ran.append((name, dict(inp)))
        return f"ran {name}"

    monkeypatch.setattr("orbweaver.agent.run_tools", run)
    return ran


def _llm(monkeypatch, responses):
    client = _RecordingAnthropic(responses)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    return client


def _ws(tmp_path):
    return LocalWorkspace("workspace:default", str(tmp_path))


async def _wait_pending(sid: UUID, tool_use_id: str):
    for _ in range(200):
        pend = [p for p in pending_approvals(sid) if p.tool_use_id == tool_use_id]
        if pend:
            return pend[0]
        await asyncio.sleep(0.01)
    raise AssertionError("permission_request never became pending")


def _by_kind(events, kind):
    return [e for e in events if e.kind == kind]


# --- agent loop -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_allow_executes_recorded_input_and_continues(
    tmp_path, monkeypatch, ask_classifier, fake_run_tools
):
    client = _llm(
        monkeypatch,
        [
            SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD))]),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="cleaned build")]),
        ],
    )
    store = reset_store_for_tests()
    sid = uuid4()
    emitted: list[dict] = []
    turn = asyncio.create_task(
        agent_turn(store, sid, "clean the build dir", _ws(tmp_path), emit=emitted.append)
    )
    pend = await _wait_pending(sid, "tu-rm")
    assert pend.input == RM_BUILD
    assert fake_run_tools == []  # held, not executed
    assert any(m["kind"] == "permission_request" for m in emitted)
    req = next(m for m in emitted if m["kind"] == "permission_request")
    assert req["payload"]["tool_use_id"] == "tu-rm"
    assert req["payload"]["name"] == "Bash"
    assert req["payload"]["input"] == RM_BUILD
    assert req["payload"]["summary"] == "rm -rf build"
    assert req["payload"]["reason"] == "destructive delete"

    assert resolve_approval(sid, "tu-rm", "allow", "once") is True
    events = await turn

    assert fake_run_tools == [("Bash", RM_BUILD)]
    results = _by_kind(events, "tool_result")
    assert results and results[0].payload["content"] == "ran Bash"
    assert not results[0].payload.get("is_error")
    response = _by_kind(events, "permission_response")
    assert response and response[0].payload["decision"] == "allow"
    assert response[0].payload["scope"] == "once"
    texts = [e.payload.get("text") for e in _by_kind(events, "assistant")]
    assert "cleaned build" in texts
    assert not any("I need your approval" in (t or "") for t in texts)
    assert len(client.calls) == 2
    assert pending_approvals(sid) == []
    assert not _by_kind(events, "permission_rule_added")


@pytest.mark.asyncio
async def test_deny_records_error_result_and_model_continues(
    tmp_path, monkeypatch, ask_classifier, fake_run_tools
):
    client = _llm(
        monkeypatch,
        [
            SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD))]),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="ok, leaving build")]),
        ],
    )
    store = reset_store_for_tests()
    sid = uuid4()
    turn = asyncio.create_task(agent_turn(store, sid, "clean up", _ws(tmp_path)))
    await _wait_pending(sid, "tu-rm")
    assert resolve_approval(sid, "tu-rm", "deny") is True
    events = await turn

    assert fake_run_tools == []
    results = _by_kind(events, "tool_result")
    assert results and results[0].payload["is_error"] is True
    assert "denied by user" in results[0].payload["content"].lower()
    assert _by_kind(events, "permission_response")[0].payload["decision"] == "deny"
    texts = [e.payload.get("text") for e in _by_kind(events, "assistant")]
    assert "ok, leaving build" in texts
    assert len(client.calls) == 2
    # The denial reaches the model as an is_error tool_result.
    blob = str(client.calls[1]["messages"])
    assert "'is_error': True" in blob
    assert "Denied by user" in blob


@pytest.mark.asyncio
async def test_timeout_records_error_and_ends_turn(
    tmp_path, monkeypatch, ask_classifier, fake_run_tools
):
    monkeypatch.setattr(settings, "orbweaver_approval_timeout_s", 0.2)
    client = _llm(monkeypatch, [SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD))])])
    store = reset_store_for_tests()
    sid = uuid4()
    events = await agent_turn(store, sid, "clean up", _ws(tmp_path))

    assert fake_run_tools == []
    results = _by_kind(events, "tool_result")
    assert results and results[0].payload["is_error"] is True
    assert "no approval decision within 0.2s" in results[0].payload["content"]
    assert _by_kind(events, "permission_response")[0].payload["decision"] == "timeout"
    texts = [e.payload.get("text") for e in _by_kind(events, "assistant")]
    assert any("timed out" in (t or "") for t in texts)
    assert len(client.calls) == 1  # the turn ended; no follow-up model call
    assert pending_approvals(sid) == []


@pytest.mark.asyncio
async def test_later_tool_uses_wait_for_the_decision(
    tmp_path, monkeypatch, ask_classifier, fake_run_tools
):
    first = SimpleNamespace(
        content=[
            _ToolUse("Glob", {"pattern": "*.py"}, uid="tu-glob"),
            _ToolUse("Bash", dict(RM_BUILD), uid="tu-rm"),
            _ToolUse("Glob", {"pattern": "*.md"}, uid="tu-glob2"),
        ]
    )
    _llm(monkeypatch, [first])
    store = reset_store_for_tests()
    sid = uuid4()
    turn = asyncio.create_task(agent_turn(store, sid, "tidy", _ws(tmp_path)))
    await _wait_pending(sid, "tu-rm")
    # Safe call before the held one already ran; the one after it has not.
    assert [n for n, _ in fake_run_tools] == ["Glob"]
    assert fake_run_tools[0][1] == {"pattern": "*.py"}
    resolve_approval(sid, "tu-rm", "allow")
    events = await turn
    assert [(n, i) for n, i in fake_run_tools] == [
        ("Glob", {"pattern": "*.py"}),
        ("Bash", RM_BUILD),
        ("Glob", {"pattern": "*.md"}),
    ]
    ids = [e.payload["tool_use_id"] for e in _by_kind(events, "tool_result")]
    assert ids == ["tu-glob", "tu-rm", "tu-glob2"]


@pytest.mark.asyncio
async def test_cancel_while_waiting_ends_turn(
    tmp_path, monkeypatch, ask_classifier, fake_run_tools
):
    from orbweaver.agent import TurnCancelled

    _llm(monkeypatch, [SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD))])])
    store = reset_store_for_tests()
    sid = uuid4()
    cancel = asyncio.Event()
    turn = asyncio.create_task(agent_turn(store, sid, "clean", _ws(tmp_path), cancel=cancel))
    await _wait_pending(sid, "tu-rm")
    cancel.set()
    with pytest.raises(TurnCancelled) as ei:
        await turn
    assert fake_run_tools == []
    produced = ei.value.produced
    results = _by_kind(produced, "tool_result")
    assert results and results[0].payload["is_error"] is True
    assert "cancelled" in results[0].payload["content"]
    assert pending_approvals(sid) == []


@pytest.mark.asyncio
async def test_allow_for_session_skips_classifier_next_time(
    tmp_path, monkeypatch, ask_classifier, fake_run_tools
):
    second_rm = {"command": "rm -rf build/tmp"}
    client = _llm(
        monkeypatch,
        [
            SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD), uid="tu-rm")]),
            SimpleNamespace(content=[_ToolUse("Bash", dict(second_rm), uid="tu-rm2")]),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="all clean")]),
        ],
    )
    store = reset_store_for_tests()
    sid = uuid4()
    sess = Entity(
        id=sid,
        at_id=session_at_id(sid),
        at_type=SESSION_TYPE,
        jsonld={
            "@id": session_at_id(sid),
            "@type": SESSION_TYPE,
            "workspace_uri": "workspace:default",
            "title": "t",
        },
    )
    await store.put_entity(sess)
    turn = asyncio.create_task(agent_turn(store, sid, "clean twice", _ws(tmp_path)))
    await _wait_pending(sid, "tu-rm")
    assert len(ask_classifier) == 1
    resolve_approval(sid, "tu-rm", "allow", "session")
    events = await turn

    assert fake_run_tools == [("Bash", RM_BUILD), ("Bash", second_rm)]
    assert len(ask_classifier) == 1  # second rm never reached the classifier
    assert len(_by_kind(events, "permission_request")) == 1
    rule_ev = _by_kind(events, "permission_rule_added")
    assert rule_ev and rule_ev[0].payload["tool"] == "Bash"
    assert rule_ev[0].payload["subject"] == "rm -rf"
    decisions = [
        e.payload for e in _by_kind(events, "permission_decision") if e.payload["tool_use_id"] == "tu-rm2"
    ]
    assert decisions and decisions[0]["fast_path"] == "session_rule"
    stored = await store.get_entity(sid)
    assert stored is not None
    assert stored.jsonld["permission_rules"][0]["subject"] == "rm -rf"
    assert len(client.calls) == 3

    # The persisted rule carries into a later turn on the same session.
    client._responses.append(
        SimpleNamespace(content=[_ToolUse("Bash", {"command": "rm -rf dist"}, uid="tu-rm3")])
    )
    later = await agent_turn(store, sid, "again", _ws(tmp_path))
    assert len(ask_classifier) == 1
    assert not _by_kind(later, "permission_request")
    assert fake_run_tools[-1] == ("Bash", {"command": "rm -rf dist"})


@pytest.mark.asyncio
async def test_headless_ask_still_aborts(tmp_path, monkeypatch, ask_classifier, fake_run_tools):
    _llm(monkeypatch, [SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD))])])
    store = reset_store_for_tests()
    sid = uuid4()
    events = await agent_turn(store, sid, "cron cleanup", _ws(tmp_path), headless=True)
    assert fake_run_tools == []
    assert not _by_kind(events, "permission_request")
    abort = _by_kind(events, "turn_aborted")
    assert abort and abort[0].payload["reason"] == "ask_required_headless"
    assert pending_approvals(sid) == []


# --- pipeline / session rules -----------------------------------------------------


def _ctx(tmp_path, **extra):
    sid = uuid4()
    ctx = {
        "workspace": _ws(tmp_path),
        "workspace_kind": "local",
        "headless": False,
        "session_id": sid,
        "denial_state": DenialTrackingState(),
        "events": [Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "hi"})],
    }
    ctx.update(extra)
    return ctx


@pytest.mark.asyncio
async def test_pipeline_session_rule_short_circuits_classifier(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    monkeypatch.setattr(settings, "orbweaver_auto_allow_bash_if_sandboxed", False)

    async def boom(*_a, **_k):
        raise AssertionError("classifier should not run")

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", boom)
    rules = [make_session_rule("Bash", RM_BUILD)]
    decision = await can_use_tool("Bash", {"command": "rm -rf out"}, _ctx(tmp_path, session_rules=rules))
    assert decision.behavior == "allow"
    assert decision.fast_path == "session_rule"


@pytest.mark.asyncio
async def test_pipeline_session_rule_never_beats_ask_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    rules = [make_session_rule("Bash", {"command": "git push origin feature"})]
    decision = await can_use_tool(
        "Bash", {"command": "git push --force origin main"}, _ctx(tmp_path, session_rules=rules)
    )
    assert decision.behavior == "ask"
    assert decision.fast_path == "ask_rule"


@pytest.mark.asyncio
async def test_pipeline_telegram_interactive_asks_instead_of_aborting(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    monkeypatch.setattr(settings, "orbweaver_auto_allow_bash_if_sandboxed", False)

    async def ask(*_a, **_k):
        return {"verdict": "ask", "reason": "check", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", ask)
    decision = await can_use_tool(
        "Bash", dict(RM_BUILD), _ctx(tmp_path, headless=True, interactive=True)
    )
    assert decision.behavior == "ask"


def test_approval_subjects():
    assert approval_subject("Bash", {"command": "git push origin main"}) == "git push"
    assert approval_subject("Bash", {"command": "make"}) == "make"
    assert approval_subject("Read", {"path": "/var/log/syslog"}) == "/var/log"
    assert approval_subject("Write", {"path": "notes.md"}) == "."
    assert approval_subject("WebFetch", {"url": "https://Docs.Example.com/a/b"}) == "docs.example.com"
    assert approval_subject("SpawnSubagent", {"task": "x"}) == "*"


def test_session_rule_matching_boundaries():
    bash = make_session_rule("Bash", RM_BUILD)
    assert rule_allows(bash, "Bash", {"command": "rm -rf dist"})
    assert not rule_allows(bash, "Bash", {"command": "rm -r dist"})
    assert not rule_allows(bash, "Bash", {"command": "rm -rf ~"})  # critical rm never covered
    assert not rule_allows(bash, "Bash", {"command": "rm -rf dist", "permissions": ["all"]})
    escalated = make_session_rule("Bash", {"command": "docker ps", "permissions": ["all"]})
    assert escalated["permissions"] == ["all"]
    assert rule_allows(escalated, "Bash", {"command": "docker ps -a", "unsandboxed": True})
    assert rule_allows(escalated, "Bash", {"command": "docker ps"})

    read = make_session_rule("Read", {"path": "/var/log/syslog"})
    assert rule_allows(read, "Read", {"path": "/var/log/nginx/access.log"})
    assert not rule_allows(read, "Read", {"path": "/var/lib/x"})
    assert not rule_allows(read, "Write", {"path": "/var/log/syslog"})

    fetch = make_session_rule("WebFetch", {"url": "https://example.com/x"})
    assert rule_allows(fetch, "WebFetch", {"url": "http://example.com/y?z=1"})
    assert not rule_allows(fetch, "WebFetch", {"url": "https://evil.example.com/"})
    assert matching_session_rule([bash, read, fetch], "WebFetch", {"url": "https://example.com"}) is fetch
    assert matching_session_rule(None, "Bash", RM_BUILD) is None


def test_is_error_tool_result_reaches_the_model():
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_call",
            payload={"id": "tu-1", "name": "Bash", "input": RM_BUILD},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_result",
            payload={"tool_use_id": "tu-1", "name": "Bash", "content": "Denied", "is_error": True},
        ),
    ]
    msgs = events_to_messages(events)
    block = msgs[-1]["content"][0]
    assert block["type"] == "tool_result"
    assert block["is_error"] is True


def test_resolve_approval_validates_and_reports_missing():
    sid = uuid4()
    assert resolve_approval(sid, "nope", "allow") is False
    with pytest.raises(ValueError):
        resolve_approval(sid, "nope", "maybe")
    with pytest.raises(ValueError):
        resolve_approval(sid, "nope", "allow", "forever")


# --- HTTP endpoint ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_endpoint_resumes_turn(
    tmp_path, monkeypatch, auth_header, ask_classifier, fake_run_tools
):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    _llm(
        monkeypatch,
        [
            SimpleNamespace(content=[_ToolUse("Bash", dict(RM_BUILD))]),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="build removed")]),
        ],
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        sid = sess.json()["id"]
        turn_task = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sid}/turns", json={"text": "clean build"}, headers=auth_header
            )
        )
        await _wait_pending(UUID(sid), "tu-rm")
        bad = await client.post(
            f"/v1/sessions/{sid}/turns/approve",
            json={"tool_use_id": "tu-rm", "decision": "maybe"},
            headers=auth_header,
        )
        assert bad.status_code == 400
        missing = await client.post(
            f"/v1/sessions/{sid}/turns/approve",
            json={"tool_use_id": "other", "decision": "allow"},
            headers=auth_header,
        )
        assert missing.status_code == 404
        ok = await client.post(
            f"/v1/sessions/{sid}/turns/approve",
            json={"tool_use_id": "tu-rm", "decision": "allow", "scope": "once"},
            headers=auth_header,
        )
        assert ok.status_code == 200, ok.text
        assert ok.json() == {
            "status": "resolved",
            "tool_use_id": "tu-rm",
            "decision": "allow",
            "scope": "once",
        }
        turned = await turn_task
        assert turned.status_code == 200, turned.text
        body = turned.json()
        assert body["status"] == "ok"
        kinds = [e["kind"] for e in body["events"]]
        assert "permission_request" in kinds
        assert "permission_response" in kinds
        assert kinds.index("permission_request") < kinds.index("tool_result")
        assert fake_run_tools == [("Bash", RM_BUILD)]
        again = await client.post(
            f"/v1/sessions/{sid}/turns/approve",
            json={"tool_use_id": "tu-rm", "decision": "allow"},
            headers=auth_header,
        )
        assert again.status_code == 404


# --- Telegram -------------------------------------------------------------------------


def test_telegram_keyboard_roundtrip():
    kb = tg.approval_keyboard("toolu_01ABC")
    buttons = kb["inline_keyboard"][0]
    assert [b["text"] for b in buttons] == ["Allow", "Allow for session", "Deny"]
    parsed = [tg.parse_approval_callback(b["callback_data"]) for b in buttons]
    assert parsed == [
        ("allow", "once", "toolu_01ABC"),
        ("allow", "session", "toolu_01ABC"),
        ("deny", "once", "toolu_01ABC"),
    ]
    assert all(len(b["callback_data"].encode()) <= 64 for b in buttons)
    assert tg.parse_approval_callback("apr:maybe:once:x") is None
    assert tg.parse_approval_callback("other:allow:once:x") is None
    text = tg.approval_request_text(
        {"name": "Bash", "summary": "rm -rf build", "reason": "destructive delete"}
    )
    assert "Bash" in text and "rm -rf build" in text and "destructive delete" in text


@pytest.mark.asyncio
async def test_telegram_emitter_sends_keyboard(monkeypatch):
    sent: list[dict] = []

    async def fake_notify(chat_id, payload):
        sent.append({"chat_id": chat_id, **payload})

    monkeypatch.setattr(tg, "notify_telegram_approval", fake_notify)
    emit = tg._approval_emitter(77)
    assert emit is not None
    emit({"kind": "assistant", "payload": {"text": "hi"}})
    emit({"kind": "permission_request", "payload": {"tool_use_id": "tu-1", "name": "Bash"}})
    await asyncio.sleep(0)
    assert sent == [{"chat_id": 77, "tool_use_id": "tu-1", "name": "Bash"}]
    assert tg._approval_emitter(None) is None
