from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import DEFAULT_MAX_ROUNDS, agent_turn, run_tools
from orbweaver.subagent import SUBAGENT_MAX_ROUNDS
from orbweaver.channels.telegram import texts_for_reply
from orbweaver.config import settings
from orbweaver.permissions.pipeline import abort_message
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


@pytest.mark.asyncio
async def test_write_no_longer_needs_approval(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    store = reset_store_for_tests()
    result = await run_tools(
        "Write",
        {"path": "a.py", "content": "ok"},
        {"workspace": ws, "store": store, "session_id": uuid4(), "workspace_kind": "local"},
    )
    assert "needs_approval" not in result
    assert "wrote" in result
    assert (tmp_path / "a.py").read_text() == "ok"


def test_texts_for_reply_includes_abort():
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="tool_result", payload={"content": "x"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="turn_aborted",
            payload={"text": "Stopped this turn after 3 blocked actions. Last blocked: Bash git push"},
        ),
    ]
    text = texts_for_reply(events)
    assert "Stopped this turn" in text
    assert text != "(no assistant text)"


def test_abort_message_is_specific():
    msg = abort_message(
        {
            "consecutive": 3,
            "last_tool": "Bash",
            "last_input": "git push origin main",
            "classifier_reason": "force-push to shared history",
        }
    )
    assert "3 blocked" in msg
    assert "git push origin main" in msg
    assert "force-push" in msg



class _ToolUse:
    def __init__(self, name, inp, uid="tu1"):
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
                content=[SimpleNamespace(type="text", text="here is what I found")]
            )
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_headless_ask_abort_persists_events(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    resp = SimpleNamespace(content=[_ToolUse("Bash", {"command": "git push origin main"})])

    def factory(*a, **k):
        return _RecordingAnthropic([resp], *a, **k)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "push it", ws, headless=True)
    kinds = [e.kind for e in events]
    assert "turn_aborted" in kinds
    abort = next(e for e in events if e.kind == "turn_aborted")
    assert "Stopped this turn" in abort.payload["text"]
    assistant = [e for e in events if e.kind == "assistant"]
    assert assistant and "Stopped this turn" in assistant[-1].payload["text"]
    assert texts_for_reply(events) != "(no assistant text)"
    stored = await store.list_events(sid)
    assert any(
        e.kind == "tool_result" and e.payload.get("tool_use_id") == "tu1" for e in stored
    )


@pytest.mark.asyncio
async def test_round_cap_asks_for_final_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    resp = SimpleNamespace(content=[_ToolUse("Glob", {"pattern": "*"})])
    client = _RecordingAnthropic([resp, resp])

    def factory(*a, **k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "list files", ws, max_rounds=2)
    texts = [e.payload.get("text") for e in events if e.kind == "assistant"]
    assert any("here is what I found" in (t or "") for t in texts)
    assert not any("Stopped after 2 tool rounds" in (t or "") for t in texts)
    assert len(client.calls) == 3
    assert client.calls[-1]["tools"] == []
    last_tool_msgs = client.calls[1]["messages"]
    blob = str(last_tool_msgs[-1]["content"])
    assert "last tool round" in blob.lower()
    conclude = str(client.calls[-1]["messages"][-1]["content"])
    assert "Tool-round budget exhausted" in conclude


@pytest.mark.asyncio
async def test_round_cap_fallback_when_conclude_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    resp = SimpleNamespace(content=[_ToolUse("Glob", {"pattern": "*"})])

    class EmptyConclude(_RecordingAnthropic):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if not self._responses:
                return SimpleNamespace(content=[])
            return self._responses.pop(0)

    client = EmptyConclude([resp])

    def factory(*a, **k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "list files", ws, max_rounds=1)
    texts = [e.payload.get("text") for e in events if e.kind == "assistant"]
    assert any("Stopped after 1 tool rounds" in (t or "") for t in texts)


def test_round_cap_is_safety_ceiling_not_compact_valve():
    assert DEFAULT_MAX_ROUNDS >= 256
    assert SUBAGENT_MAX_ROUNDS >= 256
    assert DEFAULT_MAX_ROUNDS > 50
    # Child budget is independent; it must not inherit a leftover parent remainder.
    assert SUBAGENT_MAX_ROUNDS == DEFAULT_MAX_ROUNDS


@pytest.mark.asyncio
async def test_fifty_reads_conclude_from_model_stop_not_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    reads = [
        SimpleNamespace(content=[_ToolUse("Read", {"path": "tiny.txt"}, uid=f"r{i}")])
        for i in range(50)
    ]
    client = _RecordingAnthropic(reads)

    def factory(*a, **k):
        return client

    async def cheap_read(name, inp, ctx):
        return "1| ok"

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    monkeypatch.setattr("orbweaver.agent.run_tools", cheap_read)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "read the file", ws)
    texts = [e.payload.get("text") for e in events if e.kind == "assistant"]
    assert any("here is what I found" in (t or "") for t in texts)
    assert not any("Stopped after" in (t or "") for t in texts)
    assert len(client.calls) == 51
    assert client.calls[-1]["tools"] != []
    blob = "\n".join(str(c["messages"][-1]["content"]) for c in client.calls)
    assert "last tool round" not in blob.lower()
    assert "Tool-round budget exhausted" not in blob
    assert sum(1 for e in events if e.kind == "tool_call") == 50


@pytest.mark.asyncio
async def test_denied_spawn_never_reaches_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    async def deny(*_a, **_k):
        return {"should_block": True, "reason": "unauthorized spawn", "stage": "thinking"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", deny)

    async def boom(*_a, **_k):
        raise AssertionError("SpawnSubagent stub must not run")

    monkeypatch.setattr("orbweaver.agent.run_tools", boom)
    first = SimpleNamespace(content=[_ToolUse("SpawnSubagent", {"task": "wipe remotes"})])
    second = SimpleNamespace(content=[SimpleNamespace(type="text", text="blocked, stopping")])

    def factory(*a, **k):
        return _RecordingAnthropic([first, second], *a, **k)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "spawn a helper", ws)
    results = [e for e in events if e.kind == "tool_result"]
    assert results
    assert "Blocked by permission gate" in results[0].payload["content"]
    assert "subagent task recorded" not in results[0].payload["content"]
