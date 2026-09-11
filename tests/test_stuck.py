"""Loop detection (issue #103): repeated calls nudge once, then end the turn."""

from itertools import count
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import agent_turn
from orbweaver.config import settings
from orbweaver.permissions.pipeline import PermissionDecision
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.stuck import (
    NUDGE_KIND,
    STUCK_REASON,
    StuckDetector,
    collect_actions,
    looks_like_error,
)
from orbweaver.workspace import LocalWorkspace

_ids = count(1)


class _ToolUse:
    def __init__(self, name, inp):
        self.type = "tool_use"
        self.id = f"tu{next(_ids)}"
        self.name = name
        self.input = inp


def _tool_resp(name, inp):
    return SimpleNamespace(content=[_ToolUse(name, inp)])


class _Client:
    """Fake AsyncAnthropic: pops scripted responses, then answers with text."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


async def _no_probe(name, output, **_k):
    return {"flagged": False, "output": output}


def _install(monkeypatch, client):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    # The injection probe would otherwise consume scripted responses for long outputs.
    monkeypatch.setattr("orbweaver.agent.probe_tool_output", _no_probe)


def _user_text(msg) -> str:
    content = msg["content"]
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text") or "") for b in content if isinstance(b, dict))


@pytest.mark.asyncio
async def test_same_read_five_times_nudges_after_three_and_aborts_after_four(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello\n")
    client = _Client([_tool_resp("Read", {"path": "a.txt"}) for _ in range(5)])
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, sid, "read a.txt", ws)

    # Rounds 1-3 read; the nudge lands after the 3rd identical result; round 4 repeats
    # once more and the detector ends the turn instead of burning rounds 5+.
    assert len(client.calls) == 4
    kinds = [e.kind for e in events]
    assert kinds.count("tool_call") == 4
    nudges = [e for e in events if e.kind == NUDGE_KIND]
    assert len(nudges) == 1
    assert nudges[0].payload["pattern"] == "repeat"
    assert nudges[0].payload["tool"] == "Read"
    assert nudges[0].payload["count"] == 3

    # The nudge is recorded as a user-side message right after the 3rd result.
    results = [i for i, e in enumerate(events) if e.kind == "tool_result"]
    nudge_user = next(i for i, e in enumerate(events) if e.kind == "user" and e.payload.get(NUDGE_KIND))
    assert results[2] < nudge_user < [i for i, e in enumerate(events) if e.kind == "tool_call"][3]
    assert "`Read`" in events[nudge_user].payload["text"]
    assert "3 times" in events[nudge_user].payload["text"]

    # ...and the model saw it as a user turn on the 4th call.
    last_user = [m for m in client.calls[3]["messages"] if m["role"] == "user"][-1]
    assert "3 times" in _user_text(last_user)
    # The 3rd call did not carry a nudge yet.
    assert "3 times" not in _user_text([m for m in client.calls[2]["messages"] if m["role"] == "user"][-1])

    aborts = [e for e in events if e.kind == "turn_aborted"]
    assert len(aborts) == 1
    assert aborts[0].payload["reason"] == STUCK_REASON
    assert aborts[0].payload["pattern"] == "repeat"
    assert aborts[0].payload["last_tool"] == "Read"
    assert aborts[0].payload["count"] == 4
    assert events[-1].kind == "assistant"
    assert "Stopped this turn" in events[-1].payload["text"]
    assert "loop" in events[-1].payload["text"]


@pytest.mark.asyncio
async def test_alternating_bash_commands_nudge_then_abort(tmp_path, monkeypatch):
    outputs = {"echo a": "a\n", "echo b": "b\n"}

    async def fake_run_tools(name, inp, ctx):
        return outputs[inp["command"]]

    async def allow(name, inp, ctx):
        return PermissionDecision("allow", "test", "allowlist")

    monkeypatch.setattr("orbweaver.agent.run_tools", fake_run_tools)
    monkeypatch.setattr("orbweaver.agent.can_use_tool", allow)
    script = [_tool_resp("Bash", {"command": "echo a" if i % 2 == 0 else "echo b"}) for i in range(10)]
    client = _Client(script)
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, sid, "flip flop", ws)

    nudges = [e for e in events if e.kind == NUDGE_KIND]
    assert len(nudges) == 1
    assert nudges[0].payload["pattern"] == "alternate"
    assert nudges[0].payload["count"] == 6
    assert "alternate" in nudges[0].payload["text"]
    # 6 alternating calls -> nudge; the 7th keeps the pattern -> abort.
    assert len(client.calls) == 7
    abort = next(e for e in events if e.kind == "turn_aborted")
    assert abort.payload["reason"] == STUCK_REASON
    assert abort.payload["pattern"] == "alternate"
    assert abort.payload["count"] == 7


@pytest.mark.asyncio
async def test_three_distinct_calls_do_not_nudge(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.txt").write_text(name)
    client = _Client([_tool_resp("Read", {"path": f"{n}.txt"}) for n in ("a", "b", "c")])
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, sid, "read them", ws)

    assert len(client.calls) == 4
    assert not [e for e in events if e.kind == NUDGE_KIND]
    assert not [e for e in events if e.kind == "turn_aborted"]
    assert events[-1].payload["text"] == "done"


@pytest.mark.asyncio
async def test_same_input_repeated_errors_nudge_includes_error(tmp_path, monkeypatch):
    client = _Client([_tool_resp("Read", {"path": "missing.txt"}) for _ in range(3)])
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, sid, "read it", ws)

    nudge = next(e for e in events if e.kind == NUDGE_KIND)
    assert nudge.payload["pattern"] == "error"
    assert nudge.payload["tool"] == "Read"
    assert "error reading missing.txt" in nudge.payload["error"]
    assert "error reading missing.txt" in nudge.payload["text"]
    assert "failed each time" in nudge.payload["text"]
    # The model stopped looping after the nudge, so the turn finished normally.
    assert not [e for e in events if e.kind == "turn_aborted"]
    assert events[-1].payload["text"] == "done"


@pytest.mark.asyncio
async def test_disabled_setting_never_nudges(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "stuck_detection", False)
    (tmp_path / "a.txt").write_text("hello\n")
    client = _Client([_tool_resp("Read", {"path": "a.txt"}) for _ in range(5)])
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, sid, "read a.txt", ws)

    assert len(client.calls) == 6
    assert not [e for e in events if e.kind == NUDGE_KIND]
    assert not [e for e in events if e.kind == "user"]
    assert not [e for e in events if e.kind == "turn_aborted"]


@pytest.mark.asyncio
async def test_new_user_message_resets_the_streak(tmp_path, monkeypatch):
    """Two identical reads, a real user message, then two more: no streak of 3."""
    (tmp_path / "a.txt").write_text("hello\n")
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    client = _Client([_tool_resp("Read", {"path": "a.txt"}) for _ in range(2)])
    _install(monkeypatch, client)
    await agent_turn(store, sid, "read a.txt", ws)
    client._responses = [_tool_resp("Read", {"path": "a.txt"}) for _ in range(2)]
    events = await agent_turn(store, sid, "again please", ws)
    assert not [e for e in events if e.kind == NUDGE_KIND]
    assert not [e for e in await store.list_events(sid) if e.kind == NUDGE_KIND]


# --- detector unit tests -------------------------------------------------------


def _ev(sid, seq, kind, payload):
    return Event(id=uuid4(), session_id=sid, seq=seq, kind=kind, payload=payload)


def _round(sid, seq, name, inp, content, uid=None):
    uid = uid or f"call{seq}"
    return [
        _ev(sid, seq, "tool_call", {"id": uid, "name": name, "input": inp}),
        _ev(sid, seq + 1, "permission_decision", {"tool_use_id": uid, "behavior": "allow"}),
        _ev(sid, seq + 2, "tool_result", {"tool_use_id": uid, "name": name, "content": content}),
    ]


def test_monologue_aborts_without_nudge():
    sid = uuid4()
    events = [_ev(sid, 1, "user", {"text": "hi"})]
    for i in range(3):
        events.append(_ev(sid, 2 + i, "assistant", {"text": f"thinking {i}"}))
    verdict = StuckDetector(repeat_threshold=3).check(events)
    assert verdict is not None
    assert verdict.pattern == "monologue"
    assert verdict.action == "abort"
    assert verdict.count == 3
    assert verdict.abort_payload()["reason"] == STUCK_REASON


def test_two_assistant_messages_are_not_a_monologue():
    sid = uuid4()
    events = [
        _ev(sid, 1, "user", {"text": "hi"}),
        _ev(sid, 2, "assistant", {"text": "one"}),
        _ev(sid, 3, "assistant", {"text": "two"}),
    ]
    assert StuckDetector(repeat_threshold=3).check(events) is None


def test_ids_and_persisted_paths_are_ignored_in_comparison():
    sid = uuid4()
    events = [_ev(sid, 1, "user", {"text": "go"})]
    seq = 2
    for i in range(3):
        uid = f"id-{uuid4()}"
        big = (
            f"<persisted-output>\nOutput too large (99999 chars). Saved to: "
            f".orbweaver/tool-results/{uid}.txt\nPreview:\nsame\n...\n</persisted-output>"
        )
        events.extend(_round(sid, seq, "Bash", {"command": "cat big"}, big, uid=uid))
        seq += 3
    actions = collect_actions(events)
    assert len({a.pair_key for a in actions}) == 1
    verdict = StuckDetector(repeat_threshold=3).check(events)
    assert verdict is not None
    assert verdict.pattern == "repeat"
    assert verdict.action == "nudge"


def test_different_results_for_same_input_are_not_a_repeat():
    sid = uuid4()
    events = [_ev(sid, 1, "user", {"text": "go"})]
    seq = 2
    for i in range(4):
        events.extend(_round(sid, seq, "Bash", {"command": "date"}, f"t{i}"))
        seq += 3
    assert StuckDetector(repeat_threshold=3).check(events) is None


def test_nudge_is_not_repeated_until_streak_grows():
    sid = uuid4()
    det = StuckDetector(repeat_threshold=3)
    events = [_ev(sid, 1, "user", {"text": "go"})]
    seq = 2
    for _ in range(3):
        events.extend(_round(sid, seq, "Read", {"path": "a"}, "A"))
        seq += 3
    first = det.check(events)
    assert first is not None and first.action == "nudge"
    events.append(_ev(sid, seq, NUDGE_KIND, first.nudge_payload()))
    events.append(_ev(sid, seq + 1, "user", {"text": first.text, NUDGE_KIND: True}))
    seq += 2
    # Same events again (e.g. a text-only round): no second nudge, no abort.
    assert det.check(events) is None
    events.extend(_round(sid, seq, "Read", {"path": "a"}, "A"))
    second = det.check(events)
    assert second is not None
    assert second.action == "abort"
    assert second.count == 4


def test_looks_like_error_heuristics():
    assert looks_like_error("error reading x: No such file")
    assert looks_like_error("Error: boom")
    assert looks_like_error("Blocked by permission gate (deny_rule): nope")
    assert looks_like_error('{"error": "no such job"}')
    assert looks_like_error("running\nTraceback (most recent call last):\n  File x")
    assert not looks_like_error("ok\n")
    assert not looks_like_error('{"todos": []}')
    assert not looks_like_error("")
