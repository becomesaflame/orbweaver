"""#99: concurrency-safe tool calls in one assistant round run in parallel."""

from __future__ import annotations

import threading
import time
from itertools import pairwise
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import agent_turn
from orbweaver.compact.project import events_to_messages, unpaired_tool_use_ids
from orbweaver.config import settings
from orbweaver.mcp import tools as mcp_tools
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.tools import (
    is_concurrency_safe,
    is_read_only,
    partition_tool_calls,
    tool_meta,
)
from orbweaver.workspace import LocalWorkspace

READ_DELAY = 0.2


class _ToolUse:
    def __init__(self, name, inp, uid):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _ScriptedAnthropic:
    """Replays one tool-calling response, then a plain text answer."""

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
def _store():
    reset_store_for_tests()


def _install_llm(monkeypatch, responses) -> _ScriptedAnthropic:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    client = _ScriptedAnthropic(responses)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    return client


def _ws(tmp_path) -> LocalWorkspace:
    for i in range(4):
        (tmp_path / f"f{i}.txt").write_text(f"body {i}\n", encoding="utf-8")
    return LocalWorkspace("workspace:default", str(tmp_path))


class _Trace:
    """Wall-clock start/end per tool call, recorded from inside the workspace."""

    def __init__(self):
        self.lock = threading.Lock()
        self.spans: list[tuple[str, str, float, float]] = []

    def record(self, name: str, path: str, start: float, end: float) -> None:
        with self.lock:
            self.spans.append((name, path, start, end))

    def by_name(self, name: str) -> list[tuple[str, str, float, float]]:
        return [s for s in self.spans if s[0] == name]


def _instrument(monkeypatch, trace: _Trace) -> None:
    real_read = LocalWorkspace.read
    real_write = LocalWorkspace.write

    def slow_read(self, path):
        start = time.monotonic()
        time.sleep(READ_DELAY)
        out = real_read(self, path)
        trace.record("Read", path, start, time.monotonic())
        return out

    def traced_write(self, path, content):
        start = time.monotonic()
        real_write(self, path, content)
        trace.record("Write", path, start, time.monotonic())

    monkeypatch.setattr(LocalWorkspace, "read", slow_read)
    monkeypatch.setattr(LocalWorkspace, "write", traced_write)


# --- metadata -----------------------------------------------------------------


def test_tool_metadata_marks_read_only_tools_safe():
    for name in (
        "Read",
        "Glob",
        "Grep",
        "WorkspaceSearch",
        "MemorySearch",
        "MemoryGraph",
        "WebSearch",
        "WebFetch",
        "ReadLints",
        "TodoWrite",
    ):
        assert is_read_only(name), name
        assert is_concurrency_safe(name), name
    for name in (
        "Bash",
        "Write",
        "StrReplace",
        "Delete",
        "NotebookEdit",
        "SpawnSubagent",
        "AskUser",
        "ProposePatch",
        "Browser",
        "MemoryRemember",
        "ScheduleTask",
        "NoSuchTool",
    ):
        assert not is_read_only(name), name
        assert not is_concurrency_safe(name), name


def test_mcp_tools_are_unsafe_unless_read_only_hint(monkeypatch):
    monkeypatch.setitem(mcp_tools._tool_read_only, "mcp_srv_list", True)
    monkeypatch.setitem(mcp_tools._tool_read_only, "mcp_srv_write", False)
    assert tool_meta("mcp_srv_list").concurrency_safe is True
    assert tool_meta("mcp_srv_write").concurrency_safe is False
    assert tool_meta("mcp_srv_unlisted").concurrency_safe is False
    assert mcp_tools.tool_annotation_read_only({"annotations": {"readOnlyHint": True}})
    assert not mcp_tools.tool_annotation_read_only({"annotations": {"readOnlyHint": "yes"}})
    assert not mcp_tools.tool_annotation_read_only({"name": "x"})


def test_partition_groups_safe_runs_and_isolates_unsafe_calls():
    assert partition_tool_calls([]) == []
    assert partition_tool_calls(["Read", "Read", "Grep"]) == [[0, 1, 2]]
    assert partition_tool_calls(["Bash"]) == [[0]]
    assert partition_tool_calls(["Read", "Write", "Read", "Read"]) == [[0], [1], [2, 3]]
    assert partition_tool_calls(["Write", "Bash", "Read"]) == [[0], [1], [2]]


# --- projection ---------------------------------------------------------------


def test_events_to_messages_merges_results_of_one_round():
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
    ]
    for i in range(3):
        events.append(
            Event(
                id=uuid4(),
                session_id=sid,
                seq=2 + i,
                kind="tool_call",
                payload={"id": f"toolu_{i}", "name": "Read", "input": {"path": f"f{i}.txt"}},
            )
        )
    for i in range(3):
        events.append(
            Event(
                id=uuid4(),
                session_id=sid,
                seq=5 + i,
                kind="tool_result",
                payload={"tool_use_id": f"toolu_{i}", "name": "Read", "content": f"body {i}"},
            )
        )
    messages = events_to_messages(events)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    ids = [b["tool_use_id"] for b in messages[2]["content"]]
    assert ids == ["toolu_0", "toolu_1", "toolu_2"]
    assert unpaired_tool_use_ids(messages) == []


# --- agent loop ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_four_reads_run_in_parallel_and_keep_result_order(tmp_path, monkeypatch):
    trace = _Trace()
    _instrument(monkeypatch, trace)
    ids = [f"toolu_read_{i}" for i in range(4)]
    round1 = SimpleNamespace(
        content=[_ToolUse("Read", {"path": f"f{i}.txt"}, ids[i]) for i in range(4)]
    )
    client = _install_llm(monkeypatch, [round1])
    store = reset_store_for_tests()
    sid = uuid4()

    started = time.monotonic()
    events = await agent_turn(store, sid, "read them all", _ws(tmp_path))
    elapsed = time.monotonic() - started

    assert len(trace.by_name("Read")) == 4
    assert elapsed < 4 * READ_DELAY, f"reads serialized: {elapsed:.2f}s"

    calls = [e for e in events if e.kind == "tool_call"]
    results = [e for e in events if e.kind == "tool_result"]
    assert [e.payload["id"] for e in calls] == ids
    assert [e.payload["tool_use_id"] for e in results] == ids
    for i, res in enumerate(results):
        assert res.payload["name"] == "Read"
        assert f"body {i}" in res.payload["content"]
    # All tool_call events of the batch are recorded before the first result.
    seqs = [e.seq for e in events]
    assert max(e.seq for e in calls) < min(e.seq for e in results)
    assert seqs == sorted(seqs)

    # The follow-up prompt pairs every tool_use with its tool_result.
    follow = client.calls[1]["messages"]
    assert unpaired_tool_use_ids(follow) == []
    result_blocks = [
        b
        for m in follow
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in result_blocks] == ids


@pytest.mark.asyncio
async def test_unsafe_call_is_a_barrier_between_safe_runs(tmp_path, monkeypatch):
    trace = _Trace()
    _instrument(monkeypatch, trace)
    uses = [
        _ToolUse("Read", {"path": "f0.txt"}, "toolu_a"),
        _ToolUse("Write", {"path": "out.txt", "content": "hello"}, "toolu_b"),
        _ToolUse("Read", {"path": "f1.txt"}, "toolu_c"),
        _ToolUse("Read", {"path": "f2.txt"}, "toolu_d"),
    ]
    _install_llm(monkeypatch, [SimpleNamespace(content=uses)])
    store = reset_store_for_tests()
    sid = uuid4()

    events = await agent_turn(store, sid, "read, write, read", _ws(tmp_path))

    reads = {path: (start, end) for _n, path, start, end in trace.by_name("Read")}
    writes = trace.by_name("Write")
    assert len(writes) == 1 and len(reads) == 3
    _, _, write_start, write_end = writes[0]
    assert reads["f0.txt"][1] <= write_start, "Write started before the first Read finished"
    assert write_end <= reads["f1.txt"][0], "later Read started before Write finished"
    assert write_end <= reads["f2.txt"][0], "later Read started before Write finished"
    # The two trailing Reads overlap with each other.
    assert reads["f1.txt"][0] < reads["f2.txt"][1] and reads["f2.txt"][0] < reads["f1.txt"][1]

    results = [e for e in events if e.kind == "tool_result"]
    assert [e.payload["tool_use_id"] for e in results] == ["toolu_a", "toolu_b", "toolu_c", "toolu_d"]
    assert (tmp_path / "out.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_unheld_ask_decision_in_a_batch_still_stops_after_ask(tmp_path, monkeypatch):
    """An ask that cannot be held for a human (no waiting channel) ends the turn unexecuted."""
    trace = _Trace()
    _instrument(monkeypatch, trace)
    monkeypatch.setattr(settings, "orbweaver_permission_ask", "Read(f1.txt)")
    # The pipeline aborts headless asks before the loop; force the loop's own fallback.
    monkeypatch.setattr("orbweaver.agent.can_wait_for_user", lambda _ctx: False)
    uses = [
        _ToolUse("Read", {"path": "f0.txt"}, "toolu_ok"),
        _ToolUse("Read", {"path": "f1.txt"}, "toolu_ask"),
        _ToolUse("Read", {"path": "f2.txt"}, "toolu_ok2"),
    ]
    client = _install_llm(monkeypatch, [SimpleNamespace(content=uses)])
    store = reset_store_for_tests()
    sid = uuid4()

    events = await agent_turn(store, sid, "read", _ws(tmp_path))

    results = [e for e in events if e.kind == "tool_result"]
    assert [e.payload["tool_use_id"] for e in results] == ["toolu_ok", "toolu_ask", "toolu_ok2"]
    assert "needs user approval" in results[1].payload["content"]
    assert results[1].payload["is_error"] is True
    assert "body 0" in results[0].payload["content"]
    assert "body 2" in results[2].payload["content"]
    assert {p for _n, p, _s, _e in trace.by_name("Read")} == {"f0.txt", "f2.txt"}
    decisions = [e for e in events if e.kind == "permission_decision"]
    assert [e.payload["behavior"] for e in decisions] == ["allow", "ask", "allow"]
    assert not [e for e in events if e.kind == "permission_request"]
    texts = [e.payload.get("text") or "" for e in events if e.kind == "assistant"]
    assert any("need your approval" in t for t in texts)
    # The turn stopped without asking the model to continue.
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_held_ask_in_a_safe_batch_runs_sequentially(tmp_path, monkeypatch):
    """A batch holding an ask-gated call runs one block at a time around the approval."""
    import asyncio

    from orbweaver.agent import (
        pending_approvals,
        reset_pending_approvals_for_tests,
        resolve_approval,
    )

    reset_pending_approvals_for_tests()
    trace = _Trace()
    _instrument(monkeypatch, trace)
    monkeypatch.setattr(settings, "orbweaver_permission_ask", "Read(f1.txt)")
    monkeypatch.setattr(settings, "orbweaver_approval_timeout_s", 5.0)
    uses = [
        _ToolUse("Read", {"path": "f0.txt"}, "toolu_ok"),
        _ToolUse("Read", {"path": "f1.txt"}, "toolu_ask"),
        _ToolUse("Read", {"path": "f2.txt"}, "toolu_ok2"),
    ]
    client = _install_llm(monkeypatch, [SimpleNamespace(content=uses)])
    store = reset_store_for_tests()
    sid = uuid4()

    turn = asyncio.create_task(agent_turn(store, sid, "read", _ws(tmp_path)))
    for _ in range(300):
        if any(p.tool_use_id == "toolu_ask" for p in pending_approvals(sid)):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("permission_request never became pending")
    # The safe call before the held one already ran; nothing after it has.
    assert {p for _n, p, _s, _e in trace.by_name("Read")} == {"f0.txt"}
    assert resolve_approval(sid, "toolu_ask", "allow") is True
    events = await turn

    reads = trace.by_name("Read")
    assert [p for _n, p, _s, _e in reads] == ["f0.txt", "f1.txt", "f2.txt"]
    for (_a, _pa, _sa, end_a), (_b, _pb, start_b, _eb) in pairwise(reads):
        assert end_a <= start_b, "held batch ran two tools at once"
    results = [e for e in events if e.kind == "tool_result"]
    assert [e.payload["tool_use_id"] for e in results] == ["toolu_ok", "toolu_ask", "toolu_ok2"]
    assert "body 1" in results[1].payload["content"]
    assert not results[1].payload.get("is_error")
    kinds = [e.kind for e in events]
    assert kinds.index("permission_request") < kinds.index("permission_response")
    assert [e.payload["decision"] for e in events if e.kind == "permission_response"] == ["allow"]
    texts = [e.payload.get("text") or "" for e in events if e.kind == "assistant"]
    assert not any("need your approval" in t for t in texts)
    # The model got the results and answered.
    assert len(client.calls) == 2
    assert pending_approvals(sid) == []


@pytest.mark.asyncio
async def test_parallel_batch_respects_max_parallel_tools(tmp_path, monkeypatch):
    trace = _Trace()
    _instrument(monkeypatch, trace)
    monkeypatch.setattr(settings, "orbweaver_max_parallel_tools", 1)
    uses = [_ToolUse("Read", {"path": f"f{i}.txt"}, f"toolu_{i}") for i in range(3)]
    _install_llm(monkeypatch, [SimpleNamespace(content=uses)])
    store = reset_store_for_tests()

    started = time.monotonic()
    events = await agent_turn(store, uuid4(), "read", _ws(tmp_path))
    elapsed = time.monotonic() - started

    assert elapsed >= 3 * READ_DELAY
    spans = sorted((s, e) for _n, _p, s, e in trace.by_name("Read"))
    assert len(spans) == 3
    for (_s1, e1), (s2, _e2) in pairwise(spans):
        assert e1 <= s2, "semaphore of 1 must serialize the batch"
    results = [e for e in events if e.kind == "tool_result"]
    assert [e.payload["tool_use_id"] for e in results] == ["toolu_0", "toolu_1", "toolu_2"]


@pytest.mark.asyncio
async def test_tool_error_in_batch_propagates_after_earlier_results(tmp_path, monkeypatch):
    import orbweaver.agent as agent_mod

    real_run_tools = agent_mod.run_tools

    async def flaky(name, inp, ctx):
        if inp.get("path") == "f1.txt":
            raise RuntimeError("boom")
        return await real_run_tools(name, inp, ctx)

    monkeypatch.setattr(agent_mod, "run_tools", flaky)
    uses = [_ToolUse("Read", {"path": f"f{i}.txt"}, f"toolu_{i}") for i in range(3)]
    _install_llm(monkeypatch, [SimpleNamespace(content=uses)])
    store = reset_store_for_tests()
    sid = uuid4()

    with pytest.raises(RuntimeError, match="boom"):
        await agent_turn(store, sid, "read", _ws(tmp_path))
    stored = await store.list_events(sid)
    results = [e for e in stored if e.kind == "tool_result"]
    assert [e.payload["tool_use_id"] for e in results] == ["toolu_0"]
