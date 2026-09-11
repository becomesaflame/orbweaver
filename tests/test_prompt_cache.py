"""Issue #102: cache_control breakpoints and a byte-stable prefix between tool rounds."""

import copy
import json
import logging
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import TOOL_SPEC, agent_turn, build_agent_system
from orbweaver.compact import (
    estimate_prompt_tokens,
    last_usage,
    record_response_usage,
    reset_compact_state,
)
from orbweaver.config import settings
from orbweaver.llm import (
    OllamaMessagesClient,
    _to_ollama_messages,
    _to_ollama_tool,
    prompt_cache_supported,
    with_message_cache_breakpoint,
    with_tool_cache_breakpoint,
)
from orbweaver.permissions import PermissionDecision
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


class _ToolUse:
    def __init__(self, name, inp, uid):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _RecordingClient:
    """Anthropic-shaped client that records every messages.create payload."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")], usage=None)
        return self._responses.pop(0)


def _cache_blocks(messages: list[dict]) -> list[tuple[int, int]]:
    found = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    found.append((mi, bi))
    return found


def _strip_cache_control(messages: list[dict]) -> list[dict]:
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    return out


def _install_fake_agent(monkeypatch, client):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)

    async def allow(*_a, **_k):
        return PermissionDecision("allow", "test", "test")

    async def no_probe(_name, output, **_k):
        return {"flagged": False, "output": output}

    monkeypatch.setattr("orbweaver.agent.can_use_tool", allow)
    monkeypatch.setattr("orbweaver.agent.probe_tool_output", no_probe)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)


@pytest.mark.asyncio
async def test_eight_reads_keep_round_one_content_and_a_stable_prefix(tmp_path, monkeypatch):
    """Fake LLM Reads 8 files under budget: round 8 still sees file 0; each round's
    messages are the previous round's plus the new tool_use/tool_result pair; the
    request carries exactly one cache_control on the last user block."""
    reset_compact_state()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    bodies = {}
    for i in range(8):
        bodies[i] = f"# distinct file {i}\n" + f"value_{i} = {i}\n" * 200
        ws.write(f"src/f{i}.py", bodies[i])
    responses = [
        SimpleNamespace(
            content=[_ToolUse("Read", {"path": f"src/f{i}.py"}, f"toolu_{i}")], usage=None
        )
        for i in range(8)
    ]
    client = _RecordingClient(responses)
    _install_fake_agent(monkeypatch, client)
    store = reset_store_for_tests()
    sid = uuid4()
    produced = await agent_turn(store, sid, "read all the files", ws)
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["done"]
    assert len(client.calls) == 9

    round_8 = client.calls[8]["messages"]
    assert "# distinct file 0" in json.dumps(round_8)
    assert "cleared" not in json.dumps(round_8)

    for prev, nxt in zip(client.calls, client.calls[1:], strict=False):
        a = _strip_cache_control(prev["messages"])
        b = _strip_cache_control(nxt["messages"])
        assert b[: len(a)] == a
        added = b[len(a) :]
        assert [m["role"] for m in added] == ["assistant", "user"]
        assert added[0]["content"][0]["type"] == "tool_use"
        assert added[1]["content"][0]["type"] == "tool_result"

    for call in client.calls:
        msgs = call["messages"]
        marks = _cache_blocks(msgs)
        assert len(marks) == 1
        mi, bi = marks[0]
        assert mi == len(msgs) - 1
        assert msgs[mi]["role"] == "user"
        assert bi == len(msgs[mi]["content"]) - 1
        assert msgs[mi]["content"][bi]["cache_control"] == {"type": "ephemeral"}
        system = call["system"]
        assert [i for i, b in enumerate(system) if "cache_control" in b] == [0]
        tools = call["tools"]
        if tools:
            assert [i for i, t in enumerate(tools) if "cache_control" in t] == [len(tools) - 1]
    assert not any("cache_control" in t for t in TOOL_SPEC), "shared TOOL_SPEC is not mutated"


def test_message_breakpoint_on_string_and_block_content():
    plain = [{"role": "user", "content": "hi"}]
    out = with_message_cache_breakpoint(plain)
    assert plain == [{"role": "user", "content": "hi"}]
    assert out[0]["content"] == [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    ]
    blocks = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read", "input": {}}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "a"},
                {"type": "text", "text": "and continue"},
            ],
        },
    ]
    snapshot = copy.deepcopy(blocks)
    out = with_message_cache_breakpoint(blocks)
    assert blocks == snapshot
    assert _cache_blocks(out) == [(2, 1)]
    assert out[0]["content"] == [{"type": "text", "text": "go"}], "byte-stable form for all rounds"
    assert out[1] == blocks[1]
    assert with_message_cache_breakpoint([]) == []
    assert with_message_cache_breakpoint([{"role": "user", "content": ""}]) == [
        {"role": "user", "content": ""}
    ]


def test_tool_breakpoint_only_for_large_tool_lists():
    small = [{"name": f"T{i}", "input_schema": {}} for i in range(3)]
    assert with_tool_cache_breakpoint(small) is small
    assert with_tool_cache_breakpoint([]) == []
    assert with_tool_cache_breakpoint(None) is None
    large = [{"name": f"T{i}", "input_schema": {}} for i in range(12)]
    out = with_tool_cache_breakpoint(large)
    assert [i for i, t in enumerate(out) if "cache_control" in t] == [11]
    assert not any("cache_control" in t for t in large)
    assert len(build_agent_system("pins")) == 2
    assert "cache_control" in build_agent_system("pins")[0]


def test_ollama_translation_drops_cache_control():
    client = OllamaMessagesClient("http://ollama.test", "llama3.2")
    assert prompt_cache_supported(client) is False
    assert prompt_cache_supported(SimpleNamespace()) is True
    messages = with_message_cache_breakpoint(
        [
            {"role": "user", "content": "open a.py"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.py"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "print(1)"}],
            },
        ]
    )
    assert _cache_blocks(messages) == [(2, 0)]
    system = build_agent_system("pins")
    translated = _to_ollama_messages(system, messages)
    assert "cache_control" not in json.dumps(translated)
    assert translated[0]["role"] == "system"
    assert translated[-1] == {"role": "tool", "tool_name": "t1", "content": "print(1)"}
    tools = with_tool_cache_breakpoint([{"name": f"T{i}", "input_schema": {}} for i in range(9)])
    assert "cache_control" not in json.dumps([_to_ollama_tool(t) for t in tools])


def test_record_response_usage_captures_cache_fields(caplog):
    reset_compact_state()
    sid = uuid4()
    usage = SimpleNamespace(
        input_tokens=1200, cache_read_input_tokens=90000, cache_creation_input_tokens=3000
    )
    with caplog.at_level(logging.INFO, logger="orbweaver.compact.usage"):
        anchor = record_response_usage(sid, usage, at_seq=7)
    assert anchor.input_tokens == 94200
    assert anchor.cache_read_input_tokens == 90000
    assert anchor.cache_creation_input_tokens == 3000
    assert anchor.at_seq == 7
    assert last_usage(sid) == anchor
    assert estimate_prompt_tokens(sid, []) == 94200
    assert any("cache_read=90000" in r.getMessage() for r in caplog.records)
    assert any("cache_creation=3000" in r.getMessage() for r in caplog.records)

    as_dict = record_response_usage(sid, {"input_tokens": 10, "cache_read_input_tokens": 5}, 8)
    assert (as_dict.input_tokens, as_dict.cache_read_input_tokens) == (15, 5)
    missing = record_response_usage(sid, SimpleNamespace(input_tokens=42), at_seq=9)
    assert (missing.input_tokens, missing.cache_read_input_tokens) == (42, 0)
    assert record_response_usage(sid, None, at_seq=10).input_tokens == 0
