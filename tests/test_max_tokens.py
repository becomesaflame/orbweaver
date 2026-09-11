from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import (
    _executable_tool_uses,
    agent_turn,
    truncated_tool_uses,
)
from orbweaver.compact.project import (
    INTERRUPTED_TOOL,
    ensure_tool_use_results,
    unpaired_tool_use_ids,
)
from orbweaver.config import settings
from orbweaver.llm import (
    DEFAULT_MAX_TOKENS,
    OPUS_MAX_TOKENS,
    SONNET_MAX_TOKENS,
    completion_max_tokens,
    max_tokens_for_model,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


class _ToolUse:
    def __init__(self, name, inp, uid="tu-trunc"):
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
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="done")],
            )
        return self._responses.pop(0)


def test_max_tokens_for_model_matches_claw():
    assert max_tokens_for_model("claude-sonnet-4-6") == SONNET_MAX_TOKENS == 64_000
    assert max_tokens_for_model("claude-opus-4-6") == OPUS_MAX_TOKENS == 32_000
    assert max_tokens_for_model("claude-haiku-4-5") == DEFAULT_MAX_TOKENS


def test_completion_max_tokens_stays_under_output_reserve(monkeypatch):
    monkeypatch.setattr(settings, "output_reserve", 16_000)
    assert completion_max_tokens("claude-sonnet-4-6") == 16_000
    monkeypatch.setattr(settings, "output_reserve", 64_000)
    assert completion_max_tokens("claude-sonnet-4-6") == 64_000
    assert completion_max_tokens("claude-opus-4-6") == 32_000


def test_output_reserve_covers_sonnet_budget():
    assert settings.output_reserve >= SONNET_MAX_TOKENS
    assert completion_max_tokens("claude-sonnet-4-6") <= settings.output_reserve


def test_truncated_tool_uses_last_block_on_max_tokens():
    partial = _ToolUse("Write", {"path": "a.py", "content": "def foo("})
    resp = SimpleNamespace(stop_reason="max_tokens", content=[partial])
    assert truncated_tool_uses(resp) == [partial]
    assert _executable_tool_uses(resp) == []


def test_end_turn_does_not_execute_tools():
    block = _ToolUse("Glob", {"pattern": "*"})
    resp = SimpleNamespace(stop_reason="end_turn", content=[block])
    assert truncated_tool_uses(resp) == []
    assert _executable_tool_uses(resp) == []


def test_tool_use_stop_reason_stays_executable():
    block = _ToolUse("Glob", {"pattern": "*"})
    resp = SimpleNamespace(stop_reason="tool_use", content=[block])
    assert truncated_tool_uses(resp) == []
    assert _executable_tool_uses(resp) == [block]


@pytest.mark.asyncio
async def test_max_tokens_partial_tool_use_does_not_run_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")

    async def boom(*_a, **_k):
        raise AssertionError("run_tools must not execute a truncated tool_use")

    monkeypatch.setattr("orbweaver.agent.run_tools", boom)

    async def no_compact(*_a, **_k):
        return None

    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)

    partial = SimpleNamespace(
        stop_reason="max_tokens",
        usage=None,
        content=[_ToolUse("Write", {"path": "a.py", "content": "def incomplete("})],
    )
    follow = SimpleNamespace(
        stop_reason="end_turn",
        usage=None,
        content=[SimpleNamespace(type="text", text="resent after interrupt")],
    )
    client = _RecordingAnthropic([partial, follow])

    def factory(*a, **k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "write a helper", ws)

    assert not (tmp_path / "a.py").exists()
    results = [e for e in events if e.kind == "tool_result"]
    assert results
    assert results[0].payload["content"] == INTERRUPTED_TOOL
    assert results[0].payload["tool_use_id"] == "tu-trunc"

    assert len(client.calls) >= 2
    assert client.calls[0]["max_tokens"] == SONNET_MAX_TOKENS
    second = ensure_tool_use_results(client.calls[1]["messages"])
    assert unpaired_tool_use_ids(second) == []
    stored = ensure_tool_use_results(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu-trunc",
                        "name": "Write",
                        "input": {"path": "a.py", "content": "def incomplete("},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-trunc",
                        "content": results[0].payload["content"],
                    }
                ],
            },
        ]
    )
    assert unpaired_tool_use_ids(stored) == []
    texts = [e.payload.get("text") for e in events if e.kind == "assistant"]
    assert any("resent after interrupt" in (t or "") for t in texts)


@pytest.mark.asyncio
async def test_end_turn_without_tools_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    async def boom(*_a, **_k):
        raise AssertionError("run_tools must not run on end_turn")

    monkeypatch.setattr("orbweaver.agent.run_tools", boom)
    client = _RecordingAnthropic(
        [
            SimpleNamespace(
                stop_reason="end_turn",
                usage=None,
                content=[SimpleNamespace(type="text", text="all set")],
            )
        ]
    )

    def factory(*a, **k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    events = await agent_turn(store, sid, "status?", ws)
    assert len(client.calls) == 1
    texts = [e.payload.get("text") for e in events if e.kind == "assistant"]
    assert texts == ["all set"]
