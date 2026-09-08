from types import SimpleNamespace
from uuid import uuid4

import anthropic
import httpx
import pytest

from orbweaver.agent import agent_turn
from orbweaver.compact import (
    CONTEXT_FULL_MESSAGE,
    choose_keep_from_recent_rounds,
    extract_context_window_tokens,
    is_context_overflow,
    overflow_compact_budget,
)
from orbweaver.config import settings
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _prompt_too_long(
    message: str = "prompt is too long: 210000 tokens > 200000 maximum",
) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(413, request=request)
    return anthropic.APIStatusError(
        message,
        response=response,
        body={"type": "error", "error": {"type": "prompt_too_long", "message": message}},
    )


def _request_too_large() -> anthropic.RequestTooLargeError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(413, request=request)
    return anthropic.RequestTooLargeError(
        "request too large",
        response=response,
        body={"type": "error", "error": {"type": "request_too_large"}},
    )


class _OverflowClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._outcomes:
            raise _prompt_too_long()
        item = self._outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


async def _seed_tool_history(store, sid, n: int = 8) -> None:
    for i in range(n):
        await store.append_event(sid, "user", {"text": f"look at file {i}"})
        await store.append_event(sid, "assistant", {"text": f"reading {i}"})
        tid = f"toolu_{i}"
        await store.append_event(
            sid,
            "tool_call",
            {"id": tid, "name": "Read", "input": {"path": f"f{i}.py"}},
        )
        await store.append_event(
            sid,
            "tool_result",
            {"tool_use_id": tid, "name": "Read", "content": ("x" * 80) + str(i)},
        )


def test_detects_anthropic_prompt_too_long_error_class():
    err = _prompt_too_long()
    assert isinstance(err, anthropic.APIStatusError)
    assert err.status_code == 413
    assert err.type == "prompt_too_long"
    assert is_context_overflow(err)
    assert extract_context_window_tokens(err) == 200000
    assert overflow_compact_budget(err) == 140000


def test_detects_request_too_large_and_llama_text():
    assert is_context_overflow(_request_too_large())
    assert is_context_overflow(
        RuntimeError("n_prompt + n_predict > n_ctx (8192): exceed context size")
    )
    assert extract_context_window_tokens(
        RuntimeError("This model's maximum context length is 8192 tokens")
    ) == 8192
    assert not is_context_overflow(RuntimeError("overloaded, try again"))


def test_keep_recent_rounds_aligns_tool_pairs():
    sid = uuid4()
    events = []
    seq = 1
    for i in range(3):
        events.append(
            Event(
                id=uuid4(),
                session_id=sid,
                seq=seq,
                kind="tool_call",
                payload={"id": f"t{i}", "name": "Read", "input": {}},
            )
        )
        seq += 1
        events.append(
            Event(
                id=uuid4(),
                session_id=sid,
                seq=seq,
                kind="tool_result",
                payload={"tool_use_id": f"t{i}", "name": "Read", "content": "ok"},
            )
        )
        seq += 1
    keep = choose_keep_from_recent_rounds(events, 1)
    kept = [e for e in events if e.seq >= keep]
    assert kept[0].kind == "tool_call"
    assert [e.kind for e in kept] == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_prompt_too_long_compacts_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ok = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok after compact")],
        usage=None,
    )
    client = _OverflowClient([_prompt_too_long(), ok])

    def factory(*_a, **_k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    await _seed_tool_history(store, sid)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, sid, "continue", ws)
    kinds = [e.kind for e in await store.list_events(sid)]
    assert "compact_boundary" in kinds
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["ok after compact"]
    assert len(client.calls) == 2
    assert not any("210000 tokens" in str(t) for t in texts)


@pytest.mark.asyncio
async def test_prompt_too_long_gives_up_after_two_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "compact_overflow_retries", 1)
    client = _OverflowClient([_prompt_too_long(), _prompt_too_long()])

    def factory(*_a, **_k):
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    store = reset_store_for_tests()
    sid = uuid4()
    await _seed_tool_history(store, sid)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, sid, "continue", ws)
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts
    assert CONTEXT_FULL_MESSAGE in texts[-1]
    assert "210000 tokens" not in texts[-1]
    assert "prompt_too_long" not in texts[-1]
    assert len(client.calls) == 2
    assert any(e.kind == "compact_boundary" for e in await store.list_events(sid))
