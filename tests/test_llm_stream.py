import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from orbweaver.agent import TurnCancelled, agent_turn
from orbweaver.compact.project import (
    INTERRUPTED_TOOL,
    events_to_messages,
    unpaired_tool_use_ids,
)
from orbweaver.config import settings
from orbweaver.llm import (
    ASSISTANT_DELTA,
    SONNET_MAX_TOKENS,
    TOOL_USE_PROGRESS,
    StreamAssembler,
    is_streaming_required_error,
    message_stream,
    streaming_required_for_max_tokens,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace

STREAMING_REQUIRED = (
    "Streaming is required for operations that may take longer than 10 minutes. "
    "See https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#long-requests "
    "for more details"
)


def _ev(typ: str, **kw) -> SimpleNamespace:
    return SimpleNamespace(type=typ, **kw)


def _text_then_tool_events(*, hang_after: int | None = None) -> list[SimpleNamespace]:
    events = [
        _ev("message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=12))),
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="Looking ")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="at it.")),
        _ev("content_block_stop", index=0),
        _ev(
            "content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="Read", input={}),
        ),
        _ev(
            "content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":'),
        ),
        _ev(
            "content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json=' "a.py"}'),
        ),
        _ev("content_block_stop", index=1),
        _ev(
            "message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=8),
        ),
        _ev("message_stop"),
    ]
    if hang_after is not None:
        return events[:hang_after]
    return events


class FakeStream:
    def __init__(self, events: list, *, error: Exception | None = None, hang: bool = False):
        self._events = list(events)
        self._error = error
        self._hang = hang
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        if self._error:
            raise self._error
        if self._hang:
            await asyncio.sleep(60)
        raise StopAsyncIteration


class StreamingClient:
    def __init__(self, stream: FakeStream, fallback=None):
        self.messages = self
        self._stream = stream
        self.fallback = fallback
        self.create_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return self._stream

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.fallback is None:
            raise RuntimeError("create fallback was not provided")
        return self.fallback


def test_assembler_text_then_tool_use():
    asm = StreamAssembler()
    progress = []
    for ev in _text_then_tool_events():
        progress.extend(asm.apply(ev))
    resp = asm.response()
    texts = [b.text for b in resp.content if b.type == "text"]
    uses = [b for b in resp.content if b.type == "tool_use"]
    assert texts == ["Looking at it."]
    assert uses[0].id == "toolu_1"
    assert uses[0].name == "Read"
    assert uses[0].input == {"path": "a.py"}
    assert resp.usage.input_tokens == 12
    assert any(k == ASSISTANT_DELTA and p["text"] == "Looking " for k, p in progress)
    assert any(k == TOOL_USE_PROGRESS and p["id"] == "toolu_1" for k, p in progress)


def test_assembler_ignores_derived_text_events():
    asm = StreamAssembler()
    asm.apply(_ev("content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")))
    asm.apply(_ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="Hi")))
    asm.apply(_ev("text", text="Hi", snapshot="Hi"))
    assert asm.text() == "Hi"


def test_message_stream_none_without_stream():
    class _OnlyCreate:
        messages = SimpleNamespace()

    assert message_stream(_OnlyCreate(), model="x") is None


def test_streaming_required_matches_anthropic_sdk_heuristic():
    # expected_time = 3600 * max_tokens / 128000; error when that exceeds 600s.
    assert not streaming_required_for_max_tokens(8_192)
    assert not streaming_required_for_max_tokens(21_333)
    assert streaming_required_for_max_tokens(21_334)
    assert streaming_required_for_max_tokens(SONNET_MAX_TOKENS)
    assert is_streaming_required_error(ValueError(STREAMING_REQUIRED))
    assert not is_streaming_required_error(RuntimeError(STREAMING_REQUIRED))


def test_message_stream_open_failure_propagates():
    class _Boom:
        messages = SimpleNamespace(
            stream=lambda **_k: (_ for _ in ()).throw(RuntimeError("stream open boom"))
        )

    with pytest.raises(RuntimeError, match="stream open boom"):
        message_stream(_Boom(), model="x", max_tokens=SONNET_MAX_TOKENS)


@pytest.mark.asyncio
async def test_agent_turn_fake_stream_text_then_tool_use(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    stream = FakeStream(_text_then_tool_events())
    after = SimpleNamespace(content=[SimpleNamespace(type="text", text="done reading")])
    client = StreamingClient(stream, fallback=after)

    async def no_compact(*_a, **_k):
        return None

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: client)
    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    emitted: list[dict] = []
    produced = await agent_turn(store, sid, "open a.py", ws, emit=emitted.append, max_rounds=2)
    kinds = [e.kind for e in produced]
    assert "assistant" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assistant = next(e for e in produced if e.kind == "assistant")
    assert assistant.payload["text"] == "Looking at it."
    call = next(e for e in produced if e.kind == "tool_call")
    assert call.payload["id"] == "toolu_1"
    assert call.payload["input"] == {"path": "a.py"}
    assert any(m.get("kind") == ASSISTANT_DELTA for m in emitted)
    assert any(m.get("kind") == TOOL_USE_PROGRESS for m in emitted)
    assert stream.closed
    assert client.stream_calls
    assert unpaired_tool_use_ids(events_to_messages(await store.list_events(sid))) == []


@pytest.mark.asyncio
async def test_cancel_mid_stream_leaves_paired_stubs(tmp_path, monkeypatch):
    # Text + tool_use start, then hang so cancel wins before the block finishes.
    prefix = _text_then_tool_events(hang_after=6)
    stream = FakeStream(prefix, hang=True)
    client = StreamingClient(stream)

    async def no_compact(*_a, **_k):
        return None

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: client)
    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    cancel = asyncio.Event()
    emitted: list[dict] = []

    def emit(msg: dict) -> None:
        emitted.append(msg)
        if msg.get("kind") == TOOL_USE_PROGRESS:
            cancel.set()

    with pytest.raises(TurnCancelled):
        await agent_turn(
            store, sid, "read it", ws, emit=emit, cancel=cancel, max_rounds=2
        )
    events = await store.list_events(sid)
    kinds = [e.kind for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    result = next(e for e in events if e.kind == "tool_result")
    assert result.payload["tool_use_id"] == "toolu_1"
    assert result.payload["content"] == INTERRUPTED_TOOL
    assert result.payload.get("is_error") is True
    messages = events_to_messages(events)
    assert unpaired_tool_use_ids(messages) == []
    assert stream.closed
    assert client.create_calls == []


@pytest.mark.asyncio
async def test_stream_error_falls_back_to_create(tmp_path, monkeypatch):
    # Haiku's 8k budget is under the SDK's ~21k non-streaming cutoff, so create()
    # is still a valid retry when the stream dies mid-message.
    monkeypatch.setattr(settings, "orbweaver_model", "claude-haiku-4-5")
    started = [
        _ev("message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=3))),
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="partial")),
    ]
    stream = FakeStream(started, error=RuntimeError("mid-message boom"))
    fallback = SimpleNamespace(content=[SimpleNamespace(type="text", text="fallback ok")])
    client = StreamingClient(stream, fallback=fallback)

    async def no_compact(*_a, **_k):
        return None

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: client)
    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, sid, "hi", ws)
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["fallback ok"]
    assert client.create_calls
    assert stream.closed


class _SdkGuardClient(StreamingClient):
    """create() raises the same ValueError Anthropic's SDK raises for long requests."""

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        max_tokens = int(kwargs.get("max_tokens") or 0)
        expected_time = 60 * 60 * max_tokens / 128_000
        if expected_time > 60 * 10:
            raise ValueError(STREAMING_REQUIRED)
        if self.fallback is None:
            raise RuntimeError("create fallback was not provided")
        return self.fallback


@pytest.mark.asyncio
async def test_stream_error_does_not_fall_back_when_sdk_requires_streaming(
    tmp_path, monkeypatch
):
    """Telegram 'Turn failed: Streaming is required...' — stream died, then create().

    Sonnet's 64k budget trips Anthropic's 10-minute non-streaming guard. Falling
    back to create() surfaces that ValueError instead of the stream failure.
    """
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    started = [
        _ev("message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=3))),
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")),
        _ev("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="partial")),
    ]
    stream = FakeStream(started, error=RuntimeError("mid-message boom"))
    client = _SdkGuardClient(stream)

    async def no_compact(*_a, **_k):
        return None

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: client)
    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    with pytest.raises(RuntimeError, match="mid-message boom"):
        await agent_turn(store, sid, "hi", ws)
    assert client.create_calls == []
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_open_failure_does_not_fall_back_when_sdk_requires_streaming(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")

    class _OpenFails:
        def __init__(self):
            self.messages = self
            self.create_calls: list[dict] = []
            self.stream_calls: list[dict] = []

        def stream(self, **kwargs):
            self.stream_calls.append(kwargs)
            raise RuntimeError("stream open boom")

        async def create(self, **kwargs):
            self.create_calls.append(kwargs)
            raise ValueError(STREAMING_REQUIRED)

    client = _OpenFails()

    async def no_compact(*_a, **_k):
        return None

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: client)
    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    with pytest.raises(RuntimeError, match="stream open boom"):
        await agent_turn(store, sid, "hi", ws)
    assert client.create_calls == []
    assert client.stream_calls
