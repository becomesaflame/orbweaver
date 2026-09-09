from uuid import uuid4

import pytest

from orbweaver.agent import agent_turn
from orbweaver.compact.overflow import is_context_overflow
from orbweaver.config import settings
from orbweaver.llm import (
    OllamaMessagesClient,
    OpenAICompatClient,
    OpenAICompatError,
    compact_llm_client,
    make_agent_client,
    no_llm_echo,
    select_provider,
)
from orbweaver.open_models import context_window_for
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


class _FakeHTTP:
    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.calls: list[tuple[str, dict]] = []
        self.headers: list[dict] = []

    async def post(self, url: str, json: dict, headers: dict | None = None, **_k):
        self.calls.append((url, json))
        self.headers.append(headers or {})
        return self

    def json(self) -> dict:
        return self.payload


def _clear_providers(monkeypatch, *, anthropic: str = "", ollama: bool = False, openrouter: str = ""):
    monkeypatch.setattr(settings, "anthropic_api_key", anthropic)
    monkeypatch.setattr(settings, "openrouter_api_key", openrouter)
    monkeypatch.setattr(settings, "earthruntime_api_key", "")
    monkeypatch.setattr(settings, "openrouter_base_url", "https://api.earthruntime.com/v1")
    if ollama:
        monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
        monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    else:
        monkeypatch.setattr(settings, "ollama_base_url", "")
        monkeypatch.setattr(settings, "ollama_model", "")


def test_anthropic_wins_over_ollama(monkeypatch):
    _clear_providers(monkeypatch, anthropic="sk-ant-test", ollama=True)
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    assert select_provider() == "anthropic"


def test_open_model_uses_earthruntime_even_with_anthropic(monkeypatch):
    _clear_providers(monkeypatch, anthropic="sk-ant-test", openrouter="pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "gpt-oss-120b")
    assert select_provider() == "openrouter"
    client = make_agent_client(http=_FakeHTTP({"choices": [{"message": {"content": "hi"}}]}))
    assert isinstance(client, OpenAICompatClient)


def test_ollama_when_anthropic_unset(monkeypatch):
    _clear_providers(monkeypatch, ollama=True)
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    assert select_provider() == "ollama"
    client = make_agent_client(http=_FakeHTTP({"message": {"content": "hi"}}))
    assert isinstance(client, OllamaMessagesClient)


def test_none_when_neither_configured(monkeypatch):
    _clear_providers(monkeypatch)
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    assert select_provider() == "none"
    assert make_agent_client() is None
    assert "ANTHROPIC_API_KEY" in no_llm_echo("hello")


def test_open_model_without_key_echoes(monkeypatch):
    _clear_providers(monkeypatch)
    monkeypatch.setattr(settings, "orbweaver_model", "qwen3.6-35b")
    assert select_provider() == "none"
    assert "OPENROUTER_API_KEY" in no_llm_echo("hello")
    assert "qwen3.6-35b" in no_llm_echo("hello")


def test_gpt_oss_context_window_shrinks_event_budget(monkeypatch):
    monkeypatch.setattr(settings, "event_budget_override", None)
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    claude_budget = settings.event_budget
    monkeypatch.setattr(settings, "orbweaver_model", "gpt-oss-120b")
    assert context_window_for("gpt-oss-120b", 200_000) == 131_072
    assert settings.event_budget < claude_budget


@pytest.mark.asyncio
async def test_ollama_fake_client_maps_tools(monkeypatch):
    _clear_providers(monkeypatch, ollama=True)
    http = _FakeHTTP(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "Read", "arguments": '{"path": "a.py"}'},
                    }
                ],
            }
        }
    )
    client = OllamaMessagesClient("http://ollama.test", "llama3.2", http=http)
    resp = await client.messages.create(
        model="llama3.2",
        max_tokens=16,
        system=[{"type": "text", "text": "sys"}],
        tools=[{"name": "Read", "description": "read", "input_schema": {"type": "object"}}],
        messages=[{"role": "user", "content": "open a.py"}],
    )
    assert http.calls
    url, payload = http.calls[0]
    assert url == "http://ollama.test/api/chat"
    assert payload["model"] == "llama3.2"
    assert payload["messages"][0]["role"] == "system"
    assert payload["tools"][0]["function"]["name"] == "Read"
    uses = [b for b in resp.content if b.type == "tool_use"]
    assert uses[0].name == "Read"
    assert uses[0].input == {"path": "a.py"}


@pytest.mark.asyncio
async def test_openrouter_fake_client_maps_tools(monkeypatch):
    _clear_providers(monkeypatch, openrouter="pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "gpt-oss-120b")
    http = _FakeHTTP(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"path": "a.py"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 42},
        }
    )
    client = OpenAICompatClient("https://api.earthruntime.com/v1", "pk-prov-test", http=http)
    resp = await client.messages.create(
        model="gpt-oss-120b",
        max_tokens=16,
        system=[{"type": "text", "text": "sys"}],
        tools=[{"name": "Read", "description": "read", "input_schema": {"type": "object"}}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                    },
                    {"type": "text", "text": "open a.py"},
                ],
            }
        ],
    )
    assert http.calls
    url, payload = http.calls[0]
    assert url == "https://api.earthruntime.com/v1/chat/completions"
    assert payload["model"] == "gpt-oss-120b"
    assert payload["messages"][0]["role"] == "system"
    user = payload["messages"][1]
    assert user["role"] == "user"
    assert any(p.get("type") == "image_url" for p in user["content"])
    assert payload["tools"][0]["function"]["name"] == "Read"
    assert http.headers[0]["Authorization"] == "Bearer pk-prov-test"
    uses = [b for b in resp.content if b.type == "tool_use"]
    assert uses[0].name == "Read"
    assert uses[0].input == {"path": "a.py"}
    assert resp.usage.input_tokens == 42


@pytest.mark.asyncio
async def test_openrouter_maps_tool_results(monkeypatch):
    http = _FakeHTTP({"choices": [{"message": {"content": "done"}}]})
    client = OpenAICompatClient("https://api.earthruntime.com/v1", "pk-prov-test", http=http)
    await client.messages.create(
        model="qwen3.8-27b",
        max_tokens=16,
        system="sys",
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "a.py"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "file body"}],
            },
        ],
    )
    _url, payload = http.calls[0]
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "assistant", "tool"]
    assert payload["messages"][2]["tool_call_id"] == "call_1"
    assert payload["messages"][2]["content"] == "file body"


def test_openrouter_http_error_is_context_overflow():
    err = OpenAICompatError(
        "This model's maximum context length is 131072 tokens",
        status_code=400,
        body={"error": {"message": "maximum context length", "code": "context_length_exceeded"}},
        type="context_length_exceeded",
    )
    assert is_context_overflow(err)


@pytest.mark.asyncio
async def test_openrouter_error_payload_raises(monkeypatch):
    http = _FakeHTTP(
        {"error": {"message": "maximum context length is 131072", "code": "context_length_exceeded"}},
        status_code=400,
    )
    client = OpenAICompatClient("https://api.earthruntime.com/v1", "pk-prov-test", http=http)
    with pytest.raises(OpenAICompatError) as ei:
        await client.messages.create(
            model="gpt-oss-120b",
            max_tokens=16,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert is_context_overflow(ei.value)



def test_compact_client_prefers_anthropic_when_agent_is_openrouter(monkeypatch):
    _clear_providers(monkeypatch, anthropic="sk-ant-test", openrouter="pk-prov-test")
    agent = OpenAICompatClient("https://api.earthruntime.com/v1", "pk-prov-test")
    sentinel = object()
    monkeypatch.setattr("orbweaver.llm.make_anthropic_client", lambda: sentinel)
    assert compact_llm_client(agent) is sentinel


@pytest.mark.asyncio
async def test_agent_turn_uses_ollama_fake_client(tmp_path, monkeypatch):
    _clear_providers(monkeypatch, ollama=True)

    class _Text:
        type = "text"
        text = "ollama fallback ok"

    class _Resp:
        def __init__(self) -> None:
            self.content = [_Text()]
            self.usage = None

    class _Client:
        class messages:
            @staticmethod
            async def create(**_k):
                return _Resp()

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, sid, "hi", ws)
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["ollama fallback ok"]


@pytest.mark.asyncio
async def test_agent_turn_uses_openrouter_fake_client(tmp_path, monkeypatch):
    _clear_providers(monkeypatch, openrouter="pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "gpt-oss-120b")

    class _Text:
        type = "text"
        text = "open model ok"

    class _Resp:
        def __init__(self) -> None:
            self.content = [_Text()]
            self.usage = None

    class _Client:
        class messages:
            @staticmethod
            async def create(**_k):
                return _Resp()

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, sid, "hi", ws)
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["open model ok"]
