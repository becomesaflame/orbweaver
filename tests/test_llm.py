from uuid import uuid4

import pytest

from orbweaver.agent import agent_turn
from orbweaver.config import settings
from orbweaver.llm import (
    OllamaMessagesClient,
    make_agent_client,
    no_llm_echo,
    select_provider,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


class _FakeHTTP:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict, **_k):
        self.calls.append((url, json))
        return self.payload


def test_anthropic_wins_over_ollama(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    assert select_provider() == "anthropic"


def test_ollama_when_anthropic_unset(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    assert select_provider() == "ollama"
    client = make_agent_client(http=_FakeHTTP({"message": {"content": "hi"}}))
    assert isinstance(client, OllamaMessagesClient)


def test_none_when_neither_configured(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    assert select_provider() == "none"
    assert make_agent_client() is None
    assert "ANTHROPIC_API_KEY" in no_llm_echo("hello")


@pytest.mark.asyncio
async def test_ollama_fake_client_maps_tools(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama.test")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
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
async def test_agent_turn_uses_ollama_fake_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama.test")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")

    class _Client:
        class messages:
            @staticmethod
            async def create(**_k):
                class _B:
                    type = "text"
                    text = "ollama fallback ok"

                class _R:
                    content = [_B()]
                    usage = None

                return _R()

        monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, sid, "hi", ws)
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["ollama fallback ok"]
