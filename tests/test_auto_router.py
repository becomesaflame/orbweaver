"""Auto router: lightweight LLM picks a catalog model from the user prompt."""

from __future__ import annotations

from uuid import uuid4

import pytest

from orbweaver.agent import agent_turn
from orbweaver.auto_router import (
    catalog_text,
    parse_router_choice,
    route_auto_model,
)
from orbweaver.config import settings
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace

CANDIDATES = [
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "glm-5.3-flash",
    "claude-opus-5",
    "qwen3.6-35b",
    "gpt-oss-120b",
]


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Text(text)]
        self.usage = None


class _ScriptedRouter:
    def __init__(self, text: str | Exception) -> None:
        self.text = text
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.text, Exception):
            raise self.text
        return _Resp(self.text)


def test_parse_router_choice_accepts_id_prefix_and_rejects_ambiguous():
    assert parse_router_choice("claude-opus-5\n", CANDIDATES) == "claude-opus-5"
    assert parse_router_choice("  `GLM-5.3-flash`  ", CANDIDATES) == "glm-5.3-flash"
    assert parse_router_choice("opus", CANDIDATES) == "claude-opus-5"
    assert parse_router_choice("I pick claude-opus-5 for this.", CANDIDATES) == "claude-opus-5"
    assert parse_router_choice("sonnet", CANDIDATES) is None
    assert parse_router_choice("gpt-4o", CANDIDATES) is None
    assert parse_router_choice("", CANDIDATES) is None


def test_catalog_text_lists_ids_and_blurbs():
    text = catalog_text(["claude-sonnet-4-6", "qwen3.6-35b"])
    assert "claude-sonnet-4-6" in text
    assert "qwen3.6-35b" in text
    assert "default coding agent" in text


@pytest.mark.asyncio
async def test_route_auto_model_uses_llm_choice(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_router_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    router = _ScriptedRouter("claude-opus-5")
    chosen = await route_auto_model("Prove this SAT encoding is correct.", client=router)
    assert chosen == "claude-opus-5"
    assert router.calls and router.calls[0]["model"] == "claude-haiku-4-5"
    assert router.calls[0]["max_tokens"] == 32
    assert "Prove this SAT encoding" in router.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_route_auto_model_falls_back_when_reply_is_unusable(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_router_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    router = _ScriptedRouter("I think you want a smart one")
    assert await route_auto_model("hi", client=router) == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_route_auto_model_falls_back_when_the_llm_errors(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_router_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    router = _ScriptedRouter(RuntimeError("haiku down"))
    assert await route_auto_model("hi", client=router) == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_route_auto_model_skips_llm_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_router_model", "")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    router = _ScriptedRouter("claude-opus-5")
    assert await route_auto_model("hard proof", client=router) == "claude-sonnet-4-6"
    assert router.calls == []


@pytest.mark.asyncio
async def test_agent_turn_auto_uses_router_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_web_model", "auto")
    monkeypatch.setattr(settings, "orbweaver_router_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")

    seen: list[str] = []

    class _Agent:
        class messages:
            @staticmethod
            async def create(**kw):
                seen.append(str(kw.get("model")))
                return _Resp("ok")

    async def _route(prompt, *, skip=None, extra="", client=None):
        del skip, extra, client
        assert "SAT encoding" in prompt
        return "claude-opus-5"

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Agent())
    monkeypatch.setattr("orbweaver.auto_router.route_auto_model", _route)
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(
        store, uuid4(), "Prove this SAT encoding is correct.", ws, channel="web"
    )
    assert seen == ["claude-opus-5"]
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["ok"]
