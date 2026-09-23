"""Daily USD spend caps per LLM provider."""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import agent_turn
from orbweaver.app import app
from orbweaver.config import Settings, settings
from orbweaver.llm import is_upstream_unavailable, select_provider
from orbweaver.pricing import estimate_prompt_usd, usage_cost_usd, usd_from_tokens
from orbweaver.spend import (
    SpendCapped,
    cap_usd,
    charged_create,
    run_charged,
    spend_key,
    spent_usd,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


@pytest.fixture
def providers(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    reset_store_for_tests()


def test_production_defaults_cap_anthropic_not_earth_runtime():
    fields = Settings.model_fields
    assert fields["orbweaver_spend_cap_anthropic_usd_day"].default == 50.0
    assert fields["orbweaver_spend_cap_openrouter_usd_day"].default == 0.0


def test_zero_cap_is_unlimited(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 0.0)
    monkeypatch.setattr(settings, "orbweaver_spend_cap_openrouter_usd_day", 0.0)
    assert cap_usd("anthropic") is None
    assert cap_usd("openrouter") is None
    assert cap_usd("ollama") is None


def test_sonnet_list_price_and_cache():
    assert usd_from_tokens("claude-sonnet-4-6", 1_000_000, 0, 0, 0) == pytest.approx(3.0)
    assert usd_from_tokens("claude-sonnet-4-6", 0, 1_000_000, 0, 0) == pytest.approx(15.0)
    assert usd_from_tokens("claude-sonnet-4-6", 0, 0, 1_000_000, 0) == pytest.approx(0.30)
    assert usd_from_tokens("claude-sonnet-4-6", 0, 0, 0, 1_000_000) == pytest.approx(3.75)
    assert estimate_prompt_usd("claude-haiku-4-5", 200_000) == pytest.approx(0.20)


def test_openrouter_prefers_usage_cost_field():
    class _U:
        cost = 0.0123
        input_tokens = 999_999

    assert usage_cost_usd("openrouter", "gpt-oss-120b", _U()) == pytest.approx(0.0123)
    assert usage_cost_usd("openrouter", "gpt-oss-120b", None) == 0.0


def test_spend_capped_is_unavailable():
    err = SpendCapped("anthropic", 50.0, 50.0)
    assert is_upstream_unavailable(err)


@pytest.mark.asyncio
async def test_preflight_refuses_anthropic_over_cap(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 1.0)
    store = reset_store_for_tests()
    await store.meta_set(spend_key("anthropic"), str(1_000_000))

    async def boom():
        raise AssertionError("must not call the provider")

    with pytest.raises(SpendCapped) as ei:
        await run_charged(
            "claude-sonnet-4-6",
            {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
            boom,
        )
    assert ei.value.provider == "anthropic"
    assert ei.value.cap_usd == 1.0


@pytest.mark.asyncio
async def test_failed_call_is_refunded(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 50.0)

    async def boom():
        raise RuntimeError("upstream down")

    with pytest.raises(RuntimeError):
        await run_charged(
            "claude-sonnet-4-6",
            {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
            boom,
        )
    assert await spent_usd("anthropic") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_settle_records_actual_and_blocks_next_call(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 1.0)

    class _Usage:
        input_tokens = 1_000_000
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        cost_usd = None

    class _Resp:
        def __init__(self) -> None:
            self.usage = _Usage()
            self.content: list = []

    async def ok():
        return _Resp()

    kwargs = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "x"}]}
    await run_charged("claude-sonnet-4-6", kwargs, ok)
    # 1M input tokens of Sonnet 4.6 is $3, over the $1 cap, recorded after the call.
    assert await spent_usd("anthropic") == pytest.approx(3.0)
    with pytest.raises(SpendCapped):
        await run_charged("claude-sonnet-4-6", kwargs, ok)


@pytest.mark.asyncio
async def test_earth_runtime_unlimited_still_records_cost(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_openrouter_usd_day", 0.0)

    class _Usage:
        cost = 1.25
        input_tokens = 10
        output_tokens = 10

    class _Resp:
        def __init__(self) -> None:
            self.usage = _Usage()
            self.content: list = []

    async def ok():
        return _Resp()

    await run_charged(
        "gpt-oss-120b",
        {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "hi"}]},
        ok,
    )
    assert select_provider("gpt-oss-120b") == "openrouter"
    assert cap_usd("openrouter") is None
    assert await spent_usd("openrouter") == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_charged_create_wraps_client(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 1.0)
    store = reset_store_for_tests()
    await store.meta_set(spend_key("anthropic"), str(1_000_000))

    class _Messages:
        @staticmethod
        async def create(**_kw):
            raise AssertionError("must not call")

    class _Client:
        messages = _Messages()

    with pytest.raises(SpendCapped):
        await charged_create(
            _Client(),
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_agent_turn_falls_back_when_anthropic_capped(tmp_path, providers, monkeypatch):
    """A $50 Anthropic cap must not stall the turn: switch to Earth Runtime."""
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 1.0)
    monkeypatch.setattr(settings, "orbweaver_spend_cap_openrouter_usd_day", 0.0)
    monkeypatch.setattr(settings, "orbweaver_web_model", "auto")
    monkeypatch.setattr(settings, "orbweaver_router_model", "")
    store = reset_store_for_tests()
    await store.meta_set(spend_key("anthropic"), str(1_000_000))

    seen: list[str] = []

    class _Text:
        type = "text"
        text = "from-earth-runtime"

    class _Resp:
        def __init__(self) -> None:
            self.content = [_Text()]
            self.usage = None

    class _Client:
        class messages:
            @staticmethod
            async def create(**kw):
                mid = str(kw.get("model"))
                seen.append(mid)
                if select_provider(mid) == "anthropic":
                    raise AssertionError("anthropic must be refused before HTTP")
                return _Resp()

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, uuid4(), "hi", ws, channel="web")
    assert seen
    assert select_provider(seen[0]) == "openrouter"
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["from-earth-runtime"]


@pytest.mark.asyncio
async def test_health_reports_spend(providers, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_spend_cap_anthropic_usd_day", 50.0)
    monkeypatch.setattr(settings, "orbweaver_spend_cap_openrouter_usd_day", 0.0)
    reset_store_for_tests()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        spend = r.json()["spend"]
        assert spend["providers"]["anthropic"]["cap_usd"] == 50.0
        assert spend["providers"]["anthropic"]["unlimited"] is False
        assert spend["providers"]["openrouter"]["unlimited"] is True
