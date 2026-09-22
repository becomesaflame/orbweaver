"""Per-channel model defaults, per-session/turn overrides and the picker API (#137)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import agent_turn
from orbweaver.app import app, reset_ws_subscribers_for_tests
from orbweaver.auth import mint_token
from orbweaver.config import settings
from orbweaver.model_routing import (
    AUTO_MODEL_ID,
    channel_default_model,
    current_model,
    fallback_model,
    format_model_id,
    is_supported_model,
    model_defaults,
    model_label,
    pick_auto_model,
    realize_turn_model,
    resolve_model_pick,
    resolve_turn_model,
    supported_models,
)
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace

WEB = "claude-sonnet-4-6"
VSCODE = "claude-opus-5"
TELEGRAM = "qwen3.6-35b"
FALLBACK = "claude-haiku-4-5"


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()
    reset_ws_subscribers_for_tests()


@pytest.fixture
def channel_models(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_model", FALLBACK)
    monkeypatch.setattr(settings, "orbweaver_web_model", WEB)
    monkeypatch.setattr(settings, "orbweaver_vscode_model", VSCODE)
    monkeypatch.setattr(settings, "orbweaver_telegram_model", TELEGRAM)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "earthruntime_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")


def _fake_client(seen: list[str]):
    class _Text:
        type = "text"
        text = "ok"

    class _Resp:
        def __init__(self) -> None:
            self.content = [_Text()]
            self.usage = None

    class _Client:
        class messages:
            @staticmethod
            async def create(**kw):
                seen.append(str(kw.get("model")))
                return _Resp()

    return _Client()


def _capture_clients(monkeypatch):
    """Record the model passed to make_agent_client and to messages.create."""
    made: list[str | None] = []
    created: list[str] = []

    def fake_make(**kw):
        made.append(kw.get("model"))
        return _fake_client(created)

    monkeypatch.setattr("orbweaver.llm.make_agent_client", fake_make)
    return made, created


# --- resolution -------------------------------------------------------------


def test_channel_defaults_web_and_telegram_are_auto_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_model", FALLBACK)
    monkeypatch.setattr(settings, "orbweaver_web_model", "")
    monkeypatch.setattr(settings, "orbweaver_vscode_model", "")
    monkeypatch.setattr(settings, "orbweaver_telegram_model", "")
    assert channel_default_model("web") == AUTO_MODEL_ID
    assert channel_default_model("telegram") == AUTO_MODEL_ID
    assert channel_default_model("vscode") == FALLBACK
    assert channel_default_model("cron") == FALLBACK
    assert channel_default_model(None) == FALLBACK
    assert model_defaults() == {
        "web": AUTO_MODEL_ID,
        "vscode": FALLBACK,
        "telegram": AUTO_MODEL_ID,
        "fallback": FALLBACK,
    }


def test_channel_defaults_split_by_channel(channel_models):
    assert channel_default_model("web") == WEB
    assert channel_default_model("vscode") == VSCODE
    assert channel_default_model("VS Code") == VSCODE
    assert channel_default_model("telegram") == TELEGRAM
    assert channel_default_model("cron") == FALLBACK
    assert channel_default_model(None) == FALLBACK


def test_resolution_order_override_session_channel(channel_models):
    assert resolve_turn_model(channel="web") == WEB
    assert resolve_turn_model(session={"model": "gpt-oss-120b"}, channel="web") == "gpt-oss-120b"
    assert (
        resolve_turn_model(override="deepseek-v4-flash-0731", session={"model": "gpt-oss-120b"}, channel="web")
        == "deepseek-v4-flash-0731"
    )
    assert resolve_turn_model(session={"model": ""}, channel="telegram") == TELEGRAM


def test_supported_models_and_routing(channel_models):
    assert is_supported_model("claude-sonnet-5")
    assert is_supported_model("gpt-oss-120b")
    assert not is_supported_model("")
    assert not is_supported_model("gpt-4o")
    assert not is_supported_model("claude bad id")
    rows = {m["id"]: m for m in supported_models()}
    # Auto, then configured defaults, then Claude ids, then the catalog.
    assert [m["id"] for m in supported_models()][:5] == [
        AUTO_MODEL_ID,
        FALLBACK,
        WEB,
        VSCODE,
        TELEGRAM,
    ]
    assert rows[AUTO_MODEL_ID]["label"] == "Auto"
    assert rows[AUTO_MODEL_ID]["provider"] == "auto" and rows[AUTO_MODEL_ID]["available"]
    assert rows[WEB]["provider"] == "anthropic" and rows[WEB]["available"]
    assert rows[TELEGRAM]["provider"] == "openrouter" and rows[TELEGRAM]["available"]
    assert rows["gpt-oss-120b"]["context_window"] == 131_072
    assert rows["gpt-oss-120b"]["label"] == "GPT-OSS 120B"
    assert rows["claude-opus-5"]["label"] == "Claude Opus 5"
    assert rows[WEB]["context_window"] == settings.context_window
    assert format_model_id("claude-opus-5") == "Claude Opus 5 (claude-opus-5)"
    assert format_model_id("auto") == "Auto"
    assert model_label("qwen3.6-35b") == "Qwen 3.6 35B"
    assert is_supported_model("auto")


def test_supported_models_marks_missing_key(channel_models, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    rows = {m["id"]: m for m in supported_models()}
    assert rows["gpt-oss-120b"] == {
        "id": "gpt-oss-120b",
        "label": "GPT-OSS 120B",
        "provider": "none",
        "available": False,
        "context_window": 131_072,
    }


def test_ollama_model_is_supported_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    assert is_supported_model("llama3.2")
    assert "llama3.2" in {m["id"] for m in supported_models()}


def test_resolve_model_pick_prefix_label_and_clear():
    ids = [
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-opus-5",
        "qwen3.6-35b",
        "qwen3.8-27b",
        "gpt-oss-120b",
    ]
    assert resolve_model_pick("default", ids) == ("", [])
    assert resolve_model_pick("clear", ids) == ("", [])
    assert resolve_model_pick("claude-opus-5", ids) == ("claude-opus-5", ["claude-opus-5"])
    assert resolve_model_pick("opus", ids)[0] == "claude-opus-5"
    assert resolve_model_pick("Claude Opus 5", ids)[0] == "claude-opus-5"
    chosen, matches = resolve_model_pick("sonnet", ids)
    assert chosen is None and set(matches) == {"claude-sonnet-5", "claude-sonnet-4-6"}
    chosen, matches = resolve_model_pick("qwen", ids)
    assert chosen is None and set(matches) == {"qwen3.6-35b", "qwen3.8-27b"}
    assert resolve_model_pick("120b", ids)[0] == "gpt-oss-120b"
    assert resolve_model_pick("auto", ["auto", "claude-opus-5"]) == ("auto", ["auto"])
    assert resolve_model_pick("Auto", ["auto", "claude-opus-5"])[0] == "auto"
    assert resolve_model_pick("nope", ids) == (None, [])
    assert resolve_model_pick("", ids) == (None, [])


def test_auto_picks_orbweaver_model_then_other_provider(channel_models):
    """Auto prefers ORBWEAVER_MODEL; a 503 should jump to a different provider."""
    from orbweaver.llm import select_provider

    assert realize_turn_model("auto") == FALLBACK
    assert pick_auto_model() == FALLBACK
    nxt = fallback_model(FALLBACK)
    assert nxt
    assert select_provider(nxt) != select_provider(FALLBACK)
    assert nxt == "glm-5.3-flash"
    # A pin stays a pin until it is unavailable.
    assert realize_turn_model("qwen3.6-35b") == "qwen3.6-35b"


def test_fallback_skips_already_tried_models(channel_models):
    from orbweaver.llm import select_provider

    first = fallback_model(FALLBACK, tried={FALLBACK})
    second = fallback_model(FALLBACK, tried={FALLBACK, first})
    assert first and second and first != second
    assert select_provider(first) != "anthropic"


# --- agent_turn -------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_turn_uses_channel_default_and_override(tmp_path, channel_models, monkeypatch):
    made, created = _capture_clients(monkeypatch)
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    await agent_turn(store, uuid4(), "hi", ws, channel="telegram")
    await agent_turn(store, uuid4(), "hi", ws, channel="vscode")
    await agent_turn(store, uuid4(), "hi", ws, channel="cron")
    await agent_turn(store, uuid4(), "hi", ws, channel="web", model="gpt-oss-120b")
    assert made == [TELEGRAM, VSCODE, FALLBACK, "gpt-oss-120b"]
    assert created == made
    # The contextvar is reset after each turn.
    assert current_model() == FALLBACK


@pytest.mark.asyncio
async def test_agent_turn_falls_back_to_other_provider_on_503(tmp_path, monkeypatch):
    """HTTP 503 is model-unavailable: switch provider instead of retrying the same id."""
    from orbweaver.llm import OpenAICompatError

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    monkeypatch.setattr(settings, "orbweaver_model", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "orbweaver_web_model", "auto")
    monkeypatch.setattr(settings, "orbweaver_telegram_model", "auto")
    monkeypatch.setattr(settings, "orbweaver_router_model", "")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(settings, "ollama_model", "")

    seen: list[str] = []

    class _Text:
        type = "text"
        text = "recovered"

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
                if mid == "claude-sonnet-4-6":
                    raise OpenAICompatError(
                        "The model is temporarily rate limited. Retry shortly.",
                        status_code=503,
                    )
                return _Resp()

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, uuid4(), "hi", ws, channel="web")
    assert seen[0] == "claude-sonnet-4-6"
    assert seen[1] == "glm-5.3-flash"
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert texts == ["recovered"]


@pytest.mark.asyncio
async def test_agent_turn_pinned_model_also_falls_back_on_503(tmp_path, channel_models, monkeypatch):
    from orbweaver.llm import OpenAICompatError

    seen: list[str] = []

    class _Text:
        type = "text"
        text = "ok"

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
                if mid == TELEGRAM:
                    raise OpenAICompatError("upstream unavailable", status_code=503)
                return _Resp()

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    await agent_turn(store, uuid4(), "hi", ws, channel="telegram")
    assert seen[0] == TELEGRAM
    assert seen[1]
    assert seen[1] != TELEGRAM
    from orbweaver.llm import select_provider

    assert select_provider(seen[1]) != "openrouter"


@pytest.mark.asyncio
async def test_event_budget_follows_turn_model(tmp_path, channel_models, monkeypatch):
    """Context-window maths use the model of the running turn, not ORBWEAVER_MODEL."""
    budgets: list[int] = []

    class _Text:
        type = "text"
        text = "ok"

    class _Resp:
        def __init__(self) -> None:
            self.content = [_Text()]
            self.usage = None

    class _Client:
        class messages:
            @staticmethod
            async def create(**_kw):
                budgets.append(settings.event_budget)
                return _Resp()

    monkeypatch.setattr("orbweaver.llm.make_agent_client", lambda **_k: _Client())
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    outside = settings.event_budget
    await agent_turn(store, uuid4(), "hi", ws, channel="web", model="gpt-oss-120b")
    await agent_turn(store, uuid4(), "hi", ws, channel="web", model="qwen3.8-27b")
    assert budgets[0] < outside < budgets[1]
    assert settings.event_budget_for("gpt-oss-120b") == budgets[0]
    assert settings.event_budget == outside


@pytest.mark.asyncio
async def test_no_provider_echo_names_the_turn_model(tmp_path, channel_models, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    produced = await agent_turn(store, uuid4(), "hi", ws, channel="telegram")
    texts = [e.payload.get("text") for e in produced if e.kind == "assistant"]
    assert len(texts) == 1 and TELEGRAM in texts[0] and "OPENROUTER_API_KEY" in texts[0]


# --- HTTP API ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_endpoint_and_health_defaults(channel_models, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        anon = await client.get("/v1/models")
        assert anon.status_code == 401
        r = await client.get("/v1/models", headers=auth_header)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["defaults"] == {
            "web": WEB,
            "vscode": VSCODE,
            "telegram": TELEGRAM,
            "fallback": FALLBACK,
        }
        ids = [m["id"] for m in body["models"]]
        assert {AUTO_MODEL_ID, WEB, VSCODE, TELEGRAM, FALLBACK, "gpt-oss-120b"} <= set(ids)
        assert any(m["id"] == "auto" and m["label"] == "Auto" for m in body["models"])
        assert any(m["id"] == "claude-opus-5" and m["label"] == "Claude Opus 5" for m in body["models"])
        health = await client.get("/health")
        assert health.json()["llm"]["model"] == WEB
        assert health.json()["llm"]["provider"] == "anthropic"
        assert health.json()["llm"]["defaults"]["telegram"] == TELEGRAM


@pytest.mark.asyncio
async def test_models_and_health_default_web_telegram_to_auto(auth_header, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "pk-prov-test")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/v1/models", headers=auth_header)
        assert r.status_code == 200
        assert r.json()["defaults"]["web"] == AUTO_MODEL_ID
        assert r.json()["defaults"]["telegram"] == AUTO_MODEL_ID
        health = await client.get("/health")
        assert health.json()["llm"]["model"] == AUTO_MODEL_ID
        assert health.json()["llm"]["provider"] == "auto"


@pytest.mark.asyncio
async def test_session_model_create_patch_and_turn(tmp_path, channel_models, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    made, _created = _capture_clients(monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "web", "model": "gpt-4o"},
            headers=auth_header,
        )
        assert bad.status_code == 400 and "/v1/models" in bad.text

        created = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "web", "model": "gpt-oss-120b"},
            headers=auth_header,
        )
        assert created.status_code == 200, created.text
        sid = created.json()["id"]
        assert created.json()["model"] == "gpt-oss-120b"
        listed = await client.get("/v1/sessions", headers=auth_header)
        row = next(s for s in listed.json()["sessions"] if s["id"] == sid)
        assert row["model"] == "gpt-oss-120b"

        # Stored model wins over the web default.
        turn = await client.post(f"/v1/sessions/{sid}/turns", json={"text": "hi"}, headers=auth_header)
        assert turn.status_code == 200, turn.text
        assert made[-1] == "gpt-oss-120b"

        # Turn-level override wins and is persisted on the session.
        turn = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "hi", "model": "qwen3.8-27b"},
            headers=auth_header,
        )
        assert turn.status_code == 200, turn.text
        assert made[-1] == "qwen3.8-27b"
        listed = await client.get("/v1/sessions", headers=auth_header)
        row = next(s for s in listed.json()["sessions"] if s["id"] == sid)
        assert row["model"] == "qwen3.8-27b"

        rejected = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "hi", "model": "not-a-model"},
            headers=auth_header,
        )
        assert rejected.status_code == 400
        assert len(made) == 2

        # PATCH switches the model; "" clears it back to the channel default.
        patched = await client.patch(f"/v1/sessions/{sid}", json={"model": "claude-sonnet-5"}, headers=auth_header)
        assert patched.status_code == 200 and patched.json()["model"] == "claude-sonnet-5"
        turn = await client.post(f"/v1/sessions/{sid}/turns", json={"text": "hi"}, headers=auth_header)
        assert turn.status_code == 200 and made[-1] == "claude-sonnet-5"
        cleared = await client.patch(f"/v1/sessions/{sid}", json={"model": ""}, headers=auth_header)
        assert cleared.status_code == 200 and cleared.json()["model"] == ""
        turn = await client.post(f"/v1/sessions/{sid}/turns", json={"text": "hi"}, headers=auth_header)
        assert turn.status_code == 200 and made[-1] == WEB


@pytest.mark.asyncio
async def test_channel_sessions_pick_their_default(tmp_path, channel_models, monkeypatch, auth_header):
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    made, _created = _capture_clients(monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for channel, expected in (("web", WEB), ("vscode", VSCODE), (None, FALLBACK)):
            body = {"workspace_uri": "workspace:default"}
            if channel:
                body["channel"] = channel
            created = await client.post("/v1/sessions", json=body, headers=auth_header)
            sid = created.json()["id"]
            turn = await client.post(f"/v1/sessions/{sid}/turns", json={"text": "hi"}, headers=auth_header)
            assert turn.status_code == 200, turn.text
            assert made[-1] == expected, channel


# --- WebSocket --------------------------------------------------------------


def test_ws_text_frame_accepts_model(tmp_path, channel_models, monkeypatch):
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    made, _created = _capture_clients(monkeypatch)
    token = mint_token("t")
    headers = {"authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        created = client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "channel": "web"},
            headers=headers,
        )
        sid = created.json()["id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/ws", subprotocols=["bearer", token]) as ws:
            assert ws.receive_json()["kind"] == "subscribed"
            ws.send_json({"text": "hi", "model": "nope"})
            err = ws.receive_json()
            assert err["kind"] == "error" and err["status"] == 400
            ws.send_json({"text": "hi", "model": "deepseek-v4-flash-0731"})
            while ws.receive_json().get("kind") != "turn_done":
                pass
        assert made == ["deepseek-v4-flash-0731"]
        row = next(s for s in client.get("/v1/sessions", headers=headers).json()["sessions"] if s["id"] == sid)
        assert row["model"] == "deepseek-v4-flash-0731"
