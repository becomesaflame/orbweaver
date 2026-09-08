import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import TOOL_SPEC, run_tools
from orbweaver.app import app
from orbweaver.config import settings
from orbweaver.hindsight import (
    enabled,
    format_recall,
    format_reflect,
    format_turn_transcript,
    reset_for_tests,
    retain_turn,
)
from orbweaver.store import Event, new_uuid, reset_store_for_tests


@pytest.fixture
def store():
    return reset_store_for_tests()


@pytest.fixture
def hs_calls(monkeypatch):
    monkeypatch.setattr(settings, "hindsight_api_url", "http://hindsight.test")
    monkeypatch.setattr(settings, "hindsight_api_key", "secret-key")
    monkeypatch.setattr(settings, "hindsight_bank_id", "personal")
    reset_for_tests()
    calls: list[tuple[str, str, dict | None]] = []

    async def fake_request(method, path, json=None, timeout=30.0):
        del timeout
        calls.append((method, path, json))
        if method == "PUT":
            return {"bank_id": "personal"}
        if method == "PATCH":
            return {"bank_id": "personal", "config": {}}
        if path.endswith("/memories/recall"):
            return {
                "results": [
                    {
                        "id": "m1",
                        "text": "airbed firmware uses co3ntrol-rs",
                        "type": "world",
                        "entities": ["co3ntrol-rs"],
                        "score": 0.9,
                    }
                ]
            }
        if path.endswith("/reflect"):
            return {
                "text": "The airbed firmware is co3ntrol-rs.",
                "based_on": {
                    "memories": [
                        {
                            "id": "m1",
                            "text": "airbed firmware uses co3ntrol-rs",
                            "type": "world",
                        }
                    ]
                },
            }
        if path.endswith("/memories"):
            return {"success": True, "items_count": 1, "async": True}
        return {}

    monkeypatch.setattr("orbweaver.hindsight.request", fake_request)
    return calls


def test_disabled_by_default():
    assert not enabled()


def test_memory_reflect_in_tool_spec():
    names = {t["name"] for t in TOOL_SPEC}
    assert "MemoryReflect" in names


def test_format_recall_entities_become_graph():
    out = format_recall(
        {
            "results": [
                {
                    "id": "m1",
                    "text": "uses co3ntrol-rs",
                    "type": "world",
                    "entities": ["co3ntrol-rs"],
                }
            ]
        }
    )
    assert out["chunk_ids"] == ["m1"]
    assert out["source"] == "hindsight"
    assert out["hits"][0]["text"] == "uses co3ntrol-rs"
    assert any(g["s"] == "co3ntrol-rs" and g["o"] == "m1" for g in out["graph"])


def test_format_reflect_extracts_citations():
    out = format_reflect(
        {
            "text": "Use co3ntrol-rs.",
            "based_on": {"memories": [{"id": "m1", "text": "fact", "type": "world"}]},
        }
    )
    assert out["text"] == "Use co3ntrol-rs."
    assert out["based_on"][0]["id"] == "m1"


def test_format_turn_transcript_prefixes_roles():
    text = format_turn_transcript("hello", "world")
    assert "user [" in text
    assert "assistant [" in text
    assert "hello" in text
    assert "world" in text


@pytest.mark.asyncio
async def test_memory_search_tool_uses_hindsight(store, hs_calls):
    sid = new_uuid()
    ctx = {"workspace": None, "store": store, "session_id": sid}
    body = json.loads(await run_tools("MemorySearch", {"query": "airbed firmware"}, ctx))
    assert body["source"] == "hindsight"
    assert "m1" in body["chunk_ids"]
    assert "co3ntrol-rs" in body["text"]
    assert any(c[0] == "POST" and str(c[1]).endswith("/memories/recall") for c in hs_calls)


@pytest.mark.asyncio
async def test_memory_reflect_tool(store, hs_calls):
    sid = new_uuid()
    ctx = {"workspace": None, "store": store, "session_id": sid}
    body = json.loads(
        await run_tools("MemoryReflect", {"query": "what firmware?", "budget": "mid"}, ctx)
    )
    assert "co3ntrol-rs" in body["text"]
    assert body["based_on"][0]["id"] == "m1"
    reflect_calls = [c for c in hs_calls if c[0] == "POST" and str(c[1]).endswith("/reflect")]
    assert reflect_calls
    assert reflect_calls[0][2]["budget"] == "mid"


@pytest.mark.asyncio
async def test_memory_reflect_requires_config(store, monkeypatch):
    monkeypatch.setattr(settings, "hindsight_api_url", "")
    sid = new_uuid()
    ctx = {"workspace": None, "store": store, "session_id": sid}
    body = json.loads(await run_tools("MemoryReflect", {"query": "x"}, ctx))
    assert "not configured" in body["error"]


@pytest.mark.asyncio
async def test_memory_remember_retains(store, hs_calls):
    sid = new_uuid()
    ctx = {"workspace": None, "store": store, "session_id": sid}
    body = json.loads(await run_tools("MemoryRemember", {"text": "prefers bubblewrap"}, ctx))
    assert body["id"] == "hindsight"
    assert any(c[0] == "POST" and str(c[1]).endswith("/memories") for c in hs_calls)
    assert await store.search_chunks("prefers bubblewrap", k=1) == []


@pytest.mark.asyncio
async def test_retain_turn_skips_subagents(store, hs_calls):
    sid = uuid4()
    produced = [
        Event(
            id=new_uuid(),
            session_id=sid,
            seq=1,
            kind="assistant",
            payload={"text": "done"},
        )
    ]
    await retain_turn(sid, "do it", produced, subagent_depth=1)
    assert not any(str(c[1]).endswith("/memories") and c[0] == "POST" for c in hs_calls)
    await retain_turn(sid, "do it", produced, subagent_depth=0, channel="telegram")
    retains = [
        c
        for c in hs_calls
        if c[0] == "POST" and str(c[1]).endswith("/memories") and c[2] and "items" in c[2]
    ]
    assert retains
    content = retains[0][2]["items"][0]["content"]
    assert "user [" in content
    assert "do it" in content
    assert "done" in content


@pytest.mark.asyncio
async def test_compact_skips_native_remember_when_hindsight(store, hs_calls, monkeypatch):
    from orbweaver.compact.pipeline import maybe_compact
    from orbweaver.store import SESSION_TYPE, Entity, session_at_id

    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:default",
            },
        )
    )
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    ev = await maybe_compact(store, sid)
    assert ev is not None
    assert await store.search_chunks("word", k=8) == []


@pytest.mark.asyncio
async def test_memory_search_api_hindsight(hs_calls, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        found = await client.post(
            "/memory/search",
            json={"query": "airbed"},
            headers=auth_header,
        )
        assert found.status_code == 200, found.text
        body = found.json()
        assert body["source"] == "hindsight"
        assert body["hits"][0]["id"] == "m1"


@pytest.mark.asyncio
async def test_memory_reflect_api(hs_calls, auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/memory/reflect",
            json={"query": "what firmware?"},
            headers=auth_header,
        )
        assert r.status_code == 200, r.text
        assert "co3ntrol-rs" in r.json()["text"]


@pytest.mark.asyncio
async def test_health_reports_hindsight(hs_calls):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["hindsight"] is True
