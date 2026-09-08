import json

import pytest

from orbweaver.config import settings
from orbweaver.agent import run_tools
from orbweaver.memory import expand_chunk_graph, remember, rewrite_search_query
from orbweaver.store import Entity, Event, PinBudgetError, reset_store_for_tests, new_uuid
from orbweaver.tokens import estimate_tokens


@pytest.fixture
def store():
    return reset_store_for_tests()


@pytest.mark.asyncio
async def test_remember_and_search(store):
    await remember(store, "the airbed firmware uses co3ntrol-rs", source="test")
    hits = await store.search_chunks("the airbed firmware uses co3ntrol-rs", k=3)
    assert hits
    assert "co3ntrol" in hits[0][0].text


@pytest.mark.asyncio
async def test_pin_cap_rejects(store, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_pinned_token_cap", 20)
    await remember(store, "tiny", pinned=True)
    with pytest.raises(PinBudgetError):
        await remember(store, "x" * 400, pinned=True)


@pytest.mark.asyncio
async def test_forget(store):
    c = await remember(store, "secret note")
    await store.forget_chunk(c.id)
    assert await store.get_chunk(c.id) is None


@pytest.mark.asyncio
async def test_truncate_events(store):
    sid = new_uuid()
    a = await store.append_event(sid, "user", {"text": "one"})
    await store.append_event(sid, "assistant", {"text": "two"})
    await store.truncate_events(sid, a.seq)
    assert await store.list_events(sid) == []


def test_search_query_uses_conversation():
    sid = new_uuid()
    events = [
        Event(id=new_uuid(), session_id=sid, seq=1, kind="user", payload={"text": "airbed valves"}),
        Event(id=new_uuid(), session_id=sid, seq=2, kind="assistant", payload={"text": "checking firmware"}),
    ]
    q = rewrite_search_query(events, "what about the other one?")
    assert "airbed" in q
    assert "other one" in q


def test_token_estimate():
    assert estimate_tokens("abcd") >= 1


@pytest.mark.asyncio
async def test_search_expands_graph_neighbors(store):
    eid = new_uuid()
    project = "urn:orbweaver:entity:project-airbed"
    await store.put_entity(
        Entity(
            id=eid,
            at_id="urn:orbweaver:entity:decision-1",
            at_type="Decision",
            jsonld={
                "@id": "urn:orbweaver:entity:decision-1",
                "@type": "Decision",
                "about": project,
            },
        )
    )
    await remember(
        store, "chose LocalWorkspace for Telegram sessions", source="test", entity_ids=[eid]
    )
    hits = await store.search_chunks("chose LocalWorkspace for Telegram sessions", k=3)
    graph = await expand_chunk_graph(store, hits)
    assert any(g["o"] == project and g["p"] == "about" for g in graph)


@pytest.mark.asyncio
async def test_memory_search_tool_includes_graph(store):
    eid = new_uuid()
    project = "urn:orbweaver:entity:project-airbed"
    await store.put_entity(
        Entity(
            id=eid,
            at_id="urn:orbweaver:entity:decision-2",
            at_type="Decision",
            jsonld={
                "@id": "urn:orbweaver:entity:decision-2",
                "@type": "Decision",
                "about": project,
            },
        )
    )
    await remember(store, "pin budget is four thousand tokens", entity_ids=[eid])
    sid = new_uuid()
    ctx = {"workspace": None, "store": store, "session_id": sid}
    expanded = await run_tools("MemorySearch", {"query": "pin budget tokens"}, ctx)
    body = json.loads(expanded)
    assert body["chunk_ids"]
    assert any(g["o"] == project for g in body["graph"])
    skipped = await run_tools(
        "MemorySearch", {"query": "pin budget tokens", "expand_graph": False}, ctx
    )
    assert json.loads(skipped)["graph"] == []
