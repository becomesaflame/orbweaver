import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.store import reset_store_for_tests


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


@pytest.mark.asyncio
async def test_memory_remember_search_pin_forget():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tok = (await client.post("/v1/auth/token", json={"sub": "t"})).json()["token"]
        headers = {"authorization": f"Bearer {tok}"}
        remembered = await client.post(
            "/memory/remember",
            json={"text": "the airbed firmware uses co3ntrol-rs", "source": "test"},
            headers=headers,
        )
        assert remembered.status_code == 200, remembered.text
        cid = remembered.json()["id"]
        found = await client.post(
            "/memory/search",
            json={"query": "the airbed firmware uses co3ntrol-rs"},
            headers=headers,
        )
        assert found.status_code == 200
        assert found.json()["hits"]
        pin = await client.post(
            "/memory/pin", json={"kind": "chunk", "id": cid, "pinned": True}, headers=headers
        )
        assert pin.status_code == 200
        forgotten = await client.post("/memory/forget", json={"id": cid}, headers=headers)
        assert forgotten.status_code == 200
        after = await client.post(
            "/memory/search",
            json={"query": "the airbed firmware uses co3ntrol-rs"},
            headers=headers,
        )
        ids = [h["id"] for h in after.json()["hits"]]
        assert cid not in ids


@pytest.mark.asyncio
async def test_entity_graph_from_jsonld():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tok = (await client.post("/v1/auth/token", json={"sub": "t"})).json()["token"]
        headers = {"authorization": f"Bearer {tok}"}
        project = "urn:orbweaver:entity:project-airbed"
        r = await client.put(
            "/memory/entities",
            json={
                "jsonld": {
                    "@id": "urn:orbweaver:entity:decision-1",
                    "@type": "Decision",
                    "about": project,
                }
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        graph = await client.get(
            "/memory/graph",
            params={"id": "urn:orbweaver:entity:decision-1"},
            headers=headers,
        )
        assert graph.status_code == 200
        triples = graph.json()["triples"]
        assert any(t["o"] == project and t["p"] == "about" for t in triples)


@pytest.mark.asyncio
async def test_memory_requires_jwt():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/memory/search", json={"query": "x"})
        assert r.status_code == 401
