import pytest

from orbweaver.agent import maybe_compact
from orbweaver.config import settings
from orbweaver.store import reset_store_for_tests, new_uuid


@pytest.mark.asyncio
async def test_compact_when_over_budget(monkeypatch):
    store = reset_store_for_tests()
    monkeypatch.setattr(settings, "event_budget_override", 80)
    monkeypatch.setattr(settings, "compact_ratio", 0.5)
    sid = new_uuid()
    from orbweaver.store import SESSION_TYPE, Entity, session_at_id

    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={"@id": session_at_id(sid), "@type": SESSION_TYPE, "workspace_uri": "workspace:default"},
        )
    )
    for i in range(40):
        await store.append_event(sid, "user", {"text": "word " * 30 + str(i)})
    ev = await maybe_compact(store, sid)
    assert ev is not None
    events = await store.list_events(sid)
    assert events[0].kind == "compact_summary"
    assert len(events) < 40
