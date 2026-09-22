from uuid import uuid4

import pytest

from orbweaver.resume import drain_paused_sessions, is_drain_paused
from orbweaver.store import SESSION_TYPE, Entity, reset_store_for_tests, session_at_id
from orbweaver.turns import DRAIN_INTERRUPT


def test_is_drain_paused_only_for_drain_interrupt():
    from orbweaver.store import Event

    sid = uuid4()
    drain = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
        Event(id=uuid4(), session_id=sid, seq=2, kind="tool_call", payload={"name": "Bash"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="turn_interrupted",
            payload={"reason": DRAIN_INTERRUPT},
        ),
    ]
    stop = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="turn_interrupted",
            payload={"reason": "stop"},
        ),
    ]
    done = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
        Event(id=uuid4(), session_id=sid, seq=2, kind="assistant", payload={"text": "ok"}),
    ]
    assert is_drain_paused(drain) is True
    assert is_drain_paused(stop) is False
    assert is_drain_paused(done) is False
    assert is_drain_paused([]) is False


@pytest.mark.asyncio
async def test_drain_paused_sessions_skips_cron_and_user_stop():
    store = reset_store_for_tests()
    web_id, cron_id, stop_id = uuid4(), uuid4(), uuid4()
    web = Entity(
        id=web_id,
        at_id=session_at_id(web_id),
        at_type=SESSION_TYPE,
        jsonld={"@id": session_at_id(web_id), "@type": SESSION_TYPE, "channel": "web"},
    )
    cron = Entity(
        id=cron_id,
        at_id=session_at_id(cron_id),
        at_type=SESSION_TYPE,
        jsonld={"@id": session_at_id(cron_id), "@type": SESSION_TYPE, "channel": "cron"},
    )
    stopped = Entity(
        id=stop_id,
        at_id=session_at_id(stop_id),
        at_type=SESSION_TYPE,
        jsonld={"@id": session_at_id(stop_id), "@type": SESSION_TYPE, "channel": "web"},
    )
    await store.put_entity(web)
    await store.put_entity(cron)
    await store.put_entity(stopped)
    await store.append_event(web_id, "user", {"text": "hi"})
    await store.append_event(web_id, "turn_interrupted", {"reason": DRAIN_INTERRUPT})
    await store.append_event(cron_id, "user", {"text": "hi"})
    await store.append_event(cron_id, "turn_interrupted", {"reason": DRAIN_INTERRUPT})
    await store.append_event(stop_id, "user", {"text": "hi"})
    await store.append_event(stop_id, "turn_interrupted", {"reason": "stop"})
    found = await drain_paused_sessions(store)
    assert [e.id for e in found] == [web_id]
