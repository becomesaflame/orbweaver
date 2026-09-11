import asyncio
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.agent import TurnCancelled, _events_to_messages, agent_turn
from orbweaver.app import app
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


async def _slow_turn(
    store,
    session_id,
    user_text,
    ws,
    workspace_kind="local",
    emit=None,
    cancel=None,
    turn_state=None,
    resume=False,
    **_kwargs,
):
    if not resume:
        ev = await store.append_event(session_id, "user", {"text": user_text})
        if turn_state is not None:
            turn_state.user_seq = ev.seq
    inject = getattr(turn_state, "inject", None)
    while True:
        if cancel is not None and cancel.is_set():
            raise TurnCancelled()
        if inject is not None and inject.is_set():
            inject.clear()
            await asyncio.sleep(0.01)
            continue
        await asyncio.sleep(0.02)


async def _open_session(client, headers):
    sess = await client.post(
        "/v1/sessions",
        json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
        headers=headers,
    )
    assert sess.status_code == 200, sess.text
    return sess.json()["id"]


async def _wait_running():
    from orbweaver import turns

    for _ in range(100):
        if turns.active():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("turn did not start")


@pytest.mark.asyncio
async def test_agent_turn_skips_append_when_already_cancelled(tmp_path):
    from orbweaver.store import get_store

    store = get_store()
    sid = uuid4()
    cancel = asyncio.Event()
    cancel.set()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    with pytest.raises(TurnCancelled):
        await agent_turn(store, sid, "never sent", ws, cancel=cancel)
    assert await store.list_events(sid) == []


@pytest.mark.asyncio
async def test_cancel_discards_in_flight_prompt(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _slow_turn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        turn_task = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sid}/turns",
                json={"text": "oops typo"},
                headers=auth_header,
            )
        )
        await _wait_running()
        cancelled = await client.post(
            f"/v1/sessions/{sid}/turns/cancel",
            json={"discard": True},
            headers=auth_header,
        )
        assert cancelled.status_code == 200, cancelled.text
        turned = await turn_task
        assert turned.json()["status"] == "discarded"
        events = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        assert events.json()["events"] == []


@pytest.mark.asyncio
async def test_stop_keeps_prompt(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _slow_turn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        turn_task = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sid}/turns",
                json={"text": "keep this"},
                headers=auth_header,
            )
        )
        await _wait_running()
        await client.post(
            f"/v1/sessions/{sid}/turns/cancel",
            json={"discard": False},
            headers=auth_header,
        )
        turned = await turn_task
        assert turned.json()["status"] == "stopped"
        events = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        kinds = [e["kind"] for e in events.json()["events"]]
        assert kinds[0] == "user"
        assert "turn_interrupted" in kinds
        assert events.json()["events"][0]["payload"]["text"] == "keep this"


@pytest.mark.asyncio
async def test_concurrent_turn_conflict(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _slow_turn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        first = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sid}/turns",
                json={"text": "first"},
                headers=auth_header,
            )
        )
        await _wait_running()
        second = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "second"},
            headers=auth_header,
        )
        assert second.status_code == 409
        await client.post(
            f"/v1/sessions/{sid}/turns/cancel",
            json={"discard": True},
            headers=auth_header,
        )
        await first


@pytest.mark.asyncio
async def test_rewind_drops_from_user_seq(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        turned = await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"text": "wire up pause"},
            headers=auth_header,
        )
        assert turned.status_code == 200, turned.text
        listed = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        user = next(e for e in listed.json()["events"] if e["kind"] == "user")
        rewound = await client.post(
            f"/v1/sessions/{sid}/rewind",
            json={"from_seq": user["seq"]},
            headers=auth_header,
        )
        assert rewound.status_code == 200, rewound.text
        assert rewound.json()["text"] == "wire up pause"
        listed = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        assert listed.json()["events"] == []


@pytest.mark.asyncio
async def test_inject_appends_user_to_open_turn(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _slow_turn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        turn_task = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sid}/turns",
                json={"text": "first prompt"},
                headers=auth_header,
            )
        )
        await _wait_running()
        injected = await client.post(
            f"/v1/sessions/{sid}/turns/inject",
            json={"text": "also handle empty input"},
            headers=auth_header,
        )
        assert injected.status_code == 200, injected.text
        assert injected.json()["status"] == "injected"
        events = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        users = [e for e in events.json()["events"] if e["kind"] == "user"]
        assert [u["payload"]["text"] for u in users] == [
            "first prompt",
            "also handle empty input",
        ]
        await client.post(
            f"/v1/sessions/{sid}/turns/cancel",
            json={"discard": True},
            headers=auth_header,
        )
        await turn_task


@pytest.mark.asyncio
async def test_inject_idle_conflicts(auth_header):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        r = await client.post(
            f"/v1/sessions/{sid}/turns/inject",
            json={"text": "nope"},
            headers=auth_header,
        )
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_continue_after_stop(tmp_path, monkeypatch, auth_header):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr("orbweaver.app.agent_turn", _slow_turn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sid = await _open_session(client, auth_header)
        turn_task = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sid}/turns",
                json={"text": "resume me"},
                headers=auth_header,
            )
        )
        await _wait_running()
        await client.post(
            f"/v1/sessions/{sid}/turns/cancel",
            json={"discard": False},
            headers=auth_header,
        )
        await turn_task
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        from orbweaver.agent import agent_turn as real_turn

        monkeypatch.setattr("orbweaver.app.agent_turn", real_turn)
        continued = await client.post(
            f"/v1/sessions/{sid}/turns/continue",
            headers=auth_header,
        )
        assert continued.status_code == 200, continued.text
        assert continued.json()["status"] == "ok"
        events = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        kinds = [e["kind"] for e in events.json()["events"]]
        assert kinds.count("user") == 1
        assert "assistant" in kinds


def test_consecutive_user_events_merge_for_the_model():
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "one"}),
        Event(id=uuid4(), session_id=sid, seq=2, kind="user", payload={"text": "two"}),
    ]
    messages = _events_to_messages(events)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "one" in messages[0]["content"]
    assert "two" in messages[0]["content"]
