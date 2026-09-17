"""Stray Continue button: a finished turn must not be classified as stopped.

Production bug: the web UI decided whether to show **Continue** by looking at
the kind of the session's *last* event and treating anything other than
``assistant`` as a stopped turn. A completed turn keeps appending bookkeeping
after the final assistant message -- ``settle_children`` records
``subagent_result`` in ``agent_turn``'s ``finally`` block, ``TodoWrite``
records ``todo_state``, cron records ``cron_result`` -- so healthy chats showed
a Continue button on every reload or chat switch.

These tests drive the real agent loop and the real events endpoint, not just
the classifier, so the trailing-event ordering is the thing under test.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from orbweaver.app import app
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.turnstate import (
    STATUS_OK,
    STATUS_STOPPED,
    STATUS_WAITING_ASK,
    session_turn_status,
)


@pytest.fixture(autouse=True)
def _store():
    reset_store_for_tests()


async def _allow(*_a, **_k):
    return {
        "verdict": "allow",
        "should_block": False,
        "should_ask": False,
        "reason": "test",
        "stage": "test",
    }


async def _passthrough_review(_parent, _calls, payload, **_k):
    return {"status": "ok", "reason": "ok", "payload": payload}


def _ev(seq: int, kind: str, payload: dict | None = None) -> Event:
    return Event(
        id=uuid4(),
        session_id=uuid4(),
        seq=seq,
        kind=kind,
        payload=payload or {},
    )


def test_trailing_subagent_result_still_counts_as_finished():
    """The exact production shape: assistant, then settle_children bookkeeping."""
    events = [
        _ev(1, "user", {"text": "do the thing"}),
        _ev(2, "assistant", {"text": "done, here is the result"}),
        _ev(3, "subagent_result", {"subagent_id": "abc", "status": "ok"}),
    ]
    assert session_turn_status(events) == STATUS_OK


def test_trailing_todo_state_still_counts_as_finished():
    events = [
        _ev(1, "user", {"text": "plan it"}),
        _ev(2, "assistant", {"text": "all items complete"}),
        _ev(3, "todo_state", {"todos": []}),
    ]
    assert session_turn_status(events) == STATUS_OK


def test_several_trailing_bookkeeping_events_still_finished():
    events = [
        _ev(1, "user", {"text": "fan out"}),
        _ev(2, "assistant", {"text": "both children reported"}),
        _ev(3, "subagent_cancelled", {"reason": "parent_turn_ended"}),
        _ev(4, "subagent_result", {"status": "cancelled"}),
        _ev(5, "todo_state", {"todos": []}),
    ]
    assert session_turn_status(events) == STATUS_OK


def test_turn_that_died_mid_round_is_still_resumable():
    """A real interrupted turn must keep offering Continue."""
    events = [
        _ev(1, "user", {"text": "read the file"}),
        _ev(2, "tool_call", {"id": "tu-1", "name": "Read"}),
        _ev(3, "tool_result", {"tool_use_id": "tu-1", "content": "text"}),
    ]
    assert session_turn_status(events) == STATUS_STOPPED


def test_user_stop_is_resumable():
    events = [
        _ev(1, "user", {"text": "long job"}),
        _ev(2, "tool_call", {"id": "tu-1", "name": "Bash"}),
        _ev(3, "turn_interrupted", {"reason": "stop"}),
    ]
    assert session_turn_status(events) == STATUS_STOPPED


def test_stopped_turn_with_trailing_bookkeeping_is_still_resumable():
    """Bookkeeping after a Stop must not flip it to finished."""
    events = [
        _ev(1, "user", {"text": "long job"}),
        _ev(2, "turn_interrupted", {"reason": "stop"}),
        _ev(3, "subagent_cancelled", {"reason": "parent_cancelled"}),
    ]
    assert session_turn_status(events) == STATUS_STOPPED


def test_cron_result_and_abort_conclude_a_turn():
    assert session_turn_status([_ev(1, "cron_result", {"text": "ran"})]) == STATUS_OK
    aborted = [
        _ev(1, "user", {"text": "loop"}),
        _ev(2, "turn_aborted", {"reason": "stuck"}),
    ]
    assert session_turn_status(aborted) == STATUS_OK


def test_unanswered_ask_user_is_waiting_not_stopped():
    events = [
        _ev(1, "user", {"text": "pick one"}),
        _ev(2, "tool_call", {"id": "tu-ask", "name": "AskUser"}),
    ]
    assert session_turn_status(events) == STATUS_WAITING_ASK


def test_answered_ask_user_then_answer_is_finished():
    events = [
        _ev(1, "user", {"text": "pick one"}),
        _ev(2, "tool_call", {"id": "tu-ask", "name": "AskUser"}),
        _ev(3, "tool_result", {"tool_use_id": "tu-ask", "content": "the blue one"}),
        _ev(4, "assistant", {"text": "going with blue"}),
    ]
    assert session_turn_status(events) == STATUS_OK


def test_empty_session_has_nothing_to_continue():
    assert session_turn_status([]) == STATUS_OK


@pytest.mark.asyncio
async def test_events_endpoint_reports_ok_after_background_subagent(
    tmp_path, monkeypatch, auth_header
):
    """End-to-end: run a turn that leaves a background child, then read /events.

    Before the fix the endpoint returned no status and the UI inferred
    "stopped" from the trailing subagent_result, so Continue appeared on a chat
    whose turn had answered normally. The assertion that matters is that
    bookkeeping really does trail the final assistant event.
    """
    import anthropic

    from orbweaver.config import settings
    from orbweaver.subagent import reset_subagent_semaphore_for_tests

    reset_subagent_semaphore_for_tests()
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(settings, "orbweaver_subagent_grace_s", 2.0)
    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", _allow)
    monkeypatch.setattr("orbweaver.subagent.review_subagent_return", _passthrough_review)

    class _ToolUse:
        def __init__(self, name, inp, uid):
            self.type = "tool_use"
            self.id = uid
            self.name = name
            self.input = inp

    class _LLM:
        """Spawns a background child on round 1, answers on round 2."""

        def __init__(self):
            self.messages = self
            self.calls = 0

        async def create(self, **kwargs):
            system = kwargs.get("system") or []
            text = (
                system
                if isinstance(system, str)
                else " ".join(str(b.get("text") or "") for b in system if isinstance(b, dict))
            )
            if "Orbweaver subagent" in text:
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="child looked around")],
                    stop_reason="end_turn",
                    usage=None,
                )
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content=[
                        _ToolUse(
                            "SpawnSubagent",
                            {"task": "look around", "background": True},
                            "tu-s",
                        )
                    ],
                    stop_reason="tool_use",
                    usage=None,
                )
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="here is the final answer")],
                stop_reason="end_turn",
                usage=None,
            )

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: _LLM())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sess = await client.post(
            "/v1/sessions",
            json={"workspace_uri": "workspace:default", "workspace_kind": "local"},
            headers=auth_header,
        )
        assert sess.status_code == 200, sess.text
        sid = sess.json()["id"]
        turned = await client.post(
            f"/v1/sessions/{sid}/turns", json={"text": "spawn and answer"}, headers=auth_header
        )
        assert turned.status_code == 200, turned.text
        assert turned.json()["status"] == "ok"

        got = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        assert got.status_code == 200, got.text
        body = got.json()
        kinds = [e["kind"] for e in body["events"]]
        # The bug only bites when bookkeeping really does trail the answer.
        assert "assistant" in kinds
        assert kinds[-1] != "assistant", kinds
        assert body["last_turn_state"] == STATUS_OK, kinds
    reset_subagent_semaphore_for_tests()


@pytest.mark.asyncio
async def test_events_endpoint_reports_stopped_for_interrupted_turn(auth_header):
    """A turn killed mid-round (gateway restart) still reports stopped."""
    from orbweaver.store import SESSION_TYPE, Entity, get_store, session_at_id

    store = get_store()
    sid = uuid4()
    await store.put_entity(
        Entity(
            id=sid,
            at_id=session_at_id(sid),
            at_type=SESSION_TYPE,
            jsonld={
                "@id": session_at_id(sid),
                "@type": SESSION_TYPE,
                "workspace_uri": "workspace:default",
                "workspace_kind": "local",
                "status": "active",
            },
        )
    )
    await store.append_event(sid, "user", {"text": "start a long job"})
    await store.append_event(sid, "tool_call", {"id": "tu-1", "name": "Bash"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        got = await client.get(f"/v1/sessions/{sid}/events", headers=auth_header)
        assert got.status_code == 200, got.text
        assert got.json()["last_turn_state"] == STATUS_STOPPED
