"""Premature hand-off guard: don't end a turn at the research/work seam.

The agent loop ends a turn when the model emits no tool calls with
``stop_reason == "end_turn"``. In practice that -- not the 256-round budget,
not the stuck detector -- is what makes a user type "continue". When the
session todo list still has open items and the closing message offers to do
the work rather than doing it, the loop nudges once and runs another round.
"""

from itertools import count
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest
from orbweaver.agent import UNFINISHED_PLAN_NUDGE, agent_turn, static_system
from orbweaver.config import settings
from orbweaver.store import reset_store_for_tests
from orbweaver.unfinished import (
    has_unfinished_todos,
    is_deferral,
    should_nudge_unfinished,
)
from orbweaver.workspace import LocalWorkspace

_ids = count(1)


# ── unit: the predicate ──────────────────────────────────────────────


def _todos(*statuses):
    return [{"content": f"t{i}", "status": s} for i, s in enumerate(statuses)]


def test_unfinished_todos_detects_open_work():
    assert has_unfinished_todos(_todos("pending"))
    assert has_unfinished_todos(_todos("completed", "in_progress"))
    assert not has_unfinished_todos(_todos("completed", "completed"))
    assert not has_unfinished_todos(_todos("completed", "cancelled"))
    assert not has_unfinished_todos([])


def test_missing_status_counts_as_pending():
    assert has_unfinished_todos([{"content": "t"}])


@pytest.mark.parametrize(
    "text",
    [
        "Tell me which direction you want and I'll implement it on a feature branch.",
        "Let me know if you want me to proceed.",
        "Want me to start on the Phosphor theme?",
        "Shall I open the PR?",
        "Should I bump the version too?",
        "Would you like me to land this?",
        "I can start the implementation now.",
        "Say the word and I'll begin.",
        "Ready to implement on your go.",
    ],
)
def test_deferral_phrasings(text):
    assert is_deferral(text)


@pytest.mark.parametrize(
    "text",
    [
        "Implemented the guard, pushed the branch, and opened draft PR #161.",
        "Blocked: the sandbox denies host writes, so I cannot create the venv.",
        "Tests pass: 412 passed, 0 failed. Version bumped to 0.50.8.",
        "I fixed the 400 and added a regression test for the same error class.",
    ],
)
def test_non_deferrals(text):
    assert not is_deferral(text)


def test_deferral_only_matches_the_tail():
    """A mid-message 'let me know' followed by real work is not a hand-off."""
    text = (
        "Let me know if this is wrong. " + "I then implemented the change. " * 40
        + "Pushed to the branch and opened draft PR #161."
    )
    assert not is_deferral(text)


def test_nudges_only_when_plan_open_and_message_defers():
    args = {"already_nudged": False, "is_last_round": False}
    assert should_nudge_unfinished(
        text="Want me to start?", todos=_todos("pending"), **args
    )
    # A finished plan with a polite sign-off is a legitimate stop.
    assert not should_nudge_unfinished(
        text="Want me to start?", todos=_todos("completed"), **args
    )
    # An open plan with a stated blocker is a legitimate stop.
    assert not should_nudge_unfinished(
        text="Blocked on credentials.", todos=_todos("pending"), **args
    )


def test_nudge_fires_at_most_once_and_never_on_the_last_round():
    assert not should_nudge_unfinished(
        text="Want me to start?",
        todos=_todos("pending"),
        already_nudged=True,
        is_last_round=False,
    )
    assert not should_nudge_unfinished(
        text="Want me to start?",
        todos=_todos("pending"),
        already_nudged=False,
        is_last_round=True,
    )


# ── prompt ───────────────────────────────────────────────────────────


def test_static_system_forbids_stopping_at_the_seam():
    sys_prompt = static_system()
    assert "seam between research and the work" in sys_prompt
    assert "carry it through" in sys_prompt
    assert "real blocker" in sys_prompt


# ── end-to-end: the loop ─────────────────────────────────────────────


def _text(s):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=s)], stop_reason="end_turn")


def _tool(name, inp):
    tu = SimpleNamespace(type="tool_use", id=f"tu{next(_ids)}", name=name, input=inp)
    return SimpleNamespace(content=[tu], stop_reason="tool_use")


class _Client:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return _text("done")
        return self._responses.pop(0)


def _install(monkeypatch, client):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)


def _user_text(msg):
    content = msg["content"]
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text") or "") for b in content if isinstance(b, dict))


@pytest.mark.asyncio
async def test_deferral_with_open_todos_gets_one_more_round(tmp_path, monkeypatch):
    """The regression: plan open + 'want me to start?' must not end the turn."""
    plan = _tool(
        "TodoWrite",
        {"todos": [{"id": "1", "content": "ship the theme", "status": "pending"}]},
    )
    client = _Client(
        [
            plan,
            _text("I've looked at the CSS. Want me to start on the implementation?"),
            _text("Implemented and pushed to the feature branch."),
        ]
    )
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, uuid4(), "ship the theme", ws)

    # The turn did not stop at the deferral; a third call carried the nudge.
    assert len(client.calls) == 3
    last_user = [m for m in client.calls[2]["messages"] if m["role"] == "user"][-1]
    assert UNFINISHED_PLAN_NUDGE in _user_text(last_user)
    assert events[-1].payload["text"] == "Implemented and pushed to the feature branch."


@pytest.mark.asyncio
async def test_nudge_fires_once_then_the_turn_may_end(tmp_path, monkeypatch):
    """A second deferral is respected -- the guard must not trap the user."""
    plan = _tool(
        "TodoWrite",
        {"todos": [{"id": "1", "content": "ship the theme", "status": "pending"}]},
    )
    client = _Client(
        [plan, _text("Want me to start?"), _text("Shall I proceed?")]
    )
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, uuid4(), "ship the theme", ws)

    assert len(client.calls) == 3
    assert events[-1].payload["text"] == "Shall I proceed?"


@pytest.mark.asyncio
async def test_completed_plan_ends_the_turn_immediately(tmp_path, monkeypatch):
    plan = _tool(
        "TodoWrite",
        {"todos": [{"id": "1", "content": "ship the theme", "status": "completed"}]},
    )
    client = _Client([plan, _text("Done. Let me know if you want changes.")])
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, uuid4(), "ship the theme", ws)

    assert len(client.calls) == 2
    assert events[-1].payload["text"] == "Done. Let me know if you want changes."


@pytest.mark.asyncio
async def test_no_todo_list_never_nudges(tmp_path, monkeypatch):
    """Sessions that never call TodoWrite keep the old behaviour exactly."""
    client = _Client([_text("Want me to start?")])
    _install(monkeypatch, client)
    store = reset_store_for_tests()
    ws = LocalWorkspace("workspace:default", str(tmp_path))

    events = await agent_turn(store, uuid4(), "hello", ws)

    assert len(client.calls) == 1
    assert events[-1].payload["text"] == "Want me to start?"
