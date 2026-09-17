"""Whether a session's last turn finished, was stopped, or is waiting on the user.

The web UI offers a **Continue** button when a session's last turn ended without
a final answer, so a turn killed mid-flight (gateway restart, crash, user Stop)
can be resumed. Deciding that from "is the very last event an ``assistant``?"
is wrong: a completed turn keeps appending bookkeeping events *after* the final
assistant message. ``settle_children`` records ``subagent_result`` /
``subagent_cancelled`` in ``agent_turn``'s ``finally`` block, and the cron
channel appends ``cron_result``. Those trailing events made a finished turn look
stopped, so the button appeared on healthy chats on every reload or chat switch.

So scan backwards past the bookkeeping kinds and classify the first event that
actually concludes (or fails to conclude) a turn. A turn that really died
mid-round still ends on ``tool_call`` / ``tool_result`` / ``user`` and stays
resumable.
"""

from __future__ import annotations

from orbweaver.store import Event

__all__ = [
    "BOOKKEEPING_KINDS",
    "CONCLUDING_KINDS",
    "STATUS_OK",
    "STATUS_STOPPED",
    "STATUS_WAITING_ASK",
    "session_turn_status",
]

STATUS_OK = "ok"
"""The last turn produced a final answer (or a cron/abort summary)."""

STATUS_STOPPED = "stopped"
"""The last turn ended without a final answer: offer Continue."""

STATUS_WAITING_ASK = "waiting_ask"
"""The agent called AskUser and the answer never arrived."""

# Kinds that end a turn with something the user can read as the outcome.
# ``turn_aborted`` is always followed by an assistant message carrying the same
# text, but treat it as concluding on its own so ordering cannot matter.
CONCLUDING_KINDS = frozenset({"assistant", "cron_result", "turn_aborted"})

# Trailing bookkeeping. None of these mean the model did or did not answer, so
# they must not decide the button. Subagent and todo/compact records are the
# ones that actually land after the final assistant message in production.
BOOKKEEPING_KINDS = frozenset(
    {
        "subagent_started",
        "subagent_finished",
        "subagent_result",
        "subagent_cancelled",
        "subagent_event",
        "todo_state",
        "compact_boundary",
        "compact_summary",
        "checkpoint_restore",
        "permission_decision",
        "permission_request",
        "permission_response",
        "permission_rule_added",
        "injection_warning",
        "patch_proposal",
        "schedule_request",
        "stuck_nudge",
        "assistant_delta",
        "tool_use_progress",
    }
)


def pending_ask(events: list[Event]) -> bool:
    """Did an ``AskUser`` tool call never get its answer?

    Thin wrapper over :func:`orbweaver.agent.pending_ask_user` so this module
    has no import cycle at module load and the two never disagree.
    """
    from orbweaver.agent import pending_ask_user

    return pending_ask_user(events) is not None


def session_turn_status(events: list[Event]) -> str:
    """Classify the session's last turn as ok / stopped / waiting_ask.

    An empty session is ``ok``: there is no turn to resume.
    """
    if not events:
        return STATUS_OK
    if pending_ask(events):
        return STATUS_WAITING_ASK
    for ev in reversed(events):
        if ev.kind in BOOKKEEPING_KINDS:
            continue
        if ev.kind in CONCLUDING_KINDS:
            return STATUS_OK
        # turn_interrupted (user Stop) or a turn that died mid-round on a
        # tool_call / tool_result / user event: resumable.
        return STATUS_STOPPED
    # Nothing but bookkeeping: no turn ran, so nothing to continue.
    return STATUS_OK
