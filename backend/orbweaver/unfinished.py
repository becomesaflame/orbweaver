"""Detect a turn that is ending while the session's own plan says work remains.

The agent loop ends a turn when the model emits no tool calls and the provider
reports ``stop_reason == "end_turn"``. Nothing else stops it: the tool-round
budget (256) and the stuck detector almost never fire in practice. So the
common "why did it stop early?" case is the model *choosing* to hand back at a
natural-sounding seam -- after research, right before the edits -- usually with
a closing line that offers to continue.

That is a behavioural failure, and a system-prompt line alone cannot fix it: a
prompt is advice the model may round off, while the TodoWrite list is concrete
session state. When the plan still has pending or in-progress items and the
model tries to stop without saying why, we nudge once and let the loop run
another round. One nudge per turn, so a deliberate stop still gets through.
"""

from __future__ import annotations

import re

__all__ = [
    "UNFINISHED_PLAN_NUDGE",
    "has_unfinished_todos",
    "is_deferral",
    "should_nudge_unfinished",
]

UNFINISHED_PLAN_NUDGE = (
    "You are ending the turn but the session todo list still has unfinished items, "
    "and your last message offers to do the work instead of doing it. "
    "The user's request to implement or fix something is the request to carry it "
    "through -- on a feature branch, to a draft PR, per ORBWEAVER.md. "
    "Do not ask whether to start, do not ask whether to commit. "
    "Continue the next pending item now, or, if you genuinely cannot proceed, "
    "say plainly what blocks you and what decision you need."
)

# Phrasings that defer work back to the user instead of doing it. Matched only
# against the tail of the final message, where a hand-off actually lands.
_DEFERRAL_PATTERNS = (
    r"\b(?:just )?(?:let me know|tell me|say the word)\b",
    r"\bif you(?:'d| would)? (?:like|want|prefer)\b",
    r"\bwant me to\b",
    r"\bshall i\b",
    r"\bshould i\b",
    r"\bwould you like\b",
    r"\bi can (?:start|begin|implement|do|take|proceed|open|land|ship)\b",
    r"\bi'?ll (?:start|begin|implement|open|land|ship)\b[^.]{0,40}\bif\b",
    r"\bready to (?:start|begin|implement|proceed)\b",
    r"\bon your (?:go|signal|say-so)\b",
)

_DEFERRAL_RE = re.compile("|".join(_DEFERRAL_PATTERNS), re.IGNORECASE)

# Only the end of the message is a hand-off. "Let me know" mid-transcript while
# work continues afterwards is not a deferral.
_TAIL_CHARS = 400


def has_unfinished_todos(todos: list[dict[str, str]]) -> bool:
    """True when the plan still has pending or in-progress items."""
    return any(
        (t.get("status") or "pending") in ("pending", "in_progress") for t in todos
    )


def is_deferral(text: str) -> bool:
    """True when the message tail hands the next step back to the user."""
    if not text:
        return False
    return bool(_DEFERRAL_RE.search(text[-_TAIL_CHARS:]))


def should_nudge_unfinished(
    *,
    text: str,
    todos: list[dict[str, str]],
    already_nudged: bool,
    is_last_round: bool,
) -> bool:
    """Whether to push one more round instead of ending the turn.

    Requires both signals: an unfinished plan *and* a deferring closing line.
    An unfinished plan with a clear blocker stated is a legitimate stop, and so
    is a deferral with everything on the plan complete.
    """
    if already_nudged or is_last_round:
        return False
    if not has_unfinished_todos(todos):
        return False
    return is_deferral(text)
