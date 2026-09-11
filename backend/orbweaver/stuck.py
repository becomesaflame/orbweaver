"""Loop detection for the agent turn (issue #103).

After each tool round ``agent_turn`` hands the session events to a
:class:`StuckDetector`. It looks at the tool calls made since the last real
user message and recognises four unproductive patterns, modelled on OpenHands'
``stuck_detector.py``:

1. **repeat** — the same tool with identical input returned the identical
   result N times in a row.
2. **error** — the same tool with identical input failed N times in a row.
3. **alternate** — two different call/result pairs alternating A,B,A,B,...
   for at least ``stuck_alternating_threshold`` calls.
4. **monologue** — N assistant messages in a row without a tool call.

The first time a streak reaches its threshold the detector asks for a
*nudge*: a one-time user-side message naming the tool and the count. When the
same streak continues past the nudge the detector asks to *abort* the turn
(``turn_aborted`` with ``reason: "stuck"``). A monologue aborts directly.

Comparison ignores ids, seq and timestamps: only ``tool_call.name`` +
``input`` and ``tool_result.content`` (as stored, i.e. after redaction and
persist) matter.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from orbweaver.config import settings
from orbweaver.store import Event

NUDGE_KIND = "stuck_nudge"
"""Event kind recorded (with the nudge text) when a nudge is injected."""

STUCK_REASON = "stuck"
"""``turn_aborted.reason`` when the detector ends the turn."""

RESULT_KINDS = frozenset({"tool_result", "MemoryRecall"})
# Bookkeeping kinds that sit between assistant messages without meaning the
# model did anything; they do not break a monologue.
_TRANSPARENT_KINDS = frozenset(
    {"permission_decision", "injection_warning", "compact_boundary", "compact_summary", NUDGE_KIND}
)
# persist_tool_result embeds the tool_use_id in the stored preview; identical
# outputs must still compare equal.
_PERSISTED_PATH_RE = re.compile(r"\.orbweaver/tool-results/[^\s]+\.txt")
_ERROR_PREFIXES = (
    "error",
    "blocked by permission gate",
    "traceback (most recent call last)",
    "this action needs user approval",
)
_ERROR_SNIPPET_CHARS = 300


def is_stuck_nudge_event(ev: Event) -> bool:
    """User-side event that carries a stuck nudge rather than a real user message."""
    return ev.kind == "user" and bool((ev.payload or {}).get(NUDGE_KIND))


def looks_like_error(content: str) -> bool:
    """Heuristic: does a tool result read like a failure?"""
    text = (content or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith(_ERROR_PREFIXES):
        return True
    if "\ntraceback (most recent call last)" in low:
        return True
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict) and bool(parsed.get("error"))
    return False


def normalize_result(content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
    return _PERSISTED_PATH_RE.sub(".orbweaver/tool-results/<id>.txt", text)


def canonical_input(inp: Any) -> str:
    return json.dumps(inp or {}, sort_keys=True, default=str)


def _digest(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass(frozen=True)
class Action:
    """One tool call with its (normalized) result."""

    name: str
    input_key: str
    result: str

    @property
    def call_key(self) -> str:
        return _digest(self.name, self.input_key)

    @property
    def pair_key(self) -> str:
        return _digest(self.name, self.input_key, self.result)

    @property
    def error(self) -> bool:
        return looks_like_error(self.result)


@dataclass(frozen=True)
class _Streak:
    pattern: str
    count: int
    threshold: int
    ref: Action
    signature: str
    label: str  # tool name(s) for the message


@dataclass(frozen=True)
class StuckVerdict:
    pattern: str  # repeat | error | alternate | monologue
    action: str  # nudge | abort
    tool: str
    count: int
    signature: str
    text: str
    error_text: str = ""

    def nudge_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pattern": self.pattern,
            "tool": self.tool,
            "count": self.count,
            "signature": self.signature,
            "text": self.text,
        }
        if self.error_text:
            payload["error"] = self.error_text
        return payload

    def abort_payload(self) -> dict[str, Any]:
        return {
            "reason": STUCK_REASON,
            "pattern": self.pattern,
            "last_tool": self.tool,
            "count": self.count,
            "text": self.text,
        }


def events_since_user(events: list[Event]) -> list[Event]:
    """Events after the last real user message (stuck nudges do not reset the window)."""
    for i in range(len(events) - 1, -1, -1):
        ev = events[i]
        if ev.kind == "user" and not is_stuck_nudge_event(ev):
            return events[i + 1 :]
    return list(events)


def collect_actions(events: list[Event]) -> list[Action]:
    """tool_call events paired with their result by tool_use_id, in call order."""
    calls: list[tuple[str, str, str]] = []
    results: dict[str, str] = {}
    for ev in events:
        p = ev.payload or {}
        if ev.kind == "tool_call":
            uid = str(p.get("id") or ev.id)
            calls.append((uid, str(p.get("name") or ""), canonical_input(p.get("input"))))
        elif ev.kind in RESULT_KINDS:
            uid = str(p.get("tool_use_id") or p.get("id") or "")
            if uid:
                content = p.get("content")
                if content is None:
                    content = p.get("text") or ""
                results[uid] = normalize_result(content)
    out: list[Action] = []
    for uid, name, input_key in calls:
        if uid in results:
            out.append(Action(name=name, input_key=input_key, result=results[uid]))
    return out


def nudged_signatures(events: list[Event]) -> set[str]:
    return {
        str((ev.payload or {}).get("signature") or "")
        for ev in events
        if ev.kind == NUDGE_KIND
    }


class StuckDetector:
    """Stateless scanner: every check re-derives the streak from stored events."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        repeat_threshold: int | None = None,
        alternating_threshold: int | None = None,
        window: int | None = None,
    ) -> None:
        self.enabled = settings.stuck_detection if enabled is None else enabled
        self.repeat_threshold = max(
            2, int(settings.stuck_repeat_threshold if repeat_threshold is None else repeat_threshold)
        )
        self.alternating_threshold = max(
            4,
            int(
                settings.stuck_alternating_threshold
                if alternating_threshold is None
                else alternating_threshold
            ),
        )
        self.window = max(
            self.alternating_threshold + 1,
            int(settings.stuck_window if window is None else window),
        )

    # -- public -----------------------------------------------------------

    def check(self, events: list[Event]) -> StuckVerdict | None:
        if not self.enabled:
            return None
        window = events_since_user(events)
        if not window:
            return None
        monologue = self._monologue(window)
        if monologue is not None:
            return monologue
        actions = collect_actions(window)[-self.window :]
        if len(actions) < self.repeat_threshold:
            return None
        nudged = nudged_signatures(window)
        for finder in (self._error_streak, self._repeat_streak, self._alternating_streak):
            found = finder(actions)
            if found is None:
                continue
            verdict = self._decide(found, nudged)
            if verdict is not None:
                return verdict
        return None

    # -- streaks ----------------------------------------------------------

    def _repeat_streak(self, actions: list[Action]) -> _Streak | None:
        ref = actions[-1]
        count = 0
        for act in reversed(actions):
            if act.pair_key != ref.pair_key:
                break
            count += 1
        if count < self.repeat_threshold:
            return None
        return _Streak(
            "repeat", count, self.repeat_threshold, ref, _digest("repeat", ref.pair_key), ref.name
        )

    def _error_streak(self, actions: list[Action]) -> _Streak | None:
        ref = actions[-1]
        if not ref.error:
            return None
        count = 0
        for act in reversed(actions):
            if act.call_key != ref.call_key or not act.error:
                break
            count += 1
        if count < self.repeat_threshold:
            return None
        return _Streak(
            "error", count, self.repeat_threshold, ref, _digest("error", ref.call_key), ref.name
        )

    def _alternating_streak(self, actions: list[Action]) -> _Streak | None:
        if len(actions) < self.alternating_threshold:
            return None
        last, prev = actions[-1], actions[-2]
        if last.pair_key == prev.pair_key:
            return None  # that is a repeat, not an alternation
        count = 2
        for i in range(len(actions) - 3, -1, -1):
            if actions[i].pair_key != actions[i + 2].pair_key:
                break
            count += 1
        if count < self.alternating_threshold:
            return None
        sig = _digest("alternate", *sorted((last.pair_key, prev.pair_key)))
        label = last.name if last.name == prev.name else f"{prev.name}`/`{last.name}"
        return _Streak("alternate", count, self.alternating_threshold, last, sig, label)

    def _monologue(self, window: list[Event]) -> StuckVerdict | None:
        count = 0
        for ev in reversed(window):
            if ev.kind == "assistant":
                count += 1
            elif ev.kind in _TRANSPARENT_KINDS:
                continue
            else:
                break
        if count < self.repeat_threshold:
            return None
        return StuckVerdict(
            pattern="monologue",
            action="abort",
            tool="",
            count=count,
            signature=_digest("monologue"),
            text=(
                f"Stopped this turn: {count} assistant messages in a row without a tool call "
                "or a user reply. Continue in a follow-up with a concrete next step."
            ),
        )

    # -- decision ---------------------------------------------------------

    def _decide(self, streak: _Streak, nudged: set[str]) -> StuckVerdict | None:
        error_text = ""
        if streak.pattern == "error":
            error_text = streak.ref.result.strip()[:_ERROR_SNIPPET_CHARS]
        if streak.signature in nudged:
            if streak.count <= streak.threshold:
                return None  # nudged already; the streak has not grown yet
            return StuckVerdict(
                pattern=streak.pattern,
                action="abort",
                tool=streak.ref.name,
                count=streak.count,
                signature=streak.signature,
                text=self._abort_text(streak, error_text),
                error_text=error_text,
            )
        return StuckVerdict(
            pattern=streak.pattern,
            action="nudge",
            tool=streak.ref.name,
            count=streak.count,
            signature=streak.signature,
            text=self._nudge_text(streak, error_text),
            error_text=error_text,
        )

    @staticmethod
    def _nudge_text(streak: _Streak, error_text: str) -> str:
        tool, count = streak.label, streak.count
        if streak.pattern == "error":
            return (
                f"You have called `{tool}` with the same input {count} times in a row and "
                f"it failed each time with: {error_text}. Repeating the exact same call will "
                "not work. Fix the input based on the error, or try a different approach."
            )
        if streak.pattern == "alternate":
            return (
                f"Your last {count} tool calls alternate between the same two `{tool}` "
                "calls and keep returning the same results. This is a loop. Do not repeat "
                "them; act on what you already have, or try a different approach."
            )
        return (
            f"You have called `{tool}` with the same input {count} times in a row and "
            "received the same result each time. Do not repeat it. Act on the result you "
            "already have, or try a different approach."
        )

    @staticmethod
    def _abort_text(streak: _Streak, error_text: str) -> str:
        tool, count = streak.label, streak.count
        if streak.pattern == "error":
            return (
                f"Stopped this turn: stuck in a loop. `{tool}` was called with the same "
                f"input {count} times and failed every time ({error_text}), even after a "
                "warning. Continue in a follow-up with a different approach."
            )
        if streak.pattern == "alternate":
            return (
                f"Stopped this turn: stuck in a loop. The last {count} tool calls alternated "
                f"between the same two `{tool}` calls with identical results, even after "
                "a warning. Continue in a follow-up with a different approach."
            )
        return (
            f"Stopped this turn: stuck in a loop. `{tool}` was called with the same input "
            f"{count} times and returned the same result every time, even after a warning. "
            "Continue in a follow-up with a different approach."
        )
