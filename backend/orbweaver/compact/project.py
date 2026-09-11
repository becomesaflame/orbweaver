"""Prompt projection: compact boundary, live window, microcompact. Does not mutate stored events."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Any

from orbweaver.config import settings
from orbweaver.image import image_read_tool_content, parse_image_read_payload, user_image_blocks
from orbweaver.permissions.prompts import INJECTION_WARNING
from orbweaver.store import Event
from orbweaver.tokens import estimate_tokens

log = logging.getLogger(__name__)

BOUNDARY_KINDS = frozenset({"compact_boundary", "compact_summary"})
# Nested AGENTS.md / CLAUDE.md / .cursor/rules discovered mid-turn; rendered on the user
# side after the tool results of the round that touched the directory.
PROJECT_INSTRUCTIONS_KIND = "project_instructions"
COMPACTABLE_TOOLS = frozenset({"Bash", "Read", "Grep", "Glob", "WebFetch", "WebSearch", "Browser"})
# Tools that change a file in place; a Read of that path is kept while it is being edited.
EDIT_TOOLS = frozenset({"Write", "StrReplace"})
MICRO_STUB = "[Old tool result content cleared]"
# Results below this many chars are not worth a prefix-cache miss to stub.
MICRO_SMALL_RESULT_CHARS = 1000
PAIR_KINDS = frozenset(
    {
        "tool_call",
        "tool_result",
        "MemoryRecall",
        "permission_decision",
        "permission_request",
        "permission_response",
        "permission_rule_added",
        "injection_warning",
        "patch_proposal",
        "schedule_request",
        "ask_user",
    }
)


def last_boundary(events: list[Event]) -> Event | None:
    for ev in reversed(events):
        if ev.kind in BOUNDARY_KINDS:
            return ev
    return None


def live_events(events: list[Event]) -> list[Event]:
    """Events the model should see: last compact summary plus the kept tail and anything after."""
    boundary = last_boundary(events)
    if boundary is None:
        return list(events)
    keep_from = int(boundary.payload.get("keep_from_seq") or (boundary.seq + 1))
    tail = [
        e
        for e in events
        if e.seq >= keep_from and e.id != boundary.id and e.kind not in BOUNDARY_KINDS
    ]
    return [boundary, *tail]


def align_keep_index(events: list[Event], idx: int) -> int:
    """Do not split a tool_use / tool_result group."""
    if not events:
        return 0
    idx = max(0, min(idx, len(events) - 1))
    while idx > 0 and events[idx].kind in PAIR_KINDS and events[idx].kind != "tool_call":
        idx -= 1
    while idx > 0 and events[idx].kind == "tool_call" and events[idx - 1].kind == "tool_call":
        idx -= 1
    return idx


def event_token_count(events: list[Event]) -> int:
    from orbweaver.compact.usage import event_token_count as _count

    return _count(events)


def choose_keep_from_seq(live: list[Event], summary: str, budget: int) -> int:
    if not live:
        return 1
    body = [e for e in live if e.kind not in BOUNDARY_KINDS]
    if not body:
        return live[-1].seq
    summary_tokens = estimate_tokens(summary)
    keep_idx = len(body) - 1
    acc = event_token_count([body[-1]])
    for i in range(len(body) - 2, -1, -1):
        piece = event_token_count([body[i]])
        if summary_tokens + acc + piece > budget:
            break
        acc += piece
        keep_idx = i
    keep_idx = align_keep_index(body, keep_idx)
    return body[keep_idx].seq


def choose_keep_from_recent_rounds(live: list[Event], n_rounds: int) -> int:
    """Keep the last n_rounds tool groups (0 = current user turn only)."""
    if not live:
        return 1
    body = [e for e in live if e.kind not in BOUNDARY_KINDS]
    if not body:
        return live[-1].seq
    starts: list[int] = []
    i = 0
    while i < len(body):
        if body[i].kind == "tool_call":
            starts.append(i)
            i += 1
            while i < len(body) and body[i].kind == "tool_call":
                i += 1
            while i < len(body) and body[i].kind in PAIR_KINDS and body[i].kind != "tool_call":
                i += 1
            continue
        i += 1
    if not starts:
        n = 1 if n_rounds <= 0 else n_rounds
        keep_idx = max(0, len(body) - n)
        keep_idx = align_keep_index(body, keep_idx)
        return body[keep_idx].seq
    if n_rounds <= 0:
        keep_idx = len(body) - 1
        for j in range(len(body) - 1, -1, -1):
            if body[j].kind == "user":
                keep_idx = j
                break
        keep_idx = align_keep_index(body, keep_idx)
        return body[keep_idx].seq
    start = starts[-n_rounds] if n_rounds <= len(starts) else starts[0]
    keep_idx = align_keep_index(body, start)
    return body[keep_idx].seq


@dataclass
class _MicroCandidate:
    ev: Event
    size: int
    round_no: int
    protected: bool


def _result_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if content is None or content == "":
        content = payload.get("text") or ""
    return content if isinstance(content, str) else str(content)


def _norm_path(raw: Any) -> str:
    text = str(raw or "").strip()
    return os.path.normpath(text) if text else ""


def _micro_stub(payload: dict[str, Any]) -> str:
    stub = MICRO_STUB
    path = payload.get("persisted_path")
    if path:
        stub = f"{stub} Full output: {path}"
    return stub


def _micro_candidates(events: list[Event]) -> tuple[list[_MicroCandidate], int]:
    """Compactable results with size, tool-round index, and edited-since-Read protection."""
    call_info: dict[str, tuple[str, str]] = {}
    edits: list[tuple[int, str]] = []
    reads: list[tuple[int, str, _MicroCandidate]] = []
    candidates: list[_MicroCandidate] = []
    round_no = 0
    prev_kind = ""
    for idx, ev in enumerate(events):
        p = ev.payload
        if ev.kind == "tool_call":
            if prev_kind != "tool_call":
                round_no += 1
            name = str(p.get("name") or "")
            path = _norm_path((p.get("input") or {}).get("path"))
            call_info[str(p.get("id") or ev.id)] = (name, path)
            if name in EDIT_TOOLS and path:
                edits.append((idx, path))
        elif ev.kind in {"tool_result", "MemoryRecall"}:
            name = str(p.get("name") or "")
            if name in COMPACTABLE_TOOLS:
                cand = _MicroCandidate(
                    ev=ev,
                    size=len(_result_text(p)),
                    round_no=round_no,
                    protected=False,
                )
                candidates.append(cand)
                if name == "Read":
                    _, path = call_info.get(str(p.get("tool_use_id") or ""), ("", ""))
                    if path:
                        reads.append((idx, path, cand))
        prev_kind = ev.kind
    for idx, path, cand in reads:
        if any(e_idx > idx and e_path == path for e_idx, e_path in edits):
            cand.protected = True
    return candidates, round_no


def microcompact_events(events: list[Event]) -> list[Event]:
    """Stub old compactable tool results in a copied list. Store rows stay intact.

    Runs only under pressure. The decision is a pure function of the event
    list (payload token estimate vs ``event_budget``), never of the API usage
    anchor: the anchor measures the already-stubbed prompt, so gating on it
    would restore and re-clear the same results on alternate rounds and the
    Anthropic prefix cache would never hit.

    Rules, in order:

    * The newest ``compact_micro_keep`` compactable results are never stubbed.
    * A ``Read`` whose path was later edited by Write/StrReplace in this window
      is never stubbed; the model is working on that file.
    * Age rule: a result older than ``compact_micro_stale_rounds`` tool rounds
      and larger than ``compact_micro_stale_chars`` is stubbed regardless of
      pressure.
    * Pressure rule: when the payload estimate exceeds
      ``compact_micro_pressure * event_budget``, clear the oldest large results
      first until the estimate is back under the line. The amount cleared is
      rounded up to whole chunks of ``(pressure - release) * event_budget`` so
      the cleared set, and therefore the message prefix, stays identical across
      the rounds it takes to grow another chunk.

    Below the pressure line nothing is stubbed, so round N+1's messages are
    round N's plus the new assistant/tool blocks.
    """
    candidates, current_round = _micro_candidates(events)
    if not candidates:
        return list(events)
    keep_n = max(1, int(settings.compact_micro_keep))
    floor_ids = {c.ev.id for c in candidates[-keep_n:]}
    eligible = [c for c in candidates if c.ev.id not in floor_ids and not c.protected]
    if not eligible:
        return list(events)

    cleared: set[object] = set()
    stale_rounds = int(settings.compact_micro_stale_rounds)
    stale_chars = max(0, int(settings.compact_micro_stale_chars))
    if stale_rounds > 0:
        for c in eligible:
            if current_round - c.round_no > stale_rounds and c.size >= stale_chars:
                cleared.add(c.ev.id)

    def saving(c: _MicroCandidate) -> int:
        before = estimate_tokens(_result_text(c.ev.payload))
        return max(0, before - estimate_tokens(_micro_stub(c.ev.payload)))

    budget = int(settings.event_budget)
    pressure = float(settings.compact_micro_pressure)
    release = float(settings.compact_micro_release)
    line = budget * pressure
    total = event_token_count(events) - sum(saving(c) for c in eligible if c.ev.id in cleared)
    if total > line:
        needed = total - line
        chunk = (pressure - release) * budget if release < pressure else 0.0
        goal = math.ceil(needed / chunk) * chunk if chunk >= 1 else needed
        saved = 0
        for c in eligible:
            if saved >= goal:
                break
            if c.ev.id in cleared or c.size < MICRO_SMALL_RESULT_CHARS:
                continue
            cleared.add(c.ev.id)
            saved += saving(c)

    if not cleared:
        return list(events)
    out: list[Event] = []
    for ev in events:
        if ev.id not in cleared:
            out.append(ev)
            continue
        stub = _micro_stub(ev.payload)
        payload = dict(ev.payload)
        payload["content"] = stub
        if "text" in payload:
            payload["text"] = stub
        out.append(replace(ev, payload=payload))
    return out


def prompt_events(events: list[Event]) -> list[Event]:
    return microcompact_events(live_events(events))


def _user_message_content(payload: dict[str, Any]) -> str | list[dict[str, Any]]:
    text = payload.get("text") or payload.get("content") or ""
    images = payload.get("images") or []
    if not images:
        return text
    blocks = user_image_blocks(images)
    if text:
        blocks.append({"type": "text", "text": str(text)})
    return blocks or str(text)


INTERRUPTED_TOOL = "Tool was interrupted before a result was recorded."
STOP_KINDS = frozenset({"turn_interrupted", "turn_aborted"})


def _flush_pending_tools(
    messages: list[dict[str, Any]],
    pending_tool: list[dict[str, Any]],
    *,
    stub_results: bool,
) -> None:
    if not pending_tool:
        return
    messages.append({"role": "assistant", "content": list(pending_tool)})
    if stub_results:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": INTERRUPTED_TOOL,
                        "is_error": True,
                    }
                    for block in pending_tool
                ],
            }
        )
    pending_tool.clear()


def _append_user_content(messages: list[dict[str, Any]], content: str | list[dict[str, Any]]) -> None:
    if not (messages and messages[-1]["role"] == "user"):
        messages.append({"role": "user", "content": content})
        return
    prev = messages[-1]["content"]
    if isinstance(content, list):
        if isinstance(prev, str):
            messages[-1]["content"] = [{"type": "text", "text": prev}, *content]
        elif isinstance(prev, list):
            prev.extend(content)
        else:
            messages.append({"role": "user", "content": content})
        return
    if isinstance(prev, str):
        messages[-1]["content"] = prev + "\n\n" + content
    elif isinstance(prev, list):
        prev.append({"type": "text", "text": content})
    else:
        messages.append({"role": "user", "content": content})


def _content_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


def _tool_use_blocks(msg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not msg or msg.get("role") != "assistant":
        return []
    return [
        b
        for b in _content_blocks(msg.get("content"))
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
    ]


def _tool_result_id_set(msg: dict[str, Any] | None) -> set[str]:
    if not msg or msg.get("role") != "user":
        return set()
    return {
        str(b.get("tool_use_id"))
        for b in _content_blocks(msg.get("content"))
        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")
    }


def _is_tool_result_message(msg: dict[str, Any] | None) -> bool:
    """User message made only of tool_result blocks (results of one assistant round)."""
    if not msg or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _stub_tool_result(tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": INTERRUPTED_TOOL,
        "is_error": True,
    }


def unpaired_tool_use_ids(messages: list[dict[str, Any]]) -> list[str]:
    """Ids of assistant tool_use blocks not covered by the next user tool_result message."""
    missing: list[str] = []
    for i, msg in enumerate(messages):
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        have = _tool_result_id_set(nxt)
        for block in _tool_use_blocks(msg):
            uid = str(block["id"])
            if uid not in have:
                missing.append(uid)
    return missing


def ensure_tool_use_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Guarantee each assistant tool_use is followed by matching tool_result blocks.

    Anthropic rejects a messages.create where a tool_use has no tool_result in the
    immediately following user message. Stub missing results rather than send that.
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        uses = _tool_use_blocks(msg)
        out.append(msg)
        if not uses:
            i += 1
            continue
        needed = [str(b["id"]) for b in uses]
        nxt = messages[i + 1] if i + 1 < n else None
        have = _tool_result_id_set(nxt)
        missing = [uid for uid in needed if uid not in have]
        if not missing:
            i += 1
            continue
        log.warning("stubbing unpaired tool_use before LLM call: %s", missing)
        stubs = [_stub_tool_result(uid) for uid in missing]
        if nxt is not None and nxt.get("role") == "user":
            merged = dict(nxt)
            rest = _content_blocks(nxt.get("content"))
            merged["content"] = stubs + rest
            out.append(merged)
            i += 2
            continue
        out.append({"role": "user", "content": stubs})
        i += 1
    return out


def flagged_tool_use_ids(events: list[Event]) -> set[str]:
    """tool_use_ids with an ``injection_warning`` event anywhere in the list.

    The probe runs off the critical path, so the warning event may land several
    events after its result (or in a later round). Rendering attaches it to the
    matching ``tool_result`` regardless of where it landed.
    """
    return {
        str(ev.payload.get("tool_use_id"))
        for ev in events
        if ev.kind == "injection_warning" and ev.payload.get("tool_use_id")
    }


def _with_injection_warning(content: Any) -> Any:
    if isinstance(content, str):
        return INJECTION_WARNING + content
    if isinstance(content, list):
        return [{"type": "text", "text": INJECTION_WARNING.strip()}, *content]
    return content


def events_to_messages(events: list[Event]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_tool: list[dict[str, Any]] = []
    flagged = flagged_tool_use_ids(events)
    for ev in events:
        k = ev.kind
        p = ev.payload
        if k == "user":
            _flush_pending_tools(messages, pending_tool, stub_results=True)
            _append_user_content(messages, _user_message_content(p))
        elif k == "assistant":
            _flush_pending_tools(messages, pending_tool, stub_results=True)
            messages.append({"role": "assistant", "content": p.get("text") or ""})
        elif k == "tool_call":
            pending_tool.append(
                {
                    "type": "tool_use",
                    "id": p.get("id") or str(ev.id),
                    "name": p.get("name"),
                    "input": p.get("input") or {},
                }
            )
        elif k in {"tool_result", "MemoryRecall"}:
            if pending_tool:
                messages.append({"role": "assistant", "content": list(pending_tool)})
                pending_tool.clear()
            content = p.get("content") or p.get("text") or json.dumps(p)[:8000]
            image_payload = parse_image_read_payload(content)
            if image_payload is None:
                image_payload = parse_image_read_payload(p)
            if image_payload is not None:
                content = image_read_tool_content(image_payload)
            elif p.get("images"):
                blocks = user_image_blocks(p.get("images") or [])
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                content = blocks or content
            tool_use_id = p.get("tool_use_id") or p.get("id") or "unknown"
            if flagged and str(tool_use_id) in flagged:
                content = _with_injection_warning(content)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
            if p.get("is_error"):
                block["is_error"] = True
            # A parallel round records tool_call, tool_call, result, result: all
            # results for one assistant message belong in one user message.
            if _is_tool_result_message(messages[-1] if messages else None):
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
        elif k in STOP_KINDS:
            _flush_pending_tools(messages, pending_tool, stub_results=True)
        elif k == PROJECT_INSTRUCTIONS_KIND:
            _flush_pending_tools(messages, pending_tool, stub_results=True)
            text = str(p.get("text") or "")
            if text:
                _append_user_content(messages, text)
        elif k == "UserCorrection":
            _flush_pending_tools(messages, pending_tool, stub_results=True)
            messages.append(
                {
                    "role": "user",
                    "content": f"User correction: {p.get('text') or json.dumps(p)}",
                }
            )
        elif k == "subagent_result":
            # Background child finished: user-side note, like an injected follow-up.
            from orbweaver.subagent import format_subagent_result

            _flush_pending_tools(messages, pending_tool, stub_results=True)
            _append_user_content(messages, format_subagent_result(p))
        elif k in BOUNDARY_KINDS:
            _flush_pending_tools(messages, pending_tool, stub_results=True)
            messages.append(
                {"role": "user", "content": f"[compacted earlier turns]\n{p.get('text') or ''}"}
            )
    _flush_pending_tools(messages, pending_tool, stub_results=True)
    return ensure_tool_use_results(messages)


def extractive_summary(dropped: list[Event]) -> str:
    sample = dropped[-40:]
    lines = ["Summary of earlier events:"]
    for e in sample:
        lines.append(f"- {e.kind}: {json.dumps(e.payload, default=str)[:240]}")
    return "\n".join(lines)
