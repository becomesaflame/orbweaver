"""Prompt projection: compact boundary, live window, microcompact. Does not mutate stored events."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from orbweaver.config import settings
from orbweaver.image import image_read_tool_content, parse_image_read_payload, user_image_blocks
from orbweaver.store import Event
from orbweaver.tokens import estimate_tokens

log = logging.getLogger(__name__)

BOUNDARY_KINDS = frozenset({"compact_boundary", "compact_summary"})
COMPACTABLE_TOOLS = frozenset({"Bash", "Read", "Grep", "Glob", "WebFetch", "WebSearch", "Browser"})
PAIR_KINDS = frozenset(
    {
        "tool_call",
        "tool_result",
        "MemoryRecall",
        "permission_decision",
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


def microcompact_events(events: list[Event]) -> list[Event]:
    """Stub old compactable tool results in a copied list. Store rows stay intact."""
    keep_n = max(1, int(settings.compact_micro_keep))
    compactable_ids: list[object] = []
    for ev in events:
        if ev.kind not in {"tool_result", "MemoryRecall"}:
            continue
        name = str(ev.payload.get("name") or "")
        if name in COMPACTABLE_TOOLS:
            compactable_ids.append(ev.id)
    drop = set(compactable_ids[:-keep_n]) if len(compactable_ids) > keep_n else set()
    if not drop:
        return list(events)
    out: list[Event] = []
    for ev in events:
        if ev.id not in drop:
            out.append(ev)
            continue
        path = ev.payload.get("persisted_path")
        stub = "[Old tool result content cleared]"
        if path:
            stub = f"{stub} Full output: {path}"
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


def events_to_messages(events: list[Event]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_tool: list[dict[str, Any]] = []
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
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": p.get("tool_use_id") or p.get("id") or "unknown",
                            "content": content,
                        }
                    ],
                }
            )
        elif k in STOP_KINDS:
            _flush_pending_tools(messages, pending_tool, stub_results=True)
        elif k == "UserCorrection":
            _flush_pending_tools(messages, pending_tool, stub_results=True)
            messages.append(
                {
                    "role": "user",
                    "content": f"User correction: {p.get('text') or json.dumps(p)}",
                }
            )
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
