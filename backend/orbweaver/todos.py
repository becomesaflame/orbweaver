"""Session-persisted todo list that survives compact projection."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from orbweaver.store import Event, Store

TODO_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
MAX_TODOS = 40


def normalize_todos(raw: Any) -> list[dict[str, str]]:
    items = raw if isinstance(raw, list) else []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        tid = str(item.get("id") or i + 1).strip()
        if not tid:
            tid = str(i + 1)
        if tid in seen:
            tid = f"{tid}-{i + 1}"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        status = str(item.get("status") or "pending").strip().lower()
        if status not in TODO_STATUSES:
            status = "pending"
        seen.add(tid)
        out.append({"id": tid, "content": content, "status": status})
        if len(out) >= MAX_TODOS:
            break
    return out


def merge_todos(
    existing: list[dict[str, str]], incoming: list[dict[str, str]]
) -> list[dict[str, str]]:
    by_id = {t["id"]: t for t in existing}
    order = [t["id"] for t in existing]
    for item in incoming:
        if item["id"] not in by_id:
            order.append(item["id"])
        by_id[item["id"]] = item
    return [by_id[i] for i in order if i in by_id][:MAX_TODOS]


def format_todos(todos: list[dict[str, str]]) -> str:
    if not todos:
        return "[session todos]\n(none)"
    lines = ["[session todos]"]
    for item in todos:
        lines.append(f"- [{item['status']}] {item['id']}: {item['content']}")
    return "\n".join(lines)


def latest_todo_state(events: list[Event]) -> Event | None:
    for ev in reversed(events):
        if ev.kind == "todo_state":
            return ev
    return None


def todos_from_events(events: list[Event]) -> list[dict[str, str]]:
    ev = latest_todo_state(events)
    if ev is None:
        return []
    return normalize_todos(ev.payload.get("todos"))


def inject_session_todos(
    messages: list[dict[str, Any]], events: list[Event]
) -> list[dict[str, Any]]:
    """Re-inject the latest session todos so a compact boundary cannot drop the plan."""
    todos = todos_from_events(events)
    if not todos:
        return messages
    extra = {"role": "user", "content": format_todos(todos)}
    if messages and messages[0].get("role") == "user":
        return [messages[0], extra, *messages[1:]]
    return [extra, *messages]


async def persist_todos(
    store: Store, session_id: UUID, inp: dict[str, Any]
) -> list[dict[str, str]]:
    incoming = normalize_todos(inp.get("todos"))
    sess = await store.get_entity(session_id)
    existing = normalize_todos((sess.jsonld if sess else {}).get("todos"))
    todos = merge_todos(existing, incoming) if inp.get("merge") else incoming
    if sess is not None:
        sess.jsonld["todos"] = todos
        await store.put_entity(sess)
    await store.append_event(session_id, "todo_state", {"todos": todos})
    return todos
