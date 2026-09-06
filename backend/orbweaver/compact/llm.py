"""Session notes and full-compact summaries. Original prompts; extractive fallback."""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID

from orbweaver.compact.project import events_to_messages, live_events
from orbweaver.config import settings
from orbweaver.store import Entity, Event, Store

log = logging.getLogger(__name__)

NOTES_TEMPLATE = """# Current task
# Files
# Decisions
# Errors
# Next step
"""

NOTES_PROMPT = (
    "Update the session notes below using the conversation. Keep the same headings. "
    "Prefer exact file paths, error strings, and the user's wording. "
    "Drop stale items. Stay under 800 words. Reply with the notes only — no tools, no preamble."
)

COMPACT_PROMPT = (
    "Write a briefing for the next turn of this coding session. Use only the conversation above. "
    "Do not call tools. Reply with markdown headings exactly:\n"
    "## Task\n## Files\n## Decisions\n## Errors\n## Next step\n"
    "Prefer exact paths, error strings, and user wording. Stay under 800 words. "
    "No preamble."
)

_ANALYSIS = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


def notes_are_usable(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if stripped == NOTES_TEMPLATE.strip():
        return False
    body = re.sub(r"^#+\s+\S[^\n]*$", "", stripped, flags=re.MULTILINE).strip()
    return bool(body)


def format_compact_summary(raw: str) -> str:
    text = _ANALYSIS.sub("", raw or "").strip()
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text or raw.strip()


def _text_of(resp: Any) -> str:
    parts = [
        getattr(b, "text", "")
        for b in getattr(resp, "content", [])
        if getattr(b, "type", None) == "text"
    ]
    return "".join(parts).strip()


async def _complete(
    client: Any, system: Any, messages: list[dict[str, Any]], max_tokens: int = 2048
) -> str:
    resp = await client.messages.create(
        model=settings.orbweaver_compact_model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return format_compact_summary(_text_of(resp))


def dropped_events(all_events: list[Event], keep_from_seq: int) -> list[Event]:
    return [e for e in all_events if e.seq < keep_from_seq]


async def update_session_notes(
    store: Store,
    session_id: UUID,
    events: list[Event],
    client: Any | None,
    system: Any,
) -> str | None:
    entity = await store.get_entity(session_id)
    previous = ""
    if entity is not None:
        previous = str(entity.jsonld.get("session_notes") or "")
    if client is None or not settings.anthropic_api_key:
        return previous if notes_are_usable(previous) else None
    digest_events = live_events(events)
    messages = events_to_messages(digest_events[-80:])
    if not messages:
        return previous if notes_are_usable(previous) else None
    prior = previous if notes_are_usable(previous) else NOTES_TEMPLATE
    messages = [
        *messages,
        {
            "role": "user",
            "content": f"{NOTES_PROMPT}\n\nCurrent notes:\n{prior}",
        },
    ]
    try:
        text = await _complete(client, system, messages)
    except Exception as e:
        log.warning("session notes update failed: %s", e)
        raise
    if not notes_are_usable(text):
        return previous if notes_are_usable(previous) else None
    text = text[: int(settings.compact_notes_max_chars)]
    if entity is not None:
        entity.jsonld["session_notes"] = text
        last = events[-1].seq if events else 0
        entity.jsonld["session_notes_until_seq"] = last
        await store.put_entity(entity)
    return text


async def llm_summary(
    events: list[Event],
    client: Any,
    system: Any,
) -> str:
    messages = events_to_messages(events)
    if not messages:
        raise ValueError("no messages to summarize")
    payload = [*messages, {"role": "user", "content": COMPACT_PROMPT}]
    last_error: Exception | None = None
    for _ in range(3):
        try:
            text = await _complete(client, system, payload, max_tokens=4096)
            if text:
                return text
            raise ValueError("empty compact summary")
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if "prompt is too long" in err or "too many tokens" in err or "too long" in err:
                cut = max(1, len(payload) // 4)
                head, instruction = payload[:-1], payload[-1]
                payload = [*head[cut:], instruction]
                if len(payload) < 2:
                    break
                continue
            raise
    if last_error:
        raise last_error
    raise ValueError("compact summary failed")


def session_entity_notes(entity: Entity | None) -> str | None:
    if entity is None:
        return None
    text = str(entity.jsonld.get("session_notes") or "")
    return text if notes_are_usable(text) else None
