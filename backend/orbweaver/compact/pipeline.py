"""Compaction pipeline: keep the event log, shrink the model prompt."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from orbweaver.compact.llm import llm_summary, session_entity_notes, update_session_notes
from orbweaver.compact.project import (
    BOUNDARY_KINDS,
    choose_keep_from_seq,
    extractive_summary,
    live_events,
    prompt_events,
)
from orbweaver.compact.usage import (
    clear_usage,
    compact_failures,
    event_token_count,
    record_compact_failure,
    record_compact_success,
)
from orbweaver.config import settings
from orbweaver.memory import remember
from orbweaver.store import Event, Store
from orbweaver.tokens import estimate_tokens

log = logging.getLogger(__name__)

SKIP_SOURCES = frozenset({"compact", "session_notes"})


async def maybe_compact(
    store: Store,
    session_id: UUID,
    *,
    client: Any | None = None,
    workspace: Any | None = None,
    system: Any | None = None,
    source: str = "turn",
) -> Event | None:
    """Append a compact_boundary if the live prompt window is over budget.

    Never deletes session events. Recursion sources (compact, session_notes) no-op.
    """
    del workspace  # persist/rehydrate happen at ingest / prompt build
    if source in SKIP_SOURCES:
        return None
    events = await store.list_events(session_id)
    projected = prompt_events(events)
    budget = int(settings.event_budget * settings.compact_ratio)
    if event_token_count(projected) <= budget:
        return None

    live = live_events(events)
    live_body = [e for e in live if e.kind not in BOUNDARY_KINDS]
    if not live_body:
        return None

    failures = compact_failures(session_id)
    max_fail = int(settings.compact_max_failures)
    allow_llm = (
        client is not None
        and bool(settings.anthropic_api_key)
        and failures < max_fail
        and system is not None
    )

    summary = None
    trigger = "extractive"
    llm_failed = False

    if allow_llm:
        try:
            notes = await update_session_notes(store, session_id, events, client, system)
            if notes:
                keep_from = choose_keep_from_seq(live, notes, budget)
                tail = [e for e in live_body if e.seq >= keep_from]
                if event_token_count(tail) + estimate_tokens(notes) <= budget and any(
                    e.seq < keep_from for e in live_body
                ):
                    summary = notes
                    trigger = "session_notes"
        except Exception as e:
            log.warning("session-notes compact failed: %s", e)
            llm_failed = True

    if summary is None and allow_llm and compact_failures(session_id) < max_fail:
        try:
            summary = await llm_summary(prompt_events(events), client, system)
            trigger = "llm"
        except Exception as e:
            log.warning("llm compact failed: %s", e)
            llm_failed = True

    if summary is None:
        entity = await store.get_entity(session_id)
        summary = session_entity_notes(entity)
        trigger = "session_notes" if summary else "extractive"
        if summary is None:
            keep_guess = choose_keep_from_seq(live, "placeholder", budget)
            dropped = [e for e in live_body if e.seq < keep_guess]
            summary = extractive_summary(dropped or live_body[-40:])

    keep_from = choose_keep_from_seq(live, summary, budget)
    if not any(e.seq < keep_from for e in live_body):
        return None

    ev = await store.append_event(
        session_id,
        "compact_boundary",
        {
            "text": summary,
            "keep_from_seq": keep_from,
            "trigger": trigger,
        },
    )
    try:
        from orbweaver.hindsight import enabled as hindsight_on

        if not hindsight_on():
            await remember(store, summary, source=f"compact:{session_id}")
    except Exception as e:
        log.warning("compact remember failed: %s", e)
    if trigger in {"llm", "session_notes"}:
        record_compact_success(session_id)
    elif llm_failed:
        record_compact_failure(session_id)
    clear_usage(session_id)
    return ev
