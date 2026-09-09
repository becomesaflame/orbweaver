"""Compaction pipeline: keep the event log, shrink the model prompt."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from orbweaver.compact.llm import llm_summary, session_entity_notes, update_session_notes
from orbweaver.compact.project import (
    BOUNDARY_KINDS,
    choose_keep_from_recent_rounds,
    choose_keep_from_seq,
    extractive_summary,
    live_events,
    prompt_events,
)
from orbweaver.compact.usage import (
    clear_usage,
    compact_failures,
    estimate_prompt_tokens,
    event_token_count,
    record_compact_failure,
    record_compact_success,
)
from orbweaver.config import settings
from orbweaver.llm import compact_llm_client
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
    force: bool = False,
    budget: int | None = None,
    keep_recent_rounds: int | None = None,
) -> Event | None:
    """Append a compact_boundary if the live prompt window is over budget.

    Trigger from last API ``usage.input_tokens`` plus a local delta
    (``estimate_prompt_tokens``), not a payload-only estimate. Payload
    estimates undercount system+tools, images, and MCP schemas, so a
    session can hit Anthropic ``prompt_too_long`` without compacting.

    Threshold is ``event_budget * compact_ratio`` (~85% of the 200k
    window after reserves). Claw Code compares cumulative input tokens
    to 100k (``CLAUDE_CODE_AUTO_COMPACT_INPUT_TOKENS``). Overflow
    retries pass force=True and a shrinking keep_recent_rounds tail
    (4 → 2 → 1 → 0).

    Never deletes session events. Recursion sources (compact, session_notes) no-op.
    """
    del workspace  # persist/rehydrate happen at ingest / prompt build
    if source in SKIP_SOURCES:
        return None
    events = await store.list_events(session_id)
    projected = prompt_events(events)
    resolved_budget = (
        int(budget) if budget is not None else int(settings.event_budget * settings.compact_ratio)
    )
    prompt_tokens = estimate_prompt_tokens(session_id, projected)
    if not force and prompt_tokens <= resolved_budget:
        return None
    # Usage includes system/tools/images the payload estimate misses. Reserve
    # that overhead so choose_keep_from_seq actually drops events.
    overhead = max(0, prompt_tokens - event_token_count(projected))
    keep_budget = max(1, resolved_budget - overhead)

    live = live_events(events)
    live_body = [e for e in live if e.kind not in BOUNDARY_KINDS]
    if not live_body:
        return None

    failures = compact_failures(session_id)
    max_fail = int(settings.compact_max_failures)
    client = compact_llm_client(client)
    allow_llm = (
        source != "overflow"
        and client is not None
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
                keep_from = choose_keep_from_seq(live, notes, keep_budget)
                tail = [e for e in live_body if e.seq >= keep_from]
                if event_token_count(tail) + estimate_tokens(notes) <= keep_budget and any(
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
            keep_guess = choose_keep_from_seq(live, "placeholder", keep_budget)
            dropped = [e for e in live_body if e.seq < keep_guess]
            summary = extractive_summary(dropped or live_body[-40:])

    keep_from = choose_keep_from_seq(live, summary, keep_budget)
    if keep_recent_rounds is not None:
        keep_from = max(keep_from, choose_keep_from_recent_rounds(live, keep_recent_rounds))
    if not any(e.seq < keep_from for e in live_body):
        return None

    ev = await store.append_event(
        session_id,
        "compact_boundary",
        {
            "text": summary,
            "keep_from_seq": keep_from,
            "trigger": "overflow" if source == "overflow" else trigger,
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
