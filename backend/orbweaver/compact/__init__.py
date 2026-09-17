"""Compaction: shrink the prompt, keep the event log."""

from orbweaver.compact.overflow import (
    CONTEXT_FULL_MESSAGE,
    ContextFullError,
    extract_context_window_tokens,
    is_context_overflow,
    overflow_compact_budget,
    should_overflow_retry,
)
from orbweaver.compact.persist import persist_tool_result
from orbweaver.compact.pipeline import maybe_compact
from orbweaver.compact.project import (
    PROJECT_INSTRUCTIONS_KIND,
    choose_keep_from_recent_rounds,
    drop_orphan_tool_results,
    ensure_tool_use_results,
    events_to_messages,
    live_events,
    microcompact_events,
    prompt_events,
)
from orbweaver.compact.rehydrate import rehydrate_messages
from orbweaver.compact.usage import (
    ProbeStats,
    estimate_prompt_tokens,
    event_token_count,
    last_usage,
    probe_stats,
    record_probe_usage,
    record_response_usage,
    record_usage,
    request_token_estimate,
    reset_compact_state,
    static_prompt_tokens,
    usage_input_tokens,
)

__all__ = [
    "CONTEXT_FULL_MESSAGE",
    "PROJECT_INSTRUCTIONS_KIND",
    "ContextFullError",
    "ProbeStats",
    "choose_keep_from_recent_rounds",
    "drop_orphan_tool_results",
    "ensure_tool_use_results",
    "estimate_prompt_tokens",
    "event_token_count",
    "events_to_messages",
    "extract_context_window_tokens",
    "is_context_overflow",
    "last_usage",
    "live_events",
    "maybe_compact",
    "microcompact_events",
    "overflow_compact_budget",
    "persist_tool_result",
    "probe_stats",
    "prompt_events",
    "record_probe_usage",
    "record_response_usage",
    "record_usage",
    "rehydrate_messages",
    "request_token_estimate",
    "reset_compact_state",
    "should_overflow_retry",
    "static_prompt_tokens",
    "usage_input_tokens",
]
