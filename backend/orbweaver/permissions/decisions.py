"""JEV (Earth Runtime Decisions API v1) as the auto-mode permission engine.

``POST /v1/decisions`` takes one plain-text ``context`` and a ``schema`` whose
fields are enum lists, and returns one constrained choice per field. There is
no chat, no free-text reason, and the context is capped at 16,384 characters,
so the two-stage transcript classifier's system prompt, the project intent
docs, and the reasoning-blind transcript are flattened into one string here
and the model answers ``verdict`` and ``reason`` as enums.

Failure is reported, never swallowed: :func:`decide_action` returns a result
with ``stage == "jev_error"`` and ``verdict == "ask"``; ``classify_action``
decides whether that falls through to the LLM classifier or holds for a human.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from orbweaver.config import settings
from orbweaver.permissions.classifier import (
    _slot,
    load_project_intent,
    to_classifier_input,
    transcript_lines,
)
from orbweaver.permissions.prompts import (
    DECISION_POLICY,
    DEFAULT_ALLOW,
    DEFAULT_ENVIRONMENT,
    DEFAULT_HARD_DENY,
    DEFAULT_SOFT_DENY,
)
from orbweaver.store import Event

log = logging.getLogger(__name__)

# "ask" first on purpose: the documented local mock (and any degenerate
# backend) always returns the first enum value, which must fail safe.
VERDICTS: list[str] = ["ask", "allow", "deny"]

# Enum stand-in for the free-text <reason> the LLM classifier produces.
REASONS: dict[str, str] = {
    "uncertain_scope_or_blast_radius": "Needs user approval: scope or blast radius is unclear",
    "allowed_user_named_action": "Allowed: the user named this action",
    "allowed_by_allow_rules": "Allowed by allow rules",
    "soft_deny_without_user_intent": "Needs user approval: soft-deny action the user did not name",
    "sandbox_override_not_confirmed": "Needs user approval: sandbox override not confirmed by the user",
    "hard_deny_secret_exfiltration": "Blocked: would exfiltrate secrets or keys",
    "hard_deny_persistence_or_logging": "Blocked: installs persistence or disables logging",
    "hard_deny_self_granting_power": "Blocked: modifies the agent's own permission or sandbox config",
}

SCHEMA: dict[str, list[str]] = {"verdict": VERDICTS, "reason": list(REASONS)}

# Hard floor so a huge pending input cannot starve the policy text.
_PENDING_MAX_CHARS = 4_000
_TRUNCATED = "…[truncated]"


class DecisionsError(Exception):
    """Non-2xx or malformed answer from the Decisions API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


def enabled() -> bool:
    return bool(settings.decisions_key)


def build_policy(extra_framing: str = "") -> str:
    framing = f"\n{extra_framing}\n" if extra_framing.strip() else ""
    return DECISION_POLICY.format(
        environment=_slot(settings.orbweaver_automode_environment, DEFAULT_ENVIRONMENT),
        hard_deny=_slot(settings.orbweaver_automode_hard_deny, DEFAULT_HARD_DENY),
        soft_deny=_slot(settings.orbweaver_automode_soft_deny, DEFAULT_SOFT_DENY),
        allow=_slot(settings.orbweaver_automode_allow, DEFAULT_ALLOW),
        extra_framing=framing,
    )


def pending_line(tool_name: str, tool_input: dict[str, Any]) -> str:
    encoded = to_classifier_input(tool_name, tool_input)
    line = json.dumps({tool_name: encoded}, ensure_ascii=False)
    if len(line) > _PENDING_MAX_CHARS:
        line = line[: _PENDING_MAX_CHARS - len(_TRUNCATED)] + _TRUNCATED
    return line


def _head(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if budget <= len(_TRUNCATED):
        return ""
    return text[: budget - len(_TRUNCATED)] + _TRUNCATED


def _tail_lines(lines: list[str], budget: int) -> str:
    """Newest lines that fit in ``budget`` chars, with an omission marker."""
    if budget <= 0 or not lines:
        return ""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + (1 if kept else 0)
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    dropped = len(lines) - len(kept)
    if dropped:
        marker = f"[… {dropped} earlier transcript line(s) omitted]"
        if kept and used + len(marker) + 1 > budget:
            used -= len(kept[-1]) + 1
            kept.pop()
            dropped += 1
            marker = f"[… {dropped} earlier transcript line(s) omitted]"
        kept.append(marker)
    kept.reverse()
    return "\n".join(kept)


def build_context(
    policy: str,
    intent: str,
    lines: list[str],
    pending: str,
    *,
    limit: int | None = None,
) -> str:
    """Flatten policy + intent + transcript + pending action under the char cap.

    The policy and the pending action are always present. The newest
    transcript lines take priority over the project docs; whatever one does
    not use, the other may.
    """
    cap = int(limit or settings.decisions_context_chars)
    cap = max(1, min(cap, 16_384))

    def _assemble(intent_text: str, transcript: str) -> str:
        # Plain concatenation: policy, intent and transcript all contain braces.
        return (
            f"{policy.strip()}\n\n"
            f"<user_project_md>\n{intent_text}\n</user_project_md>\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            f"<pending_action>\n{pending}\n</pending_action>\n\n"
            "Decide verdict and reason for the pending action."
        )

    remaining = cap - len(_assemble("", ""))
    if remaining < 0:
        # Policy alone overflows: nothing sensible to drop but the extras.
        return _assemble("", "")[:cap]
    intent = intent.strip()
    intent_reserve = min(len(intent), remaining // 3)
    transcript = _tail_lines(lines, remaining - intent_reserve)
    intent_text = _head(intent, remaining - len(transcript))
    return _assemble(intent_text, transcript)


async def request_decision(
    context: str,
    schema: dict[str, list[str]] | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """POST one decision. Raises :class:`DecisionsError` on any failure."""
    key = settings.decisions_key
    if not key:
        raise DecisionsError("DECISIONS_API_KEY is not set", code="no_key")
    body = {
        "context": context,
        "schema": schema or SCHEMA,
        "decoding": (settings.decisions_decoding or "parallel_constrained").strip(),
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.decisions_timeout_s, transport=transport
        ) as client:
            resp = await client.post(settings.decisions_api_url, json=body, headers=headers)
    except httpx.HTTPError as e:
        raise DecisionsError(f"transport error: {e}", code="transport") from e
    try:
        data: Any = resp.json()
    except ValueError:
        data = None
    if resp.status_code >= 400 or not isinstance(data, dict):
        err = (data or {}).get("error") if isinstance(data, dict) else None
        err = err if isinstance(err, dict) else {}
        code = str(err.get("code") or resp.status_code)
        msg = str(err.get("message") or resp.text[:200] or "no body")
        raise DecisionsError(
            f"HTTP {resp.status_code} {code}: {msg}",
            status_code=resp.status_code,
            code=code,
            request_id=err.get("request_id"),
        )
    return data


def parse_decision(data: dict[str, Any]) -> tuple[str, str, float | None]:
    """Return (verdict, reason_key, probability) or raise DecisionsError."""
    decision = data.get("decision")
    if not isinstance(decision, dict):
        raise DecisionsError("response has no decision object", code="malformed")
    v = decision.get("verdict")
    verdict = v.get("value") if isinstance(v, dict) else v
    if verdict not in VERDICTS:
        raise DecisionsError(f"verdict {verdict!r} not in schema", code="malformed")
    r = decision.get("reason")
    reason = r.get("value") if isinstance(r, dict) else r
    if reason not in REASONS:
        reason = "uncertain_scope_or_blast_radius"
    prob = v.get("probability") if isinstance(v, dict) else None
    probability = float(prob) if isinstance(prob, (int, float)) else None
    return str(verdict), str(reason), probability


def _result(verdict: str, reason: str, stage: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "verdict": verdict,
        "should_block": verdict == "deny",
        "should_ask": verdict == "ask",
        "reason": reason,
        "stage": stage,
        "engine": "jev",
    }
    out.update(extra)
    return out


async def decide_action(
    events: list[Event],
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    workspace=None,
    extra_framing: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Classify one pending action through JEV.

    Same result shape as ``classify_action``; ``stage`` is ``jev`` on a real
    verdict and ``jev_error`` when the API could not answer (verdict ``ask``).
    """
    intent = load_project_intent(workspace) if workspace is not None else ""
    context = build_context(
        build_policy(extra_framing),
        intent,
        transcript_lines(events),
        pending_line(tool_name, tool_input),
    )
    try:
        data = await request_decision(context, transport=transport)
        verdict, reason_key, probability = parse_decision(data)
    except DecisionsError as e:
        log.warning("jev decision failed for %s: %s", tool_name, e)
        return _result(
            "ask",
            f"JEV unavailable — needs user approval: {e}",
            "jev_error",
            error_code=e.code,
            status_code=e.status_code,
        )
    request_id = data.get("request_id")
    log.info(
        "jev %s -> %s (%s) p=%s request_id=%s mode=%s",
        tool_name,
        verdict,
        reason_key,
        probability,
        request_id,
        data.get("mode"),
    )
    if verdict == "allow":
        text = "Allowed by JEV decision"
    else:
        text = REASONS[reason_key]
        if reason_key.startswith("allowed_"):
            text = "Needs user approval" if verdict == "ask" else "Blocked by JEV decision"
    if probability is not None:
        text = f"{text} (p={probability:.2f})"
    return _result(
        verdict,
        text,
        "jev",
        reason_key=reason_key,
        probability=probability,
        request_id=request_id,
        mode=data.get("mode"),
    )
