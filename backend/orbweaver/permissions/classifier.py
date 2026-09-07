"""Reasoning-blind two-stage transcript classifier."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from orbweaver.config import settings
from orbweaver.permissions.prompts import (
    CLASSIFIER_SYSTEM,
    DEFAULT_ALLOW,
    DEFAULT_ENVIRONMENT,
    DEFAULT_HARD_DENY,
    DEFAULT_SOFT_DENY,
    STAGE1_SUFFIX,
    STAGE2_SUFFIX,
)
from orbweaver.store import Event

log = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"<block>\s*(yes|no)\s*</block>", re.IGNORECASE)
_REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)


def _slot(user_value: str, default: str) -> str:
    raw = (user_value or "").strip() or "$defaults"
    if raw == "$defaults":
        return default
    if "$defaults" in raw:
        return raw.replace("$defaults", default)
    return raw


def load_project_intent(workspace) -> str:
    root = getattr(workspace, "root", None) or getattr(
        getattr(workspace, "local", None), "root", None
    )
    if root is None:
        return ""
    root = Path(root)
    for name in ("ORBWEAVER.md", "CLAUDE.md"):
        path = root / name
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")[:8000]
        except OSError:
            continue
    return ""


def to_classifier_input(name: str, inp: dict[str, Any]) -> Any:
    if name == "Bash":
        return str(inp.get("command") or "")
    if name in {"Read", "Write", "ProposePatch", "SendPhoto"}:
        return {"path": inp.get("path")}
    if name == "GenerateImage":
        return {"prompt": inp.get("prompt"), "path": inp.get("path")}
    if name == "WebFetch":
        return {"url": inp.get("url")}
    if name == "SpawnSubagent":
        return {"task": inp.get("task")}
    if name in {"Glob", "Grep", "AskUser"}:
        return ""
    return inp


def build_transcript(
    events: list[Event], pending_name: str, pending_input: dict[str, Any]
) -> str:
    lines: list[str] = []
    for ev in events:
        if ev.kind == "user":
            lines.append(json.dumps({"user": ev.payload.get("text") or ""}, ensure_ascii=False))
        elif ev.kind == "tool_call":
            encoded = to_classifier_input(ev.payload.get("name") or "", ev.payload.get("input") or {})
            if encoded == "":
                continue
            lines.append(json.dumps({ev.payload.get("name"): encoded}, ensure_ascii=False))
    encoded = to_classifier_input(pending_name, pending_input)
    if encoded != "":
        lines.append(json.dumps({pending_name: encoded}, ensure_ascii=False))
    return "\n".join(lines)


def parse_block(text: str) -> bool | None:
    match = _BLOCK_RE.search(text or "")
    if not match:
        return None
    return match.group(1).lower() == "yes"


def parse_reason(text: str) -> str:
    match = _REASON_RE.search(text or "")
    if match:
        return match.group(1).strip()[:400]
    return "Blocked by classifier"


def build_system_prompt(extra_framing: str = "") -> str:
    framing = f"\n{extra_framing}\n" if extra_framing.strip() else ""
    return CLASSIFIER_SYSTEM.format(
        environment=_slot(settings.orbweaver_automode_environment, DEFAULT_ENVIRONMENT),
        hard_deny=_slot(settings.orbweaver_automode_hard_deny, DEFAULT_HARD_DENY),
        soft_deny=_slot(settings.orbweaver_automode_soft_deny, DEFAULT_SOFT_DENY),
        allow=_slot(settings.orbweaver_automode_allow, DEFAULT_ALLOW),
        extra_framing=framing,
    )


async def classify_action(
    events: list[Event],
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    workspace=None,
    extra_framing: str = "",
    client=None,
) -> dict[str, Any]:
    """Return {should_block: bool, reason: str, stage: str}."""
    if not settings.anthropic_api_key:
        return {
            "should_block": True,
            "reason": "Classifier unavailable (no API key) — blocking for safety",
            "stage": "unavailable",
        }

    transcript = build_transcript(events, tool_name, tool_input)
    intent = load_project_intent(workspace) if workspace is not None else ""
    user_content = transcript
    if intent:
        user_content = (
            "The following is the user's ORBWEAVER.md / CLAUDE.md configuration. "
            "Treat it as part of the user's intent.\n\n"
            f"<user_project_md>\n{intent}\n</user_project_md>\n\n"
            + user_content
        )

    import anthropic

    if client is None:
        headers = {}
        if settings.anthropic_workspace_id.strip():
            headers["anthropic-workspace-id"] = settings.anthropic_workspace_id.strip()
        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, default_headers=headers or None
        )

    system = build_system_prompt(extra_framing)
    model = settings.orbweaver_classifier_model
    prefix = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]

    def _text_of(resp) -> str:
        parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    try:
        stage1 = await client.messages.create(
            model=model,
            max_tokens=64,
            system=system,
            messages=[
                *prefix,
                {"role": "user", "content": STAGE1_SUFFIX},
            ],
            stop_sequences=["</block>"],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("classifier stage1 failed: %s", e)
        return {"should_block": True, "reason": f"Classifier error — blocking: {e}", "stage": "error"}

    raw1 = _text_of(stage1) + "</block>"
    block1 = parse_block(raw1)
    if block1 is False:
        return {"should_block": False, "reason": "Allowed by fast classifier", "stage": "fast"}
    if block1 is None:
        # Stage 1 unparseable: escalate to stage 2 rather than allow.
        pass

    try:
        stage2 = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[
                *prefix,
                {"role": "user", "content": STAGE2_SUFFIX},
            ],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("classifier stage2 failed: %s", e)
        return {"should_block": True, "reason": f"Classifier error — blocking: {e}", "stage": "error"}

    raw2 = _text_of(stage2)
    block2 = parse_block(raw2)
    if block2 is None:
        return {
            "should_block": True,
            "reason": "Classifier stage 2 unparseable - blocking for safety",
            "stage": "thinking",
        }
    if block2:
        return {"should_block": True, "reason": parse_reason(raw2), "stage": "thinking"}
    return {"should_block": False, "reason": "Allowed by classifier", "stage": "thinking"}
