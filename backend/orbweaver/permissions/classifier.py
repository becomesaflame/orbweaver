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

_BLOCK_RE = re.compile(r"<block>\s*(yes|no|ask)\s*</block>", re.IGNORECASE)
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
        cmd = str(inp.get("command") or "")
        perms: list[str] = []
        raw = inp.get("permissions") or []
        if isinstance(raw, str):
            raw = [raw]
        for item in raw:
            val = str(item).strip().lower()
            if val in {"all", "full_network"}:
                perms.append(val)
        if inp.get("unsandboxed") is True or str(inp.get("unsandboxed")).lower() in {"1", "true"}:
            perms.append("all")
        if perms:
            return {"command": cmd, "permissions": sorted(set(perms))}
        return cmd
    if name in {"Read", "Write", "ProposePatch", "SendPhoto", "Delete"}:
        return {"path": inp.get("path")}
    if name == "ReadLints":
        return {"paths": inp.get("paths") or inp.get("path")}
    if name == "GenerateImage":
        return {"prompt": inp.get("prompt"), "path": inp.get("path")}
    if name == "WebFetch":
        return {"url": inp.get("url")}
    if name == "WebSearch":
        return {"query": inp.get("query")}
    if name == "MemoryGraph":
        return {"id": inp.get("id"), "depth": inp.get("depth")}
    if name == "SpawnSubagent":
        return {"task": inp.get("task")}
    if name in {"Glob", "Grep", "AskUser", "TodoWrite"}:
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


def parse_verdict(text: str) -> str | None:
    """Map <block>yes|no|ask</block> to deny|allow|ask."""
    match = _BLOCK_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1).lower()
    if raw == "yes":
        return "deny"
    if raw == "no":
        return "allow"
    return "ask"


def parse_block(text: str) -> bool | None:
    verdict = parse_verdict(text)
    if verdict is None:
        return None
    return verdict == "deny"


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


def _classified(verdict: str, reason: str, stage: str) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "should_block": verdict == "deny",
        "should_ask": verdict == "ask",
        "reason": reason,
        "stage": stage,
    }


async def classify_action(
    events: list[Event],
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    workspace=None,
    extra_framing: str = "",
    client=None,
) -> dict[str, Any]:
    """Return {verdict: allow|ask|deny, should_block, should_ask, reason, stage}."""
    if not settings.anthropic_api_key:
        return _classified("ask", "Classifier unavailable (no API key) — needs user approval", "unavailable")

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
    except Exception as e:
        log.warning("classifier stage1 failed: %s", e)
        return _classified("ask", f"Classifier error — needs user approval: {e}", "error")

    raw1 = _text_of(stage1) + "</block>"
    verdict1 = parse_verdict(raw1)
    if verdict1 == "allow":
        return _classified("allow", "Allowed by fast classifier", "fast")

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
    except Exception as e:
        log.warning("classifier stage2 failed: %s", e)
        return _classified("ask", f"Classifier error — needs user approval: {e}", "error")

    raw2 = _text_of(stage2)
    verdict2 = parse_verdict(raw2)
    if verdict2 is None:
        return _classified("ask", "Classifier stage 2 unparseable — needs user approval", "thinking")
    if verdict2 == "deny":
        return _classified("deny", parse_reason(raw2), "thinking")
    if verdict2 == "ask":
        reason = parse_reason(raw2)
        if reason == "Blocked by classifier":
            reason = "Needs user approval"
        return _classified("ask", reason, "thinking")
    return _classified("allow", "Allowed by classifier", "thinking")
