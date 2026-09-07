"""Ordered permission pipeline (deny → ask → allowlist → edits → sandbox → classifier)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from orbweaver.config import settings
from orbweaver.permissions.classifier import classify_action
from orbweaver.permissions.denial import DenialTrackingState, denial_state_for
from orbweaver.permissions.prompts import DELEGATION_FRAMING
from orbweaver.permissions.rules import (
    SAFE_ALLOWLIST,
    allow_rules,
    ask_rules,
    deny_rules,
    in_project_path,
    is_critical_rm,
    matching_rule,
    path_is_always_denied,
)
from orbweaver.sandbox.bwrap import sandbox_available
from orbweaver.store import Event


@dataclass
class PermissionDecision:
    behavior: str  # allow | deny | ask
    reason: str
    fast_path: str
    classifier_reason: str | None = None


class TurnAborted(Exception):
    """Headless session hit denial limits or an ask rule with no human."""

    def __init__(self, message: str, payload: dict[str, Any]):
        super().__init__(message)
        self.message = message
        self.payload = payload


def summarize_input(name: str, inp: dict[str, Any]) -> str:
    if name == "Bash":
        return str(inp.get("command") or "")[:240]
    if name in {"Read", "Write", "ProposePatch", "SendPhoto"}:
        return str(inp.get("path") or "")[:240]
    if name == "GenerateImage":
        return str(inp.get("prompt") or "")[:240]
    if name == "WebFetch":
        return str(inp.get("url") or "")[:240]
    if name == "SpawnSubagent":
        return str(inp.get("task") or "")[:240]
    return name


def abort_message(payload: dict[str, Any]) -> str:
    n = payload.get("consecutive") or payload.get("total") or 1
    last = payload.get("last_tool") or "unknown"
    detail = payload.get("last_input") or ""
    why = payload.get("classifier_reason") or payload.get("reason") or "blocked"
    label = f"{last} {detail}".strip()
    return (
        f"Stopped this turn after {n} blocked actions. Last blocked: {label} — {why}. "
        "Say what you want done, or switch this session out of auto mode."
    )


def _abort(ctx: dict[str, Any], reason_code: str, decision: PermissionDecision, name: str, inp: dict[str, Any]):
    state: DenialTrackingState = ctx["denial_state"]
    payload = {
        "reason": reason_code,
        "consecutive": state.consecutive_denials,
        "total": state.total_denials,
        "last_tool": name,
        "last_input": summarize_input(name, inp),
        "classifier_reason": decision.classifier_reason or decision.reason,
        "fast_path": decision.fast_path,
    }
    text = abort_message(payload)
    payload["text"] = text
    raise TurnAborted(text, payload)


def bash_sandboxable(inp: dict[str, Any], workspace_kind: str) -> bool:
    if inp.get("unsandboxed") is True or str(inp.get("unsandboxed")).lower() in {"1", "true"}:
        return False
    if is_critical_rm(str(inp.get("command") or "")):
        return False
    if workspace_kind == "docker":
        return True
    if not settings.orbweaver_sandbox:
        return False
    return sandbox_available()


async def can_use_tool(name: str, inp: dict[str, Any], ctx: dict[str, Any]) -> PermissionDecision:
    workspace = ctx.get("workspace")
    workspace_kind = str(ctx.get("workspace_kind") or "local")
    headless = bool(ctx.get("headless"))
    events: list[Event] = ctx.get("events") or []
    mode = (settings.orbweaver_permission_mode or "auto").strip().lower()

    if name in {"Read", "Write", "ProposePatch", "SendPhoto"}:
        path = str(inp.get("path") or "")
        if path_is_always_denied(path):
            return PermissionDecision("deny", f"path denied: {path}", "deny_rule")
        if name != "Read" and workspace and not in_project_path(path, workspace):
            return PermissionDecision("deny", f"path denied: {path}", "deny_rule")

    if name == "GenerateImage":
        out_path = str(inp.get("path") or "attachments/generated.png")
        if path_is_always_denied(out_path) or (workspace and not in_project_path(out_path, workspace)):
            return PermissionDecision("deny", f"path denied: {out_path}", "deny_rule")

    deny_hit = matching_rule(deny_rules(), name, inp)
    if deny_hit:
        return PermissionDecision("deny", f"denied by rule {deny_hit[0]}({deny_hit[1] or '*'})", "deny_rule")

    ask_hit = matching_rule(ask_rules(), name, inp)
    if ask_hit:
        decision = PermissionDecision(
            "ask",
            f"ask rule {ask_hit[0]}({ask_hit[1] or '*'})",
            "ask_rule",
        )
        if headless:
            _abort(ctx, "ask_required_headless", decision, name, inp)
        return decision

    allow_hit = matching_rule(allow_rules(), name, inp, strip_dangerous=(mode == "auto"))
    if allow_hit:
        return PermissionDecision("allow", f"allow rule {allow_hit[0]}", "allow_rule")

    if name in SAFE_ALLOWLIST:
        if name == "Read" and path_is_always_denied(str(inp.get("path") or "")):
            return PermissionDecision("deny", "path denied", "deny_rule")
        return PermissionDecision("allow", "safe tool allowlist", "allowlist")

    if name in {"Write", "ProposePatch"} and workspace and in_project_path(str(inp.get("path") or ""), workspace):
        return PermissionDecision("allow", "in-project file edit", "acceptEdits")

    if name == "Bash" and settings.orbweaver_auto_allow_bash_if_sandboxed and bash_sandboxable(inp, workspace_kind):
        return PermissionDecision("allow", "sandboxed bash auto-allow", "sandbox")

    if (
        name == "Bash"
        and not bash_sandboxable(inp, workspace_kind)
        and workspace_kind != "docker"
        and settings.orbweaver_sandbox
        and settings.orbweaver_sandbox_fail_if_unavailable
        and not sandbox_available()
        and not (inp.get("unsandboxed") is True)
        and not is_critical_rm(str(inp.get("command") or ""))
    ):
        return PermissionDecision(
            "deny",
            "sandbox_unavailable: bwrap missing or blocked; bash refused",
            "sandbox",
        )

    extra = DELEGATION_FRAMING if name == "SpawnSubagent" else ""
    fast = "handoff" if name == "SpawnSubagent" else "classifier"
    result = await classify_action(
        events, name, inp, workspace=workspace, extra_framing=extra
    )
    state: DenialTrackingState = ctx.get("denial_state") or denial_state_for(
        ctx.get("session_id") or UUID(int=0)
    )
    if result.get("should_block"):
        state.record_denial()
        decision = PermissionDecision(
            "deny",
            result.get("reason") or "Blocked by classifier",
            fast,
            classifier_reason=result.get("reason"),
        )
        if state.should_fallback():
            if headless:
                _abort(ctx, "classifier_denial_limit", decision, name, inp)
            return PermissionDecision(
                "ask",
                decision.reason,
                fast,
                classifier_reason=decision.classifier_reason,
            )
        return decision
    state.record_success()
    return PermissionDecision("allow", result.get("reason") or "classifier allow", fast)
