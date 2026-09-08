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
    in_working_set,
    is_critical_rm,
    is_protected_git_push,
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
    if name in {"Read", "Write", "ProposePatch", "NotebookEdit", "SendPhoto", "Delete"}:
        return str(inp.get("path") or "")[:240]
    if name == "ReadLints":
        paths = inp.get("paths") or inp.get("path") or ""
        if isinstance(paths, list):
            return " ".join(str(p) for p in paths)[:240]
        return str(paths)[:240]
    if name == "GenerateImage":
        return str(inp.get("prompt") or "")[:240]
    if name == "WebFetch":
        return str(inp.get("url") or "")[:240]
    if name == "Browser":
        from orbweaver.browser import summarize_browser

        return summarize_browser(inp)
    if name == "WebSearch":
        return str(inp.get("query") or "")[:240]
    if name == "MemoryGraph":
        return str(inp.get("id") or "")[:240]
    if name == "SpawnSubagent":
        return str(inp.get("task") or "")[:240]
    if name.startswith("mcp_"):
        from orbweaver.permissions.rules import json_fallback

        return json_fallback(inp)[:240]
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


def bash_permissions(inp: dict[str, Any]) -> frozenset[str]:
    out: set[str] = set()
    if inp.get("unsandboxed") is True or str(inp.get("unsandboxed")).lower() in {"1", "true"}:
        out.add("all")
    raw = inp.get("permissions") or []
    if isinstance(raw, str):
        raw = [raw]
    for item in raw:
        val = str(item).strip().lower()
        if val in {"all", "full_network"}:
            out.add(val)
    return frozenset(out)


def _classifier_verdict(result: dict[str, Any]) -> str:
    verdict = result.get("verdict")
    if verdict in {"allow", "ask", "deny"}:
        return verdict
    if result.get("should_ask"):
        return "ask"
    if result.get("should_block"):
        return "deny"
    return "allow"


def bash_sandboxable(inp: dict[str, Any], workspace_kind: str) -> bool:
    perms = bash_permissions(inp)
    if perms & {"all", "full_network"}:
        return False
    if is_critical_rm(str(inp.get("command") or "")):
        return False
    del workspace_kind
    if not settings.orbweaver_sandbox:
        return False
    return sandbox_available()


async def can_use_tool(name: str, inp: dict[str, Any], ctx: dict[str, Any]) -> PermissionDecision:
    workspace = ctx.get("workspace")
    workspace_kind = str(ctx.get("workspace_kind") or "local")
    headless = bool(ctx.get("headless"))
    events: list[Event] = ctx.get("events") or []
    mode = (settings.orbweaver_permission_mode or "auto").strip().lower()

    if name in {"Read", "Write", "ProposePatch", "NotebookEdit", "SendPhoto", "Delete"}:
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
    cmd = str(inp.get("command") or "") if name == "Bash" else ""
    if (
        ask_hit
        and name == "Bash"
        and (ask_hit[1] or "").replace(" ", "") in {"gitpush*", "gitpush"}
        and not is_protected_git_push(cmd)
    ):
        # Legacy ORBWEAVER_PERMISSION_ASK=Bash(git push *) must not block feature branches.
        ask_hit = None
    if name == "Bash" and is_protected_git_push(cmd):
        ask_hit = ask_hit or ("Bash", "protected git push")
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
        if name == "Read" and workspace and not in_working_set(str(inp.get("path") or ""), workspace):
            if not getattr(workspace, "host_reads", True):
                return PermissionDecision("deny", f"path denied: {inp.get('path')}", "deny_rule")
        elif name == "ReadLints":
            from orbweaver.lints import lint_paths_from_input

            lint_paths = lint_paths_from_input(inp)
            if any(path_is_always_denied(p) for p in lint_paths):
                return PermissionDecision("deny", "path denied", "deny_rule")
            outside = [
                p for p in lint_paths if workspace and not in_working_set(p, workspace)
            ]
            if outside and not getattr(workspace, "host_reads", True):
                return PermissionDecision("deny", f"path denied: {outside[0]}", "deny_rule")
            if not outside:
                return PermissionDecision("allow", "safe tool allowlist", "allowlist")
        elif name in {"Glob", "Grep"}:
            glob_pat = str(inp.get("glob") or inp.get("pattern") or "")
            if (
                workspace
                and glob_pat.startswith(("/", "~"))
                and not in_working_set(glob_pat, workspace)
            ):
                pass
            else:
                return PermissionDecision("allow", "safe tool allowlist", "allowlist")
        else:
            return PermissionDecision("allow", "safe tool allowlist", "allowlist")

    if name in {"Write", "ProposePatch", "NotebookEdit", "Delete"} and workspace and in_project_path(str(inp.get("path") or ""), workspace):
        return PermissionDecision("allow", "in-project file edit", "acceptEdits")

    if name == "Bash" and settings.orbweaver_auto_allow_bash_if_sandboxed and bash_sandboxable(inp, workspace_kind):
        return PermissionDecision("allow", "sandboxed bash auto-allow", "sandbox")

    if (
        name == "Bash"
        and not bash_sandboxable(inp, workspace_kind)
        and settings.orbweaver_sandbox
        and settings.orbweaver_sandbox_fail_if_unavailable
        and not sandbox_available()
        and not bash_permissions(inp)
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
    verdict = _classifier_verdict(result)
    reason = result.get("reason") or "classifier"
    if verdict == "allow":
        state.record_success()
        return PermissionDecision("allow", reason, fast)
    if verdict == "ask":
        decision = PermissionDecision("ask", reason, fast, classifier_reason=reason)
        if headless:
            _abort(ctx, "ask_required_headless", decision, name, inp)
        return decision
    state.record_denial()
    decision = PermissionDecision(
        "deny",
        reason,
        fast,
        classifier_reason=reason,
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
