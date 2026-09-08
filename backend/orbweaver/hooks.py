"""Optional PreToolUse / PostToolUse hooks from layered .orbweaver/hooks.json."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbweaver.permissions.pipeline import TurnAborted
from orbweaver.permissions.rules import rule_matches
from orbweaver.sandbox.policy import expand_path

log = logging.getLogger(__name__)

HOOK_EVENTS = ("PreToolUse", "PostToolUse", "PostToolUseFailure")
DEFAULT_TIMEOUT_SEC = 15.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class HookSpec:
    event: str
    matcher: str
    hook_type: str
    command: str
    timeout: float = DEFAULT_TIMEOUT_SEC


@dataclass(frozen=True)
class HooksConfig:
    hooks: tuple[HookSpec, ...] = ()

    def matching(self, event: str, tool: str, inp: dict[str, Any]) -> tuple[HookSpec, ...]:
        return tuple(
            h for h in self.hooks if h.event == event and matcher_hits(h.matcher, tool, inp)
        )


@dataclass(frozen=True)
class PreToolUseOutcome:
    action: str  # proceed | deny | cancel
    tool_input: dict[str, Any]
    reason: str = ""


def matcher_hits(matcher: str, tool: str, inp: dict[str, Any]) -> bool:
    text = (matcher or "").strip()
    if text in {"", "*"}:
        return True
    if "(" in text and text.endswith(")"):
        rule_tool, rest = text.split("(", 1)
        return rule_matches(tool, inp, rule_tool.strip(), rest[:-1].strip() or None)
    if text == tool:
        return True
    try:
        return bool(re.fullmatch(text, tool))
    except re.error:
        return False


def _as_timeout(raw: Any) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SEC
    if val <= 0:
        return DEFAULT_TIMEOUT_SEC
    return min(val, 120.0)


def _hook_from_entry(event: str, raw: Any) -> list[HookSpec]:
    if not isinstance(raw, dict):
        return []
    matcher = str(raw.get("matcher") or raw.get("if") or "*")
    nested = raw.get("hooks")
    specs: list[HookSpec] = []
    if isinstance(nested, list):
        for item in nested:
            specs.extend(_hook_from_entry(event, {**raw, "hooks": None, **item}))
        return specs
    hook_type = str(raw.get("type") or "command").strip().lower() or "command"
    command = str(raw.get("command") or "").strip()
    if hook_type == "prompt":
        return [
            HookSpec(
                event=event,
                matcher=matcher,
                hook_type="prompt",
                command=command,
                timeout=_as_timeout(raw.get("timeout")),
            )
        ]
    if not command:
        return []
    return [
        HookSpec(
            event=event,
            matcher=matcher,
            hook_type="command",
            command=command,
            timeout=_as_timeout(raw.get("timeout")),
        )
    ]


def _hooks_from_file(data: dict[str, Any]) -> list[HookSpec]:
    blob = data.get("hooks") if isinstance(data.get("hooks"), dict) else data
    if not isinstance(blob, dict):
        return []
    out: list[HookSpec] = []
    for event in HOOK_EVENTS:
        raw_list = blob.get(event)
        if not isinstance(raw_list, list):
            continue
        for entry in raw_list:
            out.extend(_hook_from_entry(event, entry))
    return out


def load_hooks_config(
    workspace_root: Path | str | None = None,
    *,
    settings=None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> HooksConfig:
    """Merge ~/.orbweaver/hooks.json, workspace .orbweaver/hooks.json, then extra file.

    Later sources append. Host hooks therefore always run before workspace hooks.
    """
    from orbweaver.config import settings as default_settings

    cfg = settings or default_settings
    env = environ if environ is not None else dict(os.environ)
    root = Path(workspace_root or getattr(cfg, "workspace_root", ".") or ".").resolve()
    home_dir = (home or Path.home()).resolve()

    sources: list[Path] = [
        home_dir / ".orbweaver" / "hooks.json",
        root / ".orbweaver" / "hooks.json",
    ]
    extra = (
        getattr(cfg, "orbweaver_hooks_config", None) or env.get("ORBWEAVER_HOOKS_CONFIG") or ""
    ).strip()
    if extra:
        sources.append(expand_path(extra, relative_to=root))

    merged: list[HookSpec] = []
    for path in sources:
        if not path.is_file():
            continue
        merged.extend(_hooks_from_file(_read_json(path)))
    return HooksConfig(hooks=tuple(merged))


def hooks_for_ctx(ctx: dict[str, Any]) -> HooksConfig:
    cached = ctx.get("hooks_config")
    if isinstance(cached, HooksConfig):
        return cached
    workspace = ctx.get("workspace")
    root = getattr(workspace, "root", None)
    cfg = load_hooks_config(root)
    ctx["hooks_config"] = cfg
    return cfg


def _workspace_root(ctx: dict[str, Any]) -> Path:
    workspace = ctx.get("workspace")
    root = getattr(workspace, "root", None)
    if root is None:
        return Path.cwd()
    return Path(root)


def _hook_env(ctx: dict[str, Any], event: str, cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["ORBWEAVER_HOOK_EVENT"] = event
    env["ORBWEAVER_WORKSPACE"] = str(cwd)
    env["ORBWEAVER_HEADLESS"] = "1" if ctx.get("headless") and not ctx.get("interactive") else "0"
    return env


def _event_payload(
    event: str,
    tool: str,
    inp: dict[str, Any],
    ctx: dict[str, Any],
    *,
    result: str | None = None,
) -> dict[str, Any]:
    cwd = str(_workspace_root(ctx))
    payload: dict[str, Any] = {
        "hook_event_name": event,
        "tool_name": tool,
        "tool_input": inp,
        "cwd": cwd,
        "headless": bool(ctx.get("headless")) and not bool(ctx.get("interactive")),
        "channel": str(ctx.get("channel") or ""),
    }
    if result is not None:
        payload["tool_result"] = result
        payload["tool_response"] = result
    return payload


def _parse_json_object(text: str) -> dict[str, Any] | None:
    blob = (text or "").strip()
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        start = blob.find("{")
        end = blob.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(blob[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _decode(raw: bytes | None) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


@dataclass
class _HookRun:
    decision: str
    reason: str
    updated_input: dict[str, Any] | None
    append: str


def _interpret_output(stdout: str, stderr: str, returncode: int) -> _HookRun:
    data = _parse_json_object(stdout)
    decision = "deny" if returncode == 2 else "proceed"
    reason = ""
    updated: dict[str, Any] | None = None
    extra = ""
    if data:
        specific = data.get("hookSpecificOutput")
        if not isinstance(specific, dict):
            specific = {}
        raw = (
            specific.get("permissionDecision")
            or data.get("permissionDecision")
            or data.get("decision")
            or ""
        )
        key = str(raw).strip().lower()
        if key in {"deny", "block"}:
            decision = "deny"
        elif key == "cancel":
            decision = "cancel"
        elif key == "ask":
            decision = "ask"
        elif key in {"allow", "approve"} and returncode != 2:
            decision = "proceed"
        if data.get("continue") is False:
            decision = "cancel"
        reason = str(
            specific.get("permissionDecisionReason")
            or data.get("reason")
            or data.get("stopReason")
            or ""
        )
        incoming = specific.get("updatedInput") or data.get("updatedInput") or data.get("updated_input")
        if isinstance(incoming, dict):
            updated = incoming
        extra = str(specific.get("additionalContext") or data.get("additionalContext") or "")
    if not reason:
        reason = (stderr or stdout).strip()
    if data is None:
        extra = "\n".join(p for p in (stdout.strip(), stderr.strip()) if p)
    return _HookRun(decision=decision, reason=reason, updated_input=updated, append=extra)


def _run_command_sync(
    command: str,
    payload: dict[str, Any],
    cwd: Path,
    timeout: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        shell=True,
        cwd=str(cwd),
        env=env,
        timeout=timeout,
        start_new_session=True,
        check=False,
    )


async def _run_command(
    spec: HookSpec,
    event: str,
    tool: str,
    inp: dict[str, Any],
    ctx: dict[str, Any],
    *,
    result: str | None = None,
) -> _HookRun:
    cwd = _workspace_root(ctx)
    payload = _event_payload(event, tool, inp, ctx, result=result)
    env = _hook_env(ctx, event, cwd)
    try:
        proc = await asyncio.to_thread(
            _run_command_sync, spec.command, payload, cwd, spec.timeout, env
        )
    except subprocess.TimeoutExpired:
        log.warning("hook timed out (%s %s)", event, tool)
        if event == "PreToolUse":
            return _HookRun("deny", "PreToolUse hook timed out; fail closed", None, "")
        return _HookRun("proceed", "hook timed out", None, "")
    except OSError as e:
        log.warning("hook failed to start (%s %s): %s", event, tool, e)
        if event == "PreToolUse":
            return _HookRun("deny", f"PreToolUse hook failed to start: {e}", None, "")
        return _HookRun("proceed", str(e), None, "")
    return _interpret_output(_decode(proc.stdout), _decode(proc.stderr), int(proc.returncode or 0))


def _prompt_denied(event: str) -> _HookRun:
    return _HookRun(
        "deny",
        f"{event} prompt hook refused (no TTY prompt; fail closed)",
        None,
        "",
    )


def hook_cancel_abort(name: str, inp: dict[str, Any], reason: str) -> TurnAborted:
    from orbweaver.permissions.pipeline import summarize_input

    text = reason or "Turn cancelled by PreToolUse hook."
    return TurnAborted(
        text,
        {
            "reason": "hook_cancel",
            "last_tool": name,
            "last_input": summarize_input(name, inp),
            "text": text,
        },
    )


async def run_pre_tool_use(
    name: str, inp: dict[str, Any], ctx: dict[str, Any]
) -> PreToolUseOutcome:
    """Run PreToolUse hooks. Deny/cancel happen here, before can_use_tool."""
    current = dict(inp)
    for spec in hooks_for_ctx(ctx).matching("PreToolUse", name, current):
        if spec.hook_type == "prompt":
            run = _prompt_denied("PreToolUse")
        else:
            run = await _run_command(spec, "PreToolUse", name, current, ctx)
        if run.updated_input is not None:
            current = dict(run.updated_input)
        if run.decision == "ask":
            return PreToolUseOutcome(
                "deny",
                current,
                run.reason or "PreToolUse hook asked for a prompt; fail closed",
            )
        if run.decision == "cancel":
            return PreToolUseOutcome("cancel", current, run.reason or "cancelled by PreToolUse hook")
        if run.decision == "deny":
            return PreToolUseOutcome(
                "deny", current, run.reason or "denied by PreToolUse hook"
            )
    return PreToolUseOutcome("proceed", current)


def is_tool_error_result(result: str) -> bool:
    return result.lstrip().lower().startswith("error")


async def apply_post_tool_use(
    name: str, inp: dict[str, Any], result: str, ctx: dict[str, Any]
) -> str:
    """Append matching PostToolUse / PostToolUseFailure hook output to the tool result."""
    event = "PostToolUseFailure" if is_tool_error_result(result) else "PostToolUse"
    extras: list[str] = []
    for spec in hooks_for_ctx(ctx).matching(event, name, inp):
        if spec.hook_type == "prompt":
            extras.append(_prompt_denied(event).reason)
            continue
        run = await _run_command(spec, event, name, inp, ctx, result=result)
        if run.append.strip():
            extras.append(run.append.strip())
    if not extras:
        return result
    return result.rstrip() + "\n" + "\n".join(extras)
