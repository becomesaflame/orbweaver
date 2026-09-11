"""Model prompt-injection probe for tool outputs (warning only).

Scope (``ORBWEAVER_INJECTION_PROBE_MODE``):

- ``off``: never probe.
- ``scoped`` (default): trust tiers. Read/Grep/Glob/WorkspaceSearch results whose
  path is inside the project roots and not under a vendored/generated dir
  (``ORBWEAVER_INJECTION_PROBE_SKIP_DIRS``) are skipped unless the output
  matches the instruction-phrasing prefilter. Bash is probed only when the
  command touches the network (``ORBWEAVER_INJECTION_PROBE_BASH_NETWORK``) or
  the output matches the prefilter (``ORBWEAVER_INJECTION_PROBE_PREFILTER``).
  WebFetch, WebSearch, Browser, memory tools, and ``mcp_*`` are always probed.
- ``all``: every eligible result is probed (behaviour before 0.33.22).

Scheduling: ``agent_turn`` collects the eligible results of one tool round and
calls :func:`start_round_probes`. Probes for the round run concurrently
(``asyncio.gather``), small results are batched into one model call, and a
flagged result appends an ``injection_warning`` event, which
``events_to_messages`` renders as a warning prefix on the matching
``tool_result``. The next LLM call waits at most
``ORBWEAVER_INJECTION_PROBE_BUDGET_S`` for the round's probes; a slower probe
keeps running in the background and its warning lands in the *following*
prompt instead. Every probe fails open.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

from orbweaver.config import settings
from orbweaver.permissions.prompts import (
    INJECTION_PROBE_BATCH_SYSTEM,
    INJECTION_PROBE_SYSTEM,
    INJECTION_WARNING,
)
from orbweaver.permissions.rules import _workspace_root, in_project_path

log = logging.getLogger(__name__)

PROBE_TOOLS = frozenset(
    {
        "WebFetch",
        "WebSearch",
        "Browser",
        "Bash",
        "Read",
        "MemorySearch",
        "MemoryReflect",
        "WorkspaceSearch",
    }
)
# Tools whose result comes from files: trusted when the path is in the project.
PATH_TOOLS = frozenset({"Read", "Grep", "Glob", "WorkspaceSearch"})
MIN_CHARS = 40
HEAD_CHARS = 4000
TAIL_CHARS = 2000
CHUNK = HEAD_CHARS  # backward-compatible alias
# Results up to this many (truncated) chars share one probe call.
BATCH_ITEM_CHARS = 2000
BATCH_MAX_ITEMS = 6
BATCH_MAX_CHARS = 8000
WARNING = INJECTION_WARNING

DEFAULT_SKIP_DIRS = (
    "node_modules,vendor,third_party,.orbweaver/tool-results,.orbweaver-tmp,"
    ".venv,venv,site-packages,__pycache__,.git,dist,build,target"
)
DEFAULT_BASH_NETWORK = [
    r"\bcurl\b",
    r"\bwget\b",
    r"\bpip3?\b",
    r"\buv\s+(pip|add|sync|tool)\b",
    r"\bpoetry\s+(add|install|update)\b",
    r"\bnpm\b",
    r"\bnpx\b",
    r"\byarn\b",
    r"\bpnpm\b",
    r"\bbun\s+(add|install|x)\b",
    r"\bcargo\s+(install|fetch|add|update)\b",
    r"\bgo\s+(get|install|mod\s+download)\b",
    r"\bgit\s+(fetch|pull|clone|ls-remote|submodule|remote\s+update)\b",
    r"\bgh\b",
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b",
    r"\bnc\b",
    r"\bncat\b",
    r"\bnetcat\b",
    r"\btelnet\b",
    r"\bhttpx?\b",
    r"\bhttpie\b",
    r"\bapt(-get)?\s+(install|update)\b",
    r"\bdocker\s+(pull|run|build)\b",
    r"https?://",
]
DEFAULT_PREFILTER = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?|rules?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"\byou\s+are\s+now\b",
    r"\bsystem\s+prompt\b",
    r"</?\s*(system|assistant|user|human)\s*>",
    r"\bIMPORTANT\s*:",
    r"\bdo\s+not\s+(tell|inform|show|reveal|mention)\s+(this\s+to\s+)?the\s+user\b",
    r"\bnew\s+instructions?\b",
    r"\bfrom\s+now\s+on\b",
    r"\bas\s+an\s+ai\b",
    r"\b(assistant|ai|agent|model)\s*:\s*(you\s+must|please|run|execute)\b",
    r"\bcurl\b[^\n]{0,120}\|\s*(ba)?sh\b",
    r"\b(run|execute)\s+(this|the\s+following)\s+command\b",
]

_INJ_RE = re.compile(r"<injection>\s*(yes|no)\s*</injection>", re.IGNORECASE)
_INJ_BATCH_RE = re.compile(
    r"<injection\s+id=\"?(\d+)\"?\s*>\s*(yes|no)\s*</injection>", re.IGNORECASE
)


# ---------------------------------------------------------------- settings ---


def probe_mode() -> str:
    mode = (settings.orbweaver_injection_probe_mode or "").strip().lower()
    if mode not in {"off", "scoped", "all"}:
        return "scoped"
    return mode


def _slot(user_value: str, default: str) -> str:
    raw = (user_value or "").strip() or "$defaults"
    return raw.replace("$defaults", default)


def _split_patterns(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in re.split(r"[;\n]", raw) if p.strip())


@lru_cache(maxsize=16)
def _compile(patterns: tuple[str, ...]) -> re.Pattern[str] | None:
    good: list[str] = []
    for p in patterns:
        try:
            re.compile(p)
        except re.error as e:
            log.warning("injection probe: ignoring bad regex %r: %s", p, e)
            continue
        good.append(f"(?:{p})")
    if not good:
        return None
    return re.compile("|".join(good), re.IGNORECASE)


def skip_dirs() -> list[str]:
    raw = _slot(settings.orbweaver_injection_probe_skip_dirs, DEFAULT_SKIP_DIRS)
    return [d.strip().strip("/") for d in raw.split(",") if d.strip().strip("/")]


def bash_network_regex() -> re.Pattern[str] | None:
    raw = _slot(settings.orbweaver_injection_probe_bash_network, ";".join(DEFAULT_BASH_NETWORK))
    return _compile(_split_patterns(raw))


def prefilter_regex() -> re.Pattern[str] | None:
    raw = _slot(settings.orbweaver_injection_probe_prefilter, ";".join(DEFAULT_PREFILTER))
    return _compile(_split_patterns(raw))


def probe_budget_s() -> float:
    try:
        return max(0.0, float(settings.orbweaver_injection_probe_budget_s))
    except (TypeError, ValueError):
        return 3.0


# ------------------------------------------------------------------- scope ---


def is_probe_tool(name: str) -> bool:
    return name in PROBE_TOOLS or name.startswith("mcp_")


def bash_touches_network(command: str) -> bool:
    rx = bash_network_regex()
    return bool(rx and rx.search(command or ""))


def looks_like_instructions(text: str) -> bool:
    """Cheap prefilter: instruction-like phrasing that warrants a model probe."""
    rx = prefilter_regex()
    return bool(rx and rx.search(text or ""))


def _under_skip_dir(path: str, workspace) -> bool:
    dirs = skip_dirs()
    if not dirs:
        return False
    root = _workspace_root(workspace)
    p = Path(path or ".").expanduser()
    if root is not None and not p.is_absolute():
        p = root / p
    try:
        p = p.resolve()
    except OSError:
        return True
    posix = p.as_posix()
    if root is not None:
        try:
            posix = p.relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    hay = f"/{posix.strip('/')}/"
    return any(f"/{d}/" in hay for d in dirs)


def path_is_trusted(path: str, workspace) -> bool:
    """In the project's writable roots and not under a vendored/generated dir."""
    rel = path or "."
    if not in_project_path(rel, workspace):
        return False
    return not _under_skip_dir(rel, workspace)


def should_probe(name: str, inp: dict[str, Any] | None, output: str, workspace=None) -> bool:
    """Decide whether this result gets a model probe under the configured mode.

    ``inp`` is the tool_use input (Read ``path``, Bash ``command``); ``workspace``
    supplies the project roots for the trust tier.
    """
    mode = probe_mode()
    if mode == "off" or not settings.anthropic_api_key:
        return False
    if not is_probe_tool(name):
        return False
    body = output or ""
    if len(body.strip()) < MIN_CHARS:
        return False
    if mode == "all":
        return True
    inp = inp or {}
    if name in PATH_TOOLS:
        # Grep/Glob/WorkspaceSearch run over the workspace root; Read names a path.
        path = str(inp.get("path") or "") if name == "Read" else ""
        if path_is_trusted(path, workspace):
            return looks_like_instructions(body)
        return True
    if name == "Bash":
        command = str(inp.get("command") or "")
        return bash_touches_network(command) or looks_like_instructions(body)
    return True


# ------------------------------------------------------------------- probe ---


def truncate_for_probe(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + "\n…\n" + text[-TAIL_CHARS:]


def parse_injection(text: str) -> bool | None:
    match = _INJ_RE.search(text or "")
    if not match:
        return None
    return match.group(1).lower() == "yes"


def parse_injection_batch(text: str, n: int) -> list[bool | None]:
    verdicts: list[bool | None] = [None] * n
    for match in _INJ_BATCH_RE.finditer(text or ""):
        idx = int(match.group(1)) - 1
        if 0 <= idx < n and verdicts[idx] is None:
            verdicts[idx] = match.group(2).lower() == "yes"
    return verdicts


def _client(client=None):
    if client is not None:
        return client
    import anthropic

    headers = {}
    if settings.anthropic_workspace_id.strip():
        headers["anthropic-workspace-id"] = settings.anthropic_workspace_id.strip()
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, default_headers=headers or None
    )


def _usage_tokens(resp) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    inp += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    inp += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    return inp, out


def _resp_text(resp) -> str:
    return "".join(
        getattr(b, "text", "")
        for b in getattr(resp, "content", [])
        if getattr(b, "type", None) == "text"
    )


def _verdict(flagged: bool, output: str, **extra: Any) -> dict[str, Any]:
    out = {"flagged": flagged, "output": WARNING + output if flagged else output}
    out.update(extra)
    return out


async def probe_tool_output(name: str, output: str, *, client=None) -> dict[str, Any]:
    """Return {flagged: bool, output: str, calls, latency_s, input_tokens, output_tokens}.

    Fail open on errors. Scope decisions live in :func:`should_probe`; this
    only skips ineligible tools, tiny bodies, and a missing API key.
    """
    if not is_probe_tool(name):
        return _verdict(False, output, calls=0)
    body = output or ""
    if len(body.strip()) < MIN_CHARS:
        return _verdict(False, output, calls=0)
    if not settings.anthropic_api_key:
        return _verdict(False, output, calls=0)
    client = _client(client)
    payload = f"tool={name}\n\n{truncate_for_probe(body)}"
    t0 = time.monotonic()
    try:
        resp = await client.messages.create(
            model=settings.orbweaver_injection_probe_model,
            max_tokens=32,
            system=INJECTION_PROBE_SYSTEM,
            messages=[{"role": "user", "content": payload}],
        )
    except Exception as e:
        log.warning("injection probe failed open: %s", e)
        return _verdict(False, output, calls=1, latency_s=time.monotonic() - t0)
    latency = time.monotonic() - t0
    inp_tok, out_tok = _usage_tokens(resp)
    flagged = parse_injection(_resp_text(resp)) is True
    return _verdict(
        flagged, output, calls=1, latency_s=latency, input_tokens=inp_tok, output_tokens=out_tok
    )


async def probe_tool_outputs(items: list[tuple[str, str]], *, client=None) -> list[dict[str, Any]]:
    """Probe several (name, output) results in one model call; per-result verdicts.

    Same return shape as :func:`probe_tool_output` for each item; ``calls``,
    latency, and tokens are attributed to the first item so sums stay correct.
    """
    if not items:
        return []
    if len(items) == 1:
        name, output = items[0]
        return [await probe_tool_output(name, output, client=client)]
    if not settings.anthropic_api_key:
        return [_verdict(False, output, calls=0) for _name, output in items]
    client = _client(client)
    parts = []
    for i, (name, output) in enumerate(items, start=1):
        parts.append(
            f'<result id="{i}" tool="{name}">\n{truncate_for_probe(output or "")}\n</result>'
        )
    payload = "\n\n".join(parts)
    t0 = time.monotonic()
    try:
        resp = await client.messages.create(
            model=settings.orbweaver_injection_probe_model,
            max_tokens=16 * len(items) + 16,
            system=INJECTION_PROBE_BATCH_SYSTEM,
            messages=[{"role": "user", "content": payload}],
        )
    except Exception as e:
        log.warning("injection probe batch failed open: %s", e)
        latency = time.monotonic() - t0
        out = [_verdict(False, output, calls=0) for _name, output in items]
        out[0].update(calls=1, latency_s=latency)
        return out
    latency = time.monotonic() - t0
    inp_tok, out_tok = _usage_tokens(resp)
    verdicts = parse_injection_batch(_resp_text(resp), len(items))
    out = [
        _verdict(v is True, output, calls=0)
        for v, (_name, output) in zip(verdicts, items, strict=True)
    ]
    out[0].update(calls=1, latency_s=latency, input_tokens=inp_tok, output_tokens=out_tok)
    return out


# ------------------------------------------------------------- scheduling ---


@dataclass
class ProbeItem:
    tool_use_id: str
    name: str
    output: str


ProbeFn = Callable[..., Awaitable[dict[str, Any]]]
ProbeBatchFn = Callable[..., Awaitable[list[dict[str, Any]]]]
FlaggedFn = Callable[[ProbeItem], Awaitable[None]]


def plan_batches(items: list[ProbeItem]) -> list[list[ProbeItem]]:
    """Large results get their own call; small ones share a call, in order."""
    batches: list[list[ProbeItem]] = []
    current: list[ProbeItem] = []
    current_chars = 0
    for item in items:
        size = len(truncate_for_probe(item.output or ""))
        if size > BATCH_ITEM_CHARS:
            batches.append([item])
            continue
        if current and (len(current) >= BATCH_MAX_ITEMS or current_chars + size > BATCH_MAX_CHARS):
            batches.append(current)
            current, current_chars = [], 0
        current.append(item)
        current_chars += size
    if current:
        batches.append(current)
    return batches


async def run_round_probes(
    items: list[ProbeItem],
    *,
    on_flagged: FlaggedFn,
    session_id: UUID | None = None,
    client=None,
    probe: ProbeFn | None = None,
    probe_batch: ProbeBatchFn | None = None,
) -> int:
    """Probe one round's results concurrently. Returns the number flagged."""
    if not items:
        return 0
    probe_fn: ProbeFn = probe or probe_tool_output
    batch_fn: ProbeBatchFn = probe_batch or probe_tool_outputs

    async def one(batch: list[ProbeItem]) -> int:
        t0 = time.monotonic()
        if len(batch) == 1:
            verdicts = [await probe_fn(batch[0].name, batch[0].output, client=client)]
        else:
            verdicts = await batch_fn([(i.name, i.output) for i in batch], client=client)
        flagged = 0
        for item, verdict in zip(batch, verdicts, strict=True):
            if verdict.get("flagged"):
                flagged += 1
                await on_flagged(item)
        if session_id is not None:
            from orbweaver.compact.usage import record_probe_usage

            record_probe_usage(
                session_id,
                calls=sum(int(v.get("calls", 1) or 0) for v in verdicts),
                results=len(batch),
                flagged=flagged,
                latency_s=time.monotonic() - t0,
                input_tokens=sum(int(v.get("input_tokens", 0) or 0) for v in verdicts),
                output_tokens=sum(int(v.get("output_tokens", 0) or 0) for v in verdicts),
            )
        return flagged

    results = await asyncio.gather(*(one(b) for b in plan_batches(items)), return_exceptions=True)
    total = 0
    for r in results:
        if isinstance(r, BaseException):
            log.warning("injection probe round failed open: %s", r)
        else:
            total += r
    return total


_BACKGROUND: set[asyncio.Task[Any]] = set()


def start_round_probes(items: list[ProbeItem], **kwargs: Any) -> asyncio.Task[int] | None:
    """Launch :func:`run_round_probes` as a task kept alive off the critical path."""
    if not items:
        return None
    task = asyncio.create_task(run_round_probes(items, **kwargs))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


async def await_round_probes(task: asyncio.Task[int] | None, budget_s: float | None = None) -> bool:
    """Wait up to the budget for a round's probes. False means they are still running."""
    if task is None or task.done():
        return True
    budget = probe_budget_s() if budget_s is None else max(0.0, budget_s)
    done, _pending = await asyncio.wait({task}, timeout=budget)
    return bool(done)
