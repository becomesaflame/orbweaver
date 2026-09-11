"""Per-tool metadata: which tools are read-only and safe to run concurrently.

Claude Code gives every tool ``isReadOnly()`` / ``isConcurrencySafe()``; Codex
routes each call through an ``RwLock`` (safe tools take the read lock, unsafe
tools take the write lock). Orbweaver keeps the same shape as data: a run of
consecutive concurrency-safe ``tool_use`` blocks executes together, and every
unsafe block is a barrier that runs alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orbweaver.mcp.tools import mcp_tool_read_only


@dataclass(frozen=True)
class ToolMeta:
    read_only: bool = False
    concurrency_safe: bool = False


# Tools that do not modify the workspace, run commands, or wait on the user.
# TodoWrite only updates the session's plan, so it is safe to run alongside reads.
READ_ONLY_TOOLS = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "WorkspaceSearch",
        "MemorySearch",
        "MemoryGraph",
        "WebSearch",
        "WebFetch",
        "ReadLints",
        "TodoWrite",
    }
)
CONCURRENCY_SAFE_TOOLS = READ_ONLY_TOOLS

# Concurrency-safe tools whose implementation is still synchronous I/O (file
# reads, ripgrep, httpx). A gather of these on one event loop would serialize;
# the agent loop moves them to a worker thread until #98 makes them async.
BLOCKING_SAFE_TOOLS = frozenset(
    {"Read", "Glob", "Grep", "WorkspaceSearch", "WebSearch", "WebFetch", "ReadLints"}
)

_SAFE_META = ToolMeta(read_only=True, concurrency_safe=True)
_UNSAFE_META = ToolMeta()


def tool_meta(name: str) -> ToolMeta:
    """Metadata for a built-in or MCP tool name. Unknown tools are unsafe."""
    if name in CONCURRENCY_SAFE_TOOLS:
        return _SAFE_META
    if name.startswith("mcp_"):
        # MCP tools are unsafe unless the server annotated them readOnlyHint: true.
        return _SAFE_META if mcp_tool_read_only(name) else _UNSAFE_META
    return _UNSAFE_META


def is_read_only(name: str) -> bool:
    return tool_meta(name).read_only


def is_concurrency_safe(name: str) -> bool:
    return tool_meta(name).concurrency_safe


def partition_tool_calls(names: Sequence[str]) -> list[list[int]]:
    """Split one round's tool calls into execution runs, as lists of indices.

    Consecutive concurrency-safe calls share a run; each unsafe call is its own
    run. Runs are returned in order, so executing them one after another and
    emitting results by index keeps the original tool_use order.
    """
    runs: list[list[int]] = []
    current: list[int] = []
    for i, name in enumerate(names):
        if is_concurrency_safe(name):
            current.append(i)
            continue
        if current:
            runs.append(current)
            current = []
        runs.append([i])
    if current:
        runs.append(current)
    return runs
