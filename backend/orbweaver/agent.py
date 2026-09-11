"""Anthropic tool-calling agent loop with pin injection, permissions, and compaction."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from orbweaver import __version__
from orbweaver.checkpoints import checkpoint_user_turn, pin_user_turn
from orbweaver.compact import (
    CONTEXT_FULL_MESSAGE,
    PROJECT_INSTRUCTIONS_KIND,
    ContextFullError,
    ensure_tool_use_results,
    events_to_messages,
    live_events,
    maybe_compact,
    persist_tool_result,
    prompt_events,
    record_response_usage,
    rehydrate_messages,
)
from orbweaver.compact.overflow import (
    OVERFLOW_KEEP_ROUNDS,
    is_context_overflow,
    overflow_compact_budget,
)
from orbweaver.config import settings
from orbweaver.image import format_image_read, hydrate_workspace_images, is_image_path
from orbweaver.instructions import instruction_blocks_for_call, seen_instruction_keys
from orbweaver.lints import read_lints
from orbweaver.llm import (
    prompt_cache_supported,
    with_message_cache_breakpoint,
    with_tool_cache_breakpoint,
)
from orbweaver.memory import graph_neighborhood, pinned_prompt, remember, rewrite_search_query
from orbweaver.permissions import (
    PermissionDecision,
    TurnAborted,
    can_use_tool,
    denial_state_for,
)
from orbweaver.permissions.injection_probe import probe_tool_output
from orbweaver.permissions.pipeline import summarize_input
from orbweaver.permissions.session_rules import (
    add_session_rule,
    make_session_rule,
    session_rules_from,
)
from orbweaver.procs import BashInterrupted
from orbweaver.readstate import ReadState, read_state_for
from orbweaver.skills import workspace_skills_prompt
from orbweaver.store import Event, Job, Store, new_uuid
from orbweaver.stuck import NUDGE_KIND, StuckDetector
from orbweaver.todos import inject_session_todos, persist_todos
from orbweaver.tools import partition_tool_calls
from orbweaver.tooltext import format_read

log = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = 48
INTERRUPTED_BY_USER = "interrupted by user"
LAST_ROUND_NUDGE = (
    "This is the last tool round of this turn. After these tool results, answer the user "
    "with what you have. Do not start new exploration. Spawn a subagent only if a narrower "
    "task can finish the work."
)
CONCLUDE_NUDGE = (
    "Tool-round budget exhausted. Answer the user now from the tool results you already "
    "have. Do not call tools."
)
GIT_NOT_DONE_NUDGE = (
    "Git ritual says NOT_DONE. The last git commit or push did not finish the rebase "
    "and did not publish HEAD. Call Bash now with `git rebase --continue` "
    "(after git add of resolved files) or `git rebase --abort`. "
    "Do not tell the user the branch or PR was updated."
)
GIT_NOT_DONE_LAST_NUDGE = (
    "This is the last tool round and Git ritual says NOT_DONE. "
    "If you can, call Bash with git rebase --continue or --abort. "
    "Otherwise tell the user the rebase is unfinished (detached HEAD / rebase-in-progress) "
    "and do not claim the PR or branch was updated."
)
GIT_NOT_DONE_CONCLUDE = (
    "Tool-round budget exhausted and Git ritual says NOT_DONE. "
    "Tell the user the rebase is unfinished (detached HEAD or rebase-in-progress). "
    "Do not claim the branch or PR was updated. Next step is git rebase --continue "
    "or --abort."
)

TOOL_SPEC = [
    {
        "name": "Read",
        "description": (
            "Read a file as numbered lines, or an image as vision (jpg/png/webp/gif). "
            "Relative paths are the session workspace. Absolute paths in extra sandbox "
            "roots are auto-allowed; other host paths are classified. Use offset "
            "(1-based line, or negative from the end) and limit to page text; do not "
            "page files with Bash. Default limit is 400 lines: Grep (or one wide Read) "
            "to find a symbol; do not take tiny windows, and do not re-Read a path "
            "already in this turn unless its result was truncated or cleared. Once you "
            "have the numbered lines, edit with StrReplace. The result says how to "
            "continue when truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "description": "1-based starting line; negative counts from the end.",
                },
                "limit": {"type": "integer", "description": "Max lines to return (default 400)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": (
            "Write a file in the session workspace. Overwriting an existing file "
            "requires that you Read it earlier this session and that it has not "
            "changed on disk since; otherwise the call is refused and you must Read "
            "first. New files skip that check. The result is a unified diff for "
            "existing files. Prefer StrReplace for partial edits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "StrReplace",
        "description": (
            "Edit an existing workspace file by replacing old_string with new_string. "
            "Read the file first: edits to a file you have not Read this session, or "
            "that changed on disk since you read it, are refused. old_string must "
            "match exactly once unless replace_all is true. When no exact match exists "
            "the tool retries with line-number prefixes stripped, then a per-line "
            "whitespace-trimmed match (the file's indentation is kept), then a "
            "first/last-line anchor for blocks of 3+ lines; the result says which tier "
            "matched and shows a unified diff. Prefer this over Write for existing "
            "files and over Bash (python, sed, perl) for source edits. To resolve a git "
            "conflict, replace the entire hunk including <<<<<<< / ======= / >>>>>>> "
            "marker lines with the resolved text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every match (default false: unique match required).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "NotebookEdit",
        "description": (
            "Edit one cell in a .ipynb notebook. Do not Write the whole notebook JSON. "
            "Read the notebook first this session or the edit is refused. action is "
            "replace (default), insert, or delete. replace can set source or "
            "search-replace with old_string/new_string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "cell_idx": {"type": "integer", "description": "0-based cell index."},
                "action": {"type": "string", "enum": ["replace", "insert", "delete"]},
                "source": {"type": "string", "description": "Full replacement or insert source."},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown", "raw"],
                    "description": "Cell type for insert (default code).",
                },
            },
            "required": ["path", "cell_idx"],
        },
    },
    {
        "name": "Delete",
        "description": (
            "Delete a file or directory in the session working set. Same deny/ask rules as "
            "Write: always-deny secrets (.env, keys), and paths outside the working set are "
            "blocked. Deleting a file requires that you Read it this session. Prefer this "
            "over Bash rm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "ProposePatch",
        "description": "Propose a search-replace patch (also applied as a review overlay in VS Code).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "new_string"],
        },
    },
    {
        "name": "Glob",
        "description": (
            "List files matching a glob, recursively from the workspace root (ripgrep --files). "
            "Skips gitignored paths. Caps at 200 files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": (
            "Search workspace file contents with ripgrep. Regex; use | for alternation "
            "(a\\|b is treated as a|b). glob and type limit files. A/B/C add context lines. "
            "Skips binary and gitignored files. Caps hits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string"},
                "type": {
                    "type": "string",
                    "description": "ripgrep file type, e.g. py, md, rust",
                },
                "A": {"type": "integer", "description": "lines after each match"},
                "B": {"type": "integer", "description": "lines before each match"},
                "C": {"type": "integer", "description": "lines before and after each match"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "ReadLints",
        "description": (
            "Return diagnostics for files you just edited. Runs ORBWEAVER_LINTER when set "
            "(use {paths} or trailing paths). Otherwise a Python AST stub reports syntax "
            "errors. Paths default to recent Write/StrReplace/ProposePatch targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace paths to check. Empty uses recently edited files.",
                }
            },
        },
    },
    {
        "name": "Skill",
        "description": (
            "Load a workspace skill (SKILL.md) or an on-demand rule by name. The system "
            "prompt lists available skills and description-only rules as name: description; "
            "call this to read the full body before following one. kind is skill (default) "
            "or rule."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill or rule name from the list."},
                "kind": {
                    "type": "string",
                    "enum": ["skill", "rule"],
                    "description": "skill loads SKILL.md (default); rule loads a .cursor/rules "
                    "or .orbweaver/rules entry that is not always-on.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "Bash",
        "description": (
            "Run a shell command in the workspace. Default timeout is 30 seconds if unspecified; "
            "pass timeout (seconds, max 600) or block_until_ms for long commands such as pytest "
            "or installs. Set background true to start a job and return a job_id immediately; "
            "later call Bash with that job_id (optional timeout to wait) to poll or collect "
            "output when it finishes. Sandboxed by default: host files are readable, "
            "writes stay in the working set, network uses a domain allowlist, Unix sockets are "
            "denied unless granted. Host reads and allowlisted sockets/domains do not need "
            "escalation. If the sandbox blocks the command, ask the user before setting "
            "permissions to [\"full_network\"] (arbitrary internet) or [\"all\"] "
            "(host writes/docker/sudo). Those overrides pause for approval. "
            "unsandboxed true aliases [\"all\"]. Every result starts with a header line "
            "(`exit <code> in <seconds>s`, or `timed out after Ns`); trust it over success "
            "substrings in the output. Oversized output is saved under "
            ".orbweaver/tool-results/ and shown as head + tail around an omission marker. "
            "Git commands get an automatic status footer (branch, HEAD, rebase-in-progress)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run. Required unless job_id is set.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 30, max 600). Job lifetime when background.",
                },
                "block_until_ms": {
                    "type": "integer",
                    "description": "Alternate timeout in milliseconds (same default/cap as timeout).",
                },
                "background": {
                    "type": "boolean",
                    "description": "If true, start the command and return a job_id without waiting.",
                },
                "job_id": {
                    "type": "string",
                    "description": "Poll/collect a previously backgrounded job instead of running a command.",
                },
                "unsandboxed": {"type": "boolean"},
                "permissions": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["full_network", "all"]},
                },
            },
        },
    },
    {
        "name": "WebFetch",
        "description": (
            "HTTP GET a known URL and return extracted readable text (HTML is stripped). "
            "Find URLs with WebSearch first. When the result says the page is "
            "JavaScript-rendered, use Browser instead of fetching nearby docs URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "Browser",
        "description": (
            "Drive a headless Chromium session to verify UI. Actions: navigate, click, type, "
            "snapshot (visible text and controls after JavaScript), screenshot (PNG under "
            "attachments/). Use this when WebFetch reports a JavaScript-rendered page or when "
            "you changed web/ and need to click through a flow. Requires "
            'pip install -e ".[browser]" and playwright install chromium. file:// and '
            "workspace-relative paths must stay in the workspace. Classified like WebFetch "
            "(not auto-allowed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "click", "type", "snapshot", "screenshot"],
                },
                "url": {
                    "type": "string",
                    "description": "navigate: http(s) URL, file://, or workspace-relative path",
                },
                "selector": {"type": "string", "description": "CSS selector for click/type"},
                "text": {"type": "string", "description": "Text to type into the selector"},
                "path": {
                    "type": "string",
                    "description": (
                        "screenshot: workspace PNG path, default attachments/browser-<id>.png"
                    ),
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "WebSearch",
        "description": (
            "Search the public web and return titled result URLs with snippets. Use this "
            "instead of guessing documentation URLs. Then WebFetch one or two promising links. "
            "Call independent searches in parallel in one round."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "description": "How many hits to return (default 8, max 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "WorkspaceSearch",
        "description": (
            "Hybrid search over project source files in the session workspace. Indexes text "
            "files on demand (skips binaries, huge files, .git, and gitignored paths). Use "
            "this to find where something is implemented. Do not MemoryRemember every file. "
            "MemorySearch is only for stored facts, not source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "description": "How many hits to return (default 8, max 20).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "MemorySearch",
        "description": (
            "Search shared memory (remembered facts), not project source. Use WorkspaceSearch "
            "to find code. When Hindsight is configured this is recall() (semantic, keyword, "
            "graph, temporal). Otherwise native vector search; graph neighbors of hit chunks "
            "are included unless expand_graph is false. Use MemoryReflect to synthesize."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "expand_graph": {
                    "type": "boolean",
                    "description": "Include JSON-LD neighbors of matching chunks (default true).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "MemoryGraph",
        "description": (
            "Walk the shared memory graph around an entity @id (same neighborhood as "
            "GET /memory/graph). Use after MemorySearch when you have an entity id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Entity @id to walk from."},
                "depth": {
                    "type": "integer",
                    "description": "Neighborhood hops (default 1, max 4).",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "MemoryRemember",
        "description": "Store a fact in shared memory (Hindsight retain when configured).",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "pinned": {"type": "boolean"}},
            "required": ["text"],
        },
    },
    {
        "name": "MemoryReflect",
        "description": (
            "Synthesize from shared memory: what is currently true, preferences, or a "
            "judgment across facts. Writes opinions into the bank. Do not use this as the "
            "user-visible answer for a whole turn — summarize for the user yourself. "
            "Use MemorySearch for raw facts. Requires Hindsight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "budget": {
                    "type": "string",
                    "enum": ["low", "mid", "high"],
                    "description": "How thorough the reflect loop is (default low).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "MemoryPin",
        "description": "Pin a chunk or entity into always-injected working memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["chunk", "entity"]},
                "id": {"type": "string"},
            },
            "required": ["kind", "id"],
        },
    },
    {
        "name": "MemoryForget",
        "description": "Tombstone a memory chunk.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "AskUser",
        "description": (
            "Ask the user a question and wait for their reply on web or Telegram. "
            "The turn pauses until they answer; that reply becomes this tool's result. "
            "Do not use this on cron or other headless sessions — those abort."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "TodoWrite",
        "description": (
            "Create or update the session todo list. Persisted on the session so compaction "
            "does not erase the plan. merge true updates by id; merge false replaces the list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                        },
                        "required": ["content"],
                    },
                },
                "merge": {
                    "type": "boolean",
                    "description": "If true, merge by id into the existing list.",
                },
            },
            "required": ["todos"],
        },
    },
    {
        "name": "SpawnSubagent",
        "description": (
            "Spawn a nested agent with its own event stream to complete a focused task. "
            "Shares this workspace and memory. Returns a summary. Nested spawns are not allowed. "
            "type/role selects a tool subset: explore (read/search), implement (default, edits), "
            "or shell (Bash). With background true it returns {subagent_id, status: running} "
            "at once and the child runs concurrently; spawn several in one round to fan out. "
            "Finished background results are delivered to you automatically after your "
            "next reply, or call SubagentWait to block for them. Children are cancelled if "
            "you end the turn without collecting them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string", "description": "Short name for the child run."},
                "type": {
                    "type": "string",
                    "enum": ["explore", "implement", "shell"],
                    "description": "Subagent role. Default implement.",
                },
                "role": {
                    "type": "string",
                    "enum": ["explore", "implement", "shell"],
                    "description": "Alias for type.",
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "If true, return immediately and run the child concurrently "
                        "(default false: wait for the result)."
                    ),
                },
                "timeout_s": {
                    "type": "number",
                    "description": (
                        "Wall-clock limit for the child in seconds (default 600). "
                        "On timeout the child is cancelled and the result is an error."
                    ),
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Tool-round budget for the child (default 24).",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "SubagentWait",
        "description": (
            "Block until background subagents finish and return their results "
            "({subagent_id, status, text} each). ids selects children; omit it to wait "
            "for every child spawned this turn whose result you have not seen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "subagent_id values from SpawnSubagent (default: all pending).",
                }
            },
        },
    },
    {
        "name": "ScheduleTask",
        "description": (
            "Ask the host to schedule a job. due_at is ISO-8601. "
            "Optional recurrence: minute, hour, or day (also 'every hour'), "
            "or a 5-field cron expression (minute hour day-of-month month "
            "day-of-week), e.g. '0 9 * * mon'. Omit recurrence for a one-shot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "due_at": {"type": "string"},
                "message": {"type": "string"},
                "recurrence": {
                    "type": "string",
                    "description": (
                        "minute/hour/day, or 5-field cron "
                        "(minute hour day month weekday)."
                    ),
                },
            },
            "required": ["due_at", "message"],
        },
    },
    {
        "name": "SendPhoto",
        "description": (
            "Send an image file from the workspace to the user on Telegram. "
            "Use this for screenshots, charts, or other images you wrote under attachments/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "caption": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "GenerateImage",
        "description": (
            "Generate an image from a text prompt and save it under attachments/. "
            "On Telegram sessions the file is also sent to the chat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "Workspace path to write, default attachments/generated-<id>.png",
                },
                "caption": {"type": "string"},
            },
            "required": ["prompt"],
        },
    },
]

PROPOSE_PATCH_TOOL = "ProposePatch"
VSCODE_CHANNEL = "vscode"


def normalize_channel(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("_", "-").replace(" ", "")
    if raw in {"vscode", "vs-code", "visualstudiocode"}:
        return VSCODE_CHANNEL
    return raw


def resolve_channel(
    jsonld: dict[str, Any] | None = None,
    *,
    channel: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Client channel for tool/prompt filtering. Only vscode keeps ProposePatch."""
    if channel and str(channel).strip():
        return normalize_channel(channel)
    extra = extra or {}
    extra_ch = extra.get("channel") or extra.get("client")
    if extra_ch and str(extra_ch).strip():
        return normalize_channel(str(extra_ch))
    jsonld = jsonld or {}
    stored = jsonld.get("channel") or jsonld.get("client")
    if stored and str(stored).strip():
        return normalize_channel(str(stored))
    if jsonld.get("telegram_user_id") is not None or jsonld.get("telegram_chat_id") is not None:
        return "telegram"
    title = str(jsonld.get("title") or "").strip().lower()
    if title == "vscode" or title.startswith(("vscode:", "vscode/")):
        return VSCODE_CHANNEL
    return ""


def channel_allows_proposepatch(channel: str) -> bool:
    return normalize_channel(channel) == VSCODE_CHANNEL


def tools_for_channel(
    channel: str,
    tool_spec: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    spec = list(tool_spec if tool_spec is not None else TOOL_SPEC)
    if channel_allows_proposepatch(channel):
        return spec
    return [t for t in spec if t.get("name") != PROPOSE_PATCH_TOOL]


async def session_tools(workspace=None, channel: str | None = None) -> list[dict[str, Any]]:
    """Built-in tools (channel-filtered) plus tools from configured MCP servers."""
    from orbweaver.mcp import mcp_tool_specs

    extra = await mcp_tool_specs(workspace)
    base = list(TOOL_SPEC) if channel is None else tools_for_channel(channel)
    return base + extra


def _events_to_messages(events: list[Event]) -> list[dict[str, Any]]:
    """Build Anthropic messages from a (possibly projected) event list."""
    return events_to_messages(events)


def _nudge_user(messages: list[dict[str, Any]], text: str) -> None:
    if messages and messages[-1].get("role") == "user":
        prev = messages[-1]["content"]
        if isinstance(prev, str):
            messages[-1]["content"] = prev + "\n\n" + text
            return
        if isinstance(prev, list):
            prev.append({"type": "text", "text": text})
            return
    messages.append({"role": "user", "content": text})


def _prompt_messages(events: list[Event], workspace, user_text: str) -> list[dict[str, Any]]:
    messages = events_to_messages(prompt_events(events))
    messages = hydrate_workspace_images(messages, workspace)
    messages = rehydrate_messages(messages, events, workspace)
    messages = inject_session_todos(messages, events)
    if not messages:
        messages = [{"role": "user", "content": user_text}]
    return ensure_tool_use_results(messages)


def static_system(channel: str = "") -> str:
    write_line = (
        "In auto mode, in-project Write and StrReplace apply immediately. "
        "Prefer StrReplace for existing files. Read a file before you edit it: "
        "Write, StrReplace, NotebookEdit and Delete refuse existing files you have "
        "not Read this session or that changed since. "
    )
    if channel_allows_proposepatch(channel):
        write_line += "Prefer ProposePatch when a visible diff overlay helps the user. "
    return (
        f"You are Orbweaver, a coding agent. The running gateway is Orbweaver {__version__} "
        f"(semantic version). If asked what version is running, answer {__version__}. "
        "Use tools to read and patch the workspace. Sandboxed Bash can read host files; "
        "do not set permissions [\"all\"] just to inspect logs or journals. "
        f"{write_line}"
        "In auto mode, in-project Delete applies immediately. TodoWrite keeps the plan "
        "on this session across compaction. After edits, ReadLints for diagnostics. "
        "Use NotebookEdit for .ipynb cells instead of rewriting the whole JSON. "
        "Use WorkspaceSearch to find code in the workspace. Use MemorySearch when past "
        "decisions or stored facts might matter, MemoryReflect to synthesize what is "
        "still true or what we prefer, and MemoryGraph to walk native entity "
        "neighborhoods. Keep pins small. If the sandbox cannot run a command, ask the user "
        "before requesting permissions [\"full_network\"] or [\"all\"]. Hard denials "
        "stay blocked; do not route around them. Call independent tools in parallel in "
        "one round. Prefer Read offset/limit and Grep (ripgrep) over Bash for paging files. "
        "Use WebSearch to find sources, then WebFetch a few result URLs; do not guess "
        "docs paths. Use Browser to verify JavaScript UI (navigate, click, type, snapshot). "
        "Configured MCP servers appear as mcp_<server>_<tool> and use the "
        "same permission pipeline as other tools. "
        "Project instructions: the workspace section below holds always-on rules; "
        "skills and on-demand rules are listed by name, so call Skill(name) before "
        "following one. Nested AGENTS.md / CLAUDE.md / .cursor/rules for a directory "
        "arrive as <project-instructions dir=...> blocks after you touch a path under it; "
        "follow them for work in that directory. "
        "Git: after clone, fetch, rebase, merge, commit, or push, read Git's state — "
        "the Bash footer 'git ritual' (status, branch, HEAD) is authoritative, not a "
        "success substring like 'Everything up-to-date'. Identify the repo (cd target "
        "or workspace root); do not rebase a nested shared clone by accident. "
        "For conflicts, keep current main and re-apply the small feature; use "
        "StrReplace on the whole hunk including <<<<<<< / ======= / >>>>>>> "
        "markers. Do not rewrite files with Python, sed, or Bash str.replace. "
        "After resolving, git add those files and "
        "git rebase --continue (or merge --continue). git commit is not continue. "
        "While detached or rebase-in-progress, git push origin <branch> updates the "
        "old branch tip, not HEAD. If the footer says NOT_DONE, the rebase is "
        "unfinished; next git must be rebase --continue or --abort, not a success "
        "claim. Do not git add -A if it would stage junk "
        "(.venv, .orbweaver-tmp). Never force-push main. Never rewrite history "
        "unless the user asked. "
        "Finish with a user-visible answer "
        "before the tool-round budget runs out; spawn a subagent for a long exploration "
        "instead of burning parent rounds."
    )


def build_agent_system(
    pins: str, extra: str = "", skills: str = "", channel: str = ""
) -> list[dict[str, Any]]:
    rest = pins or "(no pinned memory)"
    if skills:
        rest = rest + "\n\n" + skills
    if extra:
        rest = rest + "\n\n" + extra
    return [
        {"type": "text", "text": static_system(channel), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": rest},
    ]


def _nested_instructions_for(
    workspace, name: str, inp: dict[str, Any], ctx: dict[str, Any]
) -> list[str]:
    """Nested AGENTS.md / CLAUDE.md / .cursor/rules blocks for paths this call touched."""
    root = getattr(workspace, "root", None)
    if root is None:
        return []
    seen = ctx.setdefault("instructions_seen", set())
    try:
        return instruction_blocks_for_call(Path(root), name, inp, seen)
    except Exception as e:  # discovery must never break the tool round
        log.warning("nested instruction discovery failed for %s: %s", name, e)
        return []


def _blocked_tool_result(decision) -> str:
    if decision.behavior == "ask":
        return (
            "This action needs user approval and was not executed. "
            f"Reason: {decision.reason}. Wait for the user to confirm, then retry."
        )
    return (
        f"Blocked by permission gate ({decision.fast_path}): {decision.reason}. "
        "Treat this boundary in good faith. Find a safer path; do not route around the block."
    )


APPROVAL_DECISIONS = frozenset({"allow", "deny"})
APPROVAL_SCOPES = frozenset({"once", "session"})
APPROVAL_TIMEOUT_NOTICE = (
    "The approval request timed out and the action was not executed. "
    "Reply to continue."
)


def _denied_tool_result(reason: str) -> str:
    return (
        f"Denied by user. This action needed approval ({reason}) and the user declined. "
        "Do not retry the same action; adapt or ask what they want instead."
    )


def _approval_timeout_result(reason: str, timeout: float) -> str:
    return (
        f"Not executed: no approval decision within {timeout:g}s "
        f"(needed approval: {reason})."
    )


def _approval_cancelled_result(reason: str) -> str:
    return f"Not executed: the turn was cancelled while waiting for approval ({reason})."


@dataclass
class PendingApproval:
    """A held tool call waiting for the user's allow / deny."""

    session_id: UUID
    tool_use_id: str
    name: str
    input: dict[str, Any]
    reason: str
    future: asyncio.Future[tuple[str, str]]


# session_id -> tool_use_id -> pending approval. One turn runs per session, so the
# session is the natural key for the HTTP/WS/Telegram decision transports.
_pending_approvals: dict[UUID, dict[str, PendingApproval]] = {}


def pending_approvals(session_id: UUID) -> list[PendingApproval]:
    return [p for p in _pending_approvals.get(session_id, {}).values() if not p.future.done()]


def resolve_approval(
    session_id: UUID, tool_use_id: str, decision: str, scope: str = "once"
) -> bool:
    """Deliver a decision to a held tool call. False when nothing is waiting for it."""
    decision = str(decision or "").strip().lower()
    scope = str(scope or "once").strip().lower() or "once"
    if decision not in APPROVAL_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(APPROVAL_DECISIONS)}")
    if scope not in APPROVAL_SCOPES:
        raise ValueError(f"scope must be one of {sorted(APPROVAL_SCOPES)}")
    pend = _pending_approvals.get(session_id, {}).get(str(tool_use_id))
    if pend is None or pend.future.done():
        return False
    pend.future.set_result((decision, scope))
    return True


def reset_pending_approvals_for_tests() -> None:
    _pending_approvals.clear()


def _register_approval(pend: PendingApproval) -> None:
    _pending_approvals.setdefault(pend.session_id, {})[pend.tool_use_id] = pend


def _unregister_approval(pend: PendingApproval) -> None:
    bucket = _pending_approvals.get(pend.session_id)
    if bucket is None:
        return
    bucket.pop(pend.tool_use_id, None)
    if not bucket:
        _pending_approvals.pop(pend.session_id, None)


async def _wait_for_approval(
    pend: PendingApproval,
    cancel: asyncio.Event | None,
    timeout: float,
) -> tuple[str, str] | None:
    """(decision, scope) once the user answers; None on timeout or cancel."""
    watchers: set[asyncio.Future[Any]] = {pend.future}
    cancel_task: asyncio.Task[Any] | None = None
    if cancel is not None:
        if cancel.is_set():
            return None
        cancel_task = asyncio.create_task(cancel.wait())
        watchers.add(cancel_task)
    try:
        done, _pending = await asyncio.wait(
            watchers, timeout=max(0.0, timeout), return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task
    if pend.future in done:
        return pend.future.result()
    return None


def can_wait_for_user(ctx: dict[str, Any]) -> bool:
    """Web and Telegram can pause for a reply; cron/subagent/other headless cannot."""
    if ctx.get("interactive"):
        return True
    return not bool(ctx.get("headless"))


def pending_ask_user(events: list[Event]) -> Event | None:
    """Unanswered AskUser tool_call, if the turn is waiting for a reply."""
    answered: set[str] = set()
    asks: list[Event] = []
    for ev in events:
        name = str((ev.payload or {}).get("name") or "")
        if ev.kind == "tool_call" and name == "AskUser":
            asks.append(ev)
        elif ev.kind in {"tool_result", "MemoryRecall"}:
            tid = (ev.payload or {}).get("tool_use_id")
            if tid:
                answered.add(str(tid))
    for ev in reversed(asks):
        uid = str((ev.payload or {}).get("id") or ev.id)
        if uid not in answered:
            return ev
    return None


def _ask_user_headless_abort(question: str) -> TurnAborted:
    text = (
        "AskUser cannot wait for a reply on this headless session. "
        "Rephrase so the work can finish without asking, or run it from web or Telegram."
    )
    return TurnAborted(
        text,
        {
            "reason": "ask_user_headless",
            "last_tool": "AskUser",
            "last_input": question[:240],
            "text": text,
        },
    )


def _read_state(ctx: dict[str, Any], ws: Any) -> ReadState | None:
    """Session ReadState (seeded from events once); None when tracking is off."""
    if not settings.edit_require_read or not hasattr(ws, "edit_target"):
        return None
    state = ctx.get("read_state")
    if not isinstance(state, ReadState):
        state = read_state_for(ctx["session_id"])
        ctx["read_state"] = state
    state.seed(ctx.get("events"), ws.read_target)
    return state


def _edit_guard(ctx: dict[str, Any], ws: Any, path: str) -> str | None:
    """Error text when ``path`` exists but was not Read this session or changed since."""
    state = _read_state(ctx, ws)
    if state is None:
        return None
    try:
        target = ws.edit_target(path)
    except PermissionError:
        return None  # the tool itself reports the policy error
    problem = state.check(target, path)
    return f"error: {problem}" if problem else None


def _note_edit(ctx: dict[str, Any], ws: Any, path: str, *, deleted: bool = False) -> None:
    state = _read_state(ctx, ws)
    if state is None:
        return
    try:
        target = ws.edit_target(path)
    except PermissionError:
        return
    if deleted:
        state.forget(target)
    else:
        state.record(target)


def _note_read(ctx: dict[str, Any], ws: Any, path: str) -> None:
    state = _read_state(ctx, ws)
    if state is None:
        return
    try:
        state.record(ws.read_target(path))
    except PermissionError:
        return


class ToolInterrupted(Exception):
    """A running tool was killed because the user stopped the turn."""

    def __init__(self, message: str = INTERRUPTED_BY_USER) -> None:
        super().__init__(message)
        self.message = message


async def _bash_call(ws: Any, cancel: asyncio.Event | None, **kwargs: Any) -> str:
    """ws.bash without blocking the loop; cancel-aware when the workspace supports it."""
    run_async = getattr(ws, "bash_async", None)
    if run_async is not None:
        return await run_async(cancel=cancel, **kwargs)
    return await asyncio.to_thread(ws.bash, **kwargs)


async def run_tools(name: str, inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Execute one tool call. Blocking work runs off the event loop.

    Everything that touches the filesystem, spawns a process, or makes a
    synchronous HTTP request goes through ``asyncio.to_thread`` or an async
    client, so one session's 10-minute ``pytest`` does not stall the gateway.
    Foreground Bash honours ``ctx["cancel"]`` and raises ToolInterrupted.
    """
    ws = ctx["workspace"]
    store: Store = ctx["store"]
    session_id: UUID = ctx["session_id"]
    if name == "Read":
        path = str(inp.get("path") or "")
        if is_image_path(path):
            return await asyncio.to_thread(format_image_read, ws, path)
        try:
            raw = await asyncio.to_thread(ws.read, path)
        except (OSError, PermissionError, UnicodeDecodeError, IsADirectoryError) as e:
            return f"error reading {inp.get('path')}: {e}"
        _note_read(ctx, ws, path)
        return format_read(raw, path=path, offset=inp.get("offset"), limit=inp.get("limit"))
    if name == "Write":
        path = str(inp.get("path") or "")
        refused = _edit_guard(ctx, ws, path)
        if refused:
            return refused
        if hasattr(ws, "write_with_diff"):
            result = await asyncio.to_thread(ws.write_with_diff, path, inp["content"])
        else:
            await asyncio.to_thread(ws.write, path, inp["content"])
            result = f"wrote {path}"
        _note_edit(ctx, ws, path)
        return result
    if name == "StrReplace":
        path = str(inp.get("path") or "")
        try:
            refused = _edit_guard(ctx, ws, path)
            if refused:
                return refused
            result = await asyncio.to_thread(
                ws.str_replace,
                path,
                str(inp.get("old_string") or ""),
                str(inp.get("new_string") if inp.get("new_string") is not None else ""),
                replace_all=bool(inp.get("replace_all")),
            )
        except (OSError, PermissionError, UnicodeDecodeError, IsADirectoryError) as e:
            return f"error replacing in {inp.get('path')}: {e}"
        if result.startswith("updated"):
            _note_edit(ctx, ws, path)
        return result
    if name == "NotebookEdit":
        from orbweaver.notebook import apply_notebook_edit

        path = str(inp.get("path") or "")
        refused = _edit_guard(ctx, ws, path)
        if refused:
            return refused
        result = await asyncio.to_thread(apply_notebook_edit, ws, inp)
        if not result.startswith("error"):
            _note_edit(ctx, ws, path)
        return result
    if name == "Delete":
        path = str(inp.get("path") or "")
        try:
            refused = _edit_guard(ctx, ws, path)
            if refused:
                return refused
            result = await asyncio.to_thread(ws.delete, path)
        except (OSError, PermissionError) as e:
            return f"error deleting {inp.get('path')}: {e}"
        _note_edit(ctx, ws, path, deleted=True)
        return result
    if name == "ProposePatch":
        result = await asyncio.to_thread(
            ws.propose_patch, inp["path"], inp.get("old_string") or "", inp["new_string"]
        )
        return json.dumps(result)[:200_000]
    if name == "Glob":
        try:
            hits = await asyncio.to_thread(ws.glob, inp["pattern"])
            return "\n".join(hits[:200])
        except (FileNotFoundError, RuntimeError, TimeoutError) as e:
            return str(e)
    if name == "Grep":
        from orbweaver.workspace import _ctx_int

        try:
            hits = await asyncio.to_thread(
                ws.grep,
                inp["pattern"],
                inp.get("glob") or "**/*",
                file_type=inp.get("type") or None,
                after=_ctx_int(inp.get("A")),
                before=_ctx_int(inp.get("B")),
                context=_ctx_int(inp.get("C")),
            )
            return "\n".join(hits)
        except (FileNotFoundError, TimeoutError) as e:
            return str(e)
    if name == "ReadLints":
        return await asyncio.to_thread(read_lints, ws, inp, ctx.get("events"))
    if name == "Skill":
        from orbweaver.instructions import load_skill_or_rule

        root = getattr(ws, "root", None)
        if root is None:
            return "error: this workspace has no local root for skills"
        return await asyncio.to_thread(
            load_skill_or_rule,
            Path(root),
            str(inp.get("name") or ""),
            str(inp.get("kind") or "skill"),
        )
    if name == "Bash":
        from orbweaver.git_ritual import annotate_bash_output, ritual_says_not_done
        from orbweaver.permissions.pipeline import bash_permissions

        perms = sorted(bash_permissions(inp))
        background = bool(inp.get("background"))
        job_id = inp.get("job_id")
        command = inp.get("command") or ""
        try:
            result = await _bash_call(
                ws,
                ctx.get("cancel"),
                command=command,
                timeout=inp.get("timeout"),
                block_until_ms=inp.get("block_until_ms"),
                background=background,
                job_id=job_id,
                unsandboxed=bool(inp.get("unsandboxed")),
                permissions=perms,
            )
        except BashInterrupted as e:
            raise ToolInterrupted() from e
        if not background and not job_id:
            result = await asyncio.to_thread(
                annotate_bash_output, Path(ws.root), command, result
            )
            ctx["git_not_done"] = ritual_says_not_done(result)
        return result
    if name == "WebFetch":
        from orbweaver.webfetch import run_webfetch

        return await run_webfetch(inp, ctx)
    if name == "Browser":
        from orbweaver.browser import run_browser

        return await run_browser(inp, ctx)
    if name == "WebSearch":
        from orbweaver.websearch import run_websearch

        return await asyncio.to_thread(run_websearch, inp)
    if name == "WorkspaceSearch":
        from orbweaver.codesearch import run_workspace_search

        return await asyncio.to_thread(run_workspace_search, ws, inp)
    if name == "MemorySearch":
        from orbweaver.hindsight import enabled as hindsight_on
        from orbweaver.hindsight import format_recall
        from orbweaver.hindsight import recall as hindsight_recall
        from orbweaver.memory import expand_chunk_graph

        events = live_events(await store.list_events(session_id))
        already: set[str] = set()
        for ev in events:
            if ev.kind == "MemoryRecall":
                already.update(ev.payload.get("chunk_ids") or [])
        q = rewrite_search_query(events, inp.get("query") or "")
        if hindsight_on():
            try:
                payload = format_recall(await hindsight_recall(q))
            except Exception as e:
                log.warning("hindsight recall failed, using native: %s", e)
            else:
                ids = [i for i in payload["chunk_ids"] if i not in already]
                payload["chunk_ids"] = ids
                return json.dumps(payload)
        hits = await store.search_chunks(q, k=8)
        lines = []
        ids = []
        kept: list[tuple] = []
        for c, score in hits:
            if str(c.id) in already:
                continue
            ids.append(str(c.id))
            kept.append((c, score))
            lines.append(f"[{c.id} score={score:.3f}] {c.text}")
        expand = inp.get("expand_graph", True)
        graph = await expand_chunk_graph(store, kept) if expand else []
        return json.dumps(
            {
                "chunk_ids": ids,
                "text": "\n".join(lines) or "(no hits)",
                "graph": graph,
            }
        )
    if name == "MemoryGraph":
        hops = inp.get("depth", 1)
        neighborhood = await graph_neighborhood(store, str(inp.get("id") or ""), hops)
        return json.dumps(neighborhood)
    if name == "MemoryRemember":
        from orbweaver.hindsight import enabled as hindsight_on
        from orbweaver.hindsight import retain as hindsight_retain
        from orbweaver.store import PinBudgetError

        pinned = bool(inp.get("pinned"))
        text = str(inp.get("text") or "")
        hs_out: dict[str, Any] | None = None
        if hindsight_on() and text.strip():
            try:
                hs_out = await hindsight_retain(
                    text,
                    context="agent",
                    retain_async=not pinned,
                )
            except Exception as e:
                if not pinned:
                    return json.dumps({"error": f"hindsight retain failed: {e}"})
                log.warning("hindsight retain failed: %s", e)
            else:
                if not pinned:
                    return json.dumps({"id": "hindsight", "hindsight": hs_out})
        try:
            chunk = await remember(store, text, source="agent", pinned=pinned)
        except PinBudgetError as e:
            return f"pin rejected: {e}"
        body: dict[str, Any] = {"id": str(chunk.id)}
        if hs_out is not None:
            body["hindsight"] = hs_out
        return json.dumps(body) if hs_out is not None else str(chunk.id)
    if name == "MemoryReflect":
        from orbweaver.hindsight import enabled as hindsight_on
        from orbweaver.hindsight import format_reflect
        from orbweaver.hindsight import reflect as hindsight_reflect

        if not hindsight_on():
            return json.dumps(
                {"error": "Hindsight is not configured (set HINDSIGHT_API_URL)."}
            )
        budget = str(inp.get("budget") or "low")
        if budget not in {"low", "mid", "high"}:
            budget = "low"
        query = str(inp.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "query is required"})
        try:
            return json.dumps(format_reflect(await hindsight_reflect(query, budget=budget)))
        except Exception as e:
            return json.dumps({"error": str(e)})
    if name == "MemoryPin":
        from orbweaver.store import PinBudgetError

        try:
            await store.set_pinned(inp["kind"], UUID(inp["id"]), True)
        except PinBudgetError as e:
            return f"pin rejected: {e}"
        except (KeyError, ValueError) as e:
            return f"error: {e}"
        return "pinned"
    if name == "MemoryForget":
        await store.forget_chunk(UUID(inp["id"]))
        return "forgotten"
    if name == "AskUser":
        question = str(inp.get("question") or "").strip()
        if not question:
            return "error: question is required"
        if not can_wait_for_user(ctx):
            raise _ask_user_headless_abort(question)
        # Interactive wait is handled in agent_turn so the reply can be the tool result.
        return json.dumps({"ask": question})
    if name == "TodoWrite":
        todos = await persist_todos(store, session_id, inp)
        return json.dumps({"todos": todos})
    if name == "SpawnSubagent":
        from orbweaver.subagent import run_subagent

        return await run_subagent(inp, ctx)
    if name == "SubagentWait":
        from orbweaver.subagent import wait_subagents

        return await wait_subagents(inp, ctx)
    if name == "ScheduleTask":
        try:
            due = datetime.fromisoformat(str(inp["due_at"]).replace("Z", "+00:00"))  # noqa: FURB162
            if due.tzinfo is None:
                due = due.replace(tzinfo=UTC)
        except ValueError as e:
            return f"invalid due_at: {e}"
        rec = inp.get("recurrence")
        if rec:
            from orbweaver.channels.cron import _parse_recurrence

            if _parse_recurrence(str(rec), due) is None:
                return (
                    "invalid recurrence: use minute, hour, day "
                    "(or 'every hour'), or a 5-field cron expression "
                    "like '0 9 * * mon'"
                )
        job = Job(
            id=new_uuid(),
            due_at=due,
            payload={"message": inp.get("message") or ""},
            recurrence=rec,
            session_id=session_id,
        )
        await store.put_job(job)
        return json.dumps({"job_id": str(job.id), "due_at": due.isoformat()})
    if name == "SendPhoto":
        from orbweaver.channels.telegram import send_session_photo

        return await send_session_photo(ctx, inp)
    if name == "GenerateImage":
        from orbweaver.channels.telegram import generate_and_maybe_send

        return await generate_and_maybe_send(ctx, inp)
    if name.startswith("mcp_"):
        from orbweaver.mcp import call_mcp_tool

        return await call_mcp_tool(name, inp, ws)
    return f"unknown tool {name}"


@dataclass(frozen=True)
class _ToolOutcome:
    """What one tool_use produced: the tool_result text plus how it was reached.

    ``executed`` is False when the gate text stands in for the tool (blocked, denied,
    not approved in time). ``approval`` names the held-call outcome that did not run
    the tool: "deny", "timeout" or "cancelled".
    """

    content: str
    persisted_path: str | None = None
    flagged: bool = False
    is_error: bool = False
    executed: bool = True
    approval: str | None = None


class TurnCancelled(Exception):
    """Raised when the user stops or discards an in-flight turn."""

    def __init__(self, produced: list[Event] | None = None):
        super().__init__("turn cancelled")
        self.produced = produced or []


class TurnInjected(Exception):
    """Current LLM call aborted so a mid-turn follow-up can join this query."""


async def _create_with_overflow_retry(
    client: Any,
    *,
    store: Store,
    session_id: UUID,
    workspace: Any,
    system: Any,
    user_text: str,
    model: str,
    tools: Any,
    cancel: asyncio.Event | None,
    produced: list[Event],
    inject: asyncio.Event | None,
    fire: Callable[[Event], None],
    nudge: str | None = None,
) -> Any:
    """Call messages.create; on context overflow, compact and retry."""
    last_error: BaseException | None = None
    retries = max(0, int(settings.compact_overflow_retries))
    for attempt in range(retries + 1):
        events = await store.list_events(session_id)
        messages = _prompt_messages(events, workspace, user_text)
        if nudge:
            _nudge_user(messages, nudge)
        messages = ensure_tool_use_results(messages)
        send_tools = tools
        if prompt_cache_supported(client):
            # Breakpoints: static system block (build_agent_system), the tools
            # array when large, and the final user block. Each round appends
            # tool results after the previous breakpoint, so the prefix is a
            # cache read. Only Anthropic sees these; the Ollama shim drops them.
            messages = with_message_cache_breakpoint(messages)
            send_tools = with_tool_cache_breakpoint(tools)
        try:
            return await _await_or_cancel(
                client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=cast(Any, system),
                    tools=cast(Any, send_tools),
                    messages=cast(Any, messages),
                ),
                cancel,
                produced,
                inject=inject,
            )
        except TurnInjected:
            raise
        except Exception as e:
            last_error = e
            if not is_context_overflow(e):
                raise
            if attempt >= retries:
                break
            keep = OVERFLOW_KEEP_ROUNDS[min(attempt, len(OVERFLOW_KEEP_ROUNDS) - 1)]
            ev = await maybe_compact(
                store,
                session_id,
                client=client,
                workspace=workspace,
                system=system,
                source="overflow",
                force=True,
                budget=overflow_compact_budget(e),
                keep_recent_rounds=keep,
            )
            if ev is not None:
                fire(ev)
            elif keep <= 0:
                break
    raise ContextFullError(CONTEXT_FULL_MESSAGE) from last_error


def _raise_if_cancelled(
    cancel: asyncio.Event | None, produced: list[Event] | None = None
) -> None:
    if cancel is not None and cancel.is_set():
        raise TurnCancelled(produced)


async def _await_or_cancel(
    coro,
    cancel: asyncio.Event | None,
    produced: list[Event] | None = None,
    inject: asyncio.Event | None = None,
):
    if cancel is None and inject is None:
        return await coro
    _raise_if_cancelled(cancel, produced)
    task = asyncio.create_task(coro)
    watchers: set[asyncio.Task] = set()
    if cancel is not None:
        watchers.add(asyncio.create_task(cancel.wait()))
    if inject is not None:
        watchers.add(asyncio.create_task(inject.wait()))
    if not watchers:
        return await task
    try:
        done, _pending = await asyncio.wait(
            {task, *watchers}, return_when=asyncio.FIRST_COMPLETED
        )
        if task not in done:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            if cancel is not None and cancel.is_set():
                raise TurnCancelled(produced)
            raise TurnInjected()
        for w in watchers:
            w.cancel()
            with suppress(asyncio.CancelledError):
                await w
        return task.result()
    except asyncio.CancelledError:
        task.cancel()
        for w in watchers:
            w.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
        raise


async def agent_turn(
    store: Store,
    session_id: UUID,
    user_text: str,
    workspace,
    workspace_kind: str = "local",
    emit: Callable[[dict[str, Any]], None] | None = None,
    cancel: asyncio.Event | None = None,
    turn_state: Any | None = None,
    resume: bool = False,
    headless: bool = False,
    interactive: bool | None = None,
    tools: list[dict[str, Any]] | None = None,
    system_extra: str = "",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    subagent_depth: int = 0,
    images: list[dict[str, str]] | None = None,
    channel: str | None = None,
) -> list[Event]:
    _raise_if_cancelled(cancel)
    wait_ok = interactive if interactive is not None else not headless
    prior = await store.list_events(session_id)
    pending = pending_ask_user(prior)
    answer_text = (user_text or "").strip()
    if pending and resume and not answer_text:
        return []
    answering = bool(pending and answer_text and not resume)
    if resume:
        if turn_state is not None:
            for ev in reversed(prior):
                if ev.kind == "user":
                    turn_state.user_seq = ev.seq
                    if not user_text:
                        user_text = str(ev.payload.get("text") or "")
                    break
    else:
        payload: dict[str, Any] = {"text": user_text}
        if answering:
            payload["ask_answer"] = True
        if images:
            payload["images"] = images
        checkpoint: dict[str, Any] | None = None
        if subagent_depth == 0:
            checkpoint = await asyncio.to_thread(checkpoint_user_turn, workspace)
            if checkpoint is not None:
                payload["checkpoint"] = checkpoint
        user_ev = await store.append_event(session_id, "user", payload)
        if checkpoint is not None:
            await asyncio.to_thread(
                pin_user_turn, workspace, checkpoint, session_id, user_ev.seq
            )
        if turn_state is not None:
            turn_state.user_seq = user_ev.seq
        if emit:
            emit(
                {
                    "kind": user_ev.kind,
                    "payload": user_ev.payload,
                    "id": str(user_ev.id),
                    "seq": user_ev.seq,
                }
            )
    produced: list[Event] = []

    def fire(ev: Event) -> None:
        produced.append(ev)
        if emit:
            emit({"kind": ev.kind, "payload": ev.payload, "id": str(ev.id), "seq": ev.seq})

    def check() -> None:
        _raise_if_cancelled(cancel, produced)

    def inject_event() -> asyncio.Event | None:
        return getattr(turn_state, "inject", None) if turn_state is not None else None

    async def record_abort(exc: TurnAborted) -> None:
        payload = dict(exc.payload)
        payload.setdefault("text", exc.message)
        fire(await store.append_event(session_id, "turn_aborted", payload))
        fire(await store.append_event(session_id, "assistant", {"text": exc.message}))

    stuck_detector = StuckDetector()

    async def check_stuck() -> None:
        """Nudge once per loop streak; raise TurnAborted when the streak outlives the nudge."""
        verdict = stuck_detector.check(await store.list_events(session_id))
        if verdict is None:
            return
        if verdict.action == "abort":
            log.warning(
                "stuck: %s loop on %s x%d, ending turn", verdict.pattern, verdict.tool, verdict.count
            )
            raise TurnAborted(verdict.text, verdict.abort_payload())
        log.info("stuck: %s loop on %s x%d, nudging", verdict.pattern, verdict.tool, verdict.count)
        fire(await store.append_event(session_id, NUDGE_KIND, verdict.nudge_payload()))
        # Same kind as injected follow-ups so events_to_messages renders it as a user turn.
        fire(
            await store.append_event(
                session_id, "user", {"text": verdict.text, NUDGE_KIND: True}
            )
        )

    if answering and pending is not None:
        uid = str((pending.payload or {}).get("id") or pending.id)
        fire(
            await store.append_event(
                session_id,
                "tool_result",
                {
                    "tool_use_id": uid,
                    "name": "AskUser",
                    "content": answer_text,
                },
            )
        )

    check()
    from orbweaver.llm import make_agent_client, no_llm_echo

    client = make_agent_client()
    if client is None:
        ev = await store.append_event(
            session_id,
            "assistant",
            {"text": no_llm_echo(user_text or "")},
        )
        fire(ev)
        return produced
    sess = await store.get_entity(session_id)
    resolved_channel = resolve_channel(sess.jsonld if sess else None, channel=channel)
    denial_state = denial_state_for(session_id)
    ctx = {
        "workspace": workspace,
        "store": store,
        "session_id": session_id,
        "workspace_kind": workspace_kind,
        "auto_write": True,
        "headless": headless,
        "interactive": wait_ok,
        "denial_state": denial_state,
        "events": [],
        "cancel": cancel,
        "fire": fire,
        "emit": emit,
        "subagent_depth": subagent_depth,
        "channel": resolved_channel,
        "session_rules": session_rules_from(sess.jsonld if sess else None),
        # Per-turn cache of nested instruction dirs / glob rules already injected. Seeded
        # from blocks still in the live prompt window so a later turn does not repeat them.
        "instructions_seen": seen_instruction_keys(live_events(prior)),
        # Background subagents spawned by this turn (str child id -> ChildRun).
        "children": {},
    }

    async def deliver_child_results() -> None:
        """Background children that finished since the last round: show the model."""
        from orbweaver.subagent import collect_finished

        for body in collect_finished(ctx):
            fire(await store.append_event(session_id, "subagent_result", body))
    tool_slots = asyncio.Semaphore(max(1, int(settings.orbweaver_max_parallel_tools)))

    async def run_allowed_tool(block: Any, inp: dict[str, Any]) -> _ToolOutcome:
        """Run one permitted tool_use with the given input and probe its output."""
        if block.name == "AskUser":
            # Waiting for the user is handled in the round loop; only an empty
            # question reaches here.
            return _ToolOutcome("error: question is required")
        async with tool_slots:
            raw = await run_tools(block.name, inp, ctx)
        text, persisted_path = persist_tool_result(workspace, str(block.id), block.name, raw)
        probed = await probe_tool_output(block.name, text)
        return _ToolOutcome(probed["output"], persisted_path, bool(probed.get("flagged")))

    async def execute_tool(block: Any, decision: PermissionDecision) -> _ToolOutcome:
        """Fast path: run an allowed tool_use; blocked (deny / unheld ask) calls get the gate text."""
        if decision.behavior != "allow":
            return _ToolOutcome(_blocked_tool_result(decision), is_error=True, executed=False)
        return await run_allowed_tool(block, dict(block.input))

    async def hold_for_approval(block: Any, decision: PermissionDecision) -> _ToolOutcome:
        """Hold an ask-gated tool_use until the user answers, then run exactly what they saw.

        Records permission_request / permission_response (and permission_rule_added
        for allow-for-session). Deny, timeout and cancel return the gate text as an
        is_error result without running the tool.
        """
        recorded_input = dict(block.input)
        pend = PendingApproval(
            session_id=session_id,
            tool_use_id=str(block.id),
            name=block.name,
            input=recorded_input,
            reason=decision.reason,
            future=asyncio.get_running_loop().create_future(),
        )
        _register_approval(pend)
        try:
            fire(
                await store.append_event(
                    session_id,
                    "permission_request",
                    {
                        "tool_use_id": block.id,
                        "name": block.name,
                        "input": recorded_input,
                        "reason": decision.reason,
                        "summary": summarize_input(block.name, recorded_input),
                    },
                )
            )
            timeout = float(settings.orbweaver_approval_timeout_s)
            verdict = await _wait_for_approval(pend, cancel, timeout)
        finally:
            _unregister_approval(pend)
        if verdict is None:
            cancelled = cancel is not None and cancel.is_set()
            outcome = "cancelled" if cancelled else "timeout"
            scope = "once"
        else:
            outcome, scope = verdict
        fire(
            await store.append_event(
                session_id,
                "permission_response",
                {
                    "tool_use_id": block.id,
                    "name": block.name,
                    "decision": outcome,
                    "scope": scope,
                },
            )
        )
        if outcome == "cancelled":
            return _ToolOutcome(
                _approval_cancelled_result(decision.reason),
                is_error=True,
                executed=False,
                approval="cancelled",
            )
        if outcome == "timeout":
            return _ToolOutcome(
                _approval_timeout_result(decision.reason, timeout),
                is_error=True,
                executed=False,
                approval="timeout",
            )
        if outcome == "deny":
            return _ToolOutcome(
                _denied_tool_result(decision.reason),
                is_error=True,
                executed=False,
                approval="deny",
            )
        if scope == "session":
            rule = make_session_rule(block.name, recorded_input)
            rules = ctx.setdefault("session_rules", [])
            if add_session_rule(rules, rule):
                current = await store.get_entity(session_id)
                if current is not None:
                    current.jsonld["permission_rules"] = list(rules)
                    await store.put_entity(current)
                fire(
                    await store.append_event(
                        session_id,
                        "permission_rule_added",
                        {
                            "tool_use_id": block.id,
                            "tool": rule["tool"],
                            "subject": rule["subject"],
                            "permissions": rule.get("permissions") or [],
                        },
                    )
                )
        check()
        # Execute exactly the input the user saw, not a re-issued call.
        return await run_allowed_tool(block, recorded_input)

    async def record_interrupted(block: Any, exc: ToolInterrupted) -> None:
        fire(
            await store.append_event(
                session_id,
                "tool_result",
                {
                    "tool_use_id": block.id,
                    "name": block.name,
                    "content": exc.message,
                    "is_error": True,
                },
            )
        )

    async def record_result(block: Any, outcome: _ToolOutcome) -> None:
        """tool_result (or MemoryRecall) for one block, plus its follow-on events."""
        if outcome.flagged:
            fire(
                await store.append_event(
                    session_id,
                    "injection_warning",
                    {"tool_use_id": block.id, "name": block.name},
                )
            )
        result = outcome.content
        kind = "MemoryRecall" if block.name == "MemorySearch" else "tool_result"
        payload: dict[str, Any] = {"tool_use_id": block.id, "name": block.name, "content": result}
        if outcome.is_error:
            payload["is_error"] = True
        if outcome.executed and outcome.persisted_path:
            payload["persisted_path"] = outcome.persisted_path
        if kind == "MemoryRecall":
            try:
                parsed = json.loads(result)
                payload["chunk_ids"] = parsed.get("chunk_ids") or []
                payload["text"] = parsed.get("text") or result
            except json.JSONDecodeError:
                payload["text"] = result
        fire(await store.append_event(session_id, kind, payload))
        if block.name == "ProposePatch" and outcome.executed:
            fire(
                await store.append_event(
                    session_id,
                    "patch_proposal",
                    {"tool_use_id": block.id, "content": result},
                )
            )
        if block.name == "ScheduleTask" and outcome.executed:
            fire(
                await store.append_event(
                    session_id, "schedule_request", {"input": dict(block.input)}
                )
            )

    pins = await pinned_prompt(store)
    system = build_agent_system(
        pins,
        system_extra,
        skills=workspace_skills_prompt(workspace),
        channel=resolved_channel,
    )
    active_tools = (
        tools if tools is not None else await session_tools(workspace, resolved_channel)
    )
    if not resume:
        await maybe_compact(
            store, session_id, client=client, workspace=workspace, system=system
        )
        if turn_state is not None:
            for ev in reversed(await store.list_events(session_id)):
                if ev.kind == "user":
                    turn_state.user_seq = ev.seq
                    break

    try:
        for round_i in range(max_rounds):
            check()
            await deliver_child_results()
            inj = inject_event()
            if inj is not None:
                inj.clear()
            nudge = None
            if round_i == max_rounds - 1:
                nudge = (
                    GIT_NOT_DONE_LAST_NUDGE
                    if ctx.get("git_not_done")
                    else LAST_ROUND_NUDGE
                )
            elif ctx.pop("git_not_done_nudge_pending", False):
                nudge = GIT_NOT_DONE_NUDGE
            try:
                resp = await _create_with_overflow_retry(
                    client,
                    store=store,
                    session_id=session_id,
                    workspace=workspace,
                    system=system,
                    user_text=user_text,
                    model=settings.orbweaver_model,
                    tools=active_tools,
                    cancel=cancel,
                    produced=produced,
                    inject=inj,
                    fire=fire,
                    nudge=nudge,
                )
            except TurnInjected:
                continue
            events = await store.list_events(session_id)
            ctx["events"] = events
            last_seq = events[-1].seq if events else 0
            record_response_usage(session_id, getattr(resp, "usage", None), last_seq)
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            if texts:
                ev = await store.append_event(session_id, "assistant", {"text": "\n".join(texts)})
                fire(ev)
            check()
            if not tool_uses:
                if (
                    ctx.get("git_not_done")
                    and not ctx.get("git_not_done_nudged")
                    and round_i < max_rounds - 1
                ):
                    ctx["git_not_done_nudged"] = True
                    ctx["git_not_done_nudge_pending"] = True
                    continue
                break
            stop_after_ask = False
            approval_timed_out = False
            waiting_ask = False
            instruction_blocks: list[str] = []
            # #99: consecutive concurrency-safe calls (Read, Grep, ...) form one
            # batch and run together; every unsafe call is a batch of one and a
            # barrier. tool_call events for a batch are recorded up front and
            # tool_result events in the original tool_use order.
            for run in partition_tool_calls([b.name for b in tool_uses]):
                batch = [tool_uses[i] for i in run]
                check()
                for block in batch:
                    call_ev = await store.append_event(
                        session_id,
                        "tool_call",
                        {"id": block.id, "name": block.name, "input": block.input},
                    )
                    fire(call_ev)
                ctx["events"] = await store.list_events(session_id)
                verdicts = await asyncio.gather(
                    *(can_use_tool(b.name, dict(b.input), ctx) for b in batch),
                    return_exceptions=True,
                )
                decisions: list[PermissionDecision] = []
                for block, verdict in zip(batch, verdicts, strict=True):
                    if isinstance(verdict, TurnAborted):
                        fire(
                            await store.append_event(
                                session_id,
                                "tool_result",
                                {
                                    "tool_use_id": block.id,
                                    "name": block.name,
                                    "content": verdict.message,
                                },
                            )
                        )
                        await record_abort(verdict)
                        return produced
                    if isinstance(verdict, BaseException):
                        raise verdict
                    decisions.append(verdict)
                    fire(
                        await store.append_event(
                            session_id,
                            "permission_decision",
                            {
                                "tool_use_id": block.id,
                                "name": block.name,
                                "behavior": verdict.behavior,
                                "reason": verdict.reason,
                                "fast_path": verdict.fast_path,
                            },
                        )
                    )
                # AskUser is never concurrency-safe, so it is always a batch of one.
                if len(batch) == 1 and batch[0].name == "AskUser" and decisions[0].behavior == "allow":
                    block = batch[0]
                    question = str(dict(block.input).get("question") or "").strip()
                    if question and not can_wait_for_user(ctx):
                        await record_abort(_ask_user_headless_abort(question))
                        return produced
                    if question:
                        fire(
                            await store.append_event(
                                session_id,
                                "ask_user",
                                {
                                    "question": question,
                                    "tool_use_id": block.id,
                                    "name": "AskUser",
                                },
                            )
                        )
                        waiting_ask = True
                        break
                needs_approval = any(d.behavior == "ask" for d in decisions)
                if needs_approval and can_wait_for_user(ctx):
                    # A held call must not run alongside anything else before the
                    # user answers: walk this batch one block at a time, in order.
                    for block, decision in zip(batch, decisions, strict=True):
                        check()
                        if decision.behavior == "ask":
                            outcome = await hold_for_approval(block, decision)
                        else:
                            try:
                                outcome = await execute_tool(block, decision)
                            except ToolInterrupted as e:
                                await record_interrupted(block, e)
                                raise TurnCancelled(produced) from e
                        if outcome.approval != "cancelled":
                            check()
                        await record_result(block, outcome)
                        if outcome.executed:
                            instruction_blocks.extend(
                                _nested_instructions_for(workspace, block.name, dict(block.input), ctx)
                            )
                        if outcome.approval == "cancelled":
                            # The is_error result is on record; now stop like any other cancel.
                            check()
                        if outcome.approval == "timeout":
                            approval_timed_out = True
                            break
                    if approval_timed_out:
                        # Later tool_uses in this round never ran; the model re-plans next turn.
                        break
                    continue
                outcomes = await asyncio.gather(
                    *(execute_tool(b, d) for b, d in zip(batch, decisions, strict=True)),
                    return_exceptions=True,
                )
                for block, got in zip(batch, outcomes, strict=True):
                    if isinstance(got, ToolInterrupted):
                        await record_interrupted(block, got)
                        raise TurnCancelled(produced) from got
                check()
                for block, decision, got in zip(batch, decisions, outcomes, strict=True):
                    if isinstance(got, BaseException):
                        raise got
                    if got.executed:
                        instruction_blocks.extend(
                            _nested_instructions_for(workspace, block.name, dict(block.input), ctx)
                        )
                    if decision.behavior == "ask":
                        # The pipeline aborts headless asks before this; an ask that
                        # could not be held for a human ends the turn unexecuted.
                        stop_after_ask = True
                    await record_result(block, got)
            if instruction_blocks:
                # User-side message for the next round (rendered after the tool results),
                # same path as injected follow-ups: persisted event, projected into the prompt.
                fire(
                    await store.append_event(
                        session_id,
                        PROJECT_INSTRUCTIONS_KIND,
                        {
                            "text": "\n\n".join(instruction_blocks),
                            "keys": sorted(ctx["instructions_seen"]),
                        },
                    )
                )
            if waiting_ask:
                break
            if approval_timed_out:
                fire(
                    await store.append_event(
                        session_id, "assistant", {"text": APPROVAL_TIMEOUT_NOTICE}
                    )
                )
                break
            if stop_after_ask:
                fire(
                    await store.append_event(
                        session_id,
                        "assistant",
                        {
                            "text": (
                                "I need your approval before continuing. "
                                "Reply with what you want done."
                            )
                        },
                    )
                )
                break
            await check_stuck()
            await maybe_compact(
                store, session_id, client=client, workspace=workspace, system=system
            )
        else:
            await deliver_child_results()
            try:
                resp = await _create_with_overflow_retry(
                    client,
                    store=store,
                    session_id=session_id,
                    workspace=workspace,
                    system=system,
                    user_text=user_text,
                    model=settings.orbweaver_model,
                    tools=[],
                    cancel=cancel,
                    produced=produced,
                    inject=inject_event(),
                    fire=fire,
                    nudge=(
                        GIT_NOT_DONE_CONCLUDE
                        if ctx.get("git_not_done")
                        else CONCLUDE_NUDGE
                    ),
                )
            except TurnInjected:
                return produced
            events = await store.list_events(session_id)
            last_seq = events[-1].seq if events else 0
            record_response_usage(session_id, getattr(resp, "usage", None), last_seq)
            texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            if texts:
                fire(
                    await store.append_event(session_id, "assistant", {"text": "\n".join(texts)})
                )
            else:
                fire(
                    await store.append_event(
                        session_id,
                        "assistant",
                        {
                            "text": (
                                f"Stopped after {max_rounds} tool rounds without a final answer. "
                                "Continue in a follow-up, or spawn a subagent with a narrower task."
                            )
                        },
                    )
                )
        return produced
    except ContextFullError as e:
        fire(await store.append_event(session_id, "assistant", {"text": e.message}))
        return produced
    except TurnAborted as e:
        await record_abort(e)
        return produced
    except TurnCancelled as e:
        e.produced = produced
        raise
    finally:
        from orbweaver.hindsight import retain_turn
        from orbweaver.subagent import settle_children

        if ctx.get("children"):
            try:
                await settle_children(
                    ctx, cancelled=bool(cancel is not None and cancel.is_set())
                )
            except Exception:
                log.exception("settling background subagents failed")
        await retain_turn(
            session_id,
            user_text,
            produced,
            subagent_depth=subagent_depth,
            channel=resolved_channel,
        )
