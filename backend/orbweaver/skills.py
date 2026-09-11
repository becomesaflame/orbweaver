"""Workspace skills and rules for the agent system prompt.

Thin facade over :mod:`orbweaver.instructions`, which is the single loader shared with
the permission classifier's project intent.
"""

from __future__ import annotations

from pathlib import Path

from orbweaver.instructions import (
    CURSOR_RULES_DIR,
    HEADER,
    MAX_FILE_BYTES,
    ORBWEAVER_RULES_DIR,
    ROOT_DOCS,
    TRUNCATION_MARK,
    build_instructions_prompt,
    parse_frontmatter,
)

__all__ = [
    "CURSOR_RULES_DIR",
    "HEADER",
    "MAX_FILE_BYTES",
    "ORBWEAVER_RULES_DIR",
    "ROOT_DOCS",
    "TRUNCATION_MARK",
    "discover_workspace_skills",
    "parse_frontmatter",
    "workspace_skills_prompt",
]


def discover_workspace_skills(root: Path | str, *, token_cap: int | None = None) -> str:
    """Always-on instructions under the global cap plus the skills / rules catalog.

    Missing files and directories are a no-op. Empty result means inject nothing.
    """
    return build_instructions_prompt(Path(root), token_cap=token_cap)


def workspace_skills_prompt(workspace, *, token_cap: int | None = None) -> str:
    root = getattr(workspace, "root", None)
    if root is None:
        return ""
    return discover_workspace_skills(root, token_cap=token_cap)
