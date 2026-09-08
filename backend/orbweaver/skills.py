"""Discover workspace skills and rules for the agent system prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orbweaver.config import settings
from orbweaver.tokens import estimate_tokens

ROOT_DOCS = ("AGENTS.md", "ORBWEAVER.md")
CURSOR_RULES_DIR = Path(".cursor") / "rules"
ORBWEAVER_RULES_DIR = Path(".orbweaver") / "rules"
MAX_FILE_BYTES = 256_000
HEADER = "# Workspace skills and rules"
TRUNCATION_MARK = "\n\n[truncated]"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML frontmatter from a markdown / .mdc body."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta: dict[str, Any] = {}
    for raw in lines[1:end]:
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip('"').strip("'")
        lowered = val.lower()
        if lowered == "true":
            meta[key] = True
        elif lowered == "false":
            meta[key] = False
        else:
            meta[key] = val
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return meta, body


def _always_apply(meta: dict[str, Any]) -> bool:
    val = meta.get("alwaysApply")
    if val is True:
        return True
    return isinstance(val, str) and val.strip().lower() == "true"


def _read_text(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_bytes()[:MAX_FILE_BYTES].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _collect_sections(root: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for name in ROOT_DOCS:
        text = _read_text(root / name)
        if not text:
            continue
        body = text.strip()
        if body:
            sections.append((name, body))

    cursor_dir = root / CURSOR_RULES_DIR
    if cursor_dir.is_dir():
        for path in sorted(p for p in cursor_dir.glob("*.mdc") if p.is_file()):
            text = _read_text(path)
            if text is None:
                continue
            meta, body = parse_frontmatter(text)
            if not _always_apply(meta):
                continue
            body = body.strip()
            if body:
                sections.append((f".cursor/rules/{path.name}", body))

    native_dir = root / ORBWEAVER_RULES_DIR
    if native_dir.is_dir():
        files = sorted(
            p
            for p in native_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".md", ".mdc"}
        )
        for path in files:
            text = _read_text(path)
            if text is None:
                continue
            meta, body = parse_frontmatter(text)
            if "alwaysApply" in meta and not _always_apply(meta):
                continue
            body = body.strip()
            if body:
                sections.append((f".orbweaver/rules/{path.name}", body))
    return sections


def _fit(sections: list[tuple[str, str]], token_cap: int) -> str:
    if token_cap <= 0 or not sections:
        return ""
    parts: list[str] = [HEADER]
    used = estimate_tokens(HEADER)
    if used >= token_cap:
        return ""
    for title, body in sections:
        remaining = token_cap - used
        if remaining <= 0:
            break
        section = f"## {title}\n{body}"
        need = estimate_tokens(section)
        if need <= remaining:
            parts.append(section)
            used += need
            continue
        prefix = f"## {title}\n"
        char_budget = remaining * 4 - len(prefix) - len(TRUNCATION_MARK)
        if char_budget < 32:
            break
        trimmed = body[:char_budget].rstrip()
        if not trimmed:
            break
        parts.append(f"{prefix}{trimmed}{TRUNCATION_MARK}")
        break
    if len(parts) == 1:
        return ""
    return "\n\n".join(parts)


def discover_workspace_skills(root: Path | str, *, token_cap: int | None = None) -> str:
    """Concatenate discoverable workspace skills/rules under a token cap.

    Missing files and directories are a no-op. Empty result means inject nothing.
    """
    cap = settings.orbweaver_skills_token_cap if token_cap is None else token_cap
    return _fit(_collect_sections(Path(root)), cap)


def workspace_skills_prompt(workspace, *, token_cap: int | None = None) -> str:
    root = getattr(workspace, "root", None)
    if root is None:
        return ""
    return discover_workspace_skills(root, token_cap=token_cap)
