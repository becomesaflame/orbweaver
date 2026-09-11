"""Project instruction discovery shared by the agent prompt and the permission classifier.

One loader covers:

* root docs (``AGENTS.md`` / ``AGENTS.override.md``, ``CLAUDE.md``, ``ORBWEAVER.md``)
  preceded by the user-level ``~/.orbweaver/AGENTS.md``;
* nested per-directory docs (``AGENTS.md`` / ``CLAUDE.md`` / ``.cursor/rules``) between the
  project root and a path a tool touched, injected lazily once per turn;
* ``.cursor/rules/*.mdc`` semantics (``alwaysApply`` / ``globs`` / ``description``) and the
  native ``.orbweaver/rules``;
* ``SKILL.md`` progressive disclosure: names and descriptions in the prompt, bodies on
  demand through the ``Skill`` tool.

The classifier's project intent and the agent's system prompt are rendered from the same
sections so both components see the same text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from orbweaver.config import settings
from orbweaver.tokens import estimate_tokens

ROOT_DOCS = ("AGENTS.md", "CLAUDE.md", "ORBWEAVER.md")
NESTED_DOCS = ("AGENTS.md", "CLAUDE.md")
OVERRIDE_DOC = "AGENTS.override.md"
USER_DOC = "AGENTS.md"
CURSOR_RULES_DIR = Path(".cursor") / "rules"
ORBWEAVER_RULES_DIR = Path(".orbweaver") / "rules"
SKILL_DIRS = (
    Path(".orbweaver") / "skills",
    Path(".cursor") / "skills",
    Path(".claude") / "skills",
)
SKILL_FILE = "SKILL.md"
EXCLUDED_DIRS = frozenset({".orbweaver", ".orbweaver-tmp", "node_modules", ".git"})
MAX_FILE_BYTES = 256_000
HEADER = "# Workspace skills and rules"
TRUNCATION_MARK = "[truncated]"
SKILLS_HEADER = "## Skills (load with the Skill tool before using one)"
RULES_HEADER = '## Rules on demand (load with Skill(name, kind="rule"))'
PATH_TOOLS = frozenset(
    {"Read", "Write", "StrReplace", "NotebookEdit", "Delete", "ProposePatch", "SendPhoto"}
)
_GLOB_CHARS = ("*", "?", "[")
_CD_RE = re.compile(r"""(?:^|[;&|(]|\s)cd\s+(?:--\s+)?(?:"([^"]+)"|'([^']+)'|([^\s;&|)]+))""")


# ---------------------------------------------------------------- frontmatter


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML frontmatter from a markdown / .mdc body.

    Supports ``key: value`` scalars, ``true`` / ``false``, inline ``[a, b]`` lists and
    block lists (``- item`` lines under a key with no inline value).
    """
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
    current: str | None = None
    for raw in lines[1:end]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current is not None and isinstance(meta.get(current), list):
            meta[current].append(_unquote(stripped[2:]))
            continue
        if ":" not in raw:
            current = None
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        if not key:
            current = None
            continue
        val = val.strip()
        if not val:
            meta[key] = []
            current = key
            continue
        current = None
        if val.startswith("[") and val.endswith("]"):
            items = [_unquote(x) for x in val[1:-1].split(",")]
            meta[key] = [x for x in items if x]
            continue
        val = _unquote(val)
        lowered = val.lower()
        if lowered == "true":
            meta[key] = True
        elif lowered == "false":
            meta[key] = False
        else:
            meta[key] = val
    for key, val in list(meta.items()):
        if isinstance(val, list) and not val:
            meta[key] = ""
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return meta, body


def _unquote(val: str) -> str:
    return val.strip().strip('"').strip("'").strip()


def _truthy(val: Any) -> bool:
    if val is True:
        return True
    return isinstance(val, str) and val.strip().lower() == "true"


def _as_list(val: Any) -> tuple[str, ...]:
    if val is None or val is True or val is False:
        return ()
    if isinstance(val, list):
        items = [str(x).strip() for x in val]
    else:
        items = [x.strip() for x in str(val).split(",")]
    return tuple(x for x in items if x)


# ---------------------------------------------------------------- filesystem


def _read_text(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_bytes()[:MAX_FILE_BYTES].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _display(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    home = Path.home()
    try:
        return "~/" + path.relative_to(home).as_posix()
    except ValueError:
        return str(path)


def instructions_data_dir() -> Path:
    raw = (settings.orbweaver_data_dir or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".orbweaver"


def user_instructions_path() -> Path:
    raw = (settings.orbweaver_user_instructions or "").strip()
    if raw:
        return Path(raw).expanduser()
    return instructions_data_dir() / USER_DOC


def item_token_cap() -> int:
    return int(settings.orbweaver_instruction_item_token_cap)


def is_excluded(rel: PurePosixPath | Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in rel.parts)


# ---------------------------------------------------------------- docs


@dataclass(frozen=True)
class Doc:
    title: str
    body: str
    source: Path


def _dir_docs(directory: Path, names: tuple[str, ...], root: Path | None) -> list[Doc]:
    """Docs in one directory; ``AGENTS.override.md`` replaces ``AGENTS.md`` (Codex-style)."""
    docs: list[Doc] = []
    for name in names:
        candidates = [directory / name]
        if name == "AGENTS.md":
            candidates.insert(0, directory / OVERRIDE_DOC)
        for path in candidates:
            text = _read_text(path)
            if text is None:
                continue
            body = text.strip()
            if body:
                docs.append(Doc(_display(path, root), body, path))
            break
    return docs


def _dedupe(docs: list[Doc]) -> list[Doc]:
    seen: set[str] = set()
    out: list[Doc] = []
    for doc in docs:
        key = " ".join(doc.body.split())
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def root_docs(root: Path | str, *, include_user: bool = True) -> list[Doc]:
    """User-level AGENTS.md first, then root AGENTS(.override).md, CLAUDE.md, ORBWEAVER.md."""
    root = Path(root)
    docs: list[Doc] = []
    if include_user:
        user_path = user_instructions_path()
        text = _read_text(user_path)
        if text and text.strip():
            docs.append(Doc(_display(user_path, None), text.strip(), user_path))
    docs.extend(_dir_docs(root, ROOT_DOCS, root))
    return _dedupe(docs)


# ---------------------------------------------------------------- rules


@dataclass(frozen=True)
class Rule:
    name: str
    title: str
    description: str
    globs: tuple[str, ...]
    always: bool
    body: str
    source: Path

    @property
    def mode(self) -> str:
        if self.always:
            return "always"
        if self.globs:
            return "globs"
        if self.description:
            return "description"
        return "manual"


def _rule_from_file(path: Path, root: Path | None, *, default_always: bool) -> Rule | None:
    text = _read_text(path)
    if text is None:
        return None
    meta, body = parse_frontmatter(text)
    body = body.strip()
    if not body:
        return None
    if "alwaysApply" in meta:
        always = _truthy(meta.get("alwaysApply"))
    else:
        always = default_always
    globs = _as_list(meta.get("globs"))
    description = str(meta.get("description") or "").strip()
    return Rule(
        name=path.stem,
        title=_display(path, root),
        description=description,
        globs=globs,
        always=always,
        body=body,
        source=path,
    )


def dir_rules(directory: Path, root: Path | None) -> list[Rule]:
    """``.cursor/rules/*.mdc`` under ``directory`` (not always-on unless ``alwaysApply``)."""
    rules_dir = directory / CURSOR_RULES_DIR
    if not rules_dir.is_dir():
        return []
    out: list[Rule] = []
    for path in sorted(p for p in rules_dir.glob("*.mdc") if p.is_file()):
        rule = _rule_from_file(path, root, default_always=False)
        if rule is not None:
            out.append(rule)
    return out


def native_rules(root: Path) -> list[Rule]:
    """``.orbweaver/rules/*.md|.mdc``: always-on unless the frontmatter says otherwise."""
    rules_dir = root / ORBWEAVER_RULES_DIR
    if not rules_dir.is_dir():
        return []
    files = sorted(
        p for p in rules_dir.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".mdc"}
    )
    out: list[Rule] = []
    for path in files:
        rule = _rule_from_file(path, root, default_always=True)
        if rule is not None:
            out.append(rule)
    return out


def root_rules(root: Path | str) -> list[Rule]:
    root = Path(root)
    return dir_rules(root, root) + native_rules(root)


def _glob_regex(pattern: str) -> re.Pattern[str]:
    pat = pattern.strip().replace("\\", "/").removeprefix("./").lstrip("/")
    out: list[str] = []
    i = 0
    while i < len(pat):
        ch = pat[i]
        if ch == "*":
            if pat.startswith("**", i):
                i += 2
                if i < len(pat) and pat[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                out.append(re.escape(ch))
            else:
                out.append(pat[i : j + 1])
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_matches(pattern: str, rel_path: str) -> bool:
    """Cursor-style glob match: bare patterns (no ``/``) match any path component tail."""
    rel = rel_path.replace("\\", "/").strip("/")
    if not pattern.strip() or not rel:
        return False
    pat = pattern.strip()
    try:
        regex = _glob_regex(pat)
    except re.error:
        return False
    if regex.match(rel):
        return True
    if "/" not in pat.rstrip("/"):
        return regex.match(rel.rsplit("/", 1)[-1]) is not None
    return False


def rule_matches_path(rule: Rule, rel_path: str) -> bool:
    return any(glob_matches(g, rel_path) for g in rule.globs)


# ---------------------------------------------------------------- skills


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    source: Path


def _skill_from_dir(directory: Path, root: Path | None) -> Skill | None:
    path = directory / SKILL_FILE
    text = _read_text(path)
    if text is None:
        return None
    meta, body = parse_frontmatter(text)
    body = body.strip()
    name = str(meta.get("name") or "").strip() or directory.name
    description = str(meta.get("description") or "").strip()
    if not description:
        for line in body.splitlines():
            first = line.strip().lstrip("#").strip()
            if first:
                description = first
                break
    if not body and not description:
        return None
    return Skill(name=name, description=description, body=body, source=path)


def _skills_in(parent: Path, root: Path | None) -> list[Skill]:
    if not parent.is_dir():
        return []
    out: list[Skill] = []
    try:
        entries = sorted(p for p in parent.iterdir() if p.is_dir())
    except OSError:
        return []
    for entry in entries:
        skill = _skill_from_dir(entry, root)
        if skill is not None:
            out.append(skill)
    return out


def load_skills(root: Path | str, *, include_user: bool = True) -> list[Skill]:
    """Project skills first (``.orbweaver/skills``, ``.cursor/skills``, ``.claude/skills``),
    then user-level ``<data dir>/skills``. The first skill with a given name wins."""
    root = Path(root)
    found: list[Skill] = []
    for sub in SKILL_DIRS:
        found.extend(_skills_in(root / sub, root))
    if include_user:
        found.extend(_skills_in(instructions_data_dir() / "skills", None))
    seen: set[str] = set()
    out: list[Skill] = []
    for skill in found:
        key = skill.name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(skill)
    return out


# ---------------------------------------------------------------- rendering


def cap_item(title: str, body: str, cap: int, *, source: str = "") -> str:
    """Truncate one section body to ``cap`` tokens with a marker naming what was cut."""
    if cap <= 0 or estimate_tokens(body) <= cap:
        return body
    where = f" Read {source} for the rest." if source else ""
    marker = f"\n\n{TRUNCATION_MARK} {title} exceeds the {cap}-token per-item cap.{where}"
    budget = max(cap * 4 - len(marker), 32)
    return body[:budget].rstrip() + marker


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body}"


def _join(parts: list[str]) -> str:
    return "\n\n".join(parts)


def fit_sections(sections: list[tuple[str, str]], token_cap: int) -> str:
    """Greedy fit under ``token_cap``; the marker names every dropped section."""
    if token_cap <= 0 or not sections:
        return ""
    parts: list[str] = [HEADER]
    used = estimate_tokens(HEADER)
    if used >= token_cap:
        return ""
    kept = 0
    for title, body in sections:
        section = _section(title, body)
        need = estimate_tokens(section) + 1
        if used + need > token_cap:
            break
        parts.append(section)
        used += need
        kept += 1
    if kept == len(sections):
        return _join(parts)

    def marker(partial: str | None, dropped: list[str]) -> str:
        bits = [f"{TRUNCATION_MARK} {token_cap}-token cap for always-on instructions reached."]
        if partial:
            bits.append(f"Cut short: {partial}.")
        if dropped:
            bits.append("Omitted entirely: " + ", ".join(dropped) + ".")
        bits.append("Read those files from the workspace when relevant.")
        return " ".join(bits)

    while True:
        rest = sections[kept:]
        title, body = rest[0]
        dropped_titles = [t for t, _ in rest[1:]]
        mark = marker(title, dropped_titles)
        remaining = token_cap - used - estimate_tokens(mark) - 1
        prefix = f"## {title}\n"
        char_budget = remaining * 4 - len(prefix)
        trimmed = body[:char_budget].rstrip() if char_budget >= 32 else ""
        if trimmed:
            candidate = [*parts, f"{prefix}{trimmed}", mark]
        else:
            mark = marker(None, [title, *dropped_titles])
            candidate = [*parts, mark]
        if estimate_tokens(_join(candidate)) <= token_cap:
            return _join(candidate)
        if kept == 0:
            return ""
        # Not enough room for the marker: drop the last kept section and retry.
        parts.pop()
        kept -= 1
        used = estimate_tokens(_join(parts)) + 1


def always_on_sections(root: Path | str, *, item_cap: int | None = None) -> list[tuple[str, str]]:
    """Root docs plus always-apply rules, each body capped per item."""
    root = Path(root)
    cap = item_token_cap() if item_cap is None else item_cap
    sections: list[tuple[str, str]] = []
    for doc in root_docs(root):
        sections.append((doc.title, cap_item(doc.title, doc.body, cap, source=doc.title)))
    for rule in root_rules(root):
        if rule.always:
            sections.append((rule.title, cap_item(rule.title, rule.body, cap, source=rule.title)))
    return sections


def project_intent_text(root: Path | str, *, item_cap: int | None = None) -> str:
    """Root instruction docs as the classifier sees them: same text as the agent prompt."""
    root = Path(root)
    cap = item_token_cap() if item_cap is None else item_cap
    parts = [
        _section(doc.title, cap_item(doc.title, doc.body, cap, source=doc.title))
        for doc in root_docs(root)
    ]
    return _join(parts)


def _catalog_line(name: str, description: str) -> str:
    desc = " ".join(description.split())
    if len(desc) > 240:
        desc = desc[:237].rstrip() + "..."
    return f"- {name}: {desc}" if desc else f"- {name}"


def catalog_text(root: Path | str) -> str:
    """Skills and description-only rules listed by name for on-demand loading."""
    root = Path(root)
    parts: list[str] = []
    skills = load_skills(root)
    if skills:
        parts.append(
            "\n".join([SKILLS_HEADER, *(_catalog_line(s.name, s.description) for s in skills)])
        )
    on_demand = [r for r in root_rules(root) if r.mode == "description"]
    if on_demand:
        parts.append(
            "\n".join([RULES_HEADER, *(_catalog_line(r.name, r.description) for r in on_demand)])
        )
    return _join(parts)


def build_instructions_prompt(
    root: Path | str, *, token_cap: int | None = None, item_cap: int | None = None
) -> str:
    """Always-on section under the global cap, then the skills / rules catalog."""
    root = Path(root)
    cap = settings.orbweaver_skills_token_cap if token_cap is None else token_cap
    body = fit_sections(always_on_sections(root, item_cap=item_cap), cap)
    catalog = catalog_text(root) if cap > 0 else ""
    if not body and not catalog:
        return ""
    if not body:
        return _join([HEADER, catalog])
    if not catalog:
        return body
    return _join([body, catalog])


# ---------------------------------------------------------------- Skill tool


def load_skill_or_rule(root: Path | str, name: str, kind: str = "skill") -> str:
    """Body of a skill (``kind=skill``) or an on-demand rule (``kind=rule``) by name."""
    root = Path(root)
    want = (name or "").strip()
    kind = (kind or "skill").strip().lower()
    if not want:
        return "error: name is required"
    if kind == "rule":
        rules = [r for r in root_rules(root) if not r.always]
        for rule in rules:
            if rule.name.lower() == want.lower() or rule.title == want:
                return f"<rule name=\"{rule.name}\" source=\"{rule.title}\">\n{rule.body}\n</rule>"
        names = ", ".join(sorted({r.name for r in rules})) or "(none)"
        return f"error: no rule named {want!r}. Available: {names}"
    skills = load_skills(root)
    for skill in skills:
        if skill.name.lower() == want.lower():
            src = _display(skill.source, root)
            return f"<skill name=\"{skill.name}\" source=\"{src}\">\n{skill.body}\n</skill>"
    names = ", ".join(s.name for s in skills) or "(none)"
    return f"error: no skill named {want!r}. Available: {names}"


# ---------------------------------------------------------------- nested (lazy) discovery


def _static_prefix(pattern: str) -> str:
    parts = pattern.replace("\\", "/").split("/")
    static: list[str] = []
    for part in parts[:-1]:
        if any(c in part for c in _GLOB_CHARS):
            break
        static.append(part)
    return "/".join(static)


def touched_paths(tool_name: str, inp: dict[str, Any]) -> list[str]:
    """Paths a tool call touches, as given (relative to the workspace or absolute)."""
    if tool_name in PATH_TOOLS:
        path = str(inp.get("path") or "").strip()
        return [path] if path else []
    if tool_name == "ReadLints":
        raw = inp.get("paths") or inp.get("path") or []
        if isinstance(raw, str):
            raw = [raw]
        return [str(p).strip() for p in raw if str(p).strip()]
    if tool_name == "Glob":
        prefix = _static_prefix(str(inp.get("pattern") or ""))
        return [prefix] if prefix else []
    if tool_name == "Grep":
        prefix = _static_prefix(str(inp.get("glob") or ""))
        return [prefix] if prefix else []
    if tool_name == "Bash":
        cmd = str(inp.get("command") or "")
        out: list[str] = []
        for m in _CD_RE.finditer(cmd):
            target = m.group(1) or m.group(2) or m.group(3) or ""
            target = target.strip()
            if target and target not in {"-", "~"} and not target.startswith("$"):
                out.append(target)
        return out
    return []


def relative_to_root(root: Path, raw: str) -> PurePosixPath | None:
    """Workspace-relative path for ``raw`` or None when it escapes / is excluded."""
    spec = (raw or "").strip()
    if not spec or spec.startswith("~"):
        return None
    try:
        path = Path(spec) if spec.startswith("/") else root / spec
        rel = Path(os.path.normpath(str(path))).relative_to(Path(os.path.normpath(str(root))))
    except (ValueError, OSError):
        return None
    posix = PurePosixPath(rel.as_posix())
    if posix.parts and posix.parts[0] == "..":
        return None
    if is_excluded(posix):
        return None
    return posix


def ancestor_dirs(root: Path, rel: PurePosixPath) -> list[PurePosixPath]:
    """Directories strictly below ``root`` on the way to ``rel`` (root-first).

    The final component is included when it is a directory on disk (Bash ``cd``, Glob
    prefix); a file's own name is not."""
    parts = [p for p in rel.parts if p not in {"", "."}]
    if not parts:
        return []
    dirs: list[PurePosixPath] = []
    for i in range(1, len(parts)):
        dirs.append(PurePosixPath(*parts[:i]))
    leaf = PurePosixPath(*parts)
    try:
        if (root / leaf).is_dir():
            dirs.append(leaf)
    except OSError:
        pass
    return dirs


def nested_instruction_blocks(
    root: Path | str, raw_path: str, seen: set[str]
) -> list[str]:
    """``<project-instructions dir=...>`` blocks for ancestors of ``raw_path`` not yet seen.

    ``seen`` is the per-turn cache; it is updated in place so each directory (and each
    glob-scoped rule) is injected at most once per turn.
    """
    root = Path(root)
    rel = relative_to_root(root, raw_path)
    if rel is None:
        return []
    cap = item_token_cap()
    blocks: list[str] = []
    rel_str = rel.as_posix()

    # Root-level glob-scoped rules that match the touched path.
    root_sections: list[str] = []
    for rule in root_rules(root):
        if rule.mode != "globs" or not rule_matches_path(rule, rel_str):
            continue
        key = f"rule:{rule.title}"
        if key in seen:
            continue
        seen.add(key)
        root_sections.append(_section(rule.title, cap_item(rule.title, rule.body, cap, source=rule.title)))
    if root_sections:
        blocks.append(_block(".", root_sections))

    for directory in ancestor_dirs(root, rel):
        dir_key = f"dir:{directory.as_posix()}"
        abs_dir = root / directory
        sections: list[str] = []
        if dir_key not in seen:
            seen.add(dir_key)
            for doc in _dedupe(_dir_docs(abs_dir, NESTED_DOCS, root)):
                sections.append(_section(doc.title, cap_item(doc.title, doc.body, cap, source=doc.title)))
        for rule in dir_rules(abs_dir, root):
            key = f"rule:{rule.title}"
            if key in seen:
                continue
            if rule.always or (rule.globs and rule_matches_path(rule, rel_str)):
                seen.add(key)
                sections.append(_section(rule.title, cap_item(rule.title, rule.body, cap, source=rule.title)))
        if sections:
            blocks.append(_block(directory.as_posix(), sections))
    return blocks


def _block(directory: str, sections: list[str]) -> str:
    return f'<project-instructions dir="{directory}">\n{_join(sections)}\n</project-instructions>'


def instruction_blocks_for_call(
    root: Path | str, tool_name: str, inp: dict[str, Any], seen: set[str]
) -> list[str]:
    blocks: list[str] = []
    for raw in touched_paths(tool_name, inp):
        blocks.extend(nested_instruction_blocks(root, raw, seen))
    return blocks


def seen_instruction_keys(events: Any) -> set[str]:
    """Cache keys from ``project_instructions`` events still in the live prompt window."""
    seen: set[str] = set()
    for ev in events or []:
        if getattr(ev, "kind", None) != "project_instructions":
            continue
        payload = getattr(ev, "payload", None) or {}
        for key in payload.get("keys") or []:
            seen.add(str(key))
    return seen
