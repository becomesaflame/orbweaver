"""Instruction discovery: shared root loader, nested docs, rule semantics, skills (#112)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import TOOL_SPEC, agent_turn, build_agent_system, run_tools
from orbweaver.compact import PROJECT_INSTRUCTIONS_KIND, events_to_messages
from orbweaver.config import settings
from orbweaver.instructions import (
    ancestor_dirs,
    build_instructions_prompt,
    glob_matches,
    load_skill_or_rule,
    load_skills,
    nested_instruction_blocks,
    parse_frontmatter,
    project_intent_text,
    relative_to_root,
    root_docs,
    touched_paths,
)
from orbweaver.permissions.classifier import load_project_intent
from orbweaver.permissions.rules import SAFE_ALLOWLIST
from orbweaver.skills import workspace_skills_prompt
from orbweaver.store import Event, reset_store_for_tests
from orbweaver.subagent import EXPLORE_TOOLS, SHELL_TOOLS
from orbweaver.tokens import estimate_tokens
from orbweaver.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _isolate_user_level(tmp_path_factory, monkeypatch):
    """Point the user-level AGENTS.md at a scratch dir so the host's file cannot leak in."""
    home = tmp_path_factory.mktemp("user-home")
    monkeypatch.setattr(settings, "orbweaver_data_dir", str(home))
    monkeypatch.setattr(settings, "orbweaver_user_instructions", "")
    yield home


def _ws(root: Path) -> LocalWorkspace:
    return LocalWorkspace("workspace:default", str(root))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- root docs


def test_claude_md_in_prompt_and_classifier_sees_same_text(tmp_path: Path):
    _write(tmp_path / "CLAUDE.md", "# Claude\nAlways run ruff before committing.\n")
    _write(tmp_path / "AGENTS.md", "# Agents\nUse pytest.\n")
    ws = _ws(tmp_path)

    prompt = workspace_skills_prompt(ws)
    assert "## CLAUDE.md" in prompt
    assert "Always run ruff before committing." in prompt
    assert prompt.index("## AGENTS.md") < prompt.index("## CLAUDE.md")

    intent = load_project_intent(ws)
    assert intent == project_intent_text(tmp_path)
    assert "Always run ruff before committing." in intent
    assert "Use pytest." in intent
    # Every classifier intent section is verbatim inside the agent's system prompt.
    for section in intent.split("\n\n## "):
        assert section.lstrip("# ") in prompt

    system = build_agent_system("", skills=prompt)
    assert "Always run ruff before committing." in system[1]["text"]


def test_root_doc_order_and_orbweaver_md(tmp_path: Path):
    _write(tmp_path / "ORBWEAVER.md", "process\n")
    _write(tmp_path / "CLAUDE.md", "claude\n")
    _write(tmp_path / "AGENTS.md", "agents\n")
    assert [d.title for d in root_docs(tmp_path)] == ["AGENTS.md", "CLAUDE.md", "ORBWEAVER.md"]


def test_user_level_agents_md_comes_first(tmp_path: Path, _isolate_user_level: Path):
    _write(_isolate_user_level / "AGENTS.md", "Prefer British spelling.\n")
    _write(tmp_path / "AGENTS.md", "Project rule.\n")

    docs = root_docs(tmp_path)
    assert [d.body for d in docs] == ["Prefer British spelling.", "Project rule."]
    assert docs[0].title.endswith("AGENTS.md")
    assert docs[0].source == _isolate_user_level / "AGENTS.md"

    prompt = build_instructions_prompt(tmp_path)
    assert prompt.index("Prefer British spelling.") < prompt.index("Project rule.")
    # Classifier intent includes the user-level file too.
    assert "Prefer British spelling." in load_project_intent(_ws(tmp_path))


def test_user_level_path_setting_overrides_data_dir(tmp_path: Path, monkeypatch):
    custom = tmp_path / "elsewhere" / "me.md"
    _write(custom, "custom user doc\n")
    monkeypatch.setattr(settings, "orbweaver_user_instructions", str(custom))
    project = tmp_path / "proj"
    project.mkdir()
    assert [d.body for d in root_docs(project)] == ["custom user doc"]


def test_agents_override_and_dedupe(tmp_path: Path):
    _write(tmp_path / "AGENTS.md", "stale agents\n")
    _write(tmp_path / "AGENTS.override.md", "override wins\n")
    _write(tmp_path / "CLAUDE.md", "override wins\n")
    _write(tmp_path / "ORBWEAVER.md", "  override   wins \n")

    docs = root_docs(tmp_path)
    assert [d.title for d in docs] == ["AGENTS.override.md"]
    assert docs[0].body == "override wins"
    assert "stale agents" not in build_instructions_prompt(tmp_path)


# ---------------------------------------------------------------- nested discovery


def test_relative_to_root_and_excluded_dirs(tmp_path: Path):
    assert relative_to_root(tmp_path, "backend/x.py").as_posix() == "backend/x.py"
    assert relative_to_root(tmp_path, str(tmp_path / "web" / "a.js")).as_posix() == "web/a.js"
    assert relative_to_root(tmp_path, "../outside.py") is None
    assert relative_to_root(tmp_path, "/etc/passwd") is None
    for bad in (".orbweaver/rules/x.md", ".orbweaver-tmp/a", "node_modules/p/i.js", ".git/HEAD"):
        assert relative_to_root(tmp_path, bad) is None, bad


def test_ancestor_dirs_between_root_and_path(tmp_path: Path):
    (tmp_path / "backend" / "pkg").mkdir(parents=True)
    file_rel = relative_to_root(tmp_path, "backend/pkg/mod.py")
    assert [d.as_posix() for d in ancestor_dirs(tmp_path, file_rel)] == ["backend", "backend/pkg"]
    dir_rel = relative_to_root(tmp_path, "backend/pkg")
    assert [d.as_posix() for d in ancestor_dirs(tmp_path, dir_rel)] == ["backend", "backend/pkg"]
    assert ancestor_dirs(tmp_path, relative_to_root(tmp_path, "README.md")) == []


def test_touched_paths_per_tool():
    assert touched_paths("Read", {"path": "backend/a.py"}) == ["backend/a.py"]
    assert touched_paths("Write", {"path": "web/index.html"}) == ["web/index.html"]
    assert touched_paths("StrReplace", {"path": "x"}) == ["x"]
    assert touched_paths("Glob", {"pattern": "backend/orbweaver/**/*.py"}) == ["backend/orbweaver"]
    assert touched_paths("Glob", {"pattern": "**/*.py"}) == []
    assert touched_paths("Grep", {"pattern": "foo", "glob": "web/src/*.ts"}) == ["web/src"]
    assert touched_paths("Bash", {"command": "cd backend && pytest -q"}) == ["backend"]
    assert touched_paths("Bash", {"command": 'ls; cd "web ui"; npm test'}) == ["web ui"]
    assert touched_paths("Bash", {"command": "cd - && cd ~ && cd $HOME"}) == []
    assert touched_paths("Bash", {"command": "echo cdrom"}) == []
    assert touched_paths("WebFetch", {"url": "https://x"}) == []


def test_nested_blocks_once_per_dir(tmp_path: Path):
    _write(tmp_path / "backend" / "AGENTS.md", "Backend: keep mypy strict.\n")
    _write(tmp_path / "backend" / "pkg" / "CLAUDE.md", "Pkg: no print().\n")
    _write(tmp_path / "backend" / "pkg" / "mod.py", "x = 1\n")
    seen: set[str] = set()

    blocks = nested_instruction_blocks(tmp_path, "backend/pkg/mod.py", seen)
    assert len(blocks) == 2
    assert blocks[0].startswith('<project-instructions dir="backend">')
    assert "## backend/AGENTS.md" in blocks[0]
    assert "keep mypy strict" in blocks[0]
    assert blocks[1].startswith('<project-instructions dir="backend/pkg">')
    assert "no print()" in blocks[1]
    assert blocks[1].endswith("</project-instructions>")

    assert nested_instruction_blocks(tmp_path, "backend/pkg/other.py", seen) == []
    assert nested_instruction_blocks(tmp_path, "backend/new.py", seen) == []


def test_nested_skips_excluded_and_root_docs(tmp_path: Path):
    _write(tmp_path / "AGENTS.md", "root doc\n")
    _write(tmp_path / ".orbweaver" / "AGENTS.md", "hidden\n")
    _write(tmp_path / "node_modules" / "AGENTS.md", "hidden\n")
    seen: set[str] = set()
    assert nested_instruction_blocks(tmp_path, ".orbweaver/tool-results/x", seen) == []
    assert nested_instruction_blocks(tmp_path, "node_modules/pkg/index.js", seen) == []
    # Root docs are already in the system prompt; touching a root file injects nothing.
    assert nested_instruction_blocks(tmp_path, "README.md", seen) == []


def test_project_instructions_event_renders_on_user_side():
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "go"}),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=2,
            kind="tool_call",
            payload={"id": "tu1", "name": "Read", "input": {"path": "backend/a.py"}},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=3,
            kind="tool_result",
            payload={"tool_use_id": "tu1", "name": "Read", "content": "1|x"},
        ),
        Event(
            id=uuid4(),
            session_id=sid,
            seq=4,
            kind=PROJECT_INSTRUCTIONS_KIND,
            payload={"text": '<project-instructions dir="backend">\nhi\n</project-instructions>'},
        ),
    ]
    messages = events_to_messages(events)
    assert messages[-1]["role"] == "user"
    blocks = messages[-1]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[-1]["type"] == "text"
    assert '<project-instructions dir="backend">' in blocks[-1]["text"]


class _ToolUse:
    def __init__(self, name, inp, uid):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _RecordingAnthropic:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


def _install_client(monkeypatch, responses) -> _RecordingAnthropic:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    client = _RecordingAnthropic(responses)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    return client


def _text_of(messages) -> str:
    out: list[str] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content:
            if block.get("type") == "text":
                out.append(block["text"])
            elif block.get("type") == "tool_result":
                inner = block.get("content")
                out.append(inner if isinstance(inner, str) else str(inner))
    return "\n".join(out)


@pytest.mark.asyncio
async def test_agent_turn_injects_nested_agents_md_once(tmp_path: Path, monkeypatch):
    _write(tmp_path / "backend" / "AGENTS.md", "Backend rule: NESTED-MARKER-42.\n")
    _write(tmp_path / "backend" / "a.py", "a = 1\n")
    _write(tmp_path / "backend" / "b.py", "b = 2\n")
    _write(tmp_path / "AGENTS.md", "Root rule.\n")
    responses = [
        SimpleNamespace(content=[_ToolUse("Read", {"path": "backend/a.py"}, "tu-1")]),
        SimpleNamespace(content=[_ToolUse("Read", {"path": "backend/b.py"}, "tu-2")]),
    ]
    client = _install_client(monkeypatch, responses)
    store = reset_store_for_tests()
    sid = uuid4()

    events = await agent_turn(store, sid, "look at backend", _ws(tmp_path))

    kinds = [e.kind for e in events]
    assert kinds.count(PROJECT_INSTRUCTIONS_KIND) == 1
    assert len(client.calls) == 3
    # Round 1: nothing touched yet, so the block is absent from the user side.
    assert "NESTED-MARKER-42" not in _text_of(client.calls[0]["messages"])
    # Round 2: block follows the first Read's result, on the user side.
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "user"
    assert _text_of(second).count('<project-instructions dir="backend">') == 1
    assert "NESTED-MARKER-42" in _text_of(second)
    # Round 3: a second Read under backend/ does not repeat the block.
    assert _text_of(client.calls[2]["messages"]).count('<project-instructions dir="backend">') == 1
    # Root AGENTS.md stays in the system prompt, not in the nested block.
    system_text = "\n".join(b["text"] for b in client.calls[0]["system"])
    assert "Root rule." in system_text
    assert "NESTED-MARKER-42" not in system_text
    # The block is a persisted event that carries its cache keys.
    ev = next(e for e in events if e.kind == PROJECT_INSTRUCTIONS_KIND)
    assert "dir:backend" in ev.payload["keys"]


@pytest.mark.asyncio
async def test_agent_turn_does_not_reinject_dirs_still_in_window(tmp_path: Path, monkeypatch):
    _write(tmp_path / "backend" / "AGENTS.md", "Backend rule.\n")
    _write(tmp_path / "backend" / "a.py", "a = 1\n")
    store = reset_store_for_tests()
    sid = uuid4()
    client = _install_client(
        monkeypatch, [SimpleNamespace(content=[_ToolUse("Read", {"path": "backend/a.py"}, "t1")])]
    )
    first = await agent_turn(store, sid, "turn one", _ws(tmp_path))
    assert [e.kind for e in first].count(PROJECT_INSTRUCTIONS_KIND) == 1

    client._responses.append(
        SimpleNamespace(content=[_ToolUse("Read", {"path": "backend/a.py"}, "t2")])
    )
    second = await agent_turn(store, sid, "turn two", _ws(tmp_path))
    assert [e.kind for e in second].count(PROJECT_INSTRUCTIONS_KIND) == 0


# ---------------------------------------------------------------- .cursor/rules semantics


def test_parse_frontmatter_lists_and_scalars():
    meta, body = parse_frontmatter(
        "---\ndescription: Python style\nglobs: *.py, backend/**\nalwaysApply: false\n---\nbody\n"
    )
    assert meta["globs"] == "*.py, backend/**"
    assert meta["alwaysApply"] is False
    assert body.strip() == "body"
    block, _ = parse_frontmatter('---\nglobs:\n  - "*.ts"\n  - web/**\n---\nx\n')
    assert block["globs"] == ["*.ts", "web/**"]
    inline, _ = parse_frontmatter("---\nglobs: [src/*.js, 'lib/**/*.js']\n---\nx\n")
    assert inline["globs"] == ["src/*.js", "lib/**/*.js"]
    legacy, _ = parse_frontmatter("---\nalwaysApply: 'true'\n---\nx\n")
    assert legacy["alwaysApply"] is True


def test_glob_matches_cursor_style():
    assert glob_matches("*.py", "backend/orbweaver/agent.py")
    assert glob_matches("backend/**/*.py", "backend/orbweaver/agent.py")
    assert glob_matches("backend/**", "backend/orbweaver/agent.py")
    assert glob_matches("**/test_*.py", "tests/test_x.py")
    assert glob_matches("web/*.html", "web/index.html")
    assert not glob_matches("web/*.html", "web/sub/index.html")
    assert not glob_matches("*.py", "web/index.html")
    assert not glob_matches("backend/**", "web/a.py")


def _rules(tmp_path: Path) -> Path:
    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    _write(rules / "always.mdc", "---\ndescription: always\nalwaysApply: true\n---\nALWAYS-BODY\n")
    _write(
        rules / "python.mdc",
        "---\ndescription: Python style\nglobs: backend/**/*.py\nalwaysApply: false\n---\nGLOBS-BODY\n",
    )
    _write(
        rules / "deploy.mdc",
        "---\ndescription: How to deploy safely\nalwaysApply: false\n---\nDEPLOY-BODY\n",
    )
    return rules


def test_rule_modes_in_system_prompt(tmp_path: Path):
    _rules(tmp_path)
    prompt = build_instructions_prompt(tmp_path)
    assert "## .cursor/rules/always.mdc" in prompt
    assert "ALWAYS-BODY" in prompt
    # globs rule: neither inlined nor catalogued; it arrives when a matching path is touched.
    assert "GLOBS-BODY" not in prompt
    assert "## Rules on demand" in prompt
    assert "- python:" not in prompt.split("## Rules on demand")[-1]
    # description-only rule: listed by name + description, body not inlined.
    assert "- deploy: How to deploy safely" in prompt
    assert "DEPLOY-BODY" not in prompt


def test_globs_rule_injected_only_for_matching_path(tmp_path: Path):
    _rules(tmp_path)
    (tmp_path / "backend").mkdir()
    seen: set[str] = set()
    assert nested_instruction_blocks(tmp_path, "web/index.html", seen) == []
    blocks = nested_instruction_blocks(tmp_path, "backend/agent.py", seen)
    assert len(blocks) == 1
    assert blocks[0].startswith('<project-instructions dir=".">')
    assert "## .cursor/rules/python.mdc" in blocks[0]
    assert "GLOBS-BODY" in blocks[0]
    assert "ALWAYS-BODY" not in blocks[0]
    assert "DEPLOY-BODY" not in blocks[0]
    # Once per turn.
    assert nested_instruction_blocks(tmp_path, "backend/other.py", seen) == []


def test_nested_cursor_rules_under_subdir(tmp_path: Path):
    _write(
        tmp_path / "web" / ".cursor" / "rules" / "ts.mdc",
        "---\nglobs: *.ts\n---\nTS-BODY\n",
    )
    _write(
        tmp_path / "web" / ".cursor" / "rules" / "web-always.mdc",
        "---\nalwaysApply: true\n---\nWEB-ALWAYS\n",
    )
    seen: set[str] = set()
    html = nested_instruction_blocks(tmp_path, "web/index.html", seen)
    assert len(html) == 1
    assert "WEB-ALWAYS" in html[0]
    assert "TS-BODY" not in html[0]
    ts = nested_instruction_blocks(tmp_path, "web/app.ts", seen)
    assert len(ts) == 1
    assert "TS-BODY" in ts[0]
    assert "WEB-ALWAYS" not in ts[0]


def test_description_rule_loaded_via_skill_tool(tmp_path: Path):
    _rules(tmp_path)
    out = load_skill_or_rule(tmp_path, "deploy", "rule")
    assert out.startswith('<rule name="deploy" source=".cursor/rules/deploy.mdc">')
    assert "DEPLOY-BODY" in out
    missing = load_skill_or_rule(tmp_path, "nope", "rule")
    assert missing.startswith("error:")
    assert "deploy" in missing


# ---------------------------------------------------------------- skills


def _skills(tmp_path: Path) -> None:
    _write(
        tmp_path / ".orbweaver" / "skills" / "release" / "SKILL.md",
        "---\nname: release\ndescription: Cut a release and tag it\n---\n# Release\nRELEASE-BODY\n",
    )
    _write(
        tmp_path / ".cursor" / "skills" / "lint-fix" / "SKILL.md",
        "# Fix lints\nLINT-BODY\n",
    )


def test_skills_listed_by_name_and_loaded_on_demand(tmp_path: Path):
    _skills(tmp_path)
    skills = load_skills(tmp_path)
    assert [(s.name, s.description) for s in skills] == [
        ("release", "Cut a release and tag it"),
        ("lint-fix", "Fix lints"),
    ]

    prompt = build_instructions_prompt(tmp_path)
    assert "## Skills" in prompt
    assert "- release: Cut a release and tag it" in prompt
    assert "- lint-fix: Fix lints" in prompt
    assert "RELEASE-BODY" not in prompt
    assert "LINT-BODY" not in prompt

    body = load_skill_or_rule(tmp_path, "release")
    assert body.startswith('<skill name="release" source=".orbweaver/skills/release/SKILL.md">')
    assert "RELEASE-BODY" in body
    assert "LINT-BODY" in load_skill_or_rule(tmp_path, "LINT-FIX", "skill")
    missing = load_skill_or_rule(tmp_path, "ghost")
    assert missing.startswith("error:")
    assert "release" in missing


def test_user_level_skills_after_project(tmp_path: Path, _isolate_user_level: Path):
    _write(_isolate_user_level / "skills" / "mine" / "SKILL.md", "---\ndescription: personal\n---\nMINE\n")
    _write(_isolate_user_level / "skills" / "release" / "SKILL.md", "---\ndescription: shadowed\n---\nNO\n")
    _skills(tmp_path)
    names = [s.name for s in load_skills(tmp_path)]
    assert names == ["release", "lint-fix", "mine"]
    assert "MINE" in load_skill_or_rule(tmp_path, "mine")
    assert "shadowed" not in build_instructions_prompt(tmp_path)


@pytest.mark.asyncio
async def test_skill_tool_wired_into_run_tools(tmp_path: Path):
    _skills(tmp_path)
    _rules(tmp_path)
    ctx = {"workspace": _ws(tmp_path), "store": reset_store_for_tests(), "session_id": uuid4()}
    assert "RELEASE-BODY" in await run_tools("Skill", {"name": "release"}, ctx)
    assert "DEPLOY-BODY" in await run_tools("Skill", {"name": "deploy", "kind": "rule"}, ctx)
    spec = next(t for t in TOOL_SPEC if t["name"] == "Skill")
    assert spec["input_schema"]["required"] == ["name"]
    assert "Skill" in SAFE_ALLOWLIST
    assert "Skill" in EXPLORE_TOOLS
    assert "Skill" in SHELL_TOOLS


# ---------------------------------------------------------------- caps


def test_large_skill_does_not_truncate_always_on_rules(tmp_path: Path):
    _write(
        tmp_path / ".orbweaver" / "skills" / "huge" / "SKILL.md",
        "---\ndescription: enormous\n---\n" + ("lorem ipsum " * 5000),
    )
    _write(tmp_path / "AGENTS.md", "Root rule stays.\n")
    _write(
        tmp_path / ".cursor" / "rules" / "z.mdc",
        "---\nalwaysApply: true\n---\nLast always-on rule survives.\n",
    )
    prompt = build_instructions_prompt(tmp_path, token_cap=4000)
    assert "Root rule stays." in prompt
    assert "Last always-on rule survives." in prompt
    assert "- huge: enormous" in prompt
    assert "lorem ipsum" not in prompt
    assert "[truncated]" not in prompt


def test_per_item_cap_names_the_item(tmp_path: Path):
    _write(tmp_path / "AGENTS.md", "alpha " * 3000)
    _write(tmp_path / "CLAUDE.md", "Short claude rule.\n")
    prompt = build_instructions_prompt(tmp_path, token_cap=4000, item_cap=100)
    assert "[truncated] AGENTS.md exceeds the 100-token per-item cap. Read AGENTS.md for the rest." in prompt
    assert "Short claude rule." in prompt
    # The classifier sees the identically capped text.
    assert project_intent_text(tmp_path, item_cap=100) in prompt


def test_global_cap_marker_names_dropped_sections(tmp_path: Path):
    _write(tmp_path / "AGENTS.md", "keep me\n")
    _write(tmp_path / "CLAUDE.md", "beta " * 200)
    _write(tmp_path / "ORBWEAVER.md", "gamma " * 200)
    prompt = build_instructions_prompt(tmp_path, token_cap=90, item_cap=2000)
    assert estimate_tokens(prompt) <= 90
    assert "keep me" in prompt
    assert "[truncated]" in prompt
    assert "Omitted entirely: ORBWEAVER.md" in prompt
    assert "gamma" not in prompt


def test_catalog_survives_zero_always_on(tmp_path: Path):
    _skills(tmp_path)
    prompt = build_instructions_prompt(tmp_path)
    assert prompt.startswith("# Workspace skills and rules")
    assert "- release:" in prompt
