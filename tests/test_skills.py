from pathlib import Path

from orbweaver.agent import build_agent_system
from orbweaver.skills import discover_workspace_skills, parse_frontmatter, workspace_skills_prompt
from orbweaver.tokens import estimate_tokens
from orbweaver.workspace import LocalWorkspace


def test_missing_files_are_noop(tmp_path: Path):
    assert discover_workspace_skills(tmp_path) == ""
    assert workspace_skills_prompt(LocalWorkspace("workspace:default", str(tmp_path))) == ""
    assert workspace_skills_prompt(object()) == ""


def test_discovers_root_docs_and_always_apply_rules(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("# Agents\nUse pytest.\n", encoding="utf-8")
    (tmp_path / "ORBWEAVER.md").write_text("# Process\nFeature branches only.\n", encoding="utf-8")
    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "always.mdc").write_text(
        "---\ndescription: always\nalwaysApply: true\n---\nNever push main.\n",
        encoding="utf-8",
    )
    (rules / "sometimes.mdc").write_text(
        "---\ndescription: optional\nalwaysApply: false\n---\nIgnore me.\n",
        encoding="utf-8",
    )
    (rules / "no-frontmatter.mdc").write_text("Bare rule.\n", encoding="utf-8")
    native = tmp_path / ".orbweaver" / "rules"
    native.mkdir(parents=True)
    (native / "house.md").write_text("Keep injection bounded.\n", encoding="utf-8")
    (native / "skip.mdc").write_text(
        "---\nalwaysApply: false\n---\nDo not inject.\n",
        encoding="utf-8",
    )

    prompt = discover_workspace_skills(tmp_path)
    assert "# Workspace skills and rules" in prompt
    assert "## AGENTS.md" in prompt
    assert "Use pytest." in prompt
    assert "## ORBWEAVER.md" in prompt
    assert "Feature branches only." in prompt
    assert "## .cursor/rules/always.mdc" in prompt
    assert "Never push main." in prompt
    assert "Ignore me." not in prompt
    assert "Bare rule." not in prompt
    assert "## .orbweaver/rules/house.md" in prompt
    assert "Keep injection bounded." in prompt
    assert "Do not inject." not in prompt

    ws = LocalWorkspace("workspace:default", str(tmp_path))
    assert "Use pytest." in workspace_skills_prompt(ws)


def test_token_cap_truncates(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("alpha " * 400, encoding="utf-8")
    (tmp_path / "ORBWEAVER.md").write_text("omega should not appear\n", encoding="utf-8")
    prompt = discover_workspace_skills(tmp_path, token_cap=80)
    assert prompt
    assert estimate_tokens(prompt) <= 80
    assert "[truncated]" in prompt
    assert "omega should not appear" not in prompt


def test_tiny_cap_injects_nothing(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("hello\n", encoding="utf-8")
    assert discover_workspace_skills(tmp_path, token_cap=1) == ""


def test_unreadable_and_empty_files_are_noop(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_bytes(b"\xff\xfe not utf-8")
    (tmp_path / "ORBWEAVER.md").write_text("   \n", encoding="utf-8")
    assert discover_workspace_skills(tmp_path) == ""


def test_parse_frontmatter_truthy_always_apply():
    meta, body = parse_frontmatter("---\nalwaysApply: true\n---\nbody\n")
    assert meta["alwaysApply"] is True
    assert body.strip() == "body"
    quoted, _ = parse_frontmatter("---\nalwaysApply: 'true'\n---\nx\n")
    assert quoted["alwaysApply"] is True
    none, raw = parse_frontmatter("no frontmatter")
    assert none == {}
    assert raw == "no frontmatter"


def test_build_agent_system_injects_skills():
    blocks = build_agent_system("pins here", skills="# Workspace skills and rules\n## AGENTS.md\nHi")
    assert "pins here" in blocks[1]["text"]
    assert "Workspace skills and rules" in blocks[1]["text"]
    empty = build_agent_system("")
    assert empty[1]["text"] == "(no pinned memory)"
    assert "Workspace skills" not in empty[1]["text"]
