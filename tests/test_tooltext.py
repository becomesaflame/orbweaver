import shutil
from uuid import uuid4

import pytest

from orbweaver.agent import run_tools
from orbweaver.compact.persist import persist_tool_result
from orbweaver.config import settings
from orbweaver.tooltext import format_read, format_webfetch, grep_regex, html_to_text
from orbweaver.workspace import LocalWorkspace


def test_format_read_pages_with_next_offset():
    text = "\n".join(f"line-{i}" for i in range(1, 501))
    out = format_read(text, path="web/index.html", limit=400)
    assert "lines 1-400 of 500" in out
    assert "1|line-1" in out
    assert "400|line-400" in out
    assert "line-401" not in out
    assert "Read offset=401" in out
    rest = format_read(text, path="web/index.html", offset=401, limit=400)
    assert "lines 401-500 of 500" in rest
    assert "401|line-401" in rest
    assert "truncated" not in rest


def test_format_read_silent_clip_is_gone():
    text = "x" * 50_000
    out = format_read(text, path="app.py")
    assert "truncated" in out
    assert "persisted-output" not in out


def test_html_to_text_keeps_article():
    html = (
        "<html><head><title>Agent limits</title><style>body{color:red}</style></head>"
        "<body><h1>Max steps</h1><p>Bedrock agents default to 5 iterations.</p>"
        "<script>window.__huge='" + ("x" * 200) + "'</script></body></html>"
    )
    text = html_to_text(html)
    assert "Max steps" in text
    assert "Bedrock agents default to 5" in text
    assert "window.__huge" not in text
    assert "color:red" not in text


def test_webfetch_js_shell_tells_agent_to_stop():
    html = (
        "<!DOCTYPE html><html id='__next_error__'><head>"
        "<script>" + ("window.payload='" + "x" * 6000 + "'") + "</script>"
        "</head><body><div id='__next'></div></body></html>"
    )
    out = format_webfetch(
        "https://docs.anthropic.com/en/docs/agents",
        200,
        "text/html",
        html,
    )
    assert "JavaScript-rendered" in out
    assert "Do not keep fetching" in out
    assert "Browser tool" in out
    assert "HTTP 200" in out


def test_webfetch_extracted_text_is_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "compact_tool_result_chars", 100)
    article = "<html><body>" + ("<p>Bedrock max iterations is five.</p>" * 80) + "</body></html>"
    body = format_webfetch("https://example.com/agents", 200, "text/html; charset=utf-8", article)
    assert "Bedrock max iterations is five." in body
    assert "<html" not in body
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    stored, rel = persist_tool_result(ws, "toolu_web", "WebFetch", body)
    assert rel is None
    assert "persisted-output" not in stored
    assert "Bedrock max iterations is five." in stored


def test_truncate_head_tail_keeps_both_ends_within_budget():
    from orbweaver.tooltext import truncate_head_tail

    body = "".join(f"line {i} " + ("x" * 60) + "\n" for i in range(4000)) + "FAILED tests/x.py::test_y"
    out = truncate_head_tail(body, max_chars=8000, path="out.txt")
    assert len(out) <= 8000
    assert out.startswith("line 0 ")
    assert out.endswith("FAILED tests/x.py::test_y")
    marker = next(ln for ln in out.splitlines() if ln.startswith("[... "))
    assert "chars omitted (" in marker
    assert marker.endswith("lines); full output saved to out.txt ...]")
    head, _, tail = out.partition(marker)
    assert head.endswith("\n")
    assert tail.startswith("\n")
    # Cuts land on line boundaries: no partial line on either side of the marker.
    assert head.splitlines()[-1].startswith("line ")
    assert tail.lstrip("\n").splitlines()[0].startswith("line ")
    # Tail gets the larger share.
    assert len(tail) > len(head)


def test_truncate_head_tail_single_line_and_small():
    from orbweaver.tooltext import truncate_head_tail

    one = "y" * 300_000
    out = truncate_head_tail(one, max_chars=8000)
    assert len(out) <= 8000
    assert out.startswith("yyy") and out.endswith("yyy")
    assert "[... 292" in out and "chars omitted ...]" in out
    small = "hello\nworld\n"
    assert truncate_head_tail(small, max_chars=8000) == small


def test_truncate_head_tail_line_cap():
    from orbweaver.tooltext import truncate_head_tail

    lines = "\n".join(f"L{i}" for i in range(5000))
    out = truncate_head_tail(lines, max_chars=1_000_000, max_lines=2000)
    assert out.startswith("L0\nL1\n")
    assert out.endswith("\nL4998\nL4999")
    assert len(out.splitlines()) <= 2000
    exact = "\n".join(f"L{i}" for i in range(2000))
    assert truncate_head_tail(exact, max_chars=1_000_000, max_lines=2000) == exact


def test_format_bash_result_header():
    from orbweaver.tooltext import format_bash_result

    assert format_bash_result("hi\n", returncode=1, elapsed_s=42.31) == "exit 1 in 42.3s\nhi\n"
    assert format_bash_result("", returncode=0, elapsed_s=0.0123) == "exit 0 in 0.01s"
    assert format_bash_result("x", returncode=-15, elapsed_s=3).startswith(
        "interrupted by signal 15 in 3.00s\n"
    )
    assert format_bash_result("x", returncode=None, elapsed_s=3).startswith("interrupted in ")
    assert format_bash_result("x", returncode=None, elapsed_s=30, timed_out_after=30) == (
        "timed out after 30s\nx"
    )
    assert len(format_bash_result("z" * 300_000, returncode=0, elapsed_s=1)) <= 200_000 + 40


def test_grep_regex_unwraps_escaped_pipes():
    rx = grep_regex(r"record_usage\|estimate_prompt_tokens\|context_window")
    assert rx is not None
    assert rx.search("def record_usage(session_id):")
    assert rx.search("estimate_prompt_tokens = 3")
    assert not rx.search("unrelated")


@pytest.mark.asyncio
async def test_run_tools_read_uses_offset(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("web/index.html", "\n".join(f"L{i}" for i in range(1, 50)))
    ctx = {"workspace": ws, "store": None, "session_id": uuid4(), "workspace_kind": "local"}
    out = await run_tools("Read", {"path": "web/index.html", "offset": 10, "limit": 5}, ctx)
    assert "10|L10" in out
    assert "14|L14" in out
    assert "9|L9" not in out


@pytest.mark.asyncio
async def test_run_tools_webfetch_extracts(monkeypatch, tmp_path):
    from orbweaver.webfetch import FetchResult

    html = "<html><body><h1>Hello</h1><p>Comparable tools use 50+ steps.</p></body></html>"

    async def fake_fetch(url, network, **_k):
        return FetchResult(url, url, 200, "text/html", html)

    monkeypatch.setattr("orbweaver.webfetch.fetch_url", fake_fetch)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    out = await run_tools(
        "WebFetch",
        {"url": "https://example.com/docs"},
        {"workspace": ws, "store": None, "session_id": uuid4(), "workspace_kind": "local"},
    )
    assert "Comparable tools use 50+ steps." in out
    assert "<h1>" not in out


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep (rg) required")
@pytest.mark.asyncio
async def test_run_tools_grep_passes_context_and_glob(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "alpha\nMATCH\nomega\n")
    ws.write("src/b.md", "MATCH\n")
    ctx = {"workspace": ws, "store": None, "session_id": uuid4(), "workspace_kind": "local"}
    out = await run_tools("Grep", {"pattern": "MATCH", "glob": "*.py", "C": 1}, ctx)
    assert "src/a.py" in out
    assert "alpha" in out
    assert "omega" in out
    assert "b.md" not in out
