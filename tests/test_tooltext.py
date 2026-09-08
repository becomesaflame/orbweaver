import shutil
from types import SimpleNamespace
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


class _Resp:
    def __init__(self, status, ctype, text):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.text = text


@pytest.mark.asyncio
async def test_run_tools_webfetch_extracts(monkeypatch, tmp_path):
    html = "<html><body><h1>Hello</h1><p>Comparable tools use 50+ steps.</p></body></html>"

    def fake_get(url, **_k):
        return _Resp(200, "text/html", html)

    monkeypatch.setattr("httpx.get", fake_get)
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
