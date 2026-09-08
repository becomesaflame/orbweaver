"""Browser tool wiring (always) and Playwright UI checks (optional extra)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from orbweaver.agent import TOOL_SPEC, run_tools
from orbweaver.browser import (
    UNAVAILABLE,
    classifier_payload,
    format_snapshot,
    playwright_available,
    pool,
    resolve_navigate_target,
    summarize_browser,
    validate_browser_input,
)
from orbweaver.compact.persist import persist_tool_result
from orbweaver.config import settings
from orbweaver.permissions.classifier import build_transcript, to_classifier_input
from orbweaver.permissions.denial import DenialTrackingState, reset_denial_states
from orbweaver.permissions.injection_probe import PROBE_TOOLS
from orbweaver.permissions.pipeline import can_use_tool
from orbweaver.store import Event
from orbweaver.workspace import LocalWorkspace

LOCAL_PAGE = """<!DOCTYPE html>
<html>
<head><title>Verify UI</title></head>
<body>
  <h1 id="title">Hello</h1>
  <p id="status">idle</p>
  <input id="name" type="text" />
  <button id="go" type="button">Go</button>
  <script>
    document.getElementById('go').addEventListener('click', () => {
      const name = document.getElementById('name').value || 'anon';
      document.getElementById('status').textContent = 'clicked:' + name;
    });
  </script>
</body>
</html>
"""


def _ctx(tmp_path: Path):
    reset_denial_states()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    return {
        "workspace": ws,
        "workspace_kind": "local",
        "headless": False,
        "session_id": sid,
        "denial_state": DenialTrackingState(),
        "events": [
            Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "check the UI"})
        ],
        "store": None,
    }


def test_browser_in_tool_spec():
    spec = next(t for t in TOOL_SPEC if t["name"] == "Browser")
    actions = spec["input_schema"]["properties"]["action"]["enum"]
    assert set(actions) == {"navigate", "click", "type", "snapshot", "screenshot"}


def test_classifier_sees_browser_tool():
    inp = {"action": "navigate", "url": "https://example.com/app", "text": "secret"}
    encoded = to_classifier_input("Browser", inp)
    assert encoded == classifier_payload(inp)
    assert encoded["action"] == "navigate"
    assert encoded["url"] == "https://example.com/app"
    assert "secret" not in str(encoded)
    sid = uuid4()
    events = [
        Event(id=uuid4(), session_id=sid, seq=1, kind="user", payload={"text": "verify the form"})
    ]
    text = build_transcript(events, "Browser", inp)
    assert "Browser" in text
    assert "https://example.com/app" in text
    assert "secret" not in text


def test_summarize_and_validate():
    assert summarize_browser({"action": "click", "selector": "#go"}) == "click #go"
    assert validate_browser_input({"action": "navigate"}) == "navigate requires url"
    assert validate_browser_input({"action": "click"}) == "click requires selector"
    assert validate_browser_input({"action": "type", "selector": "#name"}) == "type requires text"
    assert validate_browser_input({"action": "zoom"}) is not None
    assert validate_browser_input({"action": "snapshot"}) is None


def test_file_url_must_stay_in_workspace(tmp_path):
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    page = tmp_path / "page.html"
    page.write_text(LOCAL_PAGE, encoding="utf-8")
    assert resolve_navigate_target("page.html", ws).startswith("file://")
    assert resolve_navigate_target(page.as_uri(), ws).startswith("file://")
    with pytest.raises(PermissionError, match="outside workspace"):
        resolve_navigate_target("/etc/passwd", ws)
    with pytest.raises(PermissionError, match="outside workspace"):
        resolve_navigate_target(Path("/etc/passwd").as_uri(), ws)
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        resolve_navigate_target("ftp://example.com", ws)
    assert resolve_navigate_target("https://example.com/app", ws) == "https://example.com/app"


def test_format_snapshot_lists_controls():
    out = format_snapshot(
        {
            "url": "file:///tmp/page.html",
            "title": "Verify UI",
            "text": "Hello\nidle",
            "nodes": [
                {"tag": "button", "id": "#go", "text": "Go"},
                {"tag": "input", "id": "#name", "type": "text"},
            ],
        }
    )
    assert "URL: file:///tmp/page.html" in out
    assert "Title: Verify UI" in out
    assert "button#go" in out
    assert "Hello" in out


def test_browser_skipped_from_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "compact_tool_result_chars", 50)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    body = "URL: file:///x\n" + ("clicked:Ada\n" * 40)
    stored, rel = persist_tool_result(ws, "toolu_browser", "Browser", body)
    assert rel is None
    assert "persisted-output" not in stored
    assert "clicked:Ada" in stored


def test_browser_is_probed():
    assert "Browser" in PROBE_TOOLS


@pytest.mark.asyncio
async def test_browser_hits_classifier_not_allowlist(tmp_path, monkeypatch):
    seen: dict = {}

    async def classify(_events, name, inp, **_k):
        seen["name"] = name
        seen["inp"] = inp
        return {"should_block": False, "reason": "ok", "stage": "fast"}

    monkeypatch.setattr("orbweaver.permissions.pipeline.classify_action", classify)
    ctx = _ctx(tmp_path)
    decision = await can_use_tool(
        "Browser",
        {"action": "navigate", "url": "https://example.com"},
        ctx,
    )
    assert decision.behavior == "allow"
    assert decision.fast_path == "classifier"
    assert seen["name"] == "Browser"
    assert seen["inp"]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_run_tools_unavailable_without_playwright(tmp_path, monkeypatch):
    monkeypatch.setattr("orbweaver.browser.playwright_available", lambda: False)
    ctx = _ctx(tmp_path)
    out = await run_tools("Browser", {"action": "snapshot"}, ctx)
    assert out == UNAVAILABLE
    assert "pip install -e" in out


@pytest.mark.asyncio
async def test_run_tools_rejects_bad_action(tmp_path):
    ctx = _ctx(tmp_path)
    out = await run_tools("Browser", {"action": "zoom"}, ctx)
    assert "unknown action" in out


class _FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.text = "idle"
        self.value = ""
        self.clicked: list[str] = []

    async def goto(self, url, **_k):
        self.url = url
        self.text = "Hello\nidle"

    async def click(self, selector, **_k):
        self.clicked.append(selector)
        self.text = f"clicked:{self.value or 'anon'}"

    async def fill(self, selector, text, **_k):
        self.value = text

    async def screenshot(self, **_k):
        return b"\x89PNG\r\n\x1a\n" + b"fake"

    async def evaluate(self, _js):
        extra = f"\n{self.value}" if self.value else ""
        return {
            "url": self.url,
            "title": "Verify UI",
            "text": self.text + extra,
            "nodes": [{"tag": "button", "id": "#go", "text": "Go"}],
        }


@pytest.mark.asyncio
async def test_mocked_session_verifies_local_page(tmp_path, monkeypatch):
    page = _FakePage()

    async def ensure(self):
        self._page = page
        return page

    monkeypatch.setattr("orbweaver.browser.playwright_available", lambda: True)
    monkeypatch.setattr("orbweaver.browser._Session._ensure", ensure)
    ctx = _ctx(tmp_path)
    (tmp_path / "page.html").write_text(LOCAL_PAGE, encoding="utf-8")
    try:
        nav = await run_tools("Browser", {"action": "navigate", "url": "page.html"}, ctx)
        assert "Hello" in nav
        assert page.url.startswith("file://")
        await run_tools("Browser", {"action": "type", "selector": "#name", "text": "Ada"}, ctx)
        clicked = await run_tools("Browser", {"action": "click", "selector": "#go"}, ctx)
        assert "clicked:Ada" in clicked
        assert page.clicked == ["#go"]
        shot = await run_tools(
            "Browser", {"action": "screenshot", "path": "attachments/ui.png"}, ctx
        )
        assert "attachments/ui.png" in shot
        png = tmp_path / "attachments" / "ui.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        await pool.close_session(ctx)


def _playwright_ready() -> bool:
    if not playwright_available():
        return False
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            browser.close()
    except Exception:
        return False
    return True


@pytest.fixture
async def browser_ctx(tmp_path):
    if not _playwright_ready():
        pytest.skip("optional [browser] extra or Chromium not installed")
    ctx = _ctx(tmp_path)
    (tmp_path / "page.html").write_text(LOCAL_PAGE, encoding="utf-8")
    try:
        yield ctx
    finally:
        await pool.close_session(ctx)


@pytest.mark.asyncio
async def test_playwright_local_page_click_and_type(browser_ctx):
    ctx = browser_ctx
    nav = await run_tools("Browser", {"action": "navigate", "url": "page.html"}, ctx)
    assert "Verify UI" in nav
    assert "idle" in nav
    assert "Hello" in nav
    typed = await run_tools(
        "Browser",
        {"action": "type", "selector": "#name", "text": "Ada"},
        ctx,
    )
    assert "Ada" in typed or "input#name" in typed
    clicked = await run_tools("Browser", {"action": "click", "selector": "#go"}, ctx)
    assert "clicked:Ada" in clicked
    snap = await run_tools("Browser", {"action": "snapshot"}, ctx)
    assert "clicked:Ada" in snap
    shot = await run_tools("Browser", {"action": "screenshot"}, ctx)
    assert "screenshot saved to" in shot
    pngs = list(Path(ctx["workspace"].root).glob("attachments/*.png"))
    assert pngs
    assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
