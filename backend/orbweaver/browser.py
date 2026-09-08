"""Headless browser tool for UI verification (optional Playwright extra)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from orbweaver.permissions.rules import in_working_set

BROWSER_ACTIONS = frozenset({"navigate", "click", "type", "snapshot", "screenshot"})
SNAPSHOT_CHAR_CAP = 24_000
NAVIGATE_TIMEOUT_MS = 15_000
ACTION_TIMEOUT_MS = 8_000
UNAVAILABLE = (
    "Browser tool unavailable. Install the optional extra: "
    'pip install -e ".[browser]" && playwright install chromium'
)
MISSING_CHROMIUM = (
    "Playwright is installed but Chromium is missing. Run: playwright install chromium"
)


def playwright_available() -> bool:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return False
    return True


def validate_browser_input(inp: dict[str, Any]) -> str | None:
    action = str(inp.get("action") or "").strip().lower()
    if action not in BROWSER_ACTIONS:
        return (
            f"unknown action {action!r}; use navigate, click, type, snapshot, or screenshot"
        )
    if action == "navigate" and not str(inp.get("url") or "").strip():
        return "navigate requires url"
    if action in {"click", "type"} and not str(inp.get("selector") or "").strip():
        return f"{action} requires selector"
    if action == "type" and inp.get("text") is None:
        return "type requires text"
    return None


def resolve_navigate_target(raw: str, workspace) -> str:
    """Return an http(s) or workspace-confined file URL."""
    spec = (raw or "").strip()
    if not spec:
        raise ValueError("url is required for navigate")
    parsed = urlparse(spec)
    scheme = (parsed.scheme or "").lower()
    if scheme in {"http", "https"}:
        return spec
    if scheme not in {"", "file"}:
        raise ValueError(f"unsupported URL scheme: {scheme}")
    if scheme == "file":
        path = Path(parsed.path)
    else:
        path = Path(spec)
        if not path.is_absolute():
            root = getattr(workspace, "root", None)
            if root is None:
                raise ValueError("workspace-relative path needs a workspace")
            path = Path(root) / spec
    path = path.expanduser().resolve()
    if not in_working_set(str(path), workspace):
        raise PermissionError(f"file URL outside workspace: {spec}")
    return path.as_uri()


def format_snapshot(info: dict[str, Any], *, char_cap: int = SNAPSHOT_CHAR_CAP) -> str:
    lines = [
        f"URL: {info.get('url') or ''}",
        f"Title: {info.get('title') or ''}",
        "Interactive:",
    ]
    nodes = info.get("nodes") or []
    if not nodes:
        lines.append("  (none)")
    for node in nodes[:80]:
        tag = node.get("tag") or "?"
        ident = node.get("id") or ""
        name = node.get("name") or ""
        typ = node.get("type") or ""
        href = node.get("href") or ""
        text = (node.get("text") or "").replace("\n", " ")
        bits = [f"  {tag}{ident}"]
        if name:
            bits.append(f"name={name}")
        if typ:
            bits.append(f"type={typ}")
        if href:
            bits.append(f"href={href}")
        if text:
            bits.append(f'"{text}"')
        lines.append(" ".join(bits))
    body = (info.get("text") or "").strip()
    lines.append("Visible text:")
    lines.append(body or "(empty)")
    out = "\n".join(lines)
    if len(out) > char_cap:
        return out[:char_cap] + f"\n\n[truncated to {char_cap} chars]"
    return out


def classifier_payload(inp: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": inp.get("action"),
        "url": inp.get("url"),
        "selector": inp.get("selector"),
    }


def summarize_browser(inp: dict[str, Any]) -> str:
    action = str(inp.get("action") or "")
    extra = str(inp.get("url") or inp.get("selector") or inp.get("path") or "")
    return f"{action} {extra}".strip()[:240]


_COLLECT_JS = """() => {
  const nodes = [];
  const sel = 'a, button, input, textarea, select, [role="button"]';
  for (const el of document.querySelectorAll(sel)) {
    const tag = el.tagName.toLowerCase();
    const id = el.id ? '#' + el.id : '';
    nodes.push({
      tag,
      id,
      name: el.getAttribute('name') || '',
      type: el.getAttribute('type') || '',
      href: el.getAttribute('href') || '',
      text: (el.innerText || el.value || el.getAttribute('aria-label') || '')
        .trim()
        .slice(0, 80),
    });
  }
  return {
    title: document.title || '',
    url: location.href || '',
    text: (document.body && document.body.innerText) || '',
    nodes,
  };
}"""


class _Session:
    def __init__(self) -> None:
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._page is not None:
            return self._page
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        try:
            self._browser = await self._pw.chromium.launch(headless=True)
        except Exception:
            await self.close()
            raise
        context = await self._browser.new_context(viewport={"width": 1280, "height": 720})
        self._page = await context.new_page()
        return self._page

    async def close(self) -> None:
        page, browser, pw = self._page, self._browser, self._pw
        self._page = self._browser = self._pw = None
        for obj, method in ((page, "close"), (browser, "close"), (pw, "stop")):
            if obj is None:
                continue
            try:
                await getattr(obj, method)()
            except Exception:
                pass

    async def run(self, inp: dict[str, Any], workspace) -> str:
        async with self._lock:
            return await self._run(inp, workspace)

    async def _run(self, inp: dict[str, Any], workspace) -> str:
        action = str(inp.get("action") or "").strip().lower()
        try:
            page = await self._ensure()
        except Exception as e:
            msg = str(e)
            if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
                return MISSING_CHROMIUM
            return f"Browser failed to start: {e}"

        try:
            if action == "navigate":
                url = resolve_navigate_target(str(inp.get("url") or ""), workspace)
                await page.goto(url, wait_until="load", timeout=NAVIGATE_TIMEOUT_MS)
                return await self._snapshot(page)
            if self._page is None:
                return "no page yet; navigate first"
            if action == "click":
                await page.click(str(inp["selector"]), timeout=ACTION_TIMEOUT_MS)
                return await self._snapshot(page)
            if action == "type":
                await page.fill(
                    str(inp["selector"]),
                    str(inp.get("text") or ""),
                    timeout=ACTION_TIMEOUT_MS,
                )
                return await self._snapshot(page)
            if action == "snapshot":
                return await self._snapshot(page)
            if action == "screenshot":
                png = await page.screenshot(type="png")
                rel = str(inp.get("path") or "").strip() or (
                    f"attachments/browser-{uuid4().hex[:8]}.png"
                )
                display = workspace.write_bytes(rel, png)
                return f"screenshot saved to {display} ({len(png)} bytes)"
        except Exception as e:
            return f"Browser {action} failed: {e}"
        return f"unknown action {action}"

    async def _snapshot(self, page: Any) -> str:
        info = await page.evaluate(_COLLECT_JS)
        return format_snapshot(info if isinstance(info, dict) else {})


class BrowserPool:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    def _key(self, ctx: dict[str, Any]) -> str:
        sid = ctx.get("session_id")
        return str(sid) if sid is not None else "default"

    async def session_for(self, ctx: dict[str, Any]) -> _Session:
        key = self._key(ctx)
        async with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                sess = _Session()
                self._sessions[key] = sess
            return sess

    async def close_session(self, ctx: dict[str, Any] | None = None, *, key: str | None = None) -> None:
        sid = key if key is not None else (self._key(ctx) if ctx else None)
        if sid is None:
            return
        async with self._lock:
            sess = self._sessions.pop(sid, None)
        if sess is not None:
            await sess.close()

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for sess in sessions:
            await sess.close()


pool = BrowserPool()


async def run_browser(inp: dict[str, Any], ctx: dict[str, Any]) -> str:
    err = validate_browser_input(inp)
    if err:
        return err
    if not playwright_available():
        return UNAVAILABLE
    workspace = ctx.get("workspace")
    sess = await pool.session_for(ctx)
    return await sess.run(inp, workspace)
