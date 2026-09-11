"""Tool execution must not block the event loop, and Stop must kill Bash (#98)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import pytest

from orbweaver.agent import (
    INTERRUPTED_BY_USER,
    ToolInterrupted,
    TurnCancelled,
    _events_to_messages,
    agent_turn,
    run_tools,
)
from orbweaver.config import settings
from orbweaver.permissions.pipeline import PermissionDecision
from orbweaver.procs import BashInterrupted
from orbweaver.sandbox.bwrap import bwrap_path
from orbweaver.store import reset_store_for_tests
from orbweaver.workspace import LocalWorkspace


def _ctx(ws, cancel: asyncio.Event | None = None) -> dict:
    return {
        "workspace": ws,
        "store": reset_store_for_tests(),
        "session_id": uuid4(),
        "workspace_kind": "local",
        "cancel": cancel,
    }


async def _tick(counter: list[int], stop: asyncio.Event, period: float = 0.1) -> None:
    while not stop.is_set():
        counter[0] += 1
        await asyncio.sleep(period)


def _procs_matching(tag: str) -> list[str]:
    out = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True, check=False).stdout
    return [ln for ln in out.split() if ln.strip()]


async def _wait_gone(tag: str, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _procs_matching(tag):
            return True
        await asyncio.sleep(0.05)
    return not _procs_matching(tag)


@pytest.mark.asyncio
async def test_bash_tool_does_not_block_event_loop(tmp_path: Path, monkeypatch):
    """A 0.1 s ticker keeps running while Bash sleeps for 2 s (issue: it fired 0 times)."""
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    counter = [0]
    stop = asyncio.Event()
    ticker = asyncio.create_task(_tick(counter, stop))
    try:
        started = time.monotonic()
        out = await run_tools("Bash", {"command": "sleep 2; echo slept"}, _ctx(ws))
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        await ticker
    assert "slept" in out
    assert elapsed >= 2
    assert counter[0] >= 10, f"ticker fired {counter[0]} times during a 2 s Bash"


@pytest.mark.asyncio
async def test_read_tool_does_not_block_event_loop(tmp_path: Path):
    """File tools yield to the loop (slow_read simulates slow storage)."""
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("src/a.py", "needle = 1\n")
    real_read = ws.read

    def slow_read(path: str) -> str:
        time.sleep(0.6)
        return real_read(path)

    ws.read = slow_read  # type: ignore[method-assign]
    counter = [0]
    stop = asyncio.Event()
    ticker = asyncio.create_task(_tick(counter, stop, period=0.05))
    try:
        out = await run_tools("Read", {"path": "src/a.py"}, _ctx(ws))
    finally:
        stop.set()
        await ticker
    assert "needle" in out
    assert counter[0] >= 5


@pytest.mark.asyncio
async def test_bash_cancel_kills_process_group(tmp_path: Path, monkeypatch):
    """Setting cancel 0.5 s into `sleep 30` returns promptly and the process is gone."""
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    cancel = asyncio.Event()
    tag = f"ow-cancel-{uuid4().hex[:8]}"
    command = f"sleep 3079 # {tag}"

    async def stop_soon() -> None:
        await asyncio.sleep(0.5)
        cancel.set()

    stopper = asyncio.create_task(stop_soon())
    started = time.monotonic()
    with pytest.raises(ToolInterrupted) as info:
        await run_tools("Bash", {"command": command}, _ctx(ws, cancel))
    elapsed = time.monotonic() - started
    await stopper
    assert info.value.message == INTERRUPTED_BY_USER
    assert 0.4 <= elapsed < 1.5, f"cancel took {elapsed:.2f}s"
    assert await _wait_gone(tag), "shell wrapper still running after cancel"
    assert await _wait_gone("sleep 3079"), "sleep child still running after cancel"


@pytest.mark.asyncio
async def test_workspace_bash_async_raises_interrupted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    cancel = asyncio.Event()
    cancel.set()
    with pytest.raises(BashInterrupted):
        await ws.bash_async("sleep 30", cancel=cancel)


@pytest.mark.asyncio
async def test_bash_async_timeout_and_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    assert (await ws.bash_async("echo hi; echo err 1>&2")).strip().splitlines() == ["hi", "err"]
    started = time.monotonic()
    out = await ws.bash_async("echo partial; sleep 5", timeout=1)
    assert time.monotonic() - started < 4
    assert "timeout: command exceeded 1s" in out
    assert "partial" in out


@pytest.mark.asyncio
async def test_bash_async_background_and_collect(tmp_path: Path, monkeypatch):
    import json

    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    started = json.loads(
        await ws.bash_async("sleep 0.3; echo async-job", background=True, timeout=10)
    )
    assert started["status"] == "running"
    counter = [0]
    stop = asyncio.Event()
    ticker = asyncio.create_task(_tick(counter, stop, period=0.05))
    try:
        done = json.loads(await ws.bash_async(job_id=started["job_id"], timeout=5))
    finally:
        stop.set()
        await ticker
    assert done["status"] == "exited"
    assert "async-job" in done["output"]
    assert counter[0] >= 3


@pytest.mark.skipif(bwrap_path() is None, reason="bubblewrap required")
@pytest.mark.asyncio
async def test_sandboxed_bash_async_runs_and_cancels(tmp_path: Path, monkeypatch):
    from orbweaver.sandbox.bwrap import run_sandboxed_async

    monkeypatch.setattr(settings, "orbweaver_sandbox", True)
    monkeypatch.setattr("orbweaver.sandbox.bwrap.is_containerized", lambda: False)
    try:
        out = await run_sandboxed_async("echo sandbox-async-ok", tmp_path, timeout=15)
    except Exception as e:  # bwrap may be blocked on this host (AppArmor, nested userns)
        pytest.skip(f"bwrap cannot run here: {e}")
    if "sandbox-async-ok" not in out:
        pytest.skip(f"bwrap cannot run here: {out[-300:]}")
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    cancel = asyncio.Event()

    async def stop_soon() -> None:
        await asyncio.sleep(0.7)
        cancel.set()

    stopper = asyncio.create_task(stop_soon())
    started = time.monotonic()
    with pytest.raises(BashInterrupted):
        await ws.bash_async("sleep 3081", timeout=30, cancel=cancel)
    await stopper
    assert time.monotonic() - started < 6
    assert await _wait_gone("sleep 3081")


class _ToolUse:
    def __init__(self, name, inp, uid="tu-async"):
        self.type = "tool_use"
        self.id = uid
        self.name = name
        self.input = inp


class _FakeAnthropic:
    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = self

    async def create(self, **_kwargs):
        if not self._responses:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_agent_turn_stop_interrupts_bash_and_records_error_result(
    tmp_path: Path, monkeypatch
):
    """Stop during Bash: the turn ends ~immediately with an is_error tool_result."""
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    async def allow(*_a, **_k):
        return PermissionDecision("allow", "test", "test")

    async def no_compact(*_a, **_k):
        return None

    async def no_probe(_name, output, **_k):
        return {"flagged": False, "output": output}

    monkeypatch.setattr("orbweaver.agent.can_use_tool", allow)
    monkeypatch.setattr("orbweaver.agent.maybe_compact", no_compact)
    monkeypatch.setattr("orbweaver.agent.probe_tool_output", no_probe)
    tag = f"ow-turn-{uuid4().hex[:8]}"
    client = _FakeAnthropic(
        [SimpleNamespace(content=[_ToolUse("Bash", {"command": f"sleep 3083 # {tag}"})])]
    )
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: client)
    store = reset_store_for_tests()
    sid = uuid4()
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    cancel = asyncio.Event()

    async def stop_soon() -> None:
        await asyncio.sleep(0.5)
        cancel.set()

    stopper = asyncio.create_task(stop_soon())
    started = time.monotonic()
    with pytest.raises(TurnCancelled) as info:
        await agent_turn(store, sid, "run the long thing", ws, cancel=cancel, max_rounds=3)
    elapsed = time.monotonic() - started
    await stopper
    assert elapsed < 1.5, f"turn took {elapsed:.2f}s to stop"
    assert await _wait_gone(tag)
    results = [e for e in info.value.produced if e.kind == "tool_result"]
    assert len(results) == 1
    assert results[0].payload["is_error"] is True
    assert results[0].payload["content"] == INTERRUPTED_BY_USER
    stored = await store.list_events(sid)
    assert [e.kind for e in stored] == ["user", "tool_call", "permission_decision", "tool_result"]
    messages = _events_to_messages(stored)
    block = messages[-1]["content"][0]
    assert block["type"] == "tool_result"
    assert block["is_error"] is True


@pytest.mark.asyncio
async def test_webfetch_uses_async_client(monkeypatch, tmp_path: Path):
    import httpx

    calls: list[str] = []

    async def fake_get(self, url, **_k):
        calls.append(url)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="<p>async fetched</p>"
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    out = await run_tools("WebFetch", {"url": "https://example.com/x"}, _ctx(ws))
    assert calls == ["https://example.com/x"]
    assert "async fetched" in out


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep required")
@pytest.mark.asyncio
async def test_sync_workspace_api_still_works(tmp_path: Path, monkeypatch):
    """Callers such as read_lints keep the synchronous LocalWorkspace.bash."""
    monkeypatch.setattr(settings, "orbweaver_sandbox", False)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    ws.write("a.txt", "x\n")
    assert ws.bash("echo sync").strip() == "sync"
    assert ws.glob("*.txt") == ["a.txt"]
