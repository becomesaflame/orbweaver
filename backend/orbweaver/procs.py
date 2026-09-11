"""Async subprocess helpers shared by raw and sandboxed Bash.

Tool execution runs on the gateway's asyncio loop. A blocking
``subprocess.run`` there stalls every session, so foreground Bash goes
through ``asyncio`` subprocesses. The helpers here wait for a process with
a timeout and a cancel event, and kill the whole process group when either
fires.
"""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from pathlib import Path
from typing import Any

OUTPUT_CAP = 200_000
GRACE_S = 2.0
DRAIN_S = 5.0

STATUS_EXITED = "exited"
STATUS_TIMEOUT = "timeout"
STATUS_INTERRUPTED = "interrupted"


class BashInterrupted(Exception):
    """The command was killed because the turn was cancelled."""

    def __init__(self, output: str = "") -> None:
        super().__init__("interrupted by user")
        self.output = output


def decode_captured(data: object) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def timeout_message(timeout: int, output: str) -> str:
    body = (output or "")[-OUTPUT_CAP:]
    prefix = f"timeout: command exceeded {timeout}s"
    return f"{prefix}\n{body}" if body else prefix


async def start_shell(command: str, cwd: Path | str) -> asyncio.subprocess.Process:
    """Start ``command`` under /bin/sh in its own process group."""
    return await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


def _signal_group(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        with suppress(ProcessLookupError, OSError):
            proc.send_signal(sig)


async def terminate_async(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM the process group, then SIGKILL if it does not exit within GRACE_S."""
    if proc.returncode is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    with suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), GRACE_S)
        return
    _signal_group(proc, signal.SIGKILL)
    with suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), GRACE_S)


async def communicate_async(
    proc: asyncio.subprocess.Process,
    timeout: float,
    *,
    cancel: asyncio.Event | None = None,
) -> tuple[str, str]:
    """Collect combined stdout+stderr with a deadline and a cancel event.

    Returns ``(output, status)`` where status is ``exited``, ``timeout``, or
    ``interrupted``. On timeout or cancel the process group is killed first.
    Cancelling the awaiting task also kills the process group.
    """
    comm = asyncio.ensure_future(proc.communicate())
    watchers: set[asyncio.Future[Any]] = set()
    if cancel is not None:
        watchers.add(asyncio.ensure_future(cancel.wait()))
    status = STATUS_EXITED
    try:
        done, _pending = await asyncio.wait(
            {comm, *watchers}, timeout=max(0.0, timeout), return_when=asyncio.FIRST_COMPLETED
        )
        if comm not in done:
            interrupted = any(w in done for w in watchers)
            status = STATUS_INTERRUPTED if interrupted else STATUS_TIMEOUT
            await terminate_async(proc)
            with suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(comm), DRAIN_S)
    except asyncio.CancelledError:
        await terminate_async(proc)
        comm.cancel()
        raise
    finally:
        for w in watchers:
            w.cancel()
        if not comm.done():
            comm.cancel()
    out = ""
    if comm.done() and not comm.cancelled() and comm.exception() is None:
        stdout, stderr = comm.result()
        out = decode_captured(stdout) + decode_captured(stderr)
    return out, status
