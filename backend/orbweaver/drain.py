"""Local-only drain API helpers and the deploy waiter.

``update-live.sh`` POSTs ``/v1/admin/drain`` on loopback, then polls until no
turns remain (or the endpoint is missing, as on the first deploy of this
code). Caddy/Tailscale requests carry forwarded-for headers and are refused.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from starlette.requests import Request

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_FORWARDED_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")

DEFAULT_TIMEOUT_S = 900.0
DEFAULT_INTERVAL_S = 1.0


def is_local_admin(request: Request) -> bool:
    """True only for a direct loopback client, not a reverse-proxied one."""
    for name in _FORWARDED_HEADERS:
        if str(request.headers.get(name) or "").strip():
            return False
    client = request.client
    host = getattr(client, "host", "") if client is not None else ""
    return host in LOOPBACK_HOSTS


def drain_state() -> dict[str, Any]:
    from orbweaver.turns import is_draining, snapshot

    turns = snapshot()
    return {"draining": is_draining(), "turns": turns, "count": len(turns)}


def wait_for_idle(
    client: httpx.Client,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    sleep=time.sleep,
) -> str:
    """POST drain, poll until empty. Return ``idle``, ``unavailable``, or ``timeout``.

    ``unavailable`` means the running gateway has no drain endpoint (or is
    already down). The caller should still restart.
    """
    try:
        posted = client.post("/v1/admin/drain")
    except httpx.TransportError as e:
        print(f"drain: gateway unreachable ({e}); restarting", flush=True)
        return "unavailable"
    if posted.status_code == 404:
        print("drain: endpoint missing on this gateway; restarting without wait", flush=True)
        return "unavailable"
    if posted.status_code != 200:
        print(
            f"drain: POST /v1/admin/drain returned {posted.status_code}; restarting without wait",
            flush=True,
        )
        return "unavailable"

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            status = client.get("/v1/admin/drain")
        except httpx.TransportError as e:
            print(f"drain: gateway unreachable while waiting ({e}); restarting", flush=True)
            return "unavailable"
        if status.status_code == 404:
            print("drain: endpoint missing on this gateway; restarting without wait", flush=True)
            return "unavailable"
        if status.status_code != 200:
            print(
                f"drain: GET /v1/admin/drain returned {status.status_code}; restarting without wait",
                flush=True,
            )
            return "unavailable"
        payload = status.json()
        turns = payload.get("turns") if isinstance(payload, dict) else None
        if not isinstance(turns, list):
            print("drain: unexpected drain payload; restarting without wait", flush=True)
            return "unavailable"
        if not turns:
            print("drain: idle", flush=True)
            return "idle"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            labels = ", ".join(
                f"{t.get('channel', '?')}:{str(t.get('session_id', ''))[:8]}"
                for t in turns
                if isinstance(t, dict)
            )
            print(f"drain: timed out with {len(turns)} turn(s) still running ({labels})", flush=True)
            return "timeout"
        print(f"drain: waiting on {len(turns)} turn(s); {remaining:.0f}s left", flush=True)
        sleep(min(interval_s, remaining))
