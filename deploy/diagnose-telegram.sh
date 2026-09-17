#!/usr/bin/env bash
# Read-only live-host dump: Telegram operator sessions, recent events that
# mention gh/auth, sandbox/SSH auth knobs. Prints to the Actions log only.
# Never prints token values or ~/.config/gh/hosts.yml contents.
set -euo pipefail

LIVE="${ORBWEAVER_LIVE:-/home/orbweaver/orbweaver}"
ENV_FILE="${LIVE}/backend/.env"
PY="${LIVE}/backend/.venv/bin/python"

echo "=== host ==="
hostname
date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
echo "live=$LIVE"
test -x "$PY" && "$PY" -c "from orbweaver import __version__; print('version', __version__)"

echo
echo "=== health ==="
curl -fsS --max-time 5 http://127.0.0.1:8080/health || echo "health failed"

echo
echo "=== journal (telegram / gh / auth, last 48h) ==="
journalctl --user -u orbweaver.service --since "48 hours ago" --no-pager 2>/dev/null \
  | rg -i 'telegram|gh auth|hosts\.yml|not logged|HTTP 40|Turn failed|operator|attach|OpenAICompat|required' \
  | tail -n 120 || echo "(no journal matches)"

echo
echo "=== SSH / sandbox auth knobs (no secrets) ==="
sudo -u orbweaver bash -lc '
  echo "SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-<unset>}"
  if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then ls -la "$SSH_AUTH_SOCK" 2>&1 || true; fi
  systemctl --user show-environment 2>/dev/null | rg -i "SSH_AUTH|ORBWEAVER_SANDBOX|GH_|GITHUB_" || true
  if [[ -f ~/.orbweaver/sandbox.json ]]; then
    python3 - <<"PY"
import json
from pathlib import Path
p = Path.home() / ".orbweaver" / "sandbox.json"
data = json.loads(p.read_text())
print("sandbox.json keys:", sorted(data))
print("allowRead:", data.get("allowRead"))
print("ssh:", data.get("ssh"))
print("env.allow:", (data.get("env") or {}).get("allow"))
PY
  else
    echo "no ~/.orbweaver/sandbox.json"
  fi
  test -e ~/.config/gh/hosts.yml && echo "hosts.yml: present" || echo "hosts.yml: missing"
  # Host-side gh (outside sandbox) — status only, no tokens.
  command -v gh >/dev/null && gh auth status 2>&1 | head -20 || echo "gh not on PATH for orbweaver"
'

echo
echo "=== Telegram session events (redacted) ==="
# Load DATABASE_URL / store settings the same way the gateway does.
set -a
# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && . "$ENV_FILE"
set +a

cd "${LIVE}/backend"
"$PY" - <<'PY'
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime, timedelta

from orbweaver.store import SESSION_TYPE, get_store, reset_store_for_tests

SECRET = re.compile(
    r"(ghp_[A-Za-z0-9]+|gho_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9-_]+|Bearer\s+\S+|token[=: ]+\S+)",
    re.I,
)
INTEREST = re.compile(
    r"\b(gh|auth|hosts\.yml|not logged|HTTP\s*40|permission denied|publickey|"
    r"SSH_AUTH|attach|operator|ListSessions|AttachSession)\b",
    re.I,
)


def scrub(text: str) -> str:
    return SECRET.sub("[REDACTED]", text or "")


async def main() -> None:
    # Prefer the live store (postgres when configured). Fall back message if unset.
    store = get_store()
    connect = getattr(store, "connect", None)
    if connect:
        await connect()
    since = datetime.now(UTC) - timedelta(hours=48)
    ops = [
        e
        for e in await store.list_entities(SESSION_TYPE)
        if (e.jsonld or {}).get("telegram_user_id") is not None
        or (e.jsonld or {}).get("role") == "operator"
        or (e.jsonld or {}).get("channel") == "telegram"
    ]
    print(f"telegram/operator sessions: {len(ops)}")
    if not ops:
        print("(none found — store may be empty in this process if ORBWEAVER_STORE!=postgres)")
        print("ORBWEAVER_STORE=", os.environ.get("ORBWEAVER_STORE", ""))
        print("DATABASE_URL set=", bool(os.environ.get("DATABASE_URL")))
        return

    for op in ops:
        j = op.jsonld or {}
        print()
        print("--- session", op.id, "---")
        print(
            "title=", j.get("title"),
            "role=", j.get("role"),
            "channel=", j.get("channel"),
            "model=", j.get("model") or "(channel default)",
            "workspace=", j.get("workspace_uri"),
            "attached=", j.get("attached_session"),
            "telegram_user_id=", j.get("telegram_user_id"),
        )
        events = await store.list_events(op.id)
        recent = [e for e in events if e.created_at.replace(tzinfo=UTC) >= since] or events[-40:]
        print(f"events_total={len(events)} showing={len(recent)}")
        for e in recent:
            payload = e.payload or {}
            blob = scrub(json.dumps(payload, default=str)[:2000])
            interesting = bool(INTEREST.search(blob)) or e.kind in {
                "user",
                "assistant",
                "turn_aborted",
                "turn_interrupted",
                "ask_user",
            }
            if not interesting and e.kind not in {"tool_call", "tool_result"}:
                continue
            # Always show user/assistant; for tools only when interesting.
            if e.kind in {"tool_call", "tool_result"} and not INTEREST.search(blob):
                continue
            text = ""
            if e.kind in {"user", "assistant", "ask_user", "turn_aborted"}:
                text = scrub(str(payload.get("text") or payload.get("question") or ""))[:800]
            elif e.kind == "tool_call":
                text = f"{payload.get('name')} {scrub(json.dumps(payload.get('input') or {})[:400])}"
            elif e.kind == "tool_result":
                text = f"{payload.get('name')} {scrub(str(payload.get('content') or '')[:600])}"
            else:
                text = blob[:600]
            mark = " ***" if INTEREST.search(text) or INTEREST.search(blob) else ""
            via = payload.get("via")
            via_s = f" via={via}" if via else ""
            print(
                f"  [{e.created_at.isoformat()}] seq={e.seq} {e.kind}{via_s}{mark}"
            )
            if text:
                for line in text.splitlines()[:12]:
                    print("   ", line[:200])

        # If attached, also dump the target session's recent interesting events.
        attached = j.get("attached_session") or ""
        if attached.startswith("urn:orbweaver:session:"):
            from uuid import UUID

            tid = UUID(attached.rsplit(":", 1)[-1])
            target = await store.get_entity(tid)
            if target:
                print()
                print("--- attached target", tid, "---")
                print(
                    "title=", (target.jsonld or {}).get("title"),
                    "channel=", (target.jsonld or {}).get("channel"),
                    "workspace=", (target.jsonld or {}).get("workspace_uri"),
                    "model=", (target.jsonld or {}).get("model"),
                )
                tevents = await store.list_events(tid)
                trecent = [
                    e for e in tevents if e.created_at.replace(tzinfo=UTC) >= since
                ] or tevents[-40:]
                for e in trecent:
                    payload = e.payload or {}
                    blob = scrub(json.dumps(payload, default=str)[:2000])
                    if e.kind not in {"user", "assistant", "tool_call", "tool_result", "turn_aborted"}:
                        continue
                    if e.kind in {"tool_call", "tool_result"} and not INTEREST.search(blob):
                        continue
                    text = scrub(
                        str(
                            payload.get("text")
                            or payload.get("content")
                            or payload.get("name")
                            or ""
                        )
                    )[:800]
                    if e.kind == "tool_call":
                        text = f"{payload.get('name')} {scrub(json.dumps(payload.get('input') or {})[:400])}"
                    mark = " ***" if INTEREST.search(text) or INTEREST.search(blob) else ""
                    via = payload.get("via")
                    via_s = f" via={via}" if via else ""
                    print(f"  [{e.created_at.isoformat()}] seq={e.seq} {e.kind}{via_s}{mark}")
                    for line in text.splitlines()[:12]:
                        print("   ", line[:200])


asyncio.run(main())
PY
