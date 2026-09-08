"""Hindsight long-term fact layer: retain, recall, reflect over HTTP.

Pins, session entities, and the event stream stay in the Orbweaver store.
When HINDSIGHT_API_URL is unset, callers must use native memory.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx

from orbweaver.config import settings
from orbweaver.store import Event

log = logging.getLogger(__name__)

RETAIN_MISSION = (
    "Always include technical decisions, architectural trade-offs, preferences, "
    "project facts, and durable personal context. Ignore greetings, "
    "acknowledgements, and ephemeral tool noise."
)
REFLECT_MISSION = (
    "You are Orbweaver's long-term memory. Ground answers in documented "
    "decisions and retained facts. Prefer simplicity. Be direct and precise. "
    "Do not invent files, commands, or APIs."
)
OBSERVATIONS_MISSION = (
    "Observations are stable facts about people, projects, and preferences. "
    "Include recurring patterns. Ignore one-off events and ephemeral state."
)
BANK_NAME = "Orbweaver"
DISPOSITION = {"skepticism": 4, "literalism": 4, "empathy": 2}

_bank_ready = False


class HindsightError(Exception):
    """HTTP or configuration failure talking to Hindsight."""


def enabled() -> bool:
    return bool(settings.hindsight_api_url.strip())


def bank_id() -> str:
    return (settings.hindsight_bank_id or "personal").strip() or "personal"


def reset_for_tests() -> None:
    global _bank_ready
    _bank_ready = False


def _base_url() -> str:
    return settings.hindsight_api_url.strip().rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    key = settings.hindsight_api_key.strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _bank_path(suffix: str) -> str:
    return f"/v1/default/banks/{bank_id()}{suffix}"


async def request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    url = _base_url()
    if not url:
        raise HindsightError("HINDSIGHT_API_URL is not set")
    async with httpx.AsyncClient(base_url=url, timeout=timeout) as client:
        resp = await client.request(method, path, json=json, headers=_headers())
    if resp.status_code >= 400:
        raise HindsightError(f"{method} {path} -> {resp.status_code}: {resp.text[:800]}")
    if not resp.content:
        return {}
    ctype = resp.headers.get("content-type") or ""
    if "json" in ctype:
        return resp.json()
    return {"text": resp.text}


async def ensure_bank() -> None:
    global _bank_ready
    if _bank_ready:
        return
    await request(
        "PUT",
        _bank_path(""),
        json={
            "name": BANK_NAME,
            "disposition": DISPOSITION,
            "mission": REFLECT_MISSION,
        },
        timeout=30.0,
    )
    await request(
        "PATCH",
        _bank_path("/config"),
        json={
            "updates": {
                "retain_mission": RETAIN_MISSION,
                "reflect_mission": REFLECT_MISSION,
                "observations_mission": OBSERVATIONS_MISSION,
                "disposition_skepticism": DISPOSITION["skepticism"],
                "disposition_literalism": DISPOSITION["literalism"],
                "disposition_empathy": DISPOSITION["empathy"],
            }
        },
        timeout=30.0,
    )
    _bank_ready = True


async def retain(
    content: str,
    *,
    context: str = "",
    document_id: str = "",
    timestamp: str | None = None,
    retain_async: bool = True,
) -> dict[str, Any]:
    await ensure_bank()
    item: dict[str, Any] = {"content": content}
    if context:
        item["context"] = context
    if document_id:
        item["document_id"] = document_id
    if timestamp:
        item["timestamp"] = timestamp
    body: dict[str, Any] = {"items": [item], "async": retain_async}
    data = await request("POST", _bank_path("/memories"), json=body, timeout=60.0)
    return data if isinstance(data, dict) else {"raw": data}


async def recall(
    query: str,
    *,
    budget: str = "mid",
    max_tokens: int = 4096,
) -> dict[str, Any]:
    await ensure_bank()
    data = await request(
        "POST",
        _bank_path("/memories/recall"),
        json={"query": query, "budget": budget, "max_tokens": max_tokens},
        timeout=30.0,
    )
    return data if isinstance(data, dict) else {"results": []}


async def reflect(
    query: str,
    *,
    budget: str = "low",
    max_tokens: int = 4096,
) -> dict[str, Any]:
    await ensure_bank()
    data = await request(
        "POST",
        _bank_path("/reflect"),
        json={
            "query": query,
            "budget": budget,
            "max_tokens": max_tokens,
            "include": {"facts": {}},
        },
        timeout=120.0,
    )
    return data if isinstance(data, dict) else {"text": str(data)}


def format_recall(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    lines: list[str] = []
    ids: list[str] = []
    graph: list[dict[str, str]] = []
    hits: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "")
        text = str(row.get("text") or "")
        kind = str(row.get("type") or "world")
        if rid:
            ids.append(rid)
            hits.append(
                {
                    "id": rid,
                    "text": text,
                    "score": float(row.get("score") or 0.0),
                    "type": kind,
                }
            )
        if text:
            lines.append(f"[{rid} type={kind}] {text}")
        for ent in row.get("entities") or []:
            name = str(ent)
            if name and rid:
                graph.append({"s": name, "p": "mentioned_in", "o": rid})
    return {
        "chunk_ids": ids,
        "text": "\n".join(lines) or "(no hits)",
        "graph": graph,
        "hits": hits,
        "source": "hindsight",
    }


def format_reflect(payload: dict[str, Any]) -> dict[str, Any]:
    based = payload.get("based_on") or {}
    memories = []
    if isinstance(based, dict):
        for row in based.get("memories") or []:
            if isinstance(row, dict):
                memories.append(
                    {
                        "id": str(row.get("id") or ""),
                        "text": str(row.get("text") or ""),
                        "type": str(row.get("type") or ""),
                    }
                )
    return {
        "text": str(payload.get("text") or "").strip() or "(empty reflect)",
        "based_on": memories,
        "source": "hindsight",
    }


def _assistant_text(events: list[Event]) -> str:
    for ev in reversed(events):
        if ev.kind == "assistant":
            return str((ev.payload or {}).get("text") or "").strip()
    return ""


def format_turn_transcript(
    user_text: str, assistant_text: str, *, when: datetime | None = None
) -> str:
    stamp = (when or datetime.now(UTC)).isoformat()
    user = (user_text or "").strip()
    assistant = (assistant_text or "").strip()
    lines = []
    if user:
        lines.append(f"user [{stamp}]: {user}")
    if assistant:
        lines.append(f"assistant [{stamp}]: {assistant}")
    return "\n".join(lines)


async def retain_turn(
    session_id: UUID,
    user_text: str,
    produced: list[Event],
    *,
    subagent_depth: int = 0,
    channel: str = "",
) -> None:
    if not enabled() or subagent_depth:
        return
    assistant = _assistant_text(produced)
    content = format_turn_transcript(user_text, assistant)
    if not content:
        return
    ctx_bits = [f"session:{session_id}"]
    if channel:
        ctx_bits.append(f"channel:{channel}")
    try:
        await retain(
            content,
            context=" ".join(ctx_bits),
            document_id=str(session_id),
            retain_async=True,
        )
    except Exception as e:
        log.warning("hindsight retain_turn failed: %s", e)
