from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from orbweaver.embeddings import embed_text
from orbweaver.store import (
    Chunk,
    Store,
    new_uuid,
)


async def pinned_prompt(store: Store) -> str:
    ents = await store.pinned_entities()
    chunks = await store.pinned_chunks()
    lines = ["# Pinned working memory"]
    for e in ents:
        lines.append(f"- ({e.at_type}) {e.at_id}: {json.dumps(e.jsonld)[:800]}")
    for c in chunks:
        lines.append(f"- [{c.id}] {c.text}")
    return "\n".join(lines) if (ents or chunks) else ""


def rewrite_search_query(events: list[Any], current: str, n: int = 6) -> str:
    recent = events[-n:] if events else []
    bits = []
    for ev in recent:
        kind = getattr(ev, "kind", "")
        payload = getattr(ev, "payload", {}) or {}
        if kind in {"user", "assistant"}:
            bits.append(str(payload.get("text") or payload.get("content") or "")[:400])
    bits.append(current)
    return "\n".join(x for x in bits if x).strip() or current


async def remember(
    store: Store, text: str, source: str = "", pinned: bool = False, entity_ids: list[UUID] | None = None
) -> Chunk:
    chunk = Chunk(
        id=new_uuid(),
        text=text,
        embedding=embed_text(text),
        entity_ids=entity_ids or [],
        source=source,
        pinned=False,
    )
    await store.put_chunk(chunk)
    if pinned:
        await store.set_pinned("chunk", chunk.id, True)
    return chunk
