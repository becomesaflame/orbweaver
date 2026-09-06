"""Postgres store. Used when ORBWEAVER_STORE=postgres."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg

from orbweaver.config import settings
from orbweaver.embeddings import embed_text
from orbweaver.store import (
    SESSION_TYPE,
    Chunk,
    Entity,
    Event,
    Job,
    ensure_pin_budget,
    jsonld_triples,
    new_uuid,
)
from orbweaver.uris import validate_workspace_uri

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text()


class PostgresStore:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(settings.database_url)
            async with self._pool.acquire() as conn:
                await conn.execute(_SCHEMA)
                await conn.execute(
                    """
                    INSERT INTO instance_meta(key, value) VALUES
                      ('embedding_model', $1),
                      ('embedding_dim', $2),
                      ('schema_version', '1'),
                      ('instance_id', $3)
                    ON CONFLICT (key) DO NOTHING
                    """,
                    settings.embedding_model,
                    str(settings.embedding_dim),
                    str(new_uuid()),
                )

    def _pool_req(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("postgres pool not connected")
        return self._pool

    async def put_entity(self, entity: Entity) -> Entity:
        if entity.at_type == SESSION_TYPE:
            uri = entity.jsonld.get("workspace_uri")
            if uri:
                validate_workspace_uri(str(uri))
        pool = self._pool_req()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO entities(id, at_id, at_type, jsonld, pinned)
                    VALUES ($1,$2,$3,$4::jsonb,$5)
                    ON CONFLICT (id) DO UPDATE SET jsonld=EXCLUDED.jsonld, pinned=EXCLUDED.pinned
                    """,
                    entity.id,
                    entity.at_id,
                    entity.at_type,
                    json.dumps(entity.jsonld),
                    entity.pinned,
                )
                await conn.execute("DELETE FROM triples WHERE subject=$1", entity.at_id)
                for s, p, o in jsonld_triples(entity):
                    await conn.execute(
                        "INSERT INTO triples(id,subject,predicate,object,graph_id) VALUES ($1,$2,$3,$4,$5)",
                        new_uuid(),
                        s,
                        p,
                        o,
                        "",
                    )
        return entity

    async def get_entity(self, uid: uuid.UUID) -> Entity | None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM entities WHERE id=$1", uid)
        return _entity(row) if row else None

    async def get_entity_by_at_id(self, at_id: str) -> Entity | None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM entities WHERE at_id=$1", at_id)
        return _entity(row) if row else None

    async def list_entities(self, at_type: str | None = None) -> list[Entity]:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            if at_type:
                rows = await conn.fetch("SELECT * FROM entities WHERE at_type=$1", at_type)
            else:
                rows = await conn.fetch("SELECT * FROM entities")
        return [_entity(r) for r in rows]

    async def put_triple(self, subject: str, predicate: str, obj: str, graph_id: str = "") -> uuid.UUID:
        tid = new_uuid()
        pool = self._pool_req()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO triples(id,subject,predicate,object,graph_id) VALUES ($1,$2,$3,$4,$5)",
                tid,
                subject,
                predicate,
                obj,
                graph_id,
            )
        return tid

    async def graph(self, at_id: str, depth: int = 1) -> list[tuple[str, str, str]]:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT subject, predicate, object FROM triples")
        triples = [(r["subject"], r["predicate"], r["object"]) for r in rows]
        frontier = {at_id}
        seen: set[str] = set()
        out: list[tuple[str, str, str]] = []
        for _ in range(max(1, depth)):
            nxt: set[str] = set()
            for s, p, o in triples:
                if s in frontier or o in frontier:
                    out.append((s, p, o))
                    nxt.add(s)
                    nxt.add(o)
            seen |= frontier
            frontier = nxt - seen
        return out

    async def put_chunk(self, chunk: Chunk) -> Chunk:
        pool = self._pool_req()
        vec = "[" + ",".join(str(x) for x in chunk.embedding) + "]"
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chunks(id,text,embedding,entity_ids,source,importance,decay_score,pinned)
                VALUES ($1,$2,$3::vector,$4,$5,$6,$7,$8)
                ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding
                """,
                chunk.id,
                chunk.text,
                vec,
                chunk.entity_ids,
                chunk.source,
                chunk.importance,
                chunk.decay_score,
                chunk.pinned,
            )
        return chunk

    async def get_chunk(self, uid: uuid.UUID) -> Chunk | None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM chunks WHERE id=$1 AND forgotten=FALSE", uid
            )
        return _chunk(row) if row else None

    async def search_chunks(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        q = embed_text(query)
        vec = "[" + ",".join(str(x) for x in q) + "]"
        pool = self._pool_req()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, 1 - (embedding <=> $1::vector) AS score
                FROM chunks WHERE forgotten=FALSE AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec,
                k,
            )
        return [(_chunk(r), float(r["score"])) for r in rows]

    async def pinned_chunks(self) -> list[Chunk]:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chunks WHERE pinned=TRUE AND forgotten=FALSE"
            )
        return [_chunk(r) for r in rows]

    async def pinned_entities(self) -> list[Entity]:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM entities WHERE pinned=TRUE")
        return [_entity(r) for r in rows]

    async def set_pinned(self, kind: str, uid: uuid.UUID, pinned: bool) -> None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            if kind == "chunk":
                row = await conn.fetchrow("SELECT * FROM chunks WHERE id=$1", uid)
                if not row:
                    raise KeyError(uid)
                if pinned:
                    await ensure_pin_budget(self, extra_text=row["text"])
                await conn.execute("UPDATE chunks SET pinned=$2 WHERE id=$1", uid, pinned)
                return
            row = await conn.fetchrow("SELECT * FROM entities WHERE id=$1", uid)
            if not row:
                raise KeyError(uid)
            if pinned:
                await ensure_pin_budget(self, extra_text=json.dumps(_json(row["jsonld"])))
            await conn.execute("UPDATE entities SET pinned=$2 WHERE id=$1", uid, pinned)

    async def forget_chunk(self, uid: uuid.UUID) -> None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE chunks SET forgotten=TRUE WHERE id=$1", uid)

    async def append_event(self, session_id: uuid.UUID, kind: str, payload: dict[str, Any]) -> Event:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            seq = await conn.fetchval(
                "SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE session_id=$1", session_id
            )
            ev = Event(
                id=new_uuid(), session_id=session_id, seq=int(seq), kind=kind, payload=payload
            )
            await conn.execute(
                "INSERT INTO events(id,session_id,seq,kind,payload) VALUES ($1,$2,$3,$4,$5::jsonb)",
                ev.id,
                ev.session_id,
                ev.seq,
                ev.kind,
                json.dumps(payload),
            )
        return ev

    async def list_events(self, session_id: uuid.UUID) -> list[Event]:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM events WHERE session_id=$1 ORDER BY seq", session_id
            )
        return [
            Event(
                id=r["id"],
                session_id=r["session_id"],
                seq=r["seq"],
                kind=r["kind"],
                payload=_json(r["payload"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def replace_events(self, session_id: uuid.UUID, events: list[Event]) -> None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM events WHERE session_id=$1", session_id)
                for ev in events:
                    await conn.execute(
                        "INSERT INTO events(id,session_id,seq,kind,payload) VALUES ($1,$2,$3,$4,$5::jsonb)",
                        ev.id,
                        ev.session_id,
                        ev.seq,
                        ev.kind,
                        json.dumps(ev.payload),
                    )

    async def truncate_events(self, session_id: uuid.UUID, from_seq: int) -> None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM events WHERE session_id=$1 AND seq>=$2",
                session_id,
                from_seq,
            )

    async def put_job(self, job: Job) -> Job:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs(id,due_at,recurrence,payload,session_id)
                VALUES ($1,$2,$3,$4::jsonb,$5)
                ON CONFLICT (id) DO UPDATE SET due_at=EXCLUDED.due_at, payload=EXCLUDED.payload
                """,
                job.id,
                job.due_at,
                job.recurrence,
                json.dumps(job.payload),
                job.session_id,
            )
        return job

    async def due_jobs(self, now: datetime) -> list[Job]:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM jobs WHERE due_at <= $1", now)
        return [
            Job(
                id=r["id"],
                due_at=r["due_at"],
                payload=_json(r["payload"]),
                recurrence=r["recurrence"],
                session_id=r["session_id"],
            )
            for r in rows
        ]

    async def reschedule_job(self, job: Job) -> None:
        await self.put_job(job)

    async def meta_get(self, key: str) -> str | None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT value FROM instance_meta WHERE key=$1", key)

    async def meta_set(self, key: str, value: str) -> None:
        pool = self._pool_req()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO instance_meta(key,value) VALUES ($1,$2)
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
                """,
                key,
                value,
            )


def _json(val: Any) -> dict[str, Any]:
    if isinstance(val, str):
        return json.loads(val)
    if isinstance(val, dict):
        return val
    return json.loads(val)


def _entity(row: asyncpg.Record) -> Entity:
    return Entity(
        id=row["id"],
        at_id=row["at_id"],
        at_type=row["at_type"],
        jsonld=_json(row["jsonld"]),
        pinned=row["pinned"],
    )


def _chunk(row: asyncpg.Record) -> Chunk:
    emb = row["embedding"]
    if isinstance(emb, str):
        embedding = [float(x) for x in emb.strip("[]").split(",") if x]
    else:
        embedding = list(emb) if emb is not None else []
    return Chunk(
        id=row["id"],
        text=row["text"],
        embedding=embedding,
        entity_ids=list(row["entity_ids"] or []),
        source=row["source"] or "",
        pinned=row["pinned"],
        importance=row["importance"],
        decay_score=row["decay_score"],
        created_at=row["created_at"],
    )
