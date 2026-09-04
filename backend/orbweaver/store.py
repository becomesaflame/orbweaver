"""In-memory and Postgres stores. Schema is UUID-keyed and host-path free."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from orbweaver.config import settings
from orbweaver.embeddings import embed_text
from orbweaver.tokens import estimate_tokens
from orbweaver.uris import WorkspaceURIError, validate_workspace_uri

SESSION_TYPE = "Session"
AGENT_TYPE = "Agent"


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def session_at_id(uid: uuid.UUID) -> str:
    return f"urn:orbweaver:session:{uid}"


def agent_at_id(uid: uuid.UUID) -> str:
    return f"urn:orbweaver:agent:{uid}"


@dataclass
class Entity:
    id: uuid.UUID
    at_id: str
    at_type: str
    jsonld: dict[str, Any]
    pinned: bool = False


def jsonld_triples(entity: Entity) -> list[tuple[str, str, str]]:
    """Index IRI-valued JSON-LD properties as graph edges."""
    out: list[tuple[str, str, str]] = []
    subject = entity.at_id

    def _add(pred: str, value: Any) -> None:
        if isinstance(value, str) and (value.startswith("urn:") or value.startswith("http")):
            out.append((subject, pred, value))
        elif isinstance(value, dict):
            nested = value.get("@id")
            if isinstance(nested, str):
                out.append((subject, pred, nested))
        elif isinstance(value, list):
            for item in value:
                _add(pred, item)

    for key, value in entity.jsonld.items():
        if key.startswith("@") or key in {"id", "workspace_uri", "title", "status"}:
            continue
        _add(key, value)
    return out


@dataclass
class Chunk:
    id: uuid.UUID
    text: str
    embedding: list[float]
    entity_ids: list[uuid.UUID] = field(default_factory=list)
    source: str = ""
    pinned: bool = False
    importance: float = 1.0
    decay_score: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Event:
    id: uuid.UUID
    session_id: uuid.UUID
    seq: int
    kind: str
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Job:
    id: uuid.UUID
    due_at: datetime
    payload: dict[str, Any]
    recurrence: str | None = None
    session_id: uuid.UUID | None = None


class Store(Protocol):
    async def put_entity(self, entity: Entity) -> Entity: ...
    async def get_entity(self, uid: uuid.UUID) -> Entity | None: ...
    async def get_entity_by_at_id(self, at_id: str) -> Entity | None: ...
    async def list_entities(self, at_type: str | None = None) -> list[Entity]: ...
    async def put_triple(self, subject: str, predicate: str, obj: str, graph_id: str = "") -> uuid.UUID: ...
    async def graph(self, at_id: str, depth: int = 1) -> list[tuple[str, str, str]]: ...
    async def put_chunk(self, chunk: Chunk) -> Chunk: ...
    async def get_chunk(self, uid: uuid.UUID) -> Chunk | None: ...
    async def search_chunks(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]: ...
    async def pinned_chunks(self) -> list[Chunk]: ...
    async def pinned_entities(self) -> list[Entity]: ...
    async def set_pinned(self, kind: str, uid: uuid.UUID, pinned: bool) -> None: ...
    async def forget_chunk(self, uid: uuid.UUID) -> None: ...
    async def append_event(self, session_id: uuid.UUID, kind: str, payload: dict[str, Any]) -> Event: ...
    async def list_events(self, session_id: uuid.UUID) -> list[Event]: ...
    async def replace_events(self, session_id: uuid.UUID, events: list[Event]) -> None: ...
    async def put_job(self, job: Job) -> Job: ...
    async def due_jobs(self, now: datetime) -> list[Job]: ...
    async def reschedule_job(self, job: Job) -> None: ...
    async def meta_get(self, key: str) -> str | None: ...
    async def meta_set(self, key: str, value: str) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self.entities: dict[uuid.UUID, Entity] = {}
        self.triples: list[tuple[uuid.UUID, str, str, str, str]] = []
        self.chunks: dict[uuid.UUID, Chunk] = {}
        self.forgotten: set[uuid.UUID] = set()
        self.events: dict[uuid.UUID, list[Event]] = {}
        self.jobs: dict[uuid.UUID, Job] = {}
        self.meta: dict[str, str] = {
            "embedding_model": settings.embedding_model,
            "embedding_dim": str(settings.embedding_dim),
            "schema_version": "1",
            "instance_id": str(new_uuid()),
        }

    async def put_entity(self, entity: Entity) -> Entity:
        if entity.at_type == SESSION_TYPE:
            uri = entity.jsonld.get("workspace_uri")
            if uri:
                validate_workspace_uri(str(uri))
        self.entities[entity.id] = entity
        self.triples = [t for t in self.triples if t[1] != entity.at_id]
        for s, p, o in jsonld_triples(entity):
            self.triples.append((new_uuid(), s, p, o, ""))
        return entity

    async def get_entity(self, uid: uuid.UUID) -> Entity | None:
        return self.entities.get(uid)

    async def get_entity_by_at_id(self, at_id: str) -> Entity | None:
        for e in self.entities.values():
            if e.at_id == at_id:
                return e
        return None

    async def list_entities(self, at_type: str | None = None) -> list[Entity]:
        rows = list(self.entities.values())
        if at_type:
            rows = [e for e in rows if e.at_type == at_type]
        return rows

    async def put_triple(self, subject: str, predicate: str, obj: str, graph_id: str = "") -> uuid.UUID:
        tid = new_uuid()
        self.triples.append((tid, subject, predicate, obj, graph_id))
        return tid

    async def graph(self, at_id: str, depth: int = 1) -> list[tuple[str, str, str]]:
        frontier = {at_id}
        seen: set[str] = set()
        out: list[tuple[str, str, str]] = []
        for _ in range(max(1, depth)):
            nxt: set[str] = set()
            for tid, s, p, o, _g in self.triples:
                if s in frontier or o in frontier:
                    out.append((s, p, o))
                    nxt.add(s)
                    nxt.add(o)
            seen |= frontier
            frontier = nxt - seen
        return out

    async def put_chunk(self, chunk: Chunk) -> Chunk:
        self.chunks[chunk.id] = chunk
        return chunk

    async def get_chunk(self, uid: uuid.UUID) -> Chunk | None:
        if uid in self.forgotten:
            return None
        return self.chunks.get(uid)

    async def search_chunks(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        q = embed_text(query)
        scored: list[tuple[Chunk, float]] = []
        for c in self.chunks.values():
            if c.id in self.forgotten:
                continue
            scored.append((c, _cosine(q, c.embedding)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def pinned_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.pinned and c.id not in self.forgotten]

    async def pinned_entities(self) -> list[Entity]:
        return [e for e in self.entities.values() if e.pinned]

    async def set_pinned(self, kind: str, uid: uuid.UUID, pinned: bool) -> None:
        if kind == "chunk":
            c = self.chunks.get(uid)
            if not c:
                raise KeyError(uid)
            if pinned:
                await ensure_pin_budget(self, extra_text=c.text)
            c.pinned = pinned
            return
        e = self.entities.get(uid)
        if not e:
            raise KeyError(uid)
        blob = json.dumps(e.jsonld)
        if pinned:
            await ensure_pin_budget(self, extra_text=blob)
        e.pinned = pinned

    async def forget_chunk(self, uid: uuid.UUID) -> None:
        self.forgotten.add(uid)

    async def append_event(self, session_id: uuid.UUID, kind: str, payload: dict[str, Any]) -> Event:
        seq = len(self.events.get(session_id, [])) + 1
        ev = Event(id=new_uuid(), session_id=session_id, seq=seq, kind=kind, payload=payload)
        self.events.setdefault(session_id, []).append(ev)
        return ev

    async def list_events(self, session_id: uuid.UUID) -> list[Event]:
        return list(self.events.get(session_id, []))

    async def replace_events(self, session_id: uuid.UUID, events: list[Event]) -> None:
        self.events[session_id] = list(events)

    async def put_job(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    async def due_jobs(self, now: datetime) -> list[Job]:
        return [j for j in self.jobs.values() if j.due_at <= now]

    async def reschedule_job(self, job: Job) -> None:
        self.jobs[job.id] = job

    async def meta_get(self, key: str) -> str | None:
        return self.meta.get(key)

    async def meta_set(self, key: str, value: str) -> None:
        self.meta[key] = value


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a[:n]) ** 0.5
    nb = sum(x * x for x in b[:n]) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def ensure_pin_budget(store: Store, extra_text: str = "") -> None:
    parts = [c.text for c in await store.pinned_chunks()]
    parts.extend(json.dumps(e.jsonld) for e in await store.pinned_entities())
    parts.append(extra_text)
    used = estimate_tokens("\n".join(parts))
    if used > settings.orbweaver_pinned_token_cap:
        raise PinBudgetError(
            f"pin would exceed cap ({used} > {settings.orbweaver_pinned_token_cap} tokens)"
        )


class PinBudgetError(ValueError):
    pass


class PinRejected(PinBudgetError, WorkspaceURIError):
    pass


_STORE: Store | None = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        if settings.orbweaver_store == "postgres":
            from orbweaver.pgstore import PostgresStore

            _STORE = PostgresStore()
        else:
            _STORE = MemoryStore()
    return _STORE


def reset_store_for_tests() -> MemoryStore:
    global _STORE
    _STORE = MemoryStore()
    return _STORE
