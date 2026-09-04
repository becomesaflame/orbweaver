CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS instance_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
  id UUID PRIMARY KEY,
  at_id TEXT UNIQUE NOT NULL,
  at_type TEXT NOT NULL,
  jsonld JSONB NOT NULL,
  pinned BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS triples (
  id UUID PRIMARY KEY,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  graph_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY,
  text TEXT NOT NULL,
  embedding vector(384),
  entity_ids UUID[] NOT NULL DEFAULT '{}',
  source TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_accessed_at TIMESTAMPTZ,
  importance REAL NOT NULL DEFAULT 1.0,
  decay_score REAL NOT NULL DEFAULT 1.0,
  pinned BOOLEAN NOT NULL DEFAULT FALSE,
  forgotten BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS events (
  id UUID PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES entities(id),
  seq BIGINT NOT NULL,
  kind TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS jobs (
  id UUID PRIMARY KEY,
  due_at TIMESTAMPTZ NOT NULL,
  recurrence TEXT,
  payload JSONB NOT NULL,
  session_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS events_session_idx ON events(session_id);
CREATE INDEX IF NOT EXISTS jobs_due_idx ON jobs(due_at);
CREATE INDEX IF NOT EXISTS triples_subject_idx ON triples(subject);
CREATE INDEX IF NOT EXISTS chunks_forgotten_idx ON chunks(forgotten) WHERE forgotten = FALSE;
