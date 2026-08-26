-- Migration: 001_initial_schema.sql
-- Initial schema setup for Deep Context Platform

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL,
    source_uri TEXT,
    doc_type TEXT NOT NULL,
    permission_scope TEXT[] NOT NULL DEFAULT ARRAY['default'],
    retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
    metadata JSONB NOT NULL DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id UUID,
    level TEXT NOT NULL CHECK (level IN ('parent', 'child')),
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    section_path TEXT,
    page_number INTEGER,
    embedding VECTOR(768),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    summary_text TEXT,
    summary_tokens INTEGER,
    summary_model TEXT DEFAULT 'qwen3-0.6b',
    generated_at TIMESTAMPTZ,
    summary_tsv TSVECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_policy (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT,
    policy_key TEXT NOT NULL,
    policy_value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, user_id, policy_key)
);

CREATE TABLE IF NOT EXISTS memory_preference (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    preference_key TEXT NOT NULL,
    preference_value JSONB NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    source TEXT NOT NULL DEFAULT 'explicit',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, preference_key)
);

CREATE TABLE IF NOT EXISTS memory_fact (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT,
    content TEXT NOT NULL,
    embedding VECTOR,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    superseded_by UUID REFERENCES memory_fact(id),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_episode (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT,
    task_type TEXT,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL,
    embedding VECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events_trace (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID,
    agent_id UUID,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    token_cost INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jobs (
    name TEXT PRIMARY KEY,
    schedule_cron TEXT NOT NULL,
    next_run_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'idle',
    max_retries INTEGER NOT NULL DEFAULT 3,
    retries INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_run_at TIMESTAMPTZ
);
