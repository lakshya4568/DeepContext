-- ============================================================================
-- Deep Context Platform — Data Model
-- Postgres 15+ with the pgvector extension.
-- Referenced from PRD.md §9 and ARCHITECTURE.md.
--
-- Design notes:
--   * Every table that can be scoped to a tenant/user carries that scope as a
--     real column, not a JSONB field, so permission filters (FR5) can be
--     enforced with a plain WHERE clause and an index, not a JSON lookup.
--   * `embedding` uses a fixed dimension (1536, matching common embedding
--     model output) — change VECTOR(1536) to match whatever embedding model
--     you actually pick, and re-check the index type below at your real
--     row count (ivfflat below ~1M rows, consider hnsw beyond that).
--   * Nothing in `memory_fact` / `memory_episode` is written directly by
--     application code — see skills/typed-memory/scripts/promotion_gate.py.
--     The schema allows it; the discipline is enforced in code, not SQL.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid()

-- ----------------------------------------------------------------------------
-- Ingestion: documents & chunks (FR2, FR5, FR6)
-- ----------------------------------------------------------------------------

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    title           TEXT NOT NULL,
    source_uri      TEXT,                       -- file path, URL, repo path, etc.
    doc_type        TEXT NOT NULL,               -- 'pdf' | 'markdown' | 'code' | 'html' | ...
    permission_scope TEXT[] NOT NULL DEFAULT ARRAY['default'], -- ACL tags checked at retrieval time
    retrieval_mode  TEXT NOT NULL DEFAULT 'hybrid', -- 'hybrid' | 'vectorless' — see FR6
    metadata        JSONB NOT NULL DEFAULT '{}',  -- author, page_count, repo, commit_sha, etc.
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_documents_tenant ON documents (tenant_id);
CREATE INDEX idx_documents_metadata_gin ON documents USING GIN (metadata);

-- Parent-child chunking (FR2): a 'parent' chunk has parent_chunk_id = NULL,
-- a 'child' chunk points at the parent it was split from. Search matches
-- children; generation is given the parent's content.
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE, -- NULL if this IS a parent
    level           TEXT NOT NULL CHECK (level IN ('parent', 'child')),
    content         TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    section_path    TEXT,                        -- e.g. "3. Retrieval > 3.2 Reranking"
    page_number     INTEGER,
    embedding       VECTOR(1536),                 -- NULL for parent-only rows if you don't embed parents
    tsv             TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_document ON chunks (document_id);
CREATE INDEX idx_chunks_parent ON chunks (parent_chunk_id);
CREATE INDEX idx_chunks_tsv ON chunks USING GIN (tsv);
-- ivfflat is fine below ~1M rows; see TECH_STACK.md §3 for the switch point.
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Optional vectorless/tree index (FR6) for PageIndex-style structured
-- navigation. One row per (document, tree node); leaves reference chunks.
CREATE TABLE document_tree_nodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_node_id  UUID REFERENCES document_tree_nodes(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,                -- section/heading title
    summary         TEXT,                         -- LLM-generated node summary used for navigation
    chunk_id        UUID REFERENCES chunks(id),    -- set on leaf nodes
    node_order      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_tree_nodes_document ON document_tree_nodes (document_id);
CREATE INDEX idx_tree_nodes_parent ON document_tree_nodes (parent_node_id);

-- ----------------------------------------------------------------------------
-- Typed memory (FR7–FR9): four distinct stores, deliberately not merged.
-- ----------------------------------------------------------------------------

-- Policy: exact-lookup only, never inferred. Written by an operator/admin
-- path, not by the promotion gate.
CREATE TABLE memory_policy (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT,                          -- NULL = tenant-wide policy
    policy_key      TEXT NOT NULL,                 -- e.g. "refund_approval_threshold"
    policy_value    JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, user_id, policy_key)
);

-- Preference: exact-lookup, user-scoped, stable. Inference requires >=2
-- corroborating observations before promotion — enforced in application code.
CREATE TABLE memory_preference (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    preference_key  TEXT NOT NULL,                 -- e.g. "response_length", "preferred_language"
    preference_value JSONB NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    source          TEXT NOT NULL DEFAULT 'explicit', -- 'explicit' | 'inferred'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, preference_key)
);

-- Semantic fact: hybrid-searchable, always carries provenance. This is the
-- table the promotion gate (FR8) writes to most often.
CREATE TABLE memory_fact (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT,                          -- NULL = tenant-scoped fact
    content         TEXT NOT NULL,                 -- the atomic claim, not the raw observation
    embedding       VECTOR(1536),
    tsv             TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    source          TEXT NOT NULL,                 -- 'user_stated' | 'tool_output' | 'inferred'
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    superseded_by   UUID REFERENCES memory_fact(id),
    expires_at      TIMESTAMPTZ,                   -- NULL = no TTL
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memory_fact_scope ON memory_fact (tenant_id, user_id);
CREATE INDEX idx_memory_fact_tsv ON memory_fact USING GIN (tsv);
CREATE INDEX idx_memory_fact_embedding ON memory_fact USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX idx_memory_fact_active ON memory_fact (expires_at) WHERE superseded_by IS NULL;

-- Episodic summary: one per completed task/session.
CREATE TABLE memory_episode (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    session_id      UUID,                          -- see sessions table below
    task_type       TEXT,                          -- freeform label for similarity matching
    summary         TEXT NOT NULL,
    outcome         TEXT,                           -- 'success' | 'partial' | 'failed'
    embedding       VECTOR(1536),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memory_episode_user ON memory_episode (user_id);
CREATE INDEX idx_memory_episode_embedding ON memory_episode USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- ----------------------------------------------------------------------------
-- Agents & sessions
-- ----------------------------------------------------------------------------

CREATE TABLE agents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'idle',  -- 'idle' | 'running' | 'blocked' | 'error'
    budgets         JSONB NOT NULL DEFAULT '{"max_turns": 50, "max_tokens": 2000000, "max_wall_clock_seconds": 3600}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    parent_session_id UUID REFERENCES sessions(id),  -- set for RLM child sessions
    user_id         TEXT NOT NULL,
    project_root    TEXT,
    kernel_ref      TEXT,                            -- opaque handle to the sandboxed kernel process
    state_snapshot  JSONB NOT NULL DEFAULT '{}',      -- compacted state, not full transcript
    status          TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'completed' | 'error' | 'deleted'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sessions_agent ON sessions (agent_id);
CREATE INDEX idx_sessions_parent ON sessions (parent_session_id);

-- RLM subagent admission handles (FR12, FR13) — matches the verified async
-- spawn/collect model, not a synchronous call.
CREATE TABLE rlm_children (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_session_id   UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    child_session_id    UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,                -- human-readable, e.g. "auth-reviewer"
    model               TEXT NOT NULL,                -- may differ from parent's model
    depth               INTEGER NOT NULL DEFAULT 1,    -- 1 = root's direct child; see FR13
    status              TEXT NOT NULL DEFAULT 'admitted', -- 'admitted' | 'running' | 'completed' | 'deleted'
    admitted_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    UNIQUE (parent_session_id, name)
);

CREATE INDEX idx_rlm_children_parent ON rlm_children (parent_session_id);

-- Messages between parent and child sessions (the only channel results
-- travel through — see ARCHITECTURE.md §5.3).
CREATE TABLE agent_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, -- sender
    receiver_role   TEXT NOT NULL CHECK (receiver_role IN ('parent', 'child')),
    receiver_name   TEXT,                             -- required when receiver_role = 'child'
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_messages_session ON agent_messages (session_id);

-- ----------------------------------------------------------------------------
-- Trace log (FR10, NFR4) — append-only, not injected directly into prompts.
-- ----------------------------------------------------------------------------

CREATE TABLE events_trace (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID REFERENCES sessions(id) ON DELETE SET NULL,
    agent_id        UUID REFERENCES agents(id) ON DELETE SET NULL,
    event_type      TEXT NOT NULL,   -- 'retrieval' | 'memory_write' | 'rlm_spawn' | 'tool_call' | 'router_decision' | ...
    payload         JSONB NOT NULL,
    token_cost      INTEGER,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_events_trace_session ON events_trace (session_id);
CREATE INDEX idx_events_trace_type_time ON events_trace (event_type, created_at);

-- ----------------------------------------------------------------------------
-- Skills registry
-- ----------------------------------------------------------------------------

CREATE TABLE skills_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,           -- matches SKILL.md frontmatter `name`
    version         TEXT NOT NULL,
    manifest        JSONB NOT NULL,                 -- parsed SKILL.md frontmatter + path
    enabled_for     TEXT[] NOT NULL DEFAULT ARRAY['default'], -- tenant/project scope
    installed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- Extension point, NOT part of v1 (see PRD.md NG5 / ARCHITECTURE.md §9).
-- Uncomment and adapt only when a real multi-hop, entity-heavy use case
-- justifies the added ingestion and maintenance cost of a graph layer.
-- ============================================================================

-- CREATE TABLE entities (
--     id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--     tenant_id       TEXT NOT NULL DEFAULT 'default',
--     entity_type     TEXT NOT NULL,        -- 'person' | 'company' | 'product' | ...
--     name            TEXT NOT NULL,
--     metadata        JSONB NOT NULL DEFAULT '{}'
-- );
--
-- CREATE TABLE relationships (
--     id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--     source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
--     target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
--     relation_type   TEXT NOT NULL,        -- 'works_at' | 'acquired' | 'uses' | ...
--     source_chunk_id UUID REFERENCES chunks(id), -- provenance: where this edge came from
--     confidence      REAL DEFAULT 1.0
-- );
