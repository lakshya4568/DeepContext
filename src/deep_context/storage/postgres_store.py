"""PostgreSQL + pgvector storage implementation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.core.types import (
    AgentMessage,
    Budgets,
    ChildStatus,
    Chunk,
    ChunkLevel,
    Document,
    DocumentTreeNode,
    ExistingMemory,
    RetrievalFilters,
    RetrievalMode,
    SessionHandle,
    SessionStatus,
)
from deep_context.storage.base import StorageInterface


class PostgresStore(StorageInterface):
    """PostgreSQL 15+ with pgvector storage implementing docs/DATA_MODEL.sql."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.postgres_dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create connection pool and verify/create schema and HNSW indexes."""

        async def init_conn(conn: asyncpg.Connection) -> None:
            await register_vector(conn)

        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            init=init_conn,
            min_size=2,
            max_size=10,
        )

        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            # Schema creation from docs/DATA_MODEL.sql
            await conn.execute("""
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
                    parent_chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
                    level TEXT NOT NULL CHECK (level IN ('parent', 'child')),
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    section_path TEXT,
                    page_number INTEGER,
                    embedding VECTOR,
                    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    summary_text TEXT,
                    summary_tokens INTEGER,
                    summary_model TEXT DEFAULT 'qwen3-0.6b',
                    generated_at TIMESTAMPTZ,
                    summary_tsv TSVECTOR,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_text TEXT;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_tokens INTEGER;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_model TEXT DEFAULT 'qwen3-0.6b';
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_tsv TSVECTOR;
                CREATE INDEX IF NOT EXISTS idx_chunks_summary_tsv ON chunks USING GIN (summary_tsv);

                CREATE OR REPLACE FUNCTION update_summary_tsv() RETURNS trigger AS $$
                BEGIN
                  NEW.summary_tsv := to_tsvector('english', COALESCE(NEW.summary_text, ''));
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                DROP TRIGGER IF EXISTS trigger_update_summary_tsv ON chunks;
                CREATE TRIGGER trigger_update_summary_tsv
                BEFORE INSERT OR UPDATE ON chunks
                FOR EACH ROW EXECUTE FUNCTION update_summary_tsv();

                CREATE TABLE IF NOT EXISTS document_tree_nodes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    parent_node_id UUID REFERENCES document_tree_nodes(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    summary TEXT,
                    chunk_id UUID REFERENCES chunks(id),
                    node_order INTEGER NOT NULL DEFAULT 0
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

                DO $$
                BEGIN
                    BEGIN
                        ALTER TABLE chunks ALTER COLUMN embedding TYPE VECTOR;
                    EXCEPTION WHEN OTHERS THEN
                        NULL;
                    END;
                    BEGIN
                        ALTER TABLE memory_fact ALTER COLUMN embedding TYPE VECTOR;
                    EXCEPTION WHEN OTHERS THEN
                        NULL;
                    END;
                    BEGIN
                        ALTER TABLE memory_episode ALTER COLUMN embedding TYPE VECTOR;
                    EXCEPTION WHEN OTHERS THEN
                        NULL;
                    END;
                END $$;

                CREATE TABLE IF NOT EXISTS agents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    budgets JSONB NOT NULL DEFAULT '{"max_turns": 50, "max_tokens": 2000000, "max_wall_clock_seconds": 3600}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    agent_id UUID,
                    parent_session_id UUID REFERENCES sessions(id),
                    user_id TEXT NOT NULL,
                    project_root TEXT,
                    kernel_ref TEXT,
                    state_snapshot JSONB NOT NULL DEFAULT '{}',
                    budgets JSONB NOT NULL DEFAULT '{}',
                    depth INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS rlm_children (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    parent_session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    child_session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    depth INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'admitted',
                    admitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    completed_at TIMESTAMPTZ,
                    UNIQUE (parent_session_id, name)
                );

                CREATE TABLE IF NOT EXISTS agent_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    target_session_id UUID NOT NULL,
                    receiver_role TEXT NOT NULL CHECK (receiver_role IN ('parent', 'child')),
                    receiver_name TEXT,
                    content TEXT NOT NULL,
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

                CREATE INDEX IF NOT EXISTS idx_jobs_due ON jobs (next_run_at);

                -- Indexes for fast query execution & pgvector cosine similarity
                CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata);
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks (parent_chunk_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_tree_nodes_document ON document_tree_nodes (document_id);
                CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent ON document_tree_nodes (parent_node_id);
                CREATE INDEX IF NOT EXISTS idx_memory_policy_tenant ON memory_policy (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_memory_preference_user ON memory_preference (user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_fact_scope ON memory_fact (tenant_id, user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_fact_tsv ON memory_fact USING GIN (tsv);
                CREATE INDEX IF NOT EXISTS idx_memory_fact_embedding ON memory_fact USING hnsw (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_memory_episode_user ON memory_episode (user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_episode_embedding ON memory_episode USING hnsw (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions (agent_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions (parent_session_id);
                CREATE INDEX IF NOT EXISTS idx_events_trace_session ON events_trace (session_id);
                """)
        logger.info("Initialized Postgres database with pgvector and HNSW indexes.")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("Postgres pool not initialized. Call initialize() first.")
        return self._pool

    async def insert_document(self, document: Document) -> str:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    id, tenant_id, title, source_uri, doc_type, permission_scope,
                    retrieval_mode, metadata, ingested_at, updated_at
                ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_uri = EXCLUDED.source_uri,
                    doc_type = EXCLUDED.doc_type,
                    permission_scope = EXCLUDED.permission_scope,
                    retrieval_mode = EXCLUDED.retrieval_mode,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                RETURNING id;
                """,
                document.id,
                document.tenant_id,
                document.title,
                document.source_uri,
                document.doc_type,
                document.permission_scope,
                document.retrieval_mode.value,
                json.dumps(document.metadata),
                document.ingested_at,
                document.updated_at,
            )
            return str(row["id"])

    async def get_document(self, document_id: str) -> Document | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM documents WHERE id = $1::uuid", document_id)
            if not row:
                return None
            return Document(
                id=str(row["id"]),
                tenant_id=row["tenant_id"],
                title=row["title"],
                source_uri=row["source_uri"],
                doc_type=row["doc_type"],
                permission_scope=list(row["permission_scope"]),
                retrieval_mode=RetrievalMode(row["retrieval_mode"]),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                ),
                ingested_at=row["ingested_at"],
                updated_at=row["updated_at"],
            )

    async def list_documents(
        self, tenant_id: str = "default", permission_scope: list[str] | None = None
    ) -> list[Document]:
        pool = self._get_pool()
        perms = permission_scope or ["default"]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM documents WHERE tenant_id = $1 AND permission_scope && $2 ORDER BY ingested_at DESC",
                tenant_id,
                perms,
            )
            return [
                Document(
                    id=str(r["id"]),
                    tenant_id=r["tenant_id"],
                    title=r["title"],
                    source_uri=r["source_uri"],
                    doc_type=r["doc_type"],
                    permission_scope=list(r["permission_scope"]),
                    retrieval_mode=RetrievalMode(r["retrieval_mode"]),
                    metadata=(
                        json.loads(r["metadata"])
                        if isinstance(r["metadata"], str)
                        else r["metadata"]
                    ),
                    ingested_at=r["ingested_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    async def list_document_summaries(self, limit: int = 50) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    d.id, d.title, d.source_uri, d.doc_type, d.retrieval_mode, d.ingested_at,
                    COUNT(c.id) FILTER (WHERE c.level = 'child') as child_chunks_count,
                    COUNT(c.id) FILTER (WHERE c.level = 'parent') as parent_chunks_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id, d.title, d.source_uri, d.doc_type, d.retrieval_mode, d.ingested_at
                ORDER BY d.ingested_at DESC
                LIMIT $1;
                """,
                limit,
            )
            return [
                {
                    "id": str(r["id"]),
                    "title": r["title"],
                    "source_uri": r["source_uri"],
                    "doc_type": r["doc_type"],
                    "retrieval_mode": r["retrieval_mode"],
                    "child_chunks_count": int(r["child_chunks_count"]),
                    "parent_chunks_count": int(r["parent_chunks_count"]),
                    "created_at": (
                        r["ingested_at"].isoformat()
                        if hasattr(r["ingested_at"], "isoformat")
                        else str(r["ingested_at"])
                    ),
                }
                for r in rows
            ]

    async def delete_document(self, document_id: str) -> bool:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            res = await conn.execute("DELETE FROM documents WHERE id = $1::uuid", document_id)
            return "DELETE 1" in res

    async def delete_all_documents(self) -> int:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM documents")
            cnt = int(row["cnt"]) if row else 0
            await conn.execute("DELETE FROM documents")
            return cnt

    async def count_chunks_for_document(self, document_id: str) -> tuple[int, int]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE level = 'child') as child_cnt,
                    COUNT(*) FILTER (WHERE level = 'parent') as parent_cnt
                FROM chunks WHERE document_id = $1::uuid;
                """,
                document_id,
            )
            if not row:
                return 0, 0
            return int(row["child_cnt"]), int(row["parent_cnt"])

    async def get_document_chunks(
        self,
        document_id: str | None = None,
        level: str = "parent",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            if document_id:
                rows = await conn.fetch(
                    """
                    SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path, c.page_number, c.summary_text, c.summary_tokens, c.summary_model, d.title as document_title
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id = $1::uuid AND c.level = $2
                    ORDER BY c.page_number ASC NULLS LAST, c.created_at ASC
                    LIMIT $3;
                    """,
                    document_id,
                    level,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path, c.page_number, c.summary_text, c.summary_tokens, c.summary_model, d.title as document_title
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.level = $1
                    ORDER BY c.page_number ASC NULLS LAST, c.created_at ASC
                    LIMIT $2;
                    """,
                    level,
                    limit,
                )
            return [
                {
                    "id": str(r["id"]),
                    "document_id": str(r["document_id"]),
                    "parent_chunk_id": str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None,
                    "title": r["document_title"],
                    "content": r["content"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
                    "summary_text": r.get("summary_text"),
                    "summary_tokens": r.get("summary_tokens"),
                    "summary_model": r.get("summary_model"),
                }
                for r in rows
            ]

    async def insert_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []
        pool = self._get_pool()
        async with pool.acquire() as conn:
            records = [
                (
                    c.id,
                    c.document_id,
                    c.parent_chunk_id,
                    c.level.value,
                    c.content,
                    c.token_count,
                    c.section_path,
                    c.page_number,
                    (np.array(c.embedding, dtype=np.float32) if c.embedding is not None else None),
                    c.summary_text,
                    c.summary_tokens,
                    c.summary_model,
                    c.generated_at,
                    c.created_at,
                )
                for c in chunks
            ]
            await conn.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, parent_chunk_id, level, content,
                    token_count, section_path, page_number, embedding,
                    summary_text, summary_tokens, summary_model, generated_at,
                    created_at
                ) VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    section_path = EXCLUDED.section_path,
                    page_number = EXCLUDED.page_number,
                    embedding = EXCLUDED.embedding,
                    summary_text = EXCLUDED.summary_text,
                    summary_tokens = EXCLUDED.summary_tokens,
                    summary_model = EXCLUDED.summary_model,
                    generated_at = EXCLUDED.generated_at;
                """,
                records,
            )
            return [c.id for c in chunks]

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM chunks WHERE id = $1::uuid", chunk_id)
            if not row:
                return None
            return Chunk(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                parent_chunk_id=(str(row["parent_chunk_id"]) if row["parent_chunk_id"] else None),
                level=ChunkLevel(row["level"]),
                content=row["content"],
                token_count=row["token_count"],
                section_path=row["section_path"],
                page_number=row["page_number"],
                embedding=(list(row["embedding"]) if row["embedding"] is not None else None),
                summary_text=row.get("summary_text"),
                summary_tokens=row.get("summary_tokens"),
                summary_model=row.get("summary_model"),
                generated_at=row.get("generated_at"),
                created_at=row["created_at"],
            )

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM chunks WHERE id = ANY($1::uuid[])", chunk_ids)
            return [
                Chunk(
                    id=str(r["id"]),
                    document_id=str(r["document_id"]),
                    parent_chunk_id=(str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None),
                    level=ChunkLevel(r["level"]),
                    content=r["content"],
                    token_count=r["token_count"],
                    section_path=r["section_path"],
                    page_number=r["page_number"],
                    embedding=(list(r["embedding"]) if r["embedding"] is not None else None),
                    summary_text=r.get("summary_text"),
                    summary_tokens=r.get("summary_tokens"),
                    summary_model=r.get("summary_model"),
                    generated_at=r.get("generated_at"),
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def search_bm25(
        self,
        query: str,
        filters: RetrievalFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        import re

        clean_query = (
            query.replace("“", '"')
            .replace("”", '"')
            .replace("‘", '"')
            .replace("’", '"')
            .replace("'", '"')
        )
        phrases = re.findall(r'"([^"]{3,})"', clean_query)
        phrase_term = phrases[0] if phrases else ""

        stopwords = {
            "who",
            "what",
            "where",
            "when",
            "why",
            "how",
            "did",
            "the",
            "and",
            "for",
            "page",
            "pages",
            "said",
            "tell",
            "about",
            "with",
            "from",
            "this",
            "that",
            "does",
        }
        words = [
            w for w in re.findall(r"\w+", clean_query.lower()) if len(w) > 2 and w not in stopwords
        ]
        or_terms = " | ".join(words[:8]) if words else (phrase_term if phrase_term else "text")
        plain_term = " ".join(words[:8]) if words else clean_query

        pool = self._get_pool()
        async with pool.acquire() as conn:
            clauses = [
                "c.level = 'child'",
                "d.tenant_id = $4",
                "d.permission_scope && $5",
            ]
            params: list[Any] = [
                plain_term,
                or_terms,
                phrase_term,
                filters.tenant_id,
                filters.permission_scope,
            ]

            if filters.document_ids:
                clauses.append(f"d.id = ANY(${len(params) + 1}::uuid[])")
                params.append(filters.document_ids)

            where_sql = " AND ".join(clauses)
            sql = f"""
                SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                       c.page_number, c.summary_text, d.title as document_title, d.source_uri,
                       (COALESCE(ts_rank(c.tsv, plainto_tsquery('english', $1)), 0.0) +
                        COALESCE(ts_rank(c.tsv, to_tsquery('english', $2)), 0.0) +
                        COALESCE(ts_rank(c.summary_tsv, plainto_tsquery('english', $1)), 0.0) +
                        COALESCE(ts_rank(c.summary_tsv, to_tsquery('english', $2)), 0.0) +
                        (CASE WHEN length($3) > 0 AND (c.content ILIKE '%' || $3 || '%' OR COALESCE(c.summary_text, '') ILIKE '%' || $3 || '%') THEN 10.0 ELSE 0.0 END)) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE {where_sql} AND (
                    (length($1) > 0 AND (c.tsv @@ plainto_tsquery('english', $1) OR (c.summary_tsv IS NOT NULL AND c.summary_tsv @@ plainto_tsquery('english', $1)))) OR
                    (length($2) > 0 AND (c.tsv @@ to_tsquery('english', $2) OR (c.summary_tsv IS NOT NULL AND c.summary_tsv @@ to_tsquery('english', $2)))) OR
                    (length($3) > 0 AND (c.content ILIKE '%' || $3 || '%' OR COALESCE(c.summary_text, '') ILIKE '%' || $3 || '%'))
                )
                ORDER BY score DESC, c.created_at DESC LIMIT {limit};
            """
            rows = await conn.fetch(sql, *params)
            return [
                {
                    "id": str(r["id"]),
                    "document_id": str(r["document_id"]),
                    "parent_chunk_id": (
                        str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None
                    ),
                    "content": r["content"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
                    "summary_text": r.get("summary_text"),
                    "document_title": r["document_title"],
                    "source_uri": r["source_uri"],
                    "score": float(r["score"]),
                }
                for r in rows
            ]

    async def search_vector(
        self,
        query_embedding: list[float],
        filters: RetrievalFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        vec = np.array(query_embedding, dtype=np.float32)
        async with pool.acquire() as conn:
            clauses = [
                "c.level = 'child'",
                "c.embedding IS NOT NULL",
                "d.tenant_id = $2",
                "d.permission_scope && $3",
            ]
            params: list[Any] = [vec, filters.tenant_id, filters.permission_scope]

            if filters.document_ids:
                clauses.append(f"d.id = ANY(${len(params) + 1}::uuid[])")
                params.append(filters.document_ids)

            where_sql = " AND ".join(clauses)
            sql = f"""
                SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                       c.page_number, c.summary_text, d.title as document_title, d.source_uri,
                       1 - (c.embedding <=> $1) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                ORDER BY c.embedding <=> $1 ASC LIMIT {limit};
            """
            try:
                rows = await conn.fetch(sql, *params)
            except Exception as e:
                if (
                    "different vector dimensions" in str(e).lower()
                    or "dimension mismatch" in str(e).lower()
                ):
                    logger.warning("Vector search skipped due to dimension mismatch: %s", e)
                    return []
                raise
            return [
                {
                    "id": str(r["id"]),
                    "document_id": str(r["document_id"]),
                    "parent_chunk_id": (
                        str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None
                    ),
                    "content": r["content"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
                    "summary_text": r.get("summary_text"),
                    "document_title": r["document_title"],
                    "source_uri": r["source_uri"],
                    "score": float(r["score"]),
                }
                for r in rows
            ]

    async def insert_tree_nodes(self, nodes: list[DocumentTreeNode]) -> list[str]:
        if not nodes:
            return []
        pool = self._get_pool()
        async with pool.acquire() as conn:
            records = [
                (
                    n.id,
                    n.document_id,
                    n.parent_node_id,
                    n.title,
                    n.summary,
                    n.chunk_id,
                    n.node_order,
                )
                for n in nodes
            ]
            await conn.executemany(
                """
                INSERT INTO document_tree_nodes (
                    id, document_id, parent_node_id, title, summary, chunk_id, node_order
                ) VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6::uuid, $7)
                ON CONFLICT (id) DO NOTHING;
                """,
                records,
            )
            return [n.id for n in nodes]

    async def get_tree_nodes_for_document(self, document_id: str) -> list[DocumentTreeNode]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_tree_nodes WHERE document_id = $1::uuid ORDER BY node_order",
                document_id,
            )
            return [
                DocumentTreeNode(
                    id=str(r["id"]),
                    document_id=str(r["document_id"]),
                    parent_node_id=(str(r["parent_node_id"]) if r["parent_node_id"] else None),
                    title=r["title"],
                    summary=r["summary"],
                    chunk_id=str(r["chunk_id"]) if r["chunk_id"] else None,
                    node_order=r["node_order"],
                )
                for r in rows
            ]

    async def get_child_tree_nodes(
        self, document_id: str, parent_node_id: str | None
    ) -> list[DocumentTreeNode]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            if parent_node_id is None:
                rows = await conn.fetch(
                    "SELECT * FROM document_tree_nodes WHERE document_id = $1::uuid AND parent_node_id IS NULL ORDER BY node_order",
                    document_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM document_tree_nodes WHERE document_id = $1::uuid AND parent_node_id = $2::uuid ORDER BY node_order",
                    document_id,
                    parent_node_id,
                )
            return [
                DocumentTreeNode(
                    id=str(r["id"]),
                    document_id=str(r["document_id"]),
                    parent_node_id=(str(r["parent_node_id"]) if r["parent_node_id"] else None),
                    title=r["title"],
                    summary=r["summary"],
                    chunk_id=str(r["chunk_id"]) if r["chunk_id"] else None,
                    node_order=r["node_order"],
                )
                for r in rows
            ]

    async def get_policy(
        self, tenant_id: str, policy_key: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            if user_id:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM memory_policy
                    WHERE tenant_id = $1 AND policy_key = $2 AND (user_id = $3 OR user_id IS NULL)
                    ORDER BY user_id DESC NULLS LAST LIMIT 1;
                    """,
                    tenant_id,
                    policy_key,
                    user_id,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT * FROM memory_policy WHERE tenant_id = $1 AND policy_key = $2 AND user_id IS NULL",
                    tenant_id,
                    policy_key,
                )
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "tenant_id": row["tenant_id"],
                "user_id": row["user_id"],
                "policy_key": row["policy_key"],
                "policy_value": (
                    json.loads(row["policy_value"])
                    if isinstance(row["policy_value"], str)
                    else row["policy_value"]
                ),
            }

    async def set_policy(
        self,
        tenant_id: str,
        policy_key: str,
        policy_value: dict[str, Any],
        user_id: str | None = None,
    ) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_policy (tenant_id, user_id, policy_key, policy_value)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (tenant_id, user_id, policy_key) DO UPDATE SET
                    policy_value = EXCLUDED.policy_value,
                    updated_at = now();
                """,
                tenant_id,
                user_id,
                policy_key,
                json.dumps(policy_value),
            )

    async def list_policies(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            if user_id:
                rows = await conn.fetch(
                    "SELECT * FROM memory_policy WHERE tenant_id = $1 AND (user_id = $2 OR user_id IS NULL)",
                    tenant_id,
                    user_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM memory_policy WHERE tenant_id = $1", tenant_id
                )
            return [
                {
                    "id": str(r["id"]),
                    "tenant_id": r["tenant_id"],
                    "user_id": r["user_id"],
                    "policy_key": r["policy_key"],
                    "policy_value": (
                        json.loads(r["policy_value"])
                        if isinstance(r["policy_value"], str)
                        else r["policy_value"]
                    ),
                }
                for r in rows
            ]

    async def get_preference(self, user_id: str, preference_key: str) -> dict[str, Any] | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory_preference WHERE user_id = $1 AND preference_key = $2",
                user_id,
                preference_key,
            )
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "user_id": row["user_id"],
                "preference_key": row["preference_key"],
                "preference_value": (
                    json.loads(row["preference_value"])
                    if isinstance(row["preference_value"], str)
                    else row["preference_value"]
                ),
                "confidence": row["confidence"],
                "source": row["source"],
            }

    async def set_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: dict[str, Any],
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_preference (user_id, preference_key, preference_value, confidence, source)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                ON CONFLICT (user_id, preference_key) DO UPDATE SET
                    preference_value = EXCLUDED.preference_value,
                    confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source,
                    updated_at = now();
                """,
                user_id,
                preference_key,
                json.dumps(preference_value),
                confidence,
                source,
            )

    async def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM memory_preference WHERE user_id = $1", user_id)
            return [
                {
                    "id": str(r["id"]),
                    "user_id": r["user_id"],
                    "preference_key": r["preference_key"],
                    "preference_value": (
                        json.loads(r["preference_value"])
                        if isinstance(r["preference_value"], str)
                        else r["preference_value"]
                    ),
                    "confidence": r["confidence"],
                    "source": r["source"],
                }
                for r in rows
            ]

    async def insert_fact(
        self,
        tenant_id: str,
        content: str,
        embedding: list[float] | None,
        source: str,
        confidence: float,
        user_id: str | None = None,
        expires_at: datetime | None = None,
        superseded_by: str | None = None,
    ) -> str:
        pool = self._get_pool()
        emb = np.array(embedding, dtype=np.float32) if embedding is not None else None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_fact (
                    tenant_id, user_id, content, embedding, source,
                    confidence, superseded_by, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::uuid, $8)
                RETURNING id;
                """,
                tenant_id,
                user_id,
                content,
                emb,
                source,
                confidence,
                superseded_by,
                expires_at,
            )
            return str(row["id"])

    async def search_facts(
        self,
        query: str,
        query_embedding: list[float] | None,
        tenant_id: str = "default",
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            if query_embedding:
                vec = np.array(query_embedding, dtype=np.float32)
                sql = """
                    SELECT id, content, source, confidence, expires_at,
                           (1 - (embedding <=> $1)) * 0.5 + confidence * 0.5 AS score
                    FROM memory_fact
                    WHERE tenant_id = $2 AND ($3::text IS NULL OR user_id = $3 OR user_id IS NULL)
                      AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > now())
                      AND embedding IS NOT NULL
                    ORDER BY score DESC LIMIT $4;
                """
                rows = await conn.fetch(sql, vec, tenant_id, user_id, limit)
            else:
                sql = """
                    SELECT id, content, source, confidence, expires_at, confidence AS score
                    FROM memory_fact
                    WHERE tenant_id = $1 AND ($2::text IS NULL OR user_id = $2 OR user_id IS NULL)
                      AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY confidence DESC LIMIT $3;
                """
                rows = await conn.fetch(sql, tenant_id, user_id, limit)
            return [
                {
                    "id": str(r["id"]),
                    "content": r["content"],
                    "source": r["source"],
                    "confidence": r["confidence"],
                    "expires_at": (r["expires_at"].isoformat() if r["expires_at"] else None),
                    "score": float(r["score"]),
                }
                for r in rows
            ]

    async def get_facts_for_scope(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[ExistingMemory]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, confidence, created_at FROM memory_fact
                WHERE tenant_id = $1 AND ($2::text IS NULL OR user_id = $2 OR user_id IS NULL)
                  AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > now());
                """,
                tenant_id,
                user_id,
            )
            return [
                ExistingMemory(
                    id=str(r["id"]),
                    content=r["content"],
                    confidence=r["confidence"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_fact SET superseded_by = $1::uuid WHERE id = $2::uuid",
                new_fact_id,
                old_fact_id,
            )

    async def insert_episode(
        self,
        user_id: str,
        summary: str,
        session_id: str | None = None,
        task_type: str | None = None,
        outcome: str = "success",
        embedding: list[float] | None = None,
    ) -> str:
        pool = self._get_pool()
        emb = np.array(embedding, dtype=np.float32) if embedding is not None else None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_episode (user_id, session_id, task_type, summary, outcome, embedding)
                VALUES ($1, $2::uuid, $3, $4, $5, $6)
                RETURNING id;
                """,
                user_id,
                session_id,
                task_type,
                summary,
                outcome,
                emb,
            )
            return str(row["id"])

    async def search_episodes(
        self, user_id: str, query_embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        vec = np.array(query_embedding, dtype=np.float32)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, summary, task_type, outcome, created_at,
                       1 - (embedding <=> $1) AS score
                FROM memory_episode
                WHERE user_id = $2 AND embedding IS NOT NULL
                ORDER BY embedding <=> $1 ASC LIMIT $3;
                """,
                vec,
                user_id,
                limit,
            )
            return [
                {
                    "id": str(r["id"]),
                    "summary": r["summary"],
                    "task_type": r["task_type"],
                    "outcome": r["outcome"],
                    "created_at": r["created_at"].isoformat(),
                    "score": float(r["score"]),
                }
                for r in rows
            ]

    async def create_session(self, session: SessionHandle, user_id: str = "default") -> None:
        pool = self._get_pool()
        budgets_json = json.dumps(
            {
                "max_turns": session.budgets.max_turns,
                "max_tokens": session.budgets.max_tokens,
                "max_wall_clock_seconds": session.budgets.max_wall_clock_seconds,
                "max_recursion_depth": session.budgets.max_recursion_depth,
            }
        )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, parent_session_id, user_id, depth, budgets, status, created_at, updated_at)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at;
                """,
                session.id,
                session.parent_session_id,
                user_id,
                session.depth,
                budgets_json,
                session.status.value,
                session.created_at,
                session.created_at,
            )

    async def get_session(self, session_id: str) -> SessionHandle | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1::uuid", session_id)
            if not row:
                return None
            b_dict = (
                json.loads(row["budgets"])
                if isinstance(row["budgets"], str)
                else (row["budgets"] or {})
            )
            budgets = Budgets(
                max_turns=b_dict.get("max_turns", 50),
                max_tokens=b_dict.get("max_tokens", 2000000),
                max_wall_clock_seconds=b_dict.get("max_wall_clock_seconds", 3600),
                max_recursion_depth=b_dict.get("max_recursion_depth", 1),
            )
            return SessionHandle(
                id=str(row["id"]),
                parent_session_id=(
                    str(row["parent_session_id"]) if row["parent_session_id"] else None
                ),
                depth=row["depth"],
                budgets=budgets,
                status=SessionStatus(row["status"]),
                created_at=row["created_at"],
            )

    async def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET status = $1, updated_at = now() WHERE id = $2::uuid",
                status.value,
                session_id,
            )

    async def insert_rlm_child(
        self,
        parent_session_id: str,
        child_session_id: str,
        name: str,
        model: str,
        depth: int,
    ) -> str:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO rlm_children (parent_session_id, child_session_id, name, model, depth)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5)
                RETURNING id;
                """,
                parent_session_id,
                child_session_id,
                name,
                model,
                depth,
            )
            return str(row["id"])

    async def update_rlm_child_status(self, child_session_id: str, status: ChildStatus) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE rlm_children SET
                    status = $1,
                    completed_at = CASE WHEN $1 IN ('completed', 'error') THEN now() ELSE completed_at END
                WHERE child_session_id = $2::uuid;
                """,
                status.value,
                child_session_id,
            )

    async def insert_agent_message(self, message: AgentMessage) -> str:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            target_id = None
            if message.receiver_role == "parent":
                r = await conn.fetchrow(
                    "SELECT parent_session_id FROM sessions WHERE id = $1::uuid",
                    message.session_id,
                )
                target_id = r["parent_session_id"] if r else None
            else:
                r = await conn.fetchrow(
                    "SELECT child_session_id FROM rlm_children WHERE parent_session_id = $1::uuid AND name = $2",
                    message.session_id,
                    message.receiver_name,
                )
                target_id = r["child_session_id"] if r else None

            if not target_id:
                raise ValueError("Target session for message could not be resolved.")

            row = await conn.fetchrow(
                """
                INSERT INTO agent_messages (session_id, target_session_id, receiver_role, receiver_name, content, created_at)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
                RETURNING id;
                """,
                message.session_id,
                target_id,
                message.receiver_role,
                message.receiver_name,
                message.content,
                message.created_at,
            )
            return str(row["id"])

    async def pop_agent_messages(self, target_session_id: str) -> list[AgentMessage]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                DELETE FROM agent_messages
                WHERE target_session_id = $1::uuid
                RETURNING session_id, receiver_role, receiver_name, content, created_at;
                """,
                target_session_id,
            )
            return [
                AgentMessage(
                    session_id=str(r["session_id"]),
                    receiver_role=r["receiver_role"],
                    receiver_name=r["receiver_name"],
                    content=r["content"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def insert_event_trace(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_id: str | None = None,
        agent_id: str | None = None,
        token_cost: int = 0,
        latency_ms: int = 0,
    ) -> int:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO events_trace (session_id, agent_id, event_type, payload, token_cost, latency_ms)
                VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5, $6)
                RETURNING id;
                """,
                session_id,
                agent_id,
                event_type,
                json.dumps(payload),
                token_cost,
                latency_ms,
            )
            return row["id"]

    async def list_event_traces(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            clauses: list[str] = []
            params: list[Any] = []
            if session_id:
                clauses.append(f"session_id = ${len(params) + 1}::uuid")
                params.append(session_id)
            if event_type:
                clauses.append(f"event_type = ${len(params) + 1}")
                params.append(event_type)

            where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"SELECT * FROM events_trace {where_clause} ORDER BY id DESC LIMIT {limit}"
            rows = await conn.fetch(sql, *params)
            return [
                {
                    "id": r["id"],
                    "session_id": str(r["session_id"]) if r["session_id"] else None,
                    "agent_id": str(r["agent_id"]) if r["agent_id"] else None,
                    "event_type": r["event_type"],
                    "payload": (
                        json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
                    ),
                    "token_cost": r["token_cost"],
                    "latency_ms": r["latency_ms"],
                    "created_at": r["created_at"].isoformat(),
                }
                for r in rows
            ]

    # -----------------------------------------------------------------------
    # Scheduler Jobs
    # -----------------------------------------------------------------------

    async def upsert_job(
        self,
        name: str,
        schedule_cron: str,
        next_run_at: datetime,
        max_retries: int = 3,
    ) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (name, schedule_cron, next_run_at, status, max_retries)
                VALUES ($1, $2, $3, 'idle', $4)
                ON CONFLICT (name) DO UPDATE SET
                    schedule_cron = EXCLUDED.schedule_cron,
                    next_run_at = EXCLUDED.next_run_at,
                    max_retries = EXCLUDED.max_retries
                """,
                name,
                schedule_cron,
                next_run_at,
                max_retries,
            )

    async def get_due_jobs(self, now: datetime) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM jobs WHERE next_run_at <= $1 AND status != 'running'",
                now,
            )
            return [
                {
                    "name": r["name"],
                    "schedule_cron": r["schedule_cron"],
                    "next_run_at": r["next_run_at"].isoformat(),
                    "status": r["status"],
                    "max_retries": r["max_retries"],
                    "retries": r["retries"],
                    "last_error": r["last_error"],
                    "last_run_at": (r["last_run_at"].isoformat() if r["last_run_at"] else None),
                }
                for r in rows
            ]

    async def mark_job_running(self, name: str) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = 'running', last_run_at = now() WHERE name = $1",
                name,
            )

    async def mark_job_done(self, name: str, next_run_at: datetime) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = 'idle', retries = 0, last_error = NULL, "
                "next_run_at = $2 WHERE name = $1",
                name,
                next_run_at,
            )

    async def mark_job_failed(self, name: str, error: str) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = 'failed', retries = retries + 1, "
                "last_error = $2 WHERE name = $1",
                name,
                error[:2000],
            )

    async def list_jobs(self) -> list[dict[str, Any]]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM jobs ORDER BY name")
            return [
                {
                    "name": r["name"],
                    "schedule_cron": r["schedule_cron"],
                    "next_run_at": r["next_run_at"].isoformat(),
                    "status": r["status"],
                    "max_retries": r["max_retries"],
                    "retries": r["retries"],
                    "last_error": r["last_error"],
                    "last_run_at": (r["last_run_at"].isoformat() if r["last_run_at"] else None),
                }
                for r in rows
            ]

    # -----------------------------------------------------------------------
    # Maintenance Operations (scheduler tasks)
    # -----------------------------------------------------------------------

    async def cleanup_orphaned_chunks(self) -> int:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            status = await conn.execute(
                "DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
            )
        # asyncpg returns a status string like 'DELETE 12'; extract the row count.
        try:
            return int(status.split()[-1])
        except (ValueError, AttributeError):
            return 0

    async def rebuild_fts_index(self) -> int:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM chunks WHERE level = 'child' AND tsv IS NOT NULL"
            )
            return row["cnt"] if row else 0

    async def backfill_missing_embeddings(self) -> int:
        from deep_context.core.llm_client import llm_client

        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, content FROM chunks WHERE level = 'child' "
                "AND embedding IS NULL LIMIT 500"
            )
            if not rows:
                return 0
            texts = [r["content"] for r in rows]
            embeddings = await llm_client.get_embeddings(texts)
            backfilled = 0
            for r, emb in zip(rows, embeddings, strict=False):
                await conn.execute(
                    "UPDATE chunks SET embedding = $2::vector WHERE id = $1::uuid",
                    str(r["id"]),
                    emb,
                )
                backfilled += 1
            return backfilled
