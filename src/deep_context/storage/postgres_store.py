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
    Chunk,
    ChunkLevel,
    Document,
    DocumentElementType,
    DocumentNode,
    EquationDataModel,
    ExistingMemory,
    FigureDataModel,
    MultimodalAsset,
    Provenance,
    RetrievalFilters,
    RetrievalMode,
    SourceCodeDataModel,
    TableDataModel,
)
from deep_context.storage.base import StorageInterface


def _clean_pg_str(val: str | None) -> str | None:
    """Strip null bytes (0x00) which are forbidden in PostgreSQL UTF-8 text."""
    if val is None:
        return None
    return val.replace("\x00", "")


def _clean_pg_json(val: Any) -> Any:
    """Recursively strip null bytes (0x00 and \\u0000) from JSON structures for PostgreSQL JSONB."""
    if isinstance(val, str):
        return val.replace("\x00", "").replace("\\u0000", "")
    elif isinstance(val, dict):
        return {(_clean_pg_str(k) or ""): _clean_pg_json(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_clean_pg_json(x) for x in val]
    return val


class PostgresStore(StorageInterface):
    """PostgreSQL 15+ with pgvector storage implementing docs/DATA_MODEL.sql."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.postgres_dsn
        self._pool: asyncpg.Pool | None = None
        self._has_pg_trgm: bool = False
        self._has_pg_search: bool = False

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
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.debug("vector extension notice: %s", e)

            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            except Exception as e:
                logger.debug("pgcrypto extension notice: %s", e)

            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                self._has_pg_trgm = True
            except Exception as e:
                logger.debug("pg_trgm extension notice: %s", e)
                self._has_pg_trgm = False

            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
                self._has_pg_search = True
            except Exception as e:
                logger.debug("pg_search extension notice: %s", e)
                self._has_pg_search = False
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
                    parent_chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
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

                ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';
                ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_parent_chunk_id_fkey;
                CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks (parent_chunk_id);
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_path TEXT;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number INTEGER;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_text TEXT;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_tokens INTEGER;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_model TEXT DEFAULT 'qwen3-0.6b';
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary_tsv TSVECTOR;
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();

                CREATE TABLE IF NOT EXISTS document_tree_nodes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    parent_node_id UUID REFERENCES document_tree_nodes(id) ON DELETE CASCADE,
                    node_type TEXT NOT NULL,
                    reading_order INTEGER NOT NULL DEFAULT 0,
                    title TEXT,
                    text TEXT,
                    raw_text TEXT,
                    section_path TEXT,
                    page_number INTEGER,
                    page_end INTEGER,
                    bbox JSONB,
                    table_data JSONB,
                    figure_data JSONB,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS source_uri TEXT;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS char_span JSONB;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS raw_ref TEXT;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS parser TEXT;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS parser_version TEXT;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS extraction_method TEXT;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS confidence REAL;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS line_range JSONB;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS equation_data JSONB;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS code_data JSONB;
                ALTER TABLE document_tree_nodes ADD COLUMN IF NOT EXISTS asset_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_tree_nodes_document ON document_tree_nodes (document_id);
                CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent ON document_tree_nodes (parent_node_id);

                CREATE TABLE IF NOT EXISTS document_assets (
                    id TEXT PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    node_id UUID REFERENCES document_tree_nodes(id) ON DELETE SET NULL,
                    asset_type TEXT NOT NULL DEFAULT 'image',
                    mime_type TEXT NOT NULL DEFAULT 'image/png',
                    width INTEGER,
                    height INTEGER,
                    byte_size INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    caption TEXT,
                    ocr_text TEXT,
                    description TEXT,
                    embedding VECTOR(768),
                    page_number INTEGER,
                    bbox JSONB,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    permission_scope TEXT[] NOT NULL DEFAULT ARRAY['default'],
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_assets_document ON document_assets (document_id);
                CREATE INDEX IF NOT EXISTS idx_assets_node ON document_assets (node_id);
                CREATE INDEX IF NOT EXISTS idx_assets_tenant ON document_assets (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON document_assets (sha256);
                CREATE INDEX IF NOT EXISTS idx_assets_embedding_hnsw ON document_assets USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);

                CREATE OR REPLACE FUNCTION update_chunks_tsv() RETURNS trigger AS $$
                BEGIN
                  NEW.search_tsv :=
                    setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.summary_text, '')), 'C');
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                DROP TRIGGER IF EXISTS trigger_update_chunks_tsv ON chunks;
                CREATE TRIGGER trigger_update_chunks_tsv
                BEFORE INSERT OR UPDATE ON chunks
                FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();

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
                    embedding VECTOR(768),
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
                    embedding VECTOR(768),
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
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id, id);
                CREATE INDEX IF NOT EXISTS idx_chunks_parent_null ON chunks (id, document_id) WHERE parent_chunk_id IS NULL;
                CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
                CREATE INDEX IF NOT EXISTS idx_chunks_search_tsv ON chunks USING GIN (search_tsv);
                CREATE INDEX IF NOT EXISTS idx_chunks_summary_tsv ON chunks USING GIN (summary_tsv);
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
                CREATE INDEX IF NOT EXISTS idx_memory_policy_tenant ON memory_policy (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_memory_preference_user ON memory_preference (user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_fact_scope ON memory_fact (tenant_id, user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_fact_tsv ON memory_fact USING GIN (tsv);
                CREATE INDEX IF NOT EXISTS idx_memory_fact_embedding ON memory_fact USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
                CREATE INDEX IF NOT EXISTS idx_memory_episode_user ON memory_episode (user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_episode_embedding ON memory_episode USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
                CREATE INDEX IF NOT EXISTS idx_events_trace_session ON events_trace (session_id);
                """)

            if self._has_pg_trgm:
                try:
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON chunks USING GIN (content gin_trgm_ops);"
                    )
                    logger.info("Created or verified GIN trigram index idx_chunks_trgm.")
                except Exception as e_trgm_idx:
                    logger.debug("pg_trgm index creation notice: %s", e_trgm_idx)

            if self._has_pg_search:
                try:
                    await conn.execute("""
                        CALL paradedb.create_bm25(
                            index_name => 'idx_chunks_bm25',
                            table_name => 'chunks',
                            key_field => 'id',
                            text_fields => '{content: {}, summary_text: {}}'
                        );
                    """)
                    logger.info("Created or verified ParadeDB BM25 index idx_chunks_bm25.")
                except Exception as e_bm25_idx:
                    logger.debug("ParadeDB BM25 index creation notice: %s", e_bm25_idx)

        logger.info("Initialized Postgres database with pgvector, FTS, and HNSW indexes.")

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
        clean_meta = _clean_pg_json(document.metadata)
        clean_title = _clean_pg_str(document.title) or ""
        clean_uri = _clean_pg_str(document.source_uri)
        clean_doctype = _clean_pg_str(document.doc_type) or "markdown"
        clean_tenant = _clean_pg_str(document.tenant_id) or "default"
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
                clean_tenant,
                clean_title,
                clean_uri,
                clean_doctype,
                document.permission_scope,
                document.retrieval_mode.value,
                json.dumps(clean_meta),
                document.ingested_at,
                document.updated_at,
            )
            return str(row["id"])

    async def insert_document_and_chunks(self, document: Document, chunks: list[Chunk]) -> str:
        """Atomically inserts document and all its chunks in a single transaction."""
        pool = self._get_pool()
        clean_meta = _clean_pg_json(document.metadata)
        clean_title = _clean_pg_str(document.title) or ""
        clean_uri = _clean_pg_str(document.source_uri)
        clean_doctype = _clean_pg_str(document.doc_type) or "markdown"
        clean_tenant = _clean_pg_str(document.tenant_id) or "default"
        async with pool.acquire() as conn:
            async with conn.transaction():
                mode_val = (
                    document.retrieval_mode.value
                    if hasattr(document.retrieval_mode, "value")
                    else str(document.retrieval_mode)
                )
                await conn.execute(
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
                        updated_at = EXCLUDED.updated_at;
                    """,
                    document.id,
                    clean_tenant,
                    clean_title,
                    clean_uri,
                    clean_doctype,
                    document.permission_scope,
                    mode_val,
                    json.dumps(clean_meta),
                    document.ingested_at,
                    document.updated_at,
                )

                if chunks:
                    insert_sql = """
                    INSERT INTO chunks (
                        id, document_id, parent_chunk_id, level, content,
                        token_count, section_path, page_number, embedding,
                        summary_text, summary_tokens, summary_model, generated_at,
                        metadata, created_at
                    ) VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        token_count = EXCLUDED.token_count,
                        section_path = EXCLUDED.section_path,
                        page_number = EXCLUDED.page_number,
                        embedding = EXCLUDED.embedding,
                        summary_text = EXCLUDED.summary_text,
                        summary_tokens = EXCLUDED.summary_tokens,
                        summary_model = EXCLUDED.summary_model,
                        generated_at = EXCLUDED.generated_at,
                        metadata = EXCLUDED.metadata;
                    """

                    def _to_rec(c: Chunk) -> tuple:
                        level_str = c.level.value if hasattr(c.level, "value") else str(c.level)
                        emb_val = (
                            np.array(c.embedding, dtype=np.float32)
                            if c.embedding is not None
                            else None
                        )
                        meta_val = json.dumps(_clean_pg_json(c.metadata)) if c.metadata else "{}"
                        return (
                            c.id,
                            c.document_id,
                            c.parent_chunk_id,
                            level_str,
                            _clean_pg_str(c.content) or "",
                            c.token_count,
                            _clean_pg_str(c.section_path),
                            c.page_number,
                            emb_val,
                            _clean_pg_str(c.summary_text),
                            c.summary_tokens,
                            _clean_pg_str(c.summary_model),
                            c.generated_at,
                            meta_val,
                            c.created_at,
                        )

                    records = [_to_rec(c) for c in chunks]
                    await conn.executemany(insert_sql, records)

                return str(document.id)

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
                    COUNT(c.id) FILTER (WHERE c.level = 'parent') as parent_chunks_count,
                    COUNT(c.id) FILTER (WHERE c.summary_text IS NOT NULL AND c.summary_text != '') as summaries_count,
                    COUNT(c.id) FILTER (WHERE c.embedding IS NOT NULL AND c.level = 'child') as embeddings_count
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
                    "summaries_count": int(r["summaries_count"]),
                    "embeddings_count": int(r["embeddings_count"]),
                    "created_at": (
                        r["ingested_at"].isoformat()
                        if hasattr(r["ingested_at"], "isoformat")
                        else str(r["ingested_at"])
                    ),
                }
                for r in rows
            ]

    async def get_unembedded_chunks(self, document_id: str) -> list[Chunk]:
        """Fetch all child chunks for a document that do not yet have embeddings."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM chunks
                WHERE document_id = $1::uuid AND level = 'child' AND embedding IS NULL
                ORDER BY page_number ASC NULLS LAST, created_at ASC;
                """,
                document_id,
            )
            return [
                Chunk(
                    id=str(r["id"]),
                    document_id=str(r["document_id"]),
                    parent_chunk_id=str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None,
                    level=ChunkLevel(r["level"]),
                    content=r["content"],
                    token_count=r["token_count"],
                    section_path=r["section_path"],
                    page_number=r["page_number"],
                    summary_text=r["summary_text"],
                    summary_tokens=r["summary_tokens"],
                    summary_model=r["summary_model"],
                    generated_at=r["generated_at"],
                )
                for r in rows
            ]

    async def get_document_chunks_detail(self, document_id: str) -> list[dict[str, Any]]:
        """Fetch all parent and child chunks with summaries and metadata for inspection."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id, c.parent_chunk_id, c.level, c.content, c.token_count,
                    c.section_path, c.page_number, c.summary_text, c.summary_tokens,
                    c.summary_model, c.generated_at,
                    (c.embedding IS NOT NULL) as has_embedding
                FROM chunks c
                WHERE c.document_id = $1::uuid
                ORDER BY c.level DESC, c.page_number ASC NULLS LAST, c.created_at ASC;
                """,
                document_id,
            )
            return [
                {
                    "id": str(r["id"]),
                    "parent_chunk_id": str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None,
                    "level": r["level"],
                    "content": r["content"],
                    "token_count": r["token_count"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
                    "summary_text": r["summary_text"],
                    "summary_tokens": r["summary_tokens"],
                    "summary_model": r["summary_model"],
                    "generated_at": (
                        r["generated_at"].isoformat()
                        if r["generated_at"] and hasattr(r["generated_at"], "isoformat")
                        else str(r["generated_at"])
                        if r["generated_at"]
                        else None
                    ),
                    "has_embedding": r["has_embedding"],
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
            async with conn.transaction():
                # Separate parent chunks (parent_chunk_id is None) from child chunks
                parents = [
                    c
                    for c in chunks
                    if c.parent_chunk_id is None
                    or (hasattr(c.level, "value") and c.level.value == "parent")
                    or str(c.level) == "parent"
                ]
                children = [c for c in chunks if c not in parents]

                insert_sql = """
                INSERT INTO chunks (
                    id, document_id, parent_chunk_id, level, content,
                    token_count, section_path, page_number, embedding,
                    summary_text, summary_tokens, summary_model, generated_at,
                    metadata, created_at
                ) VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    section_path = EXCLUDED.section_path,
                    page_number = EXCLUDED.page_number,
                    embedding = EXCLUDED.embedding,
                    summary_text = EXCLUDED.summary_text,
                    summary_tokens = EXCLUDED.summary_tokens,
                    summary_model = EXCLUDED.summary_model,
                    generated_at = EXCLUDED.generated_at,
                    metadata = EXCLUDED.metadata;
                """

                def _to_record(c: Chunk) -> tuple:
                    level_str = c.level.value if hasattr(c.level, "value") else str(c.level)
                    emb_val = (
                        np.array(c.embedding, dtype=np.float32) if c.embedding is not None else None
                    )
                    meta_val = json.dumps(_clean_pg_json(c.metadata)) if c.metadata else "{}"
                    return (
                        c.id,
                        c.document_id,
                        c.parent_chunk_id,
                        level_str,
                        _clean_pg_str(c.content) or "",
                        c.token_count,
                        _clean_pg_str(c.section_path),
                        c.page_number,
                        emb_val,
                        _clean_pg_str(c.summary_text),
                        c.summary_tokens,
                        _clean_pg_str(c.summary_model),
                        c.generated_at,
                        meta_val,
                        c.created_at,
                    )

                # 1. Insert parents first so foreign key constraints on parent_chunk_id are always satisfied
                if parents:
                    parent_records = [_to_record(c) for c in parents]
                    await conn.executemany(insert_sql, parent_records)

                # 2. Insert children second
                if children:
                    child_records = [_to_record(c) for c in children]
                    await conn.executemany(insert_sql, child_records)

                return [c.id for c in chunks]

    async def update_chunk_summaries_batch(
        self, updates: list[tuple[str, str, int, str, Any]]
    ) -> None:
        """Incrementally update summaries for a batch of chunks (chunk_id, summary_text, tokens, model, gen_time)."""
        if not updates:
            return
        pool = self._get_pool()
        clean_updates = [
            (
                chunk_id,
                _clean_pg_str(sum_text) or "",
                tokens,
                _clean_pg_str(model) or "",
                gen_time,
            )
            for chunk_id, sum_text, tokens, model, gen_time in updates
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                UPDATE chunks SET
                    summary_text = $2,
                    summary_tokens = $3,
                    summary_model = $4,
                    generated_at = $5
                WHERE id = $1::uuid;
                """,
                clean_updates,
            )

    async def update_chunk_embeddings_batch(self, updates: list[tuple[str, list[float]]]) -> None:
        """Incrementally update embeddings for a batch of chunks (chunk_id, embedding_vector)."""
        if not updates:
            return
        pool = self._get_pool()
        records = [
            (chunk_id, np.array(emb, dtype=np.float32) if emb is not None else None)
            for chunk_id, emb in updates
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                UPDATE chunks SET
                    embedding = $2
                WHERE id = $1::uuid;
                """,
                records,
            )

    def _to_float_list(self, val: Any) -> list[float] | None:
        if val is None:
            return None
        if hasattr(val, "to_numpy"):
            return val.to_numpy().tolist()
        if isinstance(val, np.ndarray):
            return val.tolist()
        try:
            return list(val)
        except Exception:
            return [float(x) for x in str(val).strip("[]").split(",") if x.strip()]

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
                embedding=self._to_float_list(row["embedding"]),
                summary_text=row.get("summary_text"),
                summary_tokens=row.get("summary_tokens"),
                summary_model=row.get("summary_model"),
                generated_at=row.get("generated_at"),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row.get("metadata"), str)
                    else row.get("metadata") or {}
                ),
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
                    embedding=self._to_float_list(r["embedding"]),
                    summary_text=r.get("summary_text"),
                    summary_tokens=r.get("summary_tokens"),
                    summary_model=r.get("summary_model"),
                    generated_at=r.get("generated_at"),
                    metadata=(
                        json.loads(r["metadata"])
                        if isinstance(r.get("metadata"), str)
                        else r.get("metadata") or {}
                    ),
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    # -----------------------------------------------------------------------
    # Document Parse Tree Nodes
    # -----------------------------------------------------------------------

    def _row_to_tree_node(self, r: Any) -> DocumentNode:
        bbox_data = json.loads(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"]
        bbox_tuple = tuple(bbox_data) if bbox_data else None

        char_span_raw = (
            json.loads(r["char_span"])
            if isinstance(r.get("char_span"), str)
            else r.get("char_span")
        )
        char_span_tuple = tuple(char_span_raw) if char_span_raw else None

        line_range_raw = (
            json.loads(r["line_range"])
            if isinstance(r.get("line_range"), str)
            else r.get("line_range")
        )
        line_range_tuple = tuple(line_range_raw) if line_range_raw else None

        prov = Provenance(
            source_uri=r.get("source_uri"),
            page_number=r["page_number"],
            page_end=r.get("page_end"),
            bbox=bbox_tuple,
            char_span=char_span_tuple,
            raw_ref=r.get("raw_ref"),
            parser=r.get("parser"),
            parser_version=r.get("parser_version"),
            extraction_method=r.get("extraction_method"),
            confidence=float(r["confidence"]) if r.get("confidence") is not None else None,
            line_range=line_range_tuple,
        )
        t_raw = (
            json.loads(r["table_data"])
            if isinstance(r.get("table_data"), str)
            else r.get("table_data")
        )
        t_data = TableDataModel.from_dict(t_raw) if t_raw else None

        f_raw = (
            json.loads(r["figure_data"])
            if isinstance(r.get("figure_data"), str)
            else r.get("figure_data")
        )
        f_data = FigureDataModel.from_dict(f_raw) if f_raw else None

        eq_raw = (
            json.loads(r["equation_data"])
            if isinstance(r.get("equation_data"), str)
            else r.get("equation_data")
        )
        eq_data = EquationDataModel.from_dict(eq_raw) if eq_raw else None

        code_raw = (
            json.loads(r["code_data"])
            if isinstance(r.get("code_data"), str)
            else r.get("code_data")
        )
        code_data = SourceCodeDataModel.from_dict(code_raw) if code_raw else None

        meta = (
            json.loads(r["metadata"])
            if isinstance(r.get("metadata"), str)
            else (r.get("metadata") or {})
        )

        return DocumentNode(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            parent_id=str(r["parent_node_id"]) if r.get("parent_node_id") else None,
            node_type=DocumentElementType(r["node_type"]),
            reading_order=r["reading_order"],
            text=r["text"] or "",
            raw_text=r.get("raw_text"),
            section_path=r.get("section_path"),
            provenance=prov,
            table_data=t_data,
            figure_data=f_data,
            equation_data=eq_data,
            code_data=code_data,
            asset_id=r.get("asset_id"),
            metadata=meta,
        )

    async def insert_tree_nodes(self, nodes: list[DocumentNode]) -> list[str]:
        if not nodes:
            return []
        pool = self._get_pool()
        insert_sql = """
        INSERT INTO document_tree_nodes (
            id, document_id, parent_node_id, node_type, reading_order,
            title, text, raw_text, section_path, page_number, page_end,
            bbox, table_data, figure_data, metadata,
            source_uri, char_span, raw_ref, parser, parser_version,
            extraction_method, confidence, line_range, equation_data, code_data, asset_id, created_at
        ) VALUES (
            $1::uuid, $2::uuid, $3::uuid, $4, $5,
            $6, $7, $8, $9, $10, $11,
            $12::jsonb, $13::jsonb, $14::jsonb, $15::jsonb,
            $16, $17::jsonb, $18, $19, $20,
            $21, $22, $23::jsonb, $24::jsonb, $25::jsonb, $26, $27
        )
        ON CONFLICT (id) DO UPDATE SET
            parent_node_id = EXCLUDED.parent_node_id,
            node_type = EXCLUDED.node_type,
            reading_order = EXCLUDED.reading_order,
            title = EXCLUDED.title,
            text = EXCLUDED.text,
            raw_text = EXCLUDED.raw_text,
            section_path = EXCLUDED.section_path,
            page_number = EXCLUDED.page_number,
            page_end = EXCLUDED.page_end,
            bbox = EXCLUDED.bbox,
            table_data = EXCLUDED.table_data,
            figure_data = EXCLUDED.figure_data,
            metadata = EXCLUDED.metadata,
            source_uri = EXCLUDED.source_uri,
            char_span = EXCLUDED.char_span,
            raw_ref = EXCLUDED.raw_ref,
            parser = EXCLUDED.parser,
            parser_version = EXCLUDED.parser_version,
            extraction_method = EXCLUDED.extraction_method,
            confidence = EXCLUDED.confidence,
            line_range = EXCLUDED.line_range,
            equation_data = EXCLUDED.equation_data,
            code_data = EXCLUDED.code_data,
            asset_id = EXCLUDED.asset_id;
        """

        now = datetime.now()

        def _to_node_rec(n: DocumentNode) -> tuple:
            bbox_json = json.dumps(n.provenance.bbox) if n.provenance.bbox else None
            char_span_json = json.dumps(n.provenance.char_span) if n.provenance.char_span else None
            line_range_json = (
                json.dumps(n.provenance.line_range) if n.provenance.line_range else None
            )
            t_data_json = (
                json.dumps(_clean_pg_json(n.table_data.to_dict())) if n.table_data else None
            )
            f_data_json = (
                json.dumps(_clean_pg_json(n.figure_data.to_dict())) if n.figure_data else None
            )
            eq_data_json = (
                json.dumps(_clean_pg_json(n.equation_data.to_dict())) if n.equation_data else None
            )
            code_data_json = (
                json.dumps(_clean_pg_json(n.code_data.to_dict())) if n.code_data else None
            )
            meta_json = json.dumps(_clean_pg_json(n.metadata)) if n.metadata else "{}"
            node_type_str = n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type)
            title = n.text[:100] if n.text else ""
            return (
                n.id,
                n.document_id,
                n.parent_id,
                node_type_str,
                n.reading_order,
                _clean_pg_str(title),
                _clean_pg_str(n.text) or "",
                _clean_pg_str(n.raw_text),
                _clean_pg_str(n.section_path),
                n.provenance.page_number,
                n.provenance.page_end,
                bbox_json,
                t_data_json,
                f_data_json,
                meta_json,
                _clean_pg_str(n.provenance.source_uri),
                char_span_json,
                _clean_pg_str(n.provenance.raw_ref),
                _clean_pg_str(n.provenance.parser),
                _clean_pg_str(n.provenance.parser_version),
                _clean_pg_str(n.provenance.extraction_method),
                n.provenance.confidence,
                line_range_json,
                eq_data_json,
                code_data_json,
                n.asset_id,
                now,
            )

        async with pool.acquire() as conn:
            records = [_to_node_rec(n) for n in nodes]
            await conn.executemany(insert_sql, records)
        return [n.id for n in nodes]

    async def get_tree_nodes(self, document_id: str) -> list[DocumentNode]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_tree_nodes WHERE document_id = $1::uuid ORDER BY reading_order ASC",
                document_id,
            )
            nodes = [self._row_to_tree_node(r) for r in rows]
            node_map = {n.id: n for n in nodes}
            for n in nodes:
                if n.parent_id and n.parent_id in node_map:
                    node_map[n.parent_id].children_ids.append(n.id)
            return nodes

    async def get_tree_node(self, node_id: str) -> DocumentNode | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM document_tree_nodes WHERE id = $1::uuid",
                node_id,
            )
            return self._row_to_tree_node(row) if row else None

    async def get_tree_nodes_by_ids(self, node_ids: list[str]) -> list[DocumentNode]:
        if not node_ids:
            return []
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_tree_nodes WHERE id::text = ANY($1::text[])",
                node_ids,
            )
            return [self._row_to_tree_node(r) for r in rows]

    # -----------------------------------------------------------------------
    # Multimodal Assets
    # -----------------------------------------------------------------------

    def _row_to_asset(self, r: Any) -> MultimodalAsset:
        bbox_data = json.loads(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"]
        bbox_tuple = tuple(bbox_data) if bbox_data else None
        emb_data = None
        if r["embedding"] is not None:
            if isinstance(r["embedding"], str):
                emb_data = json.loads(r["embedding"])
            else:
                emb_data = list(r["embedding"])
        scopes = (
            json.loads(r["permission_scope"])
            if isinstance(r["permission_scope"], str)
            else list(r["permission_scope"] or ["default"])
        )
        meta = (
            json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {})
        )
        return MultimodalAsset(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            node_id=str(r["node_id"]) if r.get("node_id") else None,
            asset_type=r["asset_type"],
            mime_type=r["mime_type"],
            width=r["width"],
            height=r["height"],
            byte_size=r["byte_size"],
            sha256=r["sha256"],
            storage_path=r["storage_path"],
            caption=r["caption"],
            ocr_text=r["ocr_text"],
            description=r["description"],
            embedding=emb_data,
            page_number=r["page_number"],
            bbox=bbox_tuple,
            tenant_id=r["tenant_id"],
            permission_scope=scopes,
            metadata=meta,
            created_at=r["created_at"],
        )

    async def insert_assets(self, assets: list[MultimodalAsset]) -> list[str]:
        if not assets:
            return []
        pool = self._get_pool()
        insert_sql = """
        INSERT INTO document_assets (
            id, document_id, node_id, asset_type, mime_type,
            width, height, byte_size, sha256, storage_path,
            caption, ocr_text, description, embedding,
            page_number, bbox, tenant_id, permission_scope,
            metadata, created_at
        ) VALUES (
            $1, $2::uuid, $3::uuid, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13, $14,
            $15, $16::jsonb, $17, $18,
            $19::jsonb, $20
        )
        ON CONFLICT (id) DO UPDATE SET
            node_id = EXCLUDED.node_id,
            asset_type = EXCLUDED.asset_type,
            mime_type = EXCLUDED.mime_type,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            byte_size = EXCLUDED.byte_size,
            sha256 = EXCLUDED.sha256,
            storage_path = EXCLUDED.storage_path,
            caption = EXCLUDED.caption,
            ocr_text = EXCLUDED.ocr_text,
            description = EXCLUDED.description,
            embedding = EXCLUDED.embedding,
            page_number = EXCLUDED.page_number,
            bbox = EXCLUDED.bbox,
            tenant_id = EXCLUDED.tenant_id,
            permission_scope = EXCLUDED.permission_scope,
            metadata = EXCLUDED.metadata;
        """

        now = datetime.now()

        def _to_asset_rec(a: MultimodalAsset) -> tuple:
            bbox_json = json.dumps(a.bbox) if a.bbox else None
            meta_json = json.dumps(_clean_pg_json(a.metadata)) if a.metadata else "{}"
            vec = np.array(a.embedding, dtype=np.float32) if a.embedding else None
            return (
                a.id,
                a.document_id,
                a.node_id,
                a.asset_type,
                a.mime_type,
                a.width,
                a.height,
                a.byte_size,
                a.sha256,
                a.storage_path,
                _clean_pg_str(a.caption),
                _clean_pg_str(a.ocr_text),
                _clean_pg_str(a.description),
                vec,
                a.page_number,
                bbox_json,
                a.tenant_id,
                a.permission_scope,
                meta_json,
                a.created_at or now,
            )

        async with pool.acquire() as conn:
            records = [_to_asset_rec(a) for a in assets]
            await conn.executemany(insert_sql, records)
        return [a.id for a in assets]

    async def insert_asset(self, asset: MultimodalAsset) -> str:
        await self.insert_assets([asset])
        return asset.id

    async def get_asset(self, asset_id: str) -> MultimodalAsset | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM document_assets WHERE id = $1", asset_id)
            return self._row_to_asset(row) if row else None

    async def get_assets_by_ids(self, asset_ids: list[str]) -> list[MultimodalAsset]:
        if not asset_ids:
            return []
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_assets WHERE id = ANY($1::text[])",
                asset_ids,
            )
            return [self._row_to_asset(r) for r in rows]

    async def get_assets_for_document(self, document_id: str) -> list[MultimodalAsset]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_assets WHERE document_id = $1::uuid ORDER BY page_number ASC, created_at ASC",
                document_id,
            )
            return [self._row_to_asset(r) for r in rows]

    async def search_assets_vector(
        self,
        query_vector: list[float],
        filters: RetrievalFilters,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        vec = np.array(query_vector, dtype=np.float32)
        async with pool.acquire() as conn:
            clauses = [
                "a.embedding IS NOT NULL",
                "a.tenant_id = $2",
                "('public' = ANY(a.permission_scope) OR a.permission_scope && $3)",
            ]
            params: list[Any] = [vec, filters.tenant_id, filters.permission_scope]
            if filters.document_ids:
                clauses.append(f"a.document_id = ANY(${len(params) + 1}::uuid[])")
                params.append(filters.document_ids)

            where_sql = " AND ".join(clauses)
            sql = f"""
                SELECT a.id, a.document_id, a.node_id, a.asset_type, a.mime_type,
                       a.width, a.height, a.byte_size, a.sha256, a.storage_path,
                       a.caption, a.ocr_text, a.description, a.page_number, a.bbox,
                       a.tenant_id, a.permission_scope, a.metadata, a.created_at,
                       d.title as document_title, d.source_uri,
                       1 - (a.embedding <=> $1) AS score
                FROM document_assets a
                JOIN documents d ON d.id = a.document_id
                WHERE {where_sql}
                ORDER BY a.embedding <=> $1 ASC LIMIT {limit};
            """
            try:
                rows = await conn.fetch(sql, *params)
            except Exception as e:
                if (
                    "different vector dimensions" in str(e).lower()
                    or "dimension mismatch" in str(e).lower()
                ):
                    logger.warning("Asset vector search skipped due to dimension mismatch: %s", e)
                    return []
                raise

            res = []
            for r in rows:
                asset = self._row_to_asset(r)
                item = asset.to_dict()
                item["score"] = float(r["score"])
                item["document_title"] = r["document_title"]
                item["source_uri"] = r["source_uri"]
                res.append(item)
            return res

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

        # Extract alphanumeric or hyphenated technical identifiers (SKUs, codes, model names)
        ident_matches = re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", clean_query)
        ident_term = ident_matches[0] if ident_matches else ""

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

            if filters.doc_types:
                clauses.append(f"d.doc_type = ANY(${len(params) + 1}::text[])")
                params.append(filters.doc_types)

            if filters.section_prefix:
                clauses.append(f"c.section_path ILIKE ${len(params) + 1} || '%'")
                params.append(filters.section_prefix)

            # 1. Try ParadeDB BM25 search if extension is enabled
            if self._has_pg_search:
                try:
                    where_sql = " AND ".join(clauses)
                    p_query = plain_term if plain_term else clean_query
                    bm25_params = list(params[3:])  # Skip plain_term, or_terms, phrase_term
                    bm25_sql = f"""
                        SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                               c.page_number, c.summary_text, d.title as document_title, d.source_uri,
                               paradedb.score(c.id) AS score
                        FROM chunks c
                        JOIN documents d ON d.id = c.document_id
                        WHERE {where_sql} AND chunks @@@ paradedb.parse(${len(bm25_params) + 1})
                        ORDER BY score DESC, c.created_at DESC LIMIT {limit};
                    """
                    bm25_rows = await conn.fetch(bm25_sql, *bm25_params, p_query)
                    if bm25_rows:
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
                            for r in bm25_rows
                        ]
                except Exception as e_pdb:
                    logger.debug("ParadeDB query notice: %s. Falling back to enhanced FTS.", e_pdb)

            # 2. Enhanced Dual-Channel FTS + Trigram + Exact Identifier Matching
            where_sql = " AND ".join(clauses)
            trgm_score_clause = "similarity(c.content, $1) * 2.0" if self._has_pg_trgm else "0.0"
            ident_match_clause = (
                f"(CASE WHEN length('{ident_term}') > 0 AND (c.content ILIKE '%{ident_term}%' OR COALESCE(c.summary_text, '') ILIKE '%{ident_term}%') THEN 8.0 ELSE 0.0 END)"
                if ident_term
                else "0.0"
            )

            sql = f"""
                SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                       c.page_number, c.summary_text, d.title as document_title, d.source_uri,
                       (COALESCE(ts_rank(COALESCE(c.search_tsv, c.tsv), plainto_tsquery('english', $1)), 0.0) +
                        COALESCE(ts_rank(COALESCE(c.search_tsv, c.tsv), to_tsquery('english', $2)), 0.0) +
                        COALESCE(ts_rank(c.summary_tsv, plainto_tsquery('english', $1)), 0.0) +
                        COALESCE(ts_rank(c.summary_tsv, to_tsquery('english', $2)), 0.0) +
                        {trgm_score_clause} +
                        {ident_match_clause} +
                        (CASE WHEN length($3) > 0 AND (c.content ILIKE '%' || $3 || '%' OR COALESCE(c.summary_text, '') ILIKE '%' || $3 || '%') THEN 10.0 ELSE 0.0 END)) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE {where_sql} AND (
                    (length($1) > 0 AND (COALESCE(c.search_tsv, c.tsv) @@ plainto_tsquery('english', $1) OR (c.summary_tsv IS NOT NULL AND c.summary_tsv @@ plainto_tsquery('english', $1)))) OR
                    (length($2) > 0 AND (COALESCE(c.search_tsv, c.tsv) @@ to_tsquery('english', $2) OR (c.summary_tsv IS NOT NULL AND c.summary_tsv @@ to_tsquery('english', $2)))) OR
                    (length($3) > 0 AND (c.content ILIKE '%' || $3 || '%' OR COALESCE(c.summary_text, '') ILIKE '%' || $3 || '%'))
                    {" OR (c.content ILIKE '%" + ident_term + "%')" if ident_term else ""}
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
        ef_search: int | None = None,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool()
        vec = np.array(query_embedding, dtype=np.float32)
        target_ef = ef_search or getattr(settings, "hnsw_ef_search", 100)
        async with pool.acquire() as conn:
            if target_ef:
                try:
                    await conn.execute(f"SET LOCAL hnsw.ef_search = {int(target_ef)};")
                except Exception:
                    pass
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

            if filters.doc_types:
                clauses.append(f"d.doc_type = ANY(${len(params) + 1}::text[])")
                params.append(filters.doc_types)

            if filters.section_prefix:
                clauses.append(f"c.section_path ILIKE ${len(params) + 1} || '%'")
                params.append(filters.section_prefix)

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

    # -----------------------------------------------------------------------
    # Convenience Aliases matching standard naming contracts
    # -----------------------------------------------------------------------
    save_document = insert_document
    save_chunks = insert_chunks
    search_vectors = search_vector
    search_fulltext = search_bm25
    get_document_chunk = get_chunk
