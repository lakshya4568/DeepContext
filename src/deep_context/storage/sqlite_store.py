"""SQLite + FTS5 + Vector Storage Implementation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import numpy as np

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


class SQLiteStore(StorageInterface):
    """Zero-configuration local storage implementing hybrid RAG with SQLite & FTS5."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.sqlite_db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable WAL mode for concurrent read/write performance
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")

        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL,
                source_uri TEXT,
                doc_type TEXT NOT NULL,
                permission_scope TEXT NOT NULL DEFAULT 'public',
                retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
                metadata TEXT NOT NULL DEFAULT '{}',
                ingested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                level TEXT NOT NULL CHECK (level IN ('parent', 'child')),
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                section_path TEXT,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                embedding TEXT,
                summary_text TEXT,
                summary_tokens INTEGER,
                summary_model TEXT,
                generated_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                document_id UNINDEXED,
                content,
                section_path,
                summary_text
            );

            CREATE TABLE IF NOT EXISTS memory_policy (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                user_id TEXT,
                policy_key TEXT NOT NULL,
                policy_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (tenant_id, user_id, policy_key)
            );

            CREATE TABLE IF NOT EXISTS memory_preference (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (user_id, preference_key)
            );

            CREATE TABLE IF NOT EXISTS memory_fact (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                user_id TEXT,
                content TEXT NOT NULL,
                embedding TEXT,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                superseded_by TEXT REFERENCES memory_fact(id),
                expires_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fact_fts USING fts5(
                id UNINDEXED,
                content
            );

            CREATE TABLE IF NOT EXISTS memory_episode (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                task_type TEXT,
                summary TEXT NOT NULL,
                outcome TEXT,
                embedding TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                token_cost INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                name TEXT PRIMARY KEY,
                schedule_cron TEXT NOT NULL,
                next_run_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'idle',
                max_retries INTEGER NOT NULL DEFAULT 3,
                retries INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                last_run_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_trace_session ON events_trace (session_id);

            CREATE TABLE IF NOT EXISTS document_tree_nodes (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_node_id TEXT REFERENCES document_tree_nodes(id) ON DELETE CASCADE,
                node_type TEXT NOT NULL,
                reading_order INTEGER NOT NULL DEFAULT 0,
                title TEXT,
                text TEXT,
                raw_text TEXT,
                section_path TEXT,
                page_number INTEGER,
                page_end INTEGER,
                bbox TEXT,
                table_data TEXT,
                figure_data TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tree_nodes_document ON document_tree_nodes (document_id);
            CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent ON document_tree_nodes (parent_node_id);

            CREATE TABLE IF NOT EXISTS document_assets (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                node_id TEXT REFERENCES document_tree_nodes(id) ON DELETE SET NULL,
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
                embedding TEXT,
                page_number INTEGER,
                bbox TEXT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                permission_scope TEXT NOT NULL DEFAULT '["default"]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_assets_document ON document_assets (document_id);
            CREATE INDEX IF NOT EXISTS idx_assets_node ON document_assets (node_id);
            CREATE INDEX IF NOT EXISTS idx_assets_tenant ON document_assets (tenant_id);
            CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON document_assets (sha256);
            """)

        # Dynamic schema auto-migration for existing SQLite databases
        async with self._conn.execute("PRAGMA table_info(chunks)") as cursor:
            chunk_cols = {row[1] for row in await cursor.fetchall()}
            if "chunk_index" not in chunk_cols:
                await self._conn.execute(
                    "ALTER TABLE chunks ADD COLUMN chunk_index INTEGER NOT NULL DEFAULT 0;"
                )
            if "section_path" not in chunk_cols:
                await self._conn.execute("ALTER TABLE chunks ADD COLUMN section_path TEXT;")
            if "page_number" not in chunk_cols:
                await self._conn.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER;")
            if "summary_text" not in chunk_cols:
                await self._conn.execute("ALTER TABLE chunks ADD COLUMN summary_text TEXT;")
            if "summary_tokens" not in chunk_cols:
                await self._conn.execute("ALTER TABLE chunks ADD COLUMN summary_tokens INTEGER;")
            if "summary_model" not in chunk_cols:
                await self._conn.execute("ALTER TABLE chunks ADD COLUMN summary_model TEXT;")
            if "generated_at" not in chunk_cols:
                await self._conn.execute("ALTER TABLE chunks ADD COLUMN generated_at TEXT;")
            if "metadata" not in chunk_cols:
                await self._conn.execute(
                    "ALTER TABLE chunks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}';"
                )

        async with self._conn.execute("PRAGMA table_info(documents)") as cursor:
            doc_cols = {row[1] for row in await cursor.fetchall()}
            if "metadata" not in doc_cols:
                await self._conn.execute(
                    "ALTER TABLE documents ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}';"
                )

        async with self._conn.execute("PRAGMA table_info(document_tree_nodes)") as cursor:
            node_cols = {row[1] for row in await cursor.fetchall()}
            for col in (
                ("source_uri", "TEXT"),
                ("char_span", "TEXT"),
                ("raw_ref", "TEXT"),
                ("parser", "TEXT"),
                ("parser_version", "TEXT"),
                ("extraction_method", "TEXT"),
                ("confidence", "REAL"),
                ("line_range", "TEXT"),
                ("equation_data", "TEXT"),
                ("code_data", "TEXT"),
                ("asset_id", "TEXT"),
            ):
                if col[0] not in node_cols:
                    await self._conn.execute(
                        f"ALTER TABLE document_tree_nodes ADD COLUMN {col[0]} {col[1]};"
                    )

        await self._conn.commit()
        logger.info("Initialized SQLite database at %s", self.db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _get_conn(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError("Database connection not initialized. Call initialize() first.")
        return self._conn

    # -----------------------------------------------------------------------
    # Documents & Chunks
    # -----------------------------------------------------------------------

    async def insert_document(self, document: Document) -> str:
        conn = self._get_conn()
        await conn.execute(
            """
            INSERT OR REPLACE INTO documents (
                id, tenant_id, title, source_uri, doc_type, permission_scope,
                retrieval_mode, metadata, ingested_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.tenant_id,
                document.title,
                document.source_uri,
                document.doc_type,
                json.dumps(document.permission_scope),
                document.retrieval_mode.value,
                json.dumps(document.metadata),
                document.ingested_at.isoformat(),
                document.updated_at.isoformat(),
            ),
        )
        await conn.commit()
        return document.id

    async def get_document(self, document_id: str) -> Document | None:
        conn = self._get_conn()
        async with conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                tenant_id=row["tenant_id"],
                title=row["title"],
                source_uri=row["source_uri"],
                doc_type=row["doc_type"],
                permission_scope=json.loads(row["permission_scope"]),
                retrieval_mode=RetrievalMode(row["retrieval_mode"]),
                metadata=json.loads(row["metadata"]),
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    async def list_documents(
        self, tenant_id: str = "default", permission_scope: list[str] | None = None
    ) -> list[Document]:
        conn = self._get_conn()
        perms = permission_scope or ["default"]
        async with conn.execute(
            "SELECT * FROM documents WHERE tenant_id = ?", (tenant_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            docs: list[Document] = []
            for r in rows:
                doc_perms = json.loads(r["permission_scope"])
                # Check permission overlap
                if any(p in perms for p in doc_perms):
                    docs.append(
                        Document(
                            id=r["id"],
                            tenant_id=r["tenant_id"],
                            title=r["title"],
                            source_uri=r["source_uri"],
                            doc_type=r["doc_type"],
                            permission_scope=doc_perms,
                            retrieval_mode=RetrievalMode(r["retrieval_mode"]),
                            metadata=json.loads(r["metadata"]),
                            ingested_at=datetime.fromisoformat(r["ingested_at"]),
                            updated_at=datetime.fromisoformat(r["updated_at"]),
                        )
                    )
            return docs

    async def list_document_summaries(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT id, title, source_uri, doc_type, retrieval_mode, ingested_at FROM documents ORDER BY ingested_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            docs = []
            for r in rows:
                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ? AND level = 'child'",
                    (r["id"],),
                ) as c_cur:
                    c_row = await c_cur.fetchone()
                    child_cnt = c_row["cnt"] if c_row else 0
                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ? AND level = 'parent'",
                    (r["id"],),
                ) as p_cur:
                    p_row = await p_cur.fetchone()
                    parent_cnt = p_row["cnt"] if p_row else 0

                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ? AND summary_text IS NOT NULL AND summary_text != ''",
                    (r["id"],),
                ) as s_cur:
                    s_row = await s_cur.fetchone()
                    summary_cnt = s_row["cnt"] if s_row else 0

                async with conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ? AND level = 'child' AND embedding IS NOT NULL",
                    (r["id"],),
                ) as e_cur:
                    e_row = await e_cur.fetchone()
                    emb_cnt = e_row["cnt"] if e_row else 0

                docs.append(
                    {
                        "id": r["id"],
                        "title": r["title"],
                        "source_uri": r["source_uri"],
                        "doc_type": r["doc_type"],
                        "retrieval_mode": r["retrieval_mode"],
                        "child_chunks_count": child_cnt,
                        "parent_chunks_count": parent_cnt,
                        "summaries_count": summary_cnt,
                        "embeddings_count": emb_cnt,
                        "created_at": r["ingested_at"],
                    }
                )
            return docs

    async def get_unembedded_chunks(self, document_id: str) -> list[Chunk]:
        """Fetch all child chunks for a document that do not yet have embeddings."""
        conn = self._get_conn()
        async with conn.execute(
            """
            SELECT * FROM chunks
            WHERE document_id = ? AND level = 'child' AND embedding IS NULL
            ORDER BY page_number ASC, created_at ASC;
            """,
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                Chunk(
                    id=r["id"],
                    document_id=r["document_id"],
                    parent_chunk_id=r["parent_chunk_id"],
                    level=ChunkLevel(r["level"]),
                    content=r["content"],
                    token_count=r["token_count"],
                    section_path=r["section_path"],
                    page_number=r["page_number"],
                    summary_text=r["summary_text"],
                    summary_tokens=r["summary_tokens"],
                    summary_model=r["summary_model"],
                    generated_at=datetime.fromisoformat(r["generated_at"])
                    if r["generated_at"]
                    else None,
                )
                for r in rows
            ]

    async def get_document_chunks_detail(self, document_id: str) -> list[dict[str, Any]]:
        """Fetch all parent and child chunks with summaries and metadata for inspection."""
        conn = self._get_conn()
        async with conn.execute(
            """
            SELECT
                id, parent_chunk_id, level, content, token_count,
                section_path, page_number, summary_text, summary_tokens,
                summary_model, generated_at,
                (embedding IS NOT NULL) as has_embedding
            FROM chunks
            WHERE document_id = ?
            ORDER BY level DESC, page_number ASC, created_at ASC
            """,
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "parent_chunk_id": r["parent_chunk_id"],
                    "level": r["level"],
                    "content": r["content"],
                    "token_count": r["token_count"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
                    "summary_text": r["summary_text"],
                    "summary_tokens": r["summary_tokens"],
                    "summary_model": r["summary_model"],
                    "generated_at": r["generated_at"],
                    "has_embedding": bool(r["has_embedding"]),
                }
                for r in rows
            ]

    async def delete_document(self, document_id: str) -> bool:
        conn = self._get_conn()
        await conn.execute("DELETE FROM document_tree_nodes WHERE document_id = ?", (document_id,))
        await conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        await conn.execute(
            "DELETE FROM chunks WHERE document_id = ? AND level = 'child'",
            (document_id,),
        )
        await conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        await conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        await conn.commit()
        return True

    async def delete_all_documents(self) -> bool:
        conn = self._get_conn()
        await conn.execute("DELETE FROM document_tree_nodes")
        await conn.execute("DELETE FROM chunks_fts")
        await conn.execute("DELETE FROM chunks WHERE level = 'child'")
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM documents")
        await conn.commit()
        return True

    async def count_chunks_for_document(self, document_id: str) -> tuple[int, int]:
        conn = self._get_conn()
        total_child = 0
        total_parent = 0
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ? AND level = 'child'",
            (document_id,),
        ) as c_cur:
            row = await c_cur.fetchone()
            if row:
                total_child = row["cnt"]
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ? AND level = 'parent'",
            (document_id,),
        ) as p_cur:
            row = await p_cur.fetchone()
            if row:
                total_parent = row["cnt"]
        return total_child, total_parent

    async def get_document_chunks(
        self,
        document_id: str | None = None,
        level: str = "parent",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        params: tuple[Any, ...]
        if document_id:
            query = """
                SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path, c.page_number, c.summary_text, c.summary_tokens, c.summary_model, d.title as document_title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = ? AND c.level = ?
                ORDER BY c.page_number ASC, c.created_at ASC
                LIMIT ?
            """
            params = (document_id, level, limit)
        else:
            query = """
                SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path, c.page_number, c.summary_text, c.summary_tokens, c.summary_model, d.title as document_title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.level = ?
                ORDER BY c.page_number ASC, c.created_at ASC
                LIMIT ?
            """
            params = (level, limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "document_id": r["document_id"],
                    "parent_chunk_id": r["parent_chunk_id"],
                    "title": r["document_title"],
                    "content": r["content"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
                    "summary_text": r["summary_text"] if "summary_text" in r.keys() else None,
                    "summary_tokens": r["summary_tokens"] if "summary_tokens" in r.keys() else None,
                    "summary_model": r["summary_model"] if "summary_model" in r.keys() else None,
                }
                for r in rows
            ]

    async def insert_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []
        conn = self._get_conn()
        chunk_params = []
        fts_params = []
        chunk_ids = []

        for c in chunks:
            chunk_ids.append(c.id)
            level_str = c.level.value if hasattr(c.level, "value") else str(c.level)
            chunk_params.append(
                (
                    c.id,
                    c.document_id,
                    c.parent_chunk_id,
                    level_str,
                    c.content,
                    c.token_count,
                    c.section_path,
                    c.page_number,
                    json.dumps(c.embedding) if c.embedding else None,
                    c.summary_text,
                    c.summary_tokens,
                    c.summary_model,
                    c.generated_at.isoformat() if c.generated_at else None,
                    json.dumps(c.metadata) if c.metadata else "{}",
                    c.created_at.isoformat(),
                )
            )
            # Only index child chunks in FTS search
            if level_str == "child" or c.level == ChunkLevel.CHILD or c.parent_chunk_id is not None:
                fts_params.append(
                    (
                        c.id,
                        c.document_id,
                        c.content,
                        c.section_path or "",
                        c.summary_text or "",
                    )
                )

        await conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id, document_id, parent_chunk_id, level, content,
                token_count, section_path, page_number, embedding,
                summary_text, summary_tokens, summary_model, generated_at,
                metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_params,
        )

        if fts_params:
            await conn.executemany(
                """
                INSERT OR REPLACE INTO chunks_fts (id, document_id, content, section_path, summary_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                fts_params,
            )

        await conn.commit()
        return chunk_ids

    async def update_chunk_summaries_batch(
        self, updates: list[tuple[str, str, int, str, Any]]
    ) -> None:
        """Incrementally update summaries for a batch of chunks in SQLite."""
        if not updates:
            return
        conn = self._get_conn()
        params = [
            (
                s_text,
                s_tokens,
                s_model,
                gen_at.isoformat() if hasattr(gen_at, "isoformat") else str(gen_at),
                c_id,
            )
            for c_id, s_text, s_tokens, s_model, gen_at in updates
        ]
        await conn.executemany(
            """
            UPDATE chunks SET
                summary_text = ?,
                summary_tokens = ?,
                summary_model = ?,
                generated_at = ?
            WHERE id = ?;
            """,
            params,
        )
        # Also update FTS summary_text
        fts_params = [(s_text or "", c_id) for c_id, s_text, _, _, _ in updates]
        await conn.executemany(
            """
            UPDATE chunks_fts SET summary_text = ? WHERE id = ?;
            """,
            fts_params,
        )
        await conn.commit()

    async def update_chunk_embeddings_batch(self, updates: list[tuple[str, list[float]]]) -> None:
        """Incrementally update embeddings for a batch of chunks in SQLite."""
        if not updates:
            return
        conn = self._get_conn()
        params = [(json.dumps(emb) if emb is not None else None, c_id) for c_id, emb in updates]
        await conn.executemany(
            """
            UPDATE chunks SET embedding = ? WHERE id = ?;
            """,
            params,
        )
        await conn.commit()

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        conn = self._get_conn()
        async with conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return Chunk(
                id=row["id"],
                document_id=row["document_id"],
                parent_chunk_id=row["parent_chunk_id"],
                level=ChunkLevel(row["level"]),
                content=row["content"],
                token_count=row["token_count"],
                section_path=row["section_path"],
                page_number=row["page_number"],
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
                summary_text=row["summary_text"] if "summary_text" in row.keys() else None,
                summary_tokens=row["summary_tokens"] if "summary_tokens" in row.keys() else None,
                summary_model=row["summary_model"] if "summary_model" in row.keys() else None,
                generated_at=(
                    datetime.fromisoformat(row["generated_at"])
                    if "generated_at" in row.keys() and row["generated_at"]
                    else None
                ),
                metadata=json.loads(row["metadata"])
                if ("metadata" in row.keys() and row["metadata"])
                else {},
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in chunk_ids)
        async with conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                Chunk(
                    id=r["id"],
                    document_id=r["document_id"],
                    parent_chunk_id=r["parent_chunk_id"],
                    level=ChunkLevel(r["level"]),
                    content=r["content"],
                    token_count=r["token_count"],
                    section_path=r["section_path"],
                    page_number=r["page_number"],
                    embedding=json.loads(r["embedding"]) if r["embedding"] else None,
                    summary_text=r["summary_text"] if "summary_text" in r.keys() else None,
                    summary_tokens=r["summary_tokens"] if "summary_tokens" in r.keys() else None,
                    summary_model=r["summary_model"] if "summary_model" in r.keys() else None,
                    generated_at=(
                        datetime.fromisoformat(r["generated_at"])
                        if "generated_at" in r.keys() and r["generated_at"]
                        else None
                    ),
                    metadata=json.loads(r["metadata"])
                    if ("metadata" in r.keys() and r["metadata"])
                    else {},
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    # -----------------------------------------------------------------------
    # Document Parse Tree Nodes
    # -----------------------------------------------------------------------

    def _row_to_tree_node(self, r: Any) -> DocumentNode:
        bbox_data = json.loads(r["bbox"]) if r["bbox"] else None
        bbox_tuple = tuple(bbox_data) if bbox_data else None
        char_span_data = (
            json.loads(r["char_span"]) if ("char_span" in r.keys() and r["char_span"]) else None
        )
        char_span_tuple = tuple(char_span_data) if char_span_data else None
        line_range_data = (
            json.loads(r["line_range"]) if ("line_range" in r.keys() and r["line_range"]) else None
        )
        line_range_tuple = tuple(line_range_data) if line_range_data else None

        prov = Provenance(
            source_uri=r["source_uri"] if "source_uri" in r.keys() else None,
            page_number=r["page_number"],
            page_end=r["page_end"] if "page_end" in r.keys() else None,
            bbox=bbox_tuple,
            char_span=char_span_tuple,
            raw_ref=r["raw_ref"] if "raw_ref" in r.keys() else None,
            parser=r["parser"] if "parser" in r.keys() else None,
            parser_version=r["parser_version"] if "parser_version" in r.keys() else None,
            extraction_method=r["extraction_method"] if "extraction_method" in r.keys() else None,
            confidence=r["confidence"] if "confidence" in r.keys() else None,
            line_range=line_range_tuple,
        )
        t_data = TableDataModel.from_dict(json.loads(r["table_data"])) if r["table_data"] else None
        f_data = (
            FigureDataModel.from_dict(json.loads(r["figure_data"])) if r["figure_data"] else None
        )
        eq_data = (
            EquationDataModel.from_dict(json.loads(r["equation_data"]))
            if ("equation_data" in r.keys() and r["equation_data"])
            else None
        )
        code_data = (
            SourceCodeDataModel.from_dict(json.loads(r["code_data"]))
            if ("code_data" in r.keys() and r["code_data"])
            else None
        )
        meta = json.loads(r["metadata"]) if ("metadata" in r.keys() and r["metadata"]) else {}
        asset_id = r["asset_id"] if "asset_id" in r.keys() else None

        return DocumentNode(
            id=r["id"],
            document_id=r["document_id"],
            parent_id=r["parent_node_id"],
            node_type=DocumentElementType(r["node_type"]),
            reading_order=r["reading_order"],
            text=r["text"] or "",
            raw_text=r["raw_text"] if "raw_text" in r.keys() else None,
            section_path=r["section_path"],
            provenance=prov,
            table_data=t_data,
            figure_data=f_data,
            equation_data=eq_data,
            code_data=code_data,
            asset_id=asset_id,
            metadata=meta,
        )

    async def insert_tree_nodes(self, nodes: list[DocumentNode]) -> list[str]:
        if not nodes:
            return []
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        params = []
        for n in nodes:
            bbox_str = json.dumps(n.provenance.bbox) if n.provenance.bbox else None
            char_span_str = json.dumps(n.provenance.char_span) if n.provenance.char_span else None
            line_range_str = (
                json.dumps(n.provenance.line_range) if n.provenance.line_range else None
            )
            t_data_str = json.dumps(n.table_data.to_dict()) if n.table_data else None
            f_data_str = json.dumps(n.figure_data.to_dict()) if n.figure_data else None
            eq_data_str = json.dumps(n.equation_data.to_dict()) if n.equation_data else None
            code_data_str = json.dumps(n.code_data.to_dict()) if n.code_data else None
            meta_str = json.dumps(n.metadata) if n.metadata else "{}"
            node_type_str = n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type)
            params.append(
                (
                    n.id,
                    n.document_id,
                    n.parent_id,
                    node_type_str,
                    n.reading_order,
                    n.text[:100] if n.text else "",
                    n.text,
                    n.raw_text,
                    n.section_path,
                    n.provenance.page_number,
                    n.provenance.page_end,
                    bbox_str,
                    t_data_str,
                    f_data_str,
                    meta_str,
                    n.provenance.source_uri,
                    char_span_str,
                    n.provenance.raw_ref,
                    n.provenance.parser,
                    n.provenance.parser_version,
                    n.provenance.extraction_method,
                    n.provenance.confidence,
                    line_range_str,
                    eq_data_str,
                    code_data_str,
                    n.asset_id,
                    now,
                )
            )

        await conn.executemany(
            """
            INSERT OR REPLACE INTO document_tree_nodes (
                id, document_id, parent_node_id, node_type, reading_order,
                title, text, raw_text, section_path, page_number, page_end,
                bbox, table_data, figure_data, metadata,
                source_uri, char_span, raw_ref, parser, parser_version,
                extraction_method, confidence, line_range, equation_data, code_data, asset_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        await conn.commit()
        return [n.id for n in nodes]

    async def get_tree_nodes(self, document_id: str) -> list[DocumentNode]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM document_tree_nodes WHERE document_id = ? ORDER BY reading_order ASC",
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            nodes = [self._row_to_tree_node(r) for r in rows]

            # Reconstruct children_ids
            node_map = {n.id: n for n in nodes}
            for n in nodes:
                if n.parent_id and n.parent_id in node_map:
                    node_map[n.parent_id].children_ids.append(n.id)

            return nodes

    async def get_tree_node(self, node_id: str) -> DocumentNode | None:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM document_tree_nodes WHERE id = ?",
            (node_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_tree_node(row) if row else None

    async def get_tree_nodes_by_ids(self, node_ids: list[str]) -> list[DocumentNode]:
        if not node_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in node_ids)
        async with conn.execute(
            f"SELECT * FROM document_tree_nodes WHERE id IN ({placeholders})",
            node_ids,
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_tree_node(r) for r in rows]

    # -----------------------------------------------------------------------
    # Multimodal Assets
    # -----------------------------------------------------------------------

    def _row_to_asset(self, r: Any) -> MultimodalAsset:
        bbox_data = json.loads(r["bbox"]) if r["bbox"] else None
        bbox_tuple = tuple(bbox_data) if bbox_data else None
        emb_data = json.loads(r["embedding"]) if r["embedding"] else None
        scopes = json.loads(r["permission_scope"]) if r["permission_scope"] else ["default"]
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        return MultimodalAsset(
            id=r["id"],
            document_id=r["document_id"],
            node_id=r["node_id"],
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
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    async def insert_assets(self, assets: list[MultimodalAsset]) -> list[str]:
        if not assets:
            return []
        conn = self._get_conn()
        params = []
        for a in assets:
            params.append(
                (
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
                    a.caption,
                    a.ocr_text,
                    a.description,
                    json.dumps(a.embedding) if a.embedding else None,
                    a.page_number,
                    json.dumps(a.bbox) if a.bbox else None,
                    a.tenant_id,
                    json.dumps(a.permission_scope),
                    json.dumps(a.metadata) if a.metadata else "{}",
                    a.created_at.isoformat()
                    if hasattr(a.created_at, "isoformat")
                    else str(a.created_at),
                )
            )
        await conn.executemany(
            """
            INSERT OR REPLACE INTO document_assets (
                id, document_id, node_id, asset_type, mime_type, width, height,
                byte_size, sha256, storage_path, caption, ocr_text, description,
                embedding, page_number, bbox, tenant_id, permission_scope,
                metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        await conn.commit()
        return [a.id for a in assets]

    async def insert_asset(self, asset: MultimodalAsset) -> str:
        ids = await self.insert_assets([asset])
        return ids[0] if ids else asset.id

    async def get_asset(self, asset_id: str) -> MultimodalAsset | None:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM document_assets WHERE id = ?",
            (asset_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_asset(row) if row else None

    async def get_assets_by_ids(self, asset_ids: list[str]) -> list[MultimodalAsset]:
        if not asset_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in asset_ids)
        async with conn.execute(
            f"SELECT * FROM document_assets WHERE id IN ({placeholders})",
            asset_ids,
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_asset(r) for r in rows]

    async def get_assets_for_document(self, document_id: str) -> list[MultimodalAsset]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM document_assets WHERE document_id = ? ORDER BY page_number ASC, created_at ASC",
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_asset(r) for r in rows]

    async def search_assets_vector(
        self,
        query_embedding: list[float],
        filters: RetrievalFilters,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        conditions = ["tenant_id = ?"]
        params: list[Any] = [filters.tenant_id]

        if filters.document_ids:
            ph = ",".join("?" for _ in filters.document_ids)
            conditions.append(f"document_id IN ({ph})")
            params.extend(filters.document_ids)

        where_clause = " AND ".join(conditions)
        query_sql = f"SELECT * FROM document_assets WHERE {where_clause} AND embedding IS NOT NULL"

        async with conn.execute(query_sql, params) as cursor:
            rows = await cursor.fetchall()

        results = []
        user_scopes = set(filters.permission_scope) if filters.permission_scope else {"default"}

        for r in rows:
            scopes = (
                set(json.loads(r["permission_scope"])) if r["permission_scope"] else {"default"}
            )
            if not (user_scopes & scopes or "public" in scopes):
                continue

            raw_emb = r["embedding"]
            if not raw_emb:
                continue
            v = np.array(json.loads(raw_emb), dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
            results.append(
                {
                    "id": r["id"],
                    "document_id": r["document_id"],
                    "node_id": r["node_id"],
                    "asset_type": r["asset_type"],
                    "mime_type": r["mime_type"],
                    "caption": r["caption"],
                    "ocr_text": r["ocr_text"],
                    "description": r["description"],
                    "page_number": r["page_number"],
                    "storage_path": r["storage_path"],
                    "score": sim,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def search_bm25(
        self,
        query: str,
        filters: RetrievalFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Perform BM25 search using SQLite FTS5 with phrase boosting and sanitized tokenization."""
        conn = self._get_conn()

        # Clean and tokenize query
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
        if not words:
            words = [query.strip()]

        clean_terms = [f'"{w}"' for w in words]

        # If query has 2+ words, also add exact phrase matching for maximum precision
        if len(words) >= 2:
            phrase = " ".join(words)
            fts_query = f'"{phrase}" OR ' + " OR ".join(clean_terms)
        else:
            fts_query = " OR ".join(clean_terms)

        sql = """
            SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                   c.page_number, c.summary_text, d.title as document_title, d.source_uri, d.permission_scope,
                   bm25(chunks_fts) as fts_rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.id
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ? AND d.tenant_id = ?
        """
        params: list[Any] = [fts_query, filters.tenant_id]

        if filters.document_ids:
            ph = ",".join("?" for _ in filters.document_ids)
            sql += f" AND d.id IN ({ph})"
            params.extend(filters.document_ids)

        if filters.doc_types:
            ph_dt = ",".join("?" for _ in filters.doc_types)
            sql += f" AND d.doc_type IN ({ph_dt})"
            params.extend(filters.doc_types)

        if filters.section_prefix:
            sql += " AND c.section_path LIKE ?"
            params.append(f"{filters.section_prefix}%")

        sql += f" ORDER BY fts_rank ASC LIMIT {limit}"

        results: list[dict[str, Any]] = []
        try:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                for r in rows:
                    doc_perms = json.loads(r["permission_scope"])
                    if any(p in filters.permission_scope for p in doc_perms):
                        # Convert SQLite BM25 (negative where lower is better) to positive score
                        fts_val = float(r["fts_rank"])
                        norm_score = -fts_val if fts_val < 0 else 1.0 / (1.0 + fts_val)
                        results.append(
                            {
                                "id": r["id"],
                                "document_id": r["document_id"],
                                "parent_chunk_id": r["parent_chunk_id"],
                                "content": r["content"],
                                "section_path": r["section_path"],
                                "page_number": r["page_number"],
                                "summary_text": r["summary_text"]
                                if "summary_text" in r.keys()
                                else None,
                                "document_title": r["document_title"],
                                "source_uri": r["source_uri"],
                                "score": norm_score,
                            }
                        )
        except Exception as e:
            logger.warning("FTS search failed or returned no results: %s", e)

        return results

    async def search_vector(
        self,
        query_embedding: list[float],
        filters: RetrievalFilters,
        limit: int = 100,
        ef_search: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform in-memory cosine vector search over stored child chunk embeddings."""
        conn = self._get_conn()
        sql = """
            SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                   c.page_number, c.summary_text, c.embedding, d.title as document_title, d.source_uri, d.permission_scope
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.level = 'child' AND c.embedding IS NOT NULL AND d.tenant_id = ?
        """
        params: list[Any] = [filters.tenant_id]

        if filters.document_ids:
            ph = ",".join("?" for _ in filters.document_ids)
            sql += f" AND d.id IN ({ph})"
            params.extend(filters.document_ids)

        if filters.doc_types:
            ph_dt = ",".join("?" for _ in filters.doc_types)
            sql += f" AND d.doc_type IN ({ph_dt})"
            params.extend(filters.doc_types)

        if filters.section_prefix:
            sql += " AND c.section_path LIKE ?"
            params.append(f"{filters.section_prefix}%")

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        candidates: list[tuple[float, dict[str, Any]]] = []

        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            for r in rows:
                doc_perms = json.loads(r["permission_scope"])
                if not any(p in filters.permission_scope for p in doc_perms):
                    continue

                emb = json.loads(r["embedding"])
                c_vec = np.array(emb, dtype=np.float32)
                if len(c_vec) != len(q_vec):
                    continue
                c_norm = np.linalg.norm(c_vec)
                if c_norm == 0:
                    continue
                c_vec = c_vec / c_norm

                cosine_sim = float(np.dot(q_vec, c_vec))
                candidates.append(
                    (
                        cosine_sim,
                        {
                            "id": r["id"],
                            "document_id": r["document_id"],
                            "parent_chunk_id": r["parent_chunk_id"],
                            "content": r["content"],
                            "section_path": r["section_path"],
                            "page_number": r["page_number"],
                            "summary_text": r["summary_text"]
                            if "summary_text" in r.keys()
                            else None,
                            "document_title": r["document_title"],
                            "source_uri": r["source_uri"],
                            "score": cosine_sim,
                        },
                    )
                )

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:limit]]

    # -----------------------------------------------------------------------
    # Typed Memory (Policy, Preference, Fact, Episode)
    # -----------------------------------------------------------------------

    async def get_policy(
        self, tenant_id: str, policy_key: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        conn = self._get_conn()
        params: tuple[Any, ...]
        if user_id:
            query = "SELECT * FROM memory_policy WHERE tenant_id = ? AND policy_key = ? AND (user_id = ? OR user_id IS NULL) ORDER BY user_id DESC LIMIT 1"
            params = (tenant_id, policy_key, user_id)
        else:
            query = "SELECT * FROM memory_policy WHERE tenant_id = ? AND policy_key = ? AND user_id IS NULL"
            params = (tenant_id, policy_key)

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "tenant_id": row["tenant_id"],
                "user_id": row["user_id"],
                "policy_key": row["policy_key"],
                "policy_value": json.loads(row["policy_value"]),
            }

    async def set_policy(
        self,
        tenant_id: str,
        policy_key: str,
        policy_value: dict[str, Any],
        user_id: str | None = None,
    ) -> None:
        conn = self._get_conn()
        import uuid

        now_iso = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            """
            INSERT OR REPLACE INTO memory_policy (id, tenant_id, user_id, policy_key, policy_value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                tenant_id,
                user_id,
                policy_key,
                json.dumps(policy_value),
                now_iso,
                now_iso,
            ),
        )
        await conn.commit()

    async def list_policies(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        params: tuple[Any, ...]
        if user_id:
            query = "SELECT * FROM memory_policy WHERE tenant_id = ? AND (user_id = ? OR user_id IS NULL)"
            params = (tenant_id, user_id)
        else:
            query = "SELECT * FROM memory_policy WHERE tenant_id = ?"
            params = (tenant_id,)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "tenant_id": r["tenant_id"],
                    "user_id": r["user_id"],
                    "policy_key": r["policy_key"],
                    "policy_value": json.loads(r["policy_value"]),
                }
                for r in rows
            ]

    async def get_preference(self, user_id: str, preference_key: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM memory_preference WHERE user_id = ? AND preference_key = ?",
            (user_id, preference_key),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "preference_key": row["preference_key"],
                "preference_value": json.loads(row["preference_value"]),
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
        conn = self._get_conn()
        import uuid

        now_iso = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            """
            INSERT OR REPLACE INTO memory_preference (id, user_id, preference_key, preference_value, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                preference_key,
                json.dumps(preference_value),
                confidence,
                source,
                now_iso,
                now_iso,
            ),
        )
        await conn.commit()

    async def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM memory_preference WHERE user_id = ?", (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "preference_key": r["preference_key"],
                    "preference_value": json.loads(r["preference_value"]),
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
        import uuid

        fact_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        exp_iso = expires_at.isoformat() if expires_at else None

        conn = self._get_conn()
        await conn.execute(
            """
            INSERT INTO memory_fact (
                id, tenant_id, user_id, content, embedding, source,
                confidence, superseded_by, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_id,
                tenant_id,
                user_id,
                content,
                json.dumps(embedding) if embedding else None,
                source,
                confidence,
                superseded_by,
                exp_iso,
                now_iso,
            ),
        )
        await conn.execute(
            "INSERT OR REPLACE INTO memory_fact_fts (id, content) VALUES (?, ?)",
            (fact_id, content),
        )
        await conn.commit()
        return fact_id

    async def search_facts(
        self,
        query: str,
        query_embedding: list[float] | None,
        tenant_id: str = "default",
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Query active facts
        if user_id:
            sql = "SELECT * FROM memory_fact WHERE tenant_id = ? AND (user_id = ? OR user_id IS NULL) AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
            params: list[Any] = [tenant_id, user_id, now_iso]
        else:
            sql = "SELECT * FROM memory_fact WHERE tenant_id = ? AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
            params = [tenant_id, now_iso]

        facts: list[dict[str, Any]] = []
        q_vec = np.array(query_embedding, dtype=np.float32) if query_embedding else None
        if q_vec is not None and np.linalg.norm(q_vec) > 0:
            q_vec = q_vec / np.linalg.norm(q_vec)

        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            for r in rows:
                score = r["confidence"]
                if q_vec is not None and r["embedding"]:
                    emb = json.loads(r["embedding"])
                    c_vec = np.array(emb, dtype=np.float32)
                    if len(c_vec) == len(q_vec):
                        c_norm = np.linalg.norm(c_vec)
                        if c_norm > 0:
                            sim = float(np.dot(q_vec, c_vec / c_norm))
                            score = 0.5 * score + 0.5 * max(sim, 0.0)

                facts.append(
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "source": r["source"],
                        "confidence": r["confidence"],
                        "expires_at": r["expires_at"],
                        "score": score,
                    }
                )

        facts.sort(key=lambda x: x["score"], reverse=True)
        return facts[:limit]

    async def get_facts_for_scope(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[ExistingMemory]:
        conn = self._get_conn()
        now_iso = datetime.now(timezone.utc).isoformat()
        params: tuple[Any, ...]
        if user_id:
            sql = "SELECT id, content, confidence, created_at FROM memory_fact WHERE tenant_id = ? AND (user_id = ? OR user_id IS NULL) AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
            params = (tenant_id, user_id, now_iso)
        else:
            sql = "SELECT id, content, confidence, created_at FROM memory_fact WHERE tenant_id = ? AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
            params = (tenant_id, now_iso)

        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [
                ExistingMemory(
                    id=r["id"],
                    content=r["content"],
                    confidence=r["confidence"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    async def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> None:
        conn = self._get_conn()
        await conn.execute(
            "UPDATE memory_fact SET superseded_by = ? WHERE id = ?",
            (new_fact_id, old_fact_id),
        )
        await conn.commit()

    async def insert_episode(
        self,
        user_id: str,
        summary: str,
        session_id: str | None = None,
        task_type: str | None = None,
        outcome: str = "success",
        embedding: list[float] | None = None,
    ) -> str:
        import uuid

        ep_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        await conn.execute(
            """
            INSERT INTO memory_episode (id, user_id, session_id, task_type, summary, outcome, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ep_id,
                user_id,
                session_id,
                task_type,
                summary,
                outcome,
                json.dumps(embedding) if embedding else None,
                now_iso,
            ),
        )
        await conn.commit()
        return ep_id

    async def search_episodes(
        self, user_id: str, query_embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        async with conn.execute(
            "SELECT * FROM memory_episode WHERE user_id = ? AND embedding IS NOT NULL",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            candidates: list[tuple[float, dict[str, Any]]] = []
            for r in rows:
                emb = json.loads(r["embedding"])
                c_vec = np.array(emb, dtype=np.float32)
                if len(c_vec) != len(q_vec):
                    continue
                c_norm = np.linalg.norm(c_vec)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q_vec, c_vec / c_norm))
                candidates.append(
                    (
                        sim,
                        {
                            "id": r["id"],
                            "summary": r["summary"],
                            "task_type": r["task_type"],
                            "outcome": r["outcome"],
                            "created_at": r["created_at"],
                            "score": sim,
                        },
                    )
                )

            candidates.sort(key=lambda x: x[0], reverse=True)
            return [c[1] for c in candidates[:limit]]

    # -----------------------------------------------------------------------
    # Events Trace Log
    # -----------------------------------------------------------------------

    async def insert_event_trace(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_id: str | None = None,
        agent_id: str | None = None,
        token_cost: int = 0,
        latency_ms: int = 0,
    ) -> int:
        conn = self._get_conn()
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = await conn.execute(
            """
            INSERT INTO events_trace (session_id, agent_id, event_type, payload, token_cost, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                agent_id,
                event_type,
                json.dumps(payload),
                token_cost,
                latency_ms,
                now_iso,
            ),
        )
        await conn.commit()
        return cursor.lastrowid or 0

    async def list_event_traces(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        conditions = []
        params = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM events_trace {where_clause} ORDER BY id DESC LIMIT {limit}"

        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "agent_id": r["agent_id"],
                    "event_type": r["event_type"],
                    "payload": json.loads(r["payload"]),
                    "token_cost": r["token_cost"],
                    "latency_ms": r["latency_ms"],
                    "created_at": r["created_at"],
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
        conn = self._get_conn()
        await conn.execute(
            """
            INSERT INTO jobs (name, schedule_cron, next_run_at, status, max_retries)
            VALUES (?, ?, ?, 'idle', ?)
            ON CONFLICT(name) DO UPDATE SET
                schedule_cron = excluded.schedule_cron,
                next_run_at = excluded.next_run_at,
                max_retries = excluded.max_retries
            """,
            (name, schedule_cron, next_run_at.isoformat(), max_retries),
        )
        await conn.commit()

    async def get_due_jobs(self, now: datetime) -> list[dict[str, Any]]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM jobs WHERE next_run_at <= ? AND status != 'running'",
            (now.isoformat(),),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def mark_job_running(self, name: str) -> None:
        conn = self._get_conn()
        await conn.execute(
            "UPDATE jobs SET status = 'running', last_run_at = ? WHERE name = ?",
            (datetime.now(timezone.utc).isoformat(), name),
        )
        await conn.commit()

    async def mark_job_done(self, name: str, next_run_at: datetime) -> None:
        conn = self._get_conn()
        await conn.execute(
            "UPDATE jobs SET status = 'idle', retries = 0, last_error = NULL, "
            "next_run_at = ? WHERE name = ?",
            (next_run_at.isoformat(), name),
        )
        await conn.commit()

    async def mark_job_failed(self, name: str, error: str) -> None:
        conn = self._get_conn()
        await conn.execute(
            "UPDATE jobs SET status = 'failed', retries = retries + 1, last_error = ? "
            "WHERE name = ?",
            (error[:2000], name),
        )
        await conn.commit()

    async def list_jobs(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        async with conn.execute("SELECT * FROM jobs ORDER BY name") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Maintenance Operations (scheduler tasks)
    # -----------------------------------------------------------------------

    async def cleanup_orphaned_chunks(self) -> int:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
        ) as cursor:
            row = await cursor.fetchone()
            count = row["cnt"] if row else 0
        if count:
            await conn.execute(
                "DELETE FROM chunks_fts WHERE document_id NOT IN (SELECT id FROM documents)"
            )
            await conn.execute(
                "DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
            )
            await conn.commit()
        return count

    async def rebuild_fts_index(self) -> int:
        conn = self._get_conn()
        await conn.execute("DELETE FROM chunks_fts")
        await conn.execute("""
            INSERT INTO chunks_fts (id, document_id, content, section_path)
            SELECT id, document_id, content, section_path FROM chunks WHERE level = 'child'
            """)
        await conn.commit()
        async with conn.execute("SELECT COUNT(*) as cnt FROM chunks_fts") as cursor:
            row = await cursor.fetchone()
            return row["cnt"] if row else 0

    async def backfill_missing_embeddings(self) -> int:
        """Re-embed child chunks with NULL embeddings using the active embedding client."""
        from deep_context.core.llm_client import llm_client

        conn = self._get_conn()
        async with conn.execute(
            "SELECT id, content FROM chunks WHERE level = 'child' AND embedding IS NULL LIMIT 500"
        ) as cursor:
            rows = await cursor.fetchall()
        if not rows:
            return 0

        texts = [r["content"] for r in rows]
        embeddings = await llm_client.get_embeddings(texts)
        backfilled = 0
        for r, emb in zip(rows, embeddings, strict=False):
            if emb is None:
                continue
            await conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                json.dumps([float(x) for x in emb]),
                r["id"],
            )
            backfilled += 1
        await conn.commit()
        return backfilled
