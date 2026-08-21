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


class SQLiteStore(StorageInterface):
    """Zero-configuration local storage implementing docs/DATA_MODEL.sql with SQLite & FTS5."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.sqlite_db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        self._conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute("PRAGMA busy_timeout = 30000;")
        await self._conn.execute("PRAGMA journal_mode = WAL;")
        await self._conn.execute("PRAGMA synchronous = NORMAL;")
        await self._conn.execute("PRAGMA foreign_keys = ON;")

        # Schema creation
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL,
                source_uri TEXT,
                doc_type TEXT NOT NULL,
                permission_scope TEXT NOT NULL DEFAULT '["default"]',
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
                embedding TEXT, -- JSON array of floats
                created_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                document_id UNINDEXED,
                content,
                section_path
            );

            CREATE TABLE IF NOT EXISTS document_tree_nodes (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_node_id TEXT REFERENCES document_tree_nodes(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                summary TEXT,
                chunk_id TEXT REFERENCES chunks(id),
                node_order INTEGER NOT NULL DEFAULT 0
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

            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'idle',
                budgets TEXT NOT NULL DEFAULT '{"max_turns": 50, "max_tokens": 2000000, "max_wall_clock_seconds": 3600}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                parent_session_id TEXT REFERENCES sessions(id),
                user_id TEXT NOT NULL,
                project_root TEXT,
                kernel_ref TEXT,
                state_snapshot TEXT NOT NULL DEFAULT '{}',
                budgets TEXT NOT NULL DEFAULT '{}',
                depth INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rlm_children (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                child_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                model TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'admitted',
                admitted_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE (parent_session_id, name)
            );

            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                target_session_id TEXT NOT NULL,
                receiver_role TEXT NOT NULL,
                receiver_name TEXT,
                content TEXT NOT NULL,
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
            """)
        await self._conn.commit()
        logger.info("Initialized SQLite database at %s", self.db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _get_conn(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError(
                "Database connection not initialized. Call initialize() first."
            )
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
        async with conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ) as cursor:
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

                docs.append(
                    {
                        "id": r["id"],
                        "title": r["title"],
                        "source_uri": r["source_uri"],
                        "doc_type": r["doc_type"],
                        "retrieval_mode": r["retrieval_mode"],
                        "child_chunks_count": child_cnt,
                        "parent_chunks_count": parent_cnt,
                        "created_at": r["ingested_at"],
                    }
                )
            return docs

    async def delete_document(self, document_id: str) -> bool:
        conn = self._get_conn()
        await conn.execute(
            "DELETE FROM document_tree_nodes WHERE document_id = ?", (document_id,)
        )
        await conn.execute(
            "DELETE FROM chunks_fts WHERE document_id = ?", (document_id,)
        )
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
                SELECT c.id, c.document_id, c.content, c.section_path, c.page_number, d.title as document_title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = ? AND c.level = ?
                ORDER BY c.page_number ASC, c.created_at ASC
                LIMIT ?
            """
            params = (document_id, level, limit)
        else:
            query = """
                SELECT c.id, c.document_id, c.content, c.section_path, c.page_number, d.title as document_title
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
                    "title": r["document_title"],
                    "content": r["content"],
                    "section_path": r["section_path"],
                    "page_number": r["page_number"],
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
            chunk_params.append(
                (
                    c.id,
                    c.document_id,
                    c.parent_chunk_id,
                    c.level.value,
                    c.content,
                    c.token_count,
                    c.section_path,
                    c.page_number,
                    json.dumps(c.embedding) if c.embedding else None,
                    c.created_at.isoformat(),
                )
            )
            # Only index child chunks in FTS search
            if c.level == ChunkLevel.CHILD:
                fts_params.append(
                    (c.id, c.document_id, c.content, c.section_path or "")
                )

        await conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id, document_id, parent_chunk_id, level, content,
                token_count, section_path, page_number, embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_params,
        )

        if fts_params:
            await conn.executemany(
                """
                INSERT OR REPLACE INTO chunks_fts (id, document_id, content, section_path)
                VALUES (?, ?, ?, ?)
                """,
                fts_params,
            )

        await conn.commit()
        return chunk_ids

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ) as cursor:
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
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

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
                   c.page_number, d.title as document_title, d.source_uri, d.permission_scope,
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
    ) -> list[dict[str, Any]]:
        """Perform in-memory cosine vector search over stored child chunk embeddings."""
        conn = self._get_conn()
        sql = """
            SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
                   c.page_number, c.embedding, d.title as document_title, d.source_uri, d.permission_scope
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.level = 'child' AND c.embedding IS NOT NULL AND d.tenant_id = ?
        """
        params: list[Any] = [filters.tenant_id]

        if filters.document_ids:
            ph = ",".join("?" for _ in filters.document_ids)
            sql += f" AND d.id IN ({ph})"
            params.extend(filters.document_ids)

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
                            "document_title": r["document_title"],
                            "source_uri": r["source_uri"],
                            "score": cosine_sim,
                        },
                    )
                )

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:limit]]

    async def insert_tree_nodes(self, nodes: list[DocumentTreeNode]) -> list[str]:
        if not nodes:
            return []
        conn = self._get_conn()
        node_params = [
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
            INSERT OR REPLACE INTO document_tree_nodes (
                id, document_id, parent_node_id, title, summary, chunk_id, node_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            node_params,
        )
        await conn.commit()
        return [n.id for n in nodes]

    async def get_tree_nodes_for_document(
        self, document_id: str
    ) -> list[DocumentTreeNode]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM document_tree_nodes WHERE document_id = ? ORDER BY node_order",
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                DocumentTreeNode(
                    id=r["id"],
                    document_id=r["document_id"],
                    parent_node_id=r["parent_node_id"],
                    title=r["title"],
                    summary=r["summary"],
                    chunk_id=r["chunk_id"],
                    node_order=r["node_order"],
                )
                for r in rows
            ]

    async def get_child_tree_nodes(
        self, document_id: str, parent_node_id: str | None
    ) -> list[DocumentTreeNode]:
        conn = self._get_conn()
        params: tuple[Any, ...]
        if parent_node_id is None:
            query = "SELECT * FROM document_tree_nodes WHERE document_id = ? AND parent_node_id IS NULL ORDER BY node_order"
            params = (document_id,)
        else:
            query = "SELECT * FROM document_tree_nodes WHERE document_id = ? AND parent_node_id = ? ORDER BY node_order"
            params = (document_id, parent_node_id)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                DocumentTreeNode(
                    id=r["id"],
                    document_id=r["document_id"],
                    parent_node_id=r["parent_node_id"],
                    title=r["title"],
                    summary=r["summary"],
                    chunk_id=r["chunk_id"],
                    node_order=r["node_order"],
                )
                for r in rows
            ]

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

    async def get_preference(
        self, user_id: str, preference_key: str
    ) -> dict[str, Any] | None:
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
    # Sessions, Agents & RLM Messaging
    # -----------------------------------------------------------------------

    async def create_session(
        self, session: SessionHandle, user_id: str = "default"
    ) -> None:
        conn = self._get_conn()
        now_iso = session.created_at.isoformat()
        budgets_json = json.dumps(
            {
                "max_turns": session.budgets.max_turns,
                "max_tokens": session.budgets.max_tokens,
                "max_wall_clock_seconds": session.budgets.max_wall_clock_seconds,
                "max_recursion_depth": session.budgets.max_recursion_depth,
            }
        )

        await conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                id, parent_session_id, user_id, depth, budgets, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.parent_session_id,
                user_id,
                session.depth,
                budgets_json,
                session.status.value,
                now_iso,
                now_iso,
            ),
        )
        await conn.commit()

    async def get_session(self, session_id: str) -> SessionHandle | None:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            b_dict = json.loads(row["budgets"]) if row["budgets"] else {}
            budgets = Budgets(
                max_turns=b_dict.get("max_turns", 50),
                max_tokens=b_dict.get("max_tokens", 2000000),
                max_wall_clock_seconds=b_dict.get("max_wall_clock_seconds", 3600),
                max_recursion_depth=b_dict.get("max_recursion_depth", 1),
            )
            return SessionHandle(
                id=row["id"],
                parent_session_id=row["parent_session_id"],
                depth=row["depth"],
                budgets=budgets,
                status=SessionStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    async def update_session_status(
        self, session_id: str, status: SessionStatus
    ) -> None:
        conn = self._get_conn()
        now_iso = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, now_iso, session_id),
        )
        await conn.commit()

    async def insert_rlm_child(
        self,
        parent_session_id: str,
        child_session_id: str,
        name: str,
        model: str,
        depth: int,
    ) -> str:
        import uuid

        cid = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        await conn.execute(
            """
            INSERT OR REPLACE INTO rlm_children (
                id, parent_session_id, child_session_id, name, model, depth, status, admitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                parent_session_id,
                child_session_id,
                name,
                model,
                depth,
                ChildStatus.ADMITTED.value,
                now_iso,
            ),
        )
        await conn.commit()
        return cid

    async def update_rlm_child_status(
        self, child_session_id: str, status: ChildStatus
    ) -> None:
        conn = self._get_conn()
        now_iso = datetime.now(timezone.utc).isoformat()
        completed_at = (
            now_iso if status in (ChildStatus.COMPLETED, ChildStatus.ERROR) else None
        )
        await conn.execute(
            "UPDATE rlm_children SET status = ?, completed_at = COALESCE(?, completed_at) WHERE child_session_id = ?",
            (status.value, completed_at, child_session_id),
        )
        await conn.commit()

    async def insert_agent_message(self, message: AgentMessage) -> str:
        import uuid

        mid = str(uuid.uuid4())
        now_iso = message.created_at.isoformat()
        conn = self._get_conn()
        # Find target session id based on sender and receiver role
        target_session_id = ""
        if message.receiver_role == "parent":
            async with conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?",
                (message.session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                target_session_id = row["parent_session_id"] if row else ""
        else:
            async with conn.execute(
                "SELECT child_session_id FROM rlm_children WHERE parent_session_id = ? AND name = ?",
                (message.session_id, message.receiver_name),
            ) as cursor:
                row = await cursor.fetchone()
                target_session_id = row["child_session_id"] if row else ""

        await conn.execute(
            """
            INSERT INTO agent_messages (
                id, session_id, target_session_id, receiver_role, receiver_name, content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mid,
                message.session_id,
                target_session_id,
                message.receiver_role,
                message.receiver_name,
                message.content,
                now_iso,
            ),
        )
        await conn.commit()
        return mid

    async def pop_agent_messages(self, target_session_id: str) -> list[AgentMessage]:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT * FROM agent_messages WHERE target_session_id = ? ORDER BY created_at ASC",
            (target_session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            messages = [
                AgentMessage(
                    session_id=r["session_id"],
                    receiver_role=r["receiver_role"],
                    receiver_name=r["receiver_name"],
                    content=r["content"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

        if messages:
            await conn.execute(
                "DELETE FROM agent_messages WHERE target_session_id = ?",
                (target_session_id,),
            )
            await conn.commit()

        return messages

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
        sql = (
            f"SELECT * FROM events_trace {where_clause} ORDER BY id DESC LIMIT {limit}"
        )

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
