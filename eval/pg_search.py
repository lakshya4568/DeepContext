"""ParadeDB pg_search BM25 as an opt-in lexical channel of the eval store (``LEXICAL_RANKING=pg_search``).

Why this exists: the application's ParadeDB integration (commit 684d533) creates its BM25 index with
``paradedb.create_bm25``, an API ParadeDB has removed, so the index is never built and every lexical query
falls back to ts_rank. Here the index is created with the current ``USING paradedb`` syntax and queried with
the ``|||`` match operator (any query term, scored with BM25). The index uses English stemming and stop-word
removal, like the tsvector channel, so the channels differ in the ranking engine and not in the text analysis.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from deep_context.core.types import RetrievalFilters

INDEX_NAME = "idx_chunks_pg_search"
_DDL_LOCK = (
    7270020  # advisory-lock key: serialise index creation across concurrent runner processes
)
_TOKENIZER = "pdb.simple('stemmer=english', 'stopwords_language=english', 'ascii_folding=true')"


async def ensure_index(pool: asyncpg.Pool) -> None:
    """Create the ParadeDB BM25 index on ``chunks`` (current schema) if it is missing."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _DDL_LOCK)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON chunks "
            f"USING paradedb (id, (content::{_TOKENIZER}))"
        )


async def search(
    pool: asyncpg.Pool, query: str, filters: RetrievalFilters, limit: int
) -> list[dict[str, Any]]:
    """Top-``limit`` child chunks by BM25; same filters and row fields as the stock lexical channels."""
    if not query.strip():
        return []
    clauses = ["c.level = 'child'", "d.tenant_id = $2", "d.permission_scope && $3"]
    params: list[Any] = [query, filters.tenant_id, filters.permission_scope]
    if filters.document_ids:
        clauses.append(f"d.id = ANY(${len(params) + 1}::uuid[])")
        params.append(filters.document_ids)
    sql = f"""
        SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path, c.page_number,
               d.title AS document_title, d.source_uri, pdb.score(c.id) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(clauses)} AND c.content ||| $1::text
        ORDER BY score DESC, c.id
        LIMIT {int(limit)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [
        {
            "id": str(r["id"]),
            "document_id": str(r["document_id"]),
            "parent_chunk_id": str(r["parent_chunk_id"]) if r["parent_chunk_id"] else None,
            "content": r["content"],
            "section_path": r["section_path"],
            "page_number": r["page_number"],
            "document_title": r["document_title"],
            "source_uri": r["source_uri"],
            "score": float(r["score"]),
        }
        for r in rows
    ]
