"""Measurement-only port of the lexical channel the application serves today (``LEXICAL_RANKING=main_fts``).

Source: ``main`` @ b720085, ``PostgresStore.search_bm25``. Its ParadeDB BM25 branch never succeeds (the index is
never created, and the query names a table that its own alias hides), so the function always ends in the branch
ported here: ts_rank over the chunk text vector, pg_trgm similarity x 2, +8 for an exact identifier (a code with a
hyphen or underscore) and +10 for a quoted phrase.

Deviations from the original: the identifier is bound as a parameter instead of being string-interpolated (same
ILIKE pattern, same result); the summary columns of ``main``'s schema do not exist in the eval schemas, which hold
fast-ingested chunks, where ``main`` leaves them NULL too, and ``search_tsv`` is what ``main``'s trigger stores for
such chunks, ``setweight(tsv, 'B')``; the doc-type and section filters do not exist in this branch.
"""

from __future__ import annotations

import re
from typing import Any

import asyncpg

from deep_context.core.types import RetrievalFilters

_DDL_LOCK = (
    7270021  # advisory-lock key: serialise CREATE EXTENSION across concurrent runner processes
)
_STOPWORDS = {
    "who", "what", "where", "when", "why", "how", "did", "the", "and", "for", "page", "pages", "said",
    "tell", "about", "with", "from", "this", "that", "does",
}  # fmt: skip


def query_terms(query: str) -> tuple[str, str, str, str]:
    """(plain_term, or_terms, phrase_term, ident_term) exactly as ``main`` derives them from the query."""
    clean = (
        query.replace("“", '"').replace("”", '"').replace("‘", '"').replace("’", '"')
        .replace("'", '"')
    )  # fmt: skip
    phrases = re.findall(r'"([^"]{3,})"', clean)
    phrase_term = phrases[0] if phrases else ""
    idents = re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", clean)
    ident_term = idents[0] if idents else ""
    words = [w for w in re.findall(r"\w+", clean.lower()) if len(w) > 2 and w not in _STOPWORDS]
    or_terms = " | ".join(words[:8]) if words else (phrase_term if phrase_term else "text")
    plain_term = " ".join(words[:8]) if words else clean
    return plain_term, or_terms, phrase_term, ident_term


async def ensure_extension(pool: asyncpg.Pool) -> None:
    """``similarity()`` needs pg_trgm in the benchmark database.

    Installed into ``public``: an extension exists once per database, and without ``WITH SCHEMA`` it would land in
    whichever per-index schema is first on the search path, invisible to every other index.
    """
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _DDL_LOCK)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")


async def search(
    pool: asyncpg.Pool, query: str, filters: RetrievalFilters, limit: int
) -> list[dict[str, Any]]:
    plain, or_terms, phrase, ident = query_terms(query)
    clauses = ["c.level = 'child'", "d.tenant_id = $4", "d.permission_scope && $5"]
    params: list[Any] = [plain, or_terms, phrase, filters.tenant_id, filters.permission_scope]
    if filters.document_ids:
        clauses.append(f"d.id = ANY(${len(params) + 1}::uuid[])")
        params.append(filters.document_ids)
    n = len(params) + 1
    params.append(ident)  # '' when the query has no identifier
    sql = f"""
        SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path, c.page_number,
               NULL::text AS summary_text, d.title AS document_title, d.source_uri,
               (COALESCE(ts_rank(setweight(c.tsv, 'B'), plainto_tsquery('english', $1)), 0.0) +
                COALESCE(ts_rank(setweight(c.tsv, 'B'), to_tsquery('english', $2)), 0.0) +
                similarity(c.content, $1) * 2.0 +
                (CASE WHEN length(${n}) > 0 AND c.content ILIKE '%' || ${n} || '%' THEN 8.0 ELSE 0.0 END) +
                (CASE WHEN length($3) > 0 AND c.content ILIKE '%' || $3 || '%' THEN 10.0 ELSE 0.0 END)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(clauses)} AND (
            (length($1) > 0 AND c.tsv @@ plainto_tsquery('english', $1)) OR
            (length($2) > 0 AND c.tsv @@ to_tsquery('english', $2)) OR
            (length($3) > 0 AND c.content ILIKE '%' || $3 || '%') OR
            (length(${n}) > 0 AND c.content ILIKE '%' || ${n} || '%')
        )
        ORDER BY score DESC, c.created_at DESC
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
            "summary_text": r["summary_text"],
            "document_title": r["document_title"],
            "source_uri": r["source_uri"],
            "score": float(r["score"]),
        }
        for r in rows
    ]
