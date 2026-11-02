"""ParadeDB pg_search BM25 as an opt-in lexical channel of the eval store (``LEXICAL_RANKING=pg_search``).

Runs the real eval store against a scratch database on the eval Postgres (Docker ParadeDB) and is skipped
when that server, or its pg_search extension, is not available.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from eval import pg

from deep_context.core.types import Chunk, ChunkLevel, Document, RetrievalFilters

SCRATCH_DB = "deepcontext_eval_pgsearch_test"


async def _server_has_pg_search() -> bool:
    if not await pg.server_available():
        return False
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        return bool(
            await admin.fetchval("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'")
        )
    finally:
        await admin.close()


HAS_PG_SEARCH = asyncio.run(_server_has_pg_search())
pytestmark = pytest.mark.skipif(
    not HAS_PG_SEARCH, reason="eval Postgres with pg_search not available"
)

DOC_A, DOC_B, DOC_C = (str(uuid.uuid4()) for _ in range(3))


async def _recreate_scratch_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{SCRATCH_DB}"')
    finally:
        await admin.close()


async def _drop_scratch_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture(scope="module", autouse=True)
def scratch_db() -> Any:
    asyncio.run(_recreate_scratch_db())
    yield
    asyncio.run(_drop_scratch_db())


def _child(doc: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        document_id=doc, level=ChunkLevel.CHILD, content=text, token_count=len(text.split()),
        page_number=page,
    )  # fmt: skip


@pytest.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Any]:
    """A fresh schema holding a tiny corpus, queried through the real eval store."""
    schema = f"t_{uuid.uuid4().hex[:10]}"
    conn = await asyncpg.connect(pg.dsn_for(SCRATCH_DB))
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await conn.close()
    s = pg.make_store_class()(pg.dsn_for(SCRATCH_DB, schema))
    await s.initialize()
    await s.insert_document(Document(id=DOC_A, title="Doc A"))
    await s.insert_document(Document(id=DOC_B, title="Doc B", tenant_id="other"))
    await s.insert_document(Document(id=DOC_C, title="Doc C"))
    parent = Chunk(
        document_id=DOC_A, level=ChunkLevel.PARENT, token_count=9, page_number=1,
        content="A parent passage that repeats dragons and dragons and dragons",
    )  # fmt: skip
    await s.insert_chunks(
        [
            _child(DOC_A, "Daenerys Targaryen hatched three dragons at the funeral pyre"),
            _child(
                DOC_A, "Ice is a Valyrian steel greatsword of House Stark, four hundred years old"
            ),
            _child(DOC_A, "The quick brown fox jumps over the lazy dog"),
            parent,
            _child(DOC_B, "Another tenant also writes about dragons hatching"),
            _child(DOC_C, "Syrio Forel taught Arya the water dance and the dragons of Valyria"),
        ]
    )
    monkeypatch.setenv("LEXICAL_RANKING", "pg_search")
    yield s
    await s.close()


async def _index_count(store: Any) -> int:
    async with store._get_pool().acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = current_schema() "
                "AND indexname = 'idx_chunks_pg_search'"
            )
        )


async def search(
    store: Any, query: str, filters: RetrievalFilters | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """Search through the store and prove the pg_search index, not the ts_rank fallback, served it."""
    hits: list[dict[str, Any]] = await store.search_bm25(
        query, filters or RetrievalFilters(), limit
    )
    assert await _index_count(store) == 1, "the pg_search index was not used"
    return hits


async def test_ranks_the_matching_chunk_first_with_descending_scores(store: Any) -> None:
    hits = await search(store, "dragons hatched at the pyre")
    assert hits[0]["content"].startswith("Daenerys")
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_results_equal_a_direct_pg_search_query(store: Any) -> None:
    query = "dragons hatched at the pyre"
    hits = await search(store, query)
    async with store._get_pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT c.id::text AS id, pdb.score(c.id) AS score FROM chunks c "
            "WHERE c.level = 'child' AND c.document_id <> $2::uuid AND c.content ||| $1 "
            "ORDER BY score DESC, c.id LIMIT 10",
            query,
            DOC_B,
        )
    assert [h["id"] for h in hits] == [r["id"] for r in rows]
    assert [h["score"] for h in hits] == pytest.approx([float(r["score"]) for r in rows])


async def test_only_child_chunks_of_the_tenant_are_returned(store: Any) -> None:
    texts = [h["content"] for h in await search(store, "dragons")]
    assert not any(t.startswith("A parent passage") for t in texts)
    assert not any(t.startswith("Another tenant") for t in texts)
    assert len(texts) == 2


async def test_english_stemming_matches_inflected_words(store: Any) -> None:
    hits = await search(store, "hatching dragon")
    assert hits and hits[0]["content"].startswith("Daenerys")


async def test_query_punctuation_is_plain_text_not_query_syntax(store: Any) -> None:
    hits = await search(store, 'What is "Ice"? (Valyrian: steel)')
    assert hits and hits[0]["content"].startswith("Ice is a Valyrian")


@pytest.mark.parametrize("query", ["", "   ", "the of and"])
async def test_blank_or_stopword_only_queries_return_nothing(store: Any, query: str) -> None:
    await search(store, "dragons")  # make sure the pg_search path is the one being exercised
    assert await store.search_bm25(query, RetrievalFilters(), limit=10) == []


async def test_document_filter_restricts_the_results(store: Any) -> None:
    hits = await search(store, "dragons", RetrievalFilters(document_ids=[DOC_C]))
    assert [h["document_id"] for h in hits] == [DOC_C]


async def test_limit_is_respected(store: Any) -> None:
    assert len(await search(store, "dragons", limit=1)) == 1


async def test_rows_carry_every_field_the_retrieval_engine_reads(store: Any) -> None:
    hits = await search(store, "dragons", limit=5)
    needed = {"id", "document_id", "parent_chunk_id", "content", "section_path", "page_number"}
    needed |= {"document_title", "source_uri", "score"}
    assert hits and needed <= set(hits[0])


async def test_default_channel_is_untouched_and_creates_no_pg_search_index(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEXICAL_RANKING", "bm25")
    hits = await store.search_bm25("dragons", RetrievalFilters(), limit=5)
    assert hits
    assert await _index_count(store) == 0


async def test_the_index_is_created_once_and_reused(store: Any) -> None:
    await search(store, "dragons", limit=5)
    await search(store, "valyrian", limit=5)
    assert await _index_count(store) == 1
