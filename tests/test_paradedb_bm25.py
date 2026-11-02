"""ParadeDB BM25 (pg_search): the index is created at startup and searches are served by it.

Integration test: needs PostgreSQL with the pg_search extension (the ParadeDB image) and is skipped otherwise.
It uses a throwaway database next to the one in POSTGRES_DSN.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from deep_context.core.config import settings
from deep_context.core.types import Chunk, ChunkLevel, Document, RetrievalFilters
from deep_context.storage.postgres_store import PostgresStore

SCRATCH_DB = "deepcontext_bm25_test"
DOC_A = "00000000-0000-0000-0000-0000000000aa"
DOC_B = "00000000-0000-0000-0000-0000000000bb"
TEXTS = [
    (DOC_A, "Daenerys Targaryen hatched three dragons at the funeral pyre"),
    (DOC_A, "Ice is a Valyrian steel greatsword of House Stark"),
    (DOC_A, "Specifications of the ZX-9000 power supply unit"),
    (DOC_B, "Another document about dragons hatching and dragons flying"),
]


def _dsn(database: str) -> str:
    return urlunsplit(urlsplit(settings.postgres_dsn)._replace(path=f"/{database}"))


async def _pg_search_available() -> bool:
    try:
        conn = await asyncpg.connect(_dsn("postgres"), timeout=5)
    except Exception:  # noqa: BLE001 - connectivity probe: any failure means "not available"
        return False
    try:
        return bool(
            await conn.fetchval("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'")
        )
    finally:
        await conn.close()


HAS_PG_SEARCH = asyncio.run(_pg_search_available())
pytestmark = pytest.mark.skipif(not HAS_PG_SEARCH, reason="PostgreSQL with pg_search not available")


async def _recreate_database() -> None:
    admin = await asyncpg.connect(_dsn("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{SCRATCH_DB}"')
    finally:
        await admin.close()


async def _drop_database() -> None:
    admin = await asyncpg.connect(_dsn("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture(scope="module", autouse=True)
def scratch_database():  # type: ignore[no-untyped-def]
    asyncio.run(_recreate_database())
    yield
    asyncio.run(_drop_database())


@pytest.fixture
async def store() -> AsyncIterator[PostgresStore]:
    s = PostgresStore(_dsn(SCRATCH_DB))
    await s.initialize()
    await s.insert_document(Document(id=DOC_A, title="Doc A"))
    await s.insert_document(Document(id=DOC_B, title="Doc B"))
    await s.insert_chunks(
        [
            Chunk(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, t)),  # deterministic: re-inserting is an upsert
                document_id=d, level=ChunkLevel.CHILD, content=t, token_count=len(t.split()), page_number=1,
            )
            for d, t in TEXTS
        ]
    )
    yield s
    await s.close()


async def _direct_scores(store: PostgresStore, text: str) -> list[float]:
    """What ParadeDB itself says: BM25 scores of the child chunks matching ``text``."""
    async with store._get_pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT pdb.score(c.id) AS score FROM chunks c WHERE c.level = 'child' "
            "AND (c.content ||| $1::text OR c.summary_text ||| $1::text) ORDER BY score DESC, c.id",
            text,
        )
    return [float(r["score"]) for r in rows]


async def test_startup_creates_the_bm25_index(store: PostgresStore) -> None:
    async with store._get_pool().acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE tablename = 'chunks' AND indexname = 'idx_chunks_bm25'"
        )
    assert n == 1
    assert store._has_pg_search is True


async def test_search_is_served_by_paradedb_not_the_fallback(store: PostgresStore) -> None:
    hits = await store.search_bm25("valyrian greatsword", RetrievalFilters(), limit=5)
    assert hits and hits[0]["content"].startswith("Ice is a Valyrian")
    assert [h["score"] for h in hits] == pytest.approx(await _direct_scores(store, "valyrian greatsword"))


async def test_stemming_and_the_document_filter_work(store: PostgresStore) -> None:
    hits = await store.search_bm25("hatching dragon", RetrievalFilters(), limit=5)
    assert {h["document_id"] for h in hits} == {DOC_A, DOC_B}
    only_b = await store.search_bm25("dragons", RetrievalFilters(document_ids=[DOC_B]), limit=5)
    assert [h["document_id"] for h in only_b] == [DOC_B]


async def test_fallback_still_serves_when_the_index_is_unavailable(store: PostgresStore) -> None:
    store._has_pg_search = False
    hits = await store.search_bm25("ZX-9000 power supply", RetrievalFilters(), limit=5)
    assert hits and hits[0]["content"].startswith("Specifications of the ZX-9000")


async def test_query_text_is_never_interpolated_into_sql(store: PostgresStore) -> None:
    store._has_pg_search = False  # the fallback builds the identifier clause
    await store.search_bm25("x'; DROP TABLE chunks; -- a-b", RetrievalFilters(), limit=5)
    async with store._get_pool().acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == len(TEXTS)
