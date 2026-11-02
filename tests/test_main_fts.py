"""Measurement-only port of the lexical channel the application serves today (``LEXICAL_RANKING=main_fts``).

``main``'s ParadeDB BM25 branch never succeeds, so its ``search_bm25`` always runs the fallback ("enhanced
FTS": ts_rank + pg_trgm similarity + exact-identifier and quoted-phrase boosts). The port is checked against
``main``'s real code separately (scratch script); these tests pin down the behaviour of the port itself.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from eval import pg
from eval.main_fts import query_terms

from deep_context.core.types import Chunk, ChunkLevel, Document, RetrievalFilters

SCRATCH_DB = "deepcontext_eval_mainfts_test"
SERVER_UP = asyncio.run(pg.server_available())
needs_pg = pytest.mark.skipif(not SERVER_UP, reason="eval Postgres is not reachable")
DOC_A, DOC_B = str(uuid.uuid4()), str(uuid.uuid4())


# ---- query parsing (pure) ------------------------------------------------------------------
def test_stopwords_and_short_words_are_dropped() -> None:
    plain, or_terms, phrase, ident = query_terms("Who did the Starks tell about Ice?")
    assert (plain, or_terms, phrase, ident) == ("starks ice", "starks | ice", "", "")


def test_curly_quoted_phrase_is_extracted() -> None:
    assert query_terms("What is “funeral pyre”?")[2] == "funeral pyre"


def test_hyphenated_identifier_is_extracted() -> None:
    assert query_terms("specs of ZX-9000 and part_no-12")[3] == "ZX-9000"


def test_query_without_usable_words_searches_for_text_like_main() -> None:
    plain, or_terms, _, _ = query_terms("?? !!")
    assert or_terms == "text" and plain == "?? !!"


def test_only_the_first_eight_words_are_used() -> None:
    words = " ".join(f"word{i}" for i in range(12))
    assert query_terms(words)[1].count("|") == 7


# ---- behaviour on the real eval store ------------------------------------------------------
async def _recreate_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{SCRATCH_DB}"')
    finally:
        await admin.close()


async def _drop_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture(scope="module", autouse=True)
def scratch_db() -> Any:
    if SERVER_UP:
        asyncio.run(_recreate_db())
    yield
    if SERVER_UP:
        asyncio.run(_drop_db())


def _child(doc: str, text: str) -> Chunk:
    return Chunk(
        document_id=doc, level=ChunkLevel.CHILD, content=text, token_count=len(text.split()),
        page_number=1,
    )  # fmt: skip


@pytest.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Any]:
    if not SERVER_UP:
        pytest.skip("eval Postgres is not reachable")
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
    await s.insert_chunks(
        [
            _child(DOC_A, "Daenerys Targaryen hatched three dragons at the funeral pyre"),
            _child(DOC_A, "Ice is a Valyrian steel greatsword of House Stark"),
            _child(DOC_A, "Specifications of the ZX-9000 power supply unit"),
            _child(DOC_A, "Power supply units are rated in watts and power supply efficiency matters for any power supply"),
            _child(DOC_A, "The funeral was long and the pyre burned all night, but no dragons came"),
            _child(DOC_B, "Another tenant also writes about dragons hatching at a funeral pyre"),
        ]
    )  # fmt: skip
    monkeypatch.setenv("LEXICAL_RANKING", "main_fts")
    yield s
    await s.close()


async def _search(store: Any, query: str, filters: RetrievalFilters | None = None, limit: int = 10):  # type: ignore[no-untyped-def]
    return await store.search_bm25(query, filters or RetrievalFilters(), limit)


@needs_pg
async def test_ranks_the_chunk_with_the_query_words_first(store: Any) -> None:
    hits = await _search(store, "dragons hatched at the pyre")
    assert hits[0]["content"].startswith("Daenerys")
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


@needs_pg
async def test_exact_identifier_outranks_a_chunk_with_more_word_matches(store: Any) -> None:
    hits = await _search(store, "ZX-9000 power supply")
    assert hits[0]["content"].startswith("Specifications of the ZX-9000")


@needs_pg
async def test_quoted_phrase_gets_the_boost(store: Any) -> None:
    hits = await _search(store, 'what about "funeral pyre"')
    assert hits[0]["content"].startswith("Daenerys")  # the only chunk with the exact phrase


@needs_pg
async def test_only_child_chunks_of_the_tenant_are_returned(store: Any) -> None:
    hits = await _search(store, "dragons funeral pyre")
    assert all(h["document_id"] == DOC_A for h in hits)


@needs_pg
async def test_limit_is_respected(store: Any) -> None:
    assert len(await _search(store, "power supply dragons funeral", limit=2)) == 2


@needs_pg
async def test_rows_carry_the_summary_field_like_main(store: Any) -> None:
    hits = await _search(store, "dragons")
    assert hits and "summary_text" in hits[0] and hits[0]["summary_text"] is None


@needs_pg
async def test_other_channels_do_not_touch_pg_trgm(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with store._get_pool().acquire() as conn:  # extensions are per database: start without it
        await conn.execute("DROP EXTENSION IF EXISTS pg_trgm CASCADE")
    monkeypatch.setenv("LEXICAL_RANKING", "bm25")
    assert await _search(store, "dragons")
    async with store._get_pool().acquire() as conn:
        n = await conn.fetchval("SELECT count(*) FROM pg_extension WHERE extname = 'pg_trgm'")
    assert n == 0
