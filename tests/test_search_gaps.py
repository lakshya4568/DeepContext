"""Tests for the search fixes: Okapi BM25 on Postgres, weighted RRF, exact-duplicate removal."""

from __future__ import annotations

import math
import uuid as uuidlib

import pytest

from deep_context.core.config import settings
from deep_context.retrieval.hybrid import deduplicate_candidates, reciprocal_rank_fusion


def _ids(lst: list[tuple[str, float]]) -> list[str]:
    return [c for c, _ in lst]


def test_rrf_default_weights_match_classic_rrf() -> None:
    a = [{"id": "x"}, {"id": "y"}]
    b = [{"id": "y"}, {"id": "z"}]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([a, b], weights=[1.0, 1.0])


def test_rrf_weights_shift_the_ranking() -> None:
    lexical = [{"id": "lex1"}, {"id": "lex2"}]
    dense = [{"id": "den1"}, {"id": "den2"}]
    # equal weights: rank-1 items tie, the lexical list is listed first -> stable order
    equal = _ids(reciprocal_rank_fusion([lexical, dense], weights=[1.0, 1.0]))
    assert set(equal[:2]) == {"lex1", "den1"}
    dense_heavy = _ids(reciprocal_rank_fusion([lexical, dense], weights=[0.25, 1.0]))
    assert dense_heavy[:2] == ["den1", "den2"]  # dense list now outvotes the lexical list
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([lexical, dense], weights=[1.0])


def test_dedup_keeps_distinct_passages_that_share_a_header(monkeypatch: pytest.MonkeyPatch) -> None:
    header = "Title: a long paper title shared by several passages of the same paper. " * 3
    cands = [
        {"id": "1", "content": header + "First passage talks about alpha."},
        {"id": "2", "content": header + "Second passage talks about beta."},
        {"id": "3", "content": header + "First passage talks about alpha."},  # exact duplicate text
    ]
    monkeypatch.setattr(settings, "dedup_mode", "content_hash")
    assert [c["id"] for c in deduplicate_candidates(cands)] == ["1", "2"]
    monkeypatch.setattr(settings, "dedup_mode", "prefix120")
    assert [c["id"] for c in deduplicate_candidates(cands)] == ["1"]  # the old rule lost passage 2


@pytest.mark.asyncio
async def test_okapi_bm25_matches_reference_formula() -> None:
    import numpy as np
    from eval import pg

    from deep_context.core.types import Chunk, ChunkLevel, Document, RetrievalFilters

    if not await pg.server_available():
        pytest.skip("PostgreSQL is not reachable")
    await pg.ensure_database()
    schema = f"test_{uuidlib.uuid4().hex[:8]}"
    await pg.ensure_schema(schema)
    store = pg.make_store_class()(pg.dsn_for(pg.EVAL_DB, schema))
    texts = [
        "alpha alpha alpha beta",
        "alpha beta beta gamma gamma gamma gamma",
        "gamma delta",
        "delta delta epsilon",
        "zeta eta theta iota kappa lambda",
    ]
    try:
        await store.initialize()
        for i, text in enumerate(texts):
            doc = Document(id=pg.doc_uuid(f"d{i}"), title=f"T{i}", doc_type="text")
            await store.insert_document(doc)
            await store.insert_chunks(
                [
                    Chunk(
                        id=str(uuidlib.uuid4()),
                        document_id=doc.id,
                        level=ChunkLevel.CHILD,
                        content=text,
                        token_count=len(text.split()),
                        embedding=np.zeros(768, dtype=np.float32).tolist(),
                    )
                ]
            )
        monkeypatch_value = settings.lexical_ranking
        settings.lexical_ranking = "bm25"
        try:
            hits = await store.search_bm25("alpha gamma", RetrievalFilters(), limit=10)
        finally:
            settings.lexical_ranking = monkeypatch_value

        n = len(texts)
        avgdl = sum(len(t.split()) for t in texts) / n
        k1, b = settings.bm25_k1, settings.bm25_b

        def ref(text: str) -> float:
            words = text.split()
            total = 0.0
            for term in ("alpha", "gamma"):
                df = sum(1 for t in texts if term in t.split())
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = words.count(term)
                if tf:
                    total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / avgdl))
            return total

        got = {h["content"]: h["score"] for h in hits}
        assert set(got) == {texts[0], texts[1], texts[2]}  # only chunks containing a query term
        for text, score in got.items():
            assert score == pytest.approx(ref(text), rel=1e-6)
        assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]
    finally:
        await store.close()
        await pg.drop_schema(schema)
