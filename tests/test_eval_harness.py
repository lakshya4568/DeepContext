"""Offline tests for the benchmark harness (no network, no API keys)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from eval.metrics import (
    contains_answer,
    exact_match,
    is_abstention,
    rank_metrics,
    rgb_correct,
    token_f1,
)
from eval.stats import bootstrap_ci, paired_test


# --------------------------------------------------------------------------- rank metrics
def test_rank_metrics_hand_computed() -> None:
    # relevant = {a, c}; ranking: x a y c  -> hits at ranks 2 and 4
    m = rank_metrics(["x", "a", "y", "c", "z"], {"a": 1, "c": 1})
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["recall@5"] == 1.0 and m["precision@5"] == pytest.approx(2 / 5)
    assert m["mrr"] == pytest.approx(1 / 2)
    assert m["map"] == pytest.approx((1 / 2 + 2 / 4) / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg@5"] == pytest.approx(dcg / idcg)
    assert m["evidence_all@5"] == 1.0 and m["evidence_all@10"] == 1.0


def test_rank_metrics_partial_evidence_and_no_relevant() -> None:
    m = rank_metrics(["a", "x", "y"], {"a": 1, "b": 1})
    assert m["recall@5"] == 0.5 and m["evidence_all@5"] == 0.0
    assert all(math.isnan(v) for v in rank_metrics(["a"], {}).values())


# --------------------------------------------------------------------------- text metrics
def test_text_metrics() -> None:
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0
    assert token_f1("sam bankman fried", ["Sam Bankman-Fried"]) > 0.0
    assert contains_answer("It was Tampa, Florida in 2021", ["Tampa, Florida"]) == 1.0
    assert rgb_correct("Premiere on Jan 2, 2022", [["January 2 2022", "Jan 2, 2022"]]) == 1.0
    assert rgb_correct("unknown", [["x"], ["y"]]) == 0.0


def test_abstention_detection() -> None:
    from eval.metrics import REFUSAL_TEMPLATE

    assert is_abstention(REFUSAL_TEMPLATE)
    assert is_abstention("")
    assert is_abstention("The provided facts do not contain information about X.")
    long_answer = "The system uses RRF with k=60. " * 30 + "It does not mention pricing."
    assert not is_abstention(long_answer)  # a long substantive answer is not a refusal


# --------------------------------------------------------------------------- statistics
def test_bootstrap_and_paired() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 0.1, 200)
    mean, lo, hi = bootstrap_ci(x, n_boot=2000, seed=0)
    assert lo < mean < hi and hi - lo < 0.1
    better = x + 0.05
    r = paired_test(x, better, n_boot=2000, seed=0)
    assert r.mean_diff == pytest.approx(0.05) and r.p_bootstrap < 0.01 and r.p_wilcoxon < 0.01
    same = paired_test(x, x, n_boot=500, seed=0)
    assert same.mean_diff == 0 and same.p_wilcoxon == 1.0


# --------------------------------------------------------------------------- gold verification
def test_docs_qa_gold_quotes_exist() -> None:
    from eval.docs_qa import ITEMS, verify_gold

    assert verify_gold() == []
    assert len({i["id"] for i in ITEMS}) == len(ITEMS)


# --------------------------------------------------------------------------- vector shim
async def test_fast_vector_search_matches_stock(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from eval.fast_vector import install_fast_vector_search

    from deep_context.core.types import Chunk, ChunkLevel, Document, RetrievalFilters
    from deep_context.storage.sqlite_store import SQLiteStore

    install_fast_vector_search()
    store = SQLiteStore(str(tmp_path / "t.db"))
    await store.initialize()
    rng = random.Random(0)
    dim = 16
    chunks = []
    for d in range(30):
        doc = Document(id=f"doc{d}", title=f"T{d}", doc_type="text")
        await store.insert_document(doc)
        for c in range(2):
            vec = [rng.gauss(0, 1) for _ in range(dim)]
            chunks.append(
                Chunk(id=f"c{d}_{c}", document_id=doc.id, level=ChunkLevel.CHILD,
                      content=f"text {d} {c}", token_count=3, embedding=vec)
            )  # fmt: skip
    await store.insert_chunks(chunks)
    q = [rng.gauss(0, 1) for _ in range(dim)]
    f = RetrievalFilters(tenant_id="default")
    stock = await store._stock_search_vector(q, f, limit=20)  # type: ignore[attr-defined]
    fast = await store.search_vector(q, f, limit=20)
    assert [r["id"] for r in fast] == [r["id"] for r in stock]
    assert [r["score"] for r in fast] == pytest.approx([r["score"] for r in stock], abs=1e-5)
    assert set(fast[0]) == set(stock[0])
    # document filter parity
    f2 = RetrievalFilters(tenant_id="default", document_ids=["doc3", "doc4"])
    assert {r["document_id"] for r in await store.search_vector(q, f2, limit=10)} == {
        "doc3",
        "doc4",
    }
    await store.close()


# --------------------------------------------------------------------------- strict guards
def test_strict_patches_forbid_mock_fallbacks() -> None:
    from eval.strict_clients import MockFallbackError, install_strict_patches

    from deep_context.core.llm_client import llm_client

    install_strict_patches()
    with pytest.raises(MockFallbackError):
        llm_client._mock_embedding("x")
    with pytest.raises(MockFallbackError):
        llm_client._mock_completion([])


async def test_strict_complete_rejects_non_gemini_models() -> None:
    from eval.strict_clients import MockFallbackError, install_strict_patches

    from deep_context.core.llm_client import llm_client

    install_strict_patches()
    with pytest.raises(MockFallbackError):
        await llm_client.complete(
            [{"role": "user", "content": "hi"}], model="meta/llama-3.1-8b-instruct"
        )


def test_embedding_text_format_matches_stock_client() -> None:
    from eval.strict_clients import VertexRuntime

    fmt = VertexRuntime.format_embedding_text
    assert (
        fmt("hello", is_query=True, title=None, task=None) == "task: search result | query: hello"
    )
    assert fmt("body", is_query=False, title="T", task=None) == "title: T | text: body"
    assert fmt("body", is_query=False, title=None, task=None) == "title: none | text: body"


def test_runtime_semaphores_are_per_event_loop() -> None:
    """Regression: RAGAS creates a fresh loop per evaluate(); a reused semaphore broke every batch."""
    import asyncio

    from eval.strict_clients import VertexRuntime

    rt = VertexRuntime.__new__(VertexRuntime)
    rt._sems = {}  # type: ignore[attr-defined]

    async def use() -> None:
        async with rt._sem("llm"):  # type: ignore[attr-defined]
            await asyncio.sleep(0)

    asyncio.run(use())
    asyncio.run(use())  # second loop must not reuse the first loop's semaphore


# --------------------------------------------------------------------------- PostgreSQL backend
def test_pg_helpers_are_deterministic_and_safe(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import uuid

    from eval import pg

    assert pg.doc_uuid("abc") == pg.doc_uuid("abc") != pg.doc_uuid("abd")
    uuid.UUID(pg.doc_uuid("abc"))  # valid UUID
    monkeypatch.setenv("EVAL_PG_DSN", "postgresql://user:secret@127.0.0.1:5432/awems")
    dsn = pg.dsn_for("deepcontext_eval", "scifact_ab12")
    assert dsn.endswith("/deepcontext_eval?search_path=scifact_ab12,public")
    assert "awems" not in dsn  # never the application database
    for bad in ("x; DROP DATABASE y", "UPPER", "a-b"):
        with pytest.raises(ValueError):
            pg.dsn_for(bad)
    assert pg.schema_name("MultiHopRAG_0329 6ea13e").startswith("multihoprag_")


async def test_postgres_store_schema_hnsw_and_search_depth() -> None:
    """Live check against the local server (skipped if unreachable): the harness subclass builds the
    schema on a fresh database, HNSW search width allows LIMIT 100, and the schema is dropped after."""
    import uuid as uuidlib

    import numpy as np
    from eval import pg

    from deep_context.core.types import Chunk, ChunkLevel, Document, RetrievalFilters

    if not await pg.server_available():
        pytest.skip("PostgreSQL is not reachable")
    await pg.ensure_database()
    schema = f"test_{uuidlib.uuid4().hex[:8]}"
    await pg.ensure_schema(schema)
    store = pg.make_store_class()(pg.dsn_for(pg.EVAL_DB, schema))
    try:
        await store.initialize()
        rng = np.random.default_rng(0)
        for i in range(30):
            doc = Document(id=pg.doc_uuid(f"d{i}"), title=f"T{i}", doc_type="text")
            await store.insert_document(doc)
            chunks = []
            for j in range(10):
                v = rng.normal(size=768).astype(np.float32)
                v /= np.linalg.norm(v)
                chunks.append(
                    Chunk(id=str(uuidlib.uuid4()), document_id=doc.id, level=ChunkLevel.CHILD,
                          content=f"alpha beta {i} {j}", token_count=4, embedding=v.tolist())
                )  # fmt: skip
            await store.insert_chunks(chunks)
        q = rng.normal(size=768).astype(np.float32)
        hits = await store.search_vector(q.tolist(), RetrievalFilters(), limit=100)
        assert len(hits) == 100  # would be 40 with pgvector's default hnsw.ef_search
        assert await store.search_bm25("alpha beta", RetrievalFilters(), limit=50)
    finally:
        await store.close()
        await pg.drop_schema(schema)
