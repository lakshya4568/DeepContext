"""Comprehensive integration tests for PostgresStore with pgvector."""

import uuid

import numpy as np
import pytest

from deep_context.core.config import settings
from deep_context.core.types import (
    Chunk,
    ChunkLevel,
    Document,
    RetrievalFilters,
    RetrievalMode,
)
from deep_context.storage.postgres_store import PostgresStore


@pytest.fixture
async def pg_store():
    """Fixture providing an initialized PostgresStore connected to the test database."""
    dsn = settings.postgres_dsn
    store = PostgresStore(dsn=dsn)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_postgres_document_crud(pg_store: PostgresStore):
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        tenant_id="test_pg",
        title="Postgres pgvector Technical Guide",
        source_uri="file:///docs/pgvector_guide.md",
        doc_type="markdown",
        permission_scope=["default", "researchers"],
        retrieval_mode=RetrievalMode.HYBRID,
        metadata={"author": "Engineering Team", "topics": ["Postgres", "pgvector", "HNSW"]},
    )
    res_id = await pg_store.insert_document(doc)
    assert res_id == doc_id

    # Fetch
    fetched = await pg_store.get_document(doc_id)
    assert fetched is not None
    assert fetched.title == "Postgres pgvector Technical Guide"
    assert "researchers" in fetched.permission_scope

    # List
    docs = await pg_store.list_documents(tenant_id="test_pg", permission_scope=["researchers"])
    assert any(d.id == doc_id for d in docs)

    # Summaries
    summaries = await pg_store.list_document_summaries(limit=10)
    assert any(s["id"] == doc_id for s in summaries)

    # Delete
    deleted = await pg_store.delete_document(doc_id)
    assert deleted is True
    assert await pg_store.get_document(doc_id) is None


@pytest.mark.asyncio
async def test_postgres_pgvector_similarity_search(pg_store: PostgresStore):
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        tenant_id="pg_vector_test",
        title="Vector Benchmarks",
        doc_type="markdown",
        permission_scope=["default"],
    )
    await pg_store.insert_document(doc)

    parent_id = str(uuid.uuid4())
    parent_chunk = Chunk(
        id=parent_id,
        document_id=doc_id,
        parent_chunk_id=None,
        level=ChunkLevel.PARENT,
        content="# Section 1: HNSW Multilayer Graph Indexing\nHNSW provides fast approximate search.",
        token_count=120,
        section_path="1. HNSW Indexing",
    )

    child_id_1 = str(uuid.uuid4())
    child_id_2 = str(uuid.uuid4())

    # Build 768-dim normalized test vectors
    vec1 = [0.0] * 768
    vec1[0] = 1.0
    vec1[1] = 0.5
    vec1 = (np.array(vec1) / np.linalg.norm(vec1)).tolist()

    vec2 = [0.0] * 768
    vec2[100] = 1.0
    vec2 = (np.array(vec2) / np.linalg.norm(vec2)).tolist()

    c1 = Chunk(
        id=child_id_1,
        document_id=doc_id,
        parent_chunk_id=parent_id,
        level=ChunkLevel.CHILD,
        content="pgvector executes cosine similarity search via 1 - (embedding <=> $1).",
        token_count=35,
        section_path="1. HNSW Indexing > 1.1 Cosine",
        embedding=vec1,
    )
    c2 = Chunk(
        id=child_id_2,
        document_id=doc_id,
        parent_chunk_id=parent_id,
        level=ChunkLevel.CHILD,
        content="Unrelated database topic on disk encryption and replication.",
        token_count=25,
        section_path="1. HNSW Indexing > 1.2 Unrelated",
        embedding=vec2,
    )

    await pg_store.insert_chunks([parent_chunk, c1, c2])

    c_child, c_parent = await pg_store.count_chunks_for_document(doc_id)
    assert c_child == 2
    assert c_parent == 1

    # Query with vector close to vec1
    query_vec = [0.0] * 768
    query_vec[0] = 0.95
    query_vec[1] = 0.45
    query_vec = (np.array(query_vec) / np.linalg.norm(query_vec)).tolist()

    filters = RetrievalFilters(tenant_id="pg_vector_test", permission_scope=["default"])
    results = await pg_store.search_vector(query_vec, filters=filters, limit=5)

    assert len(results) >= 1
    top = results[0]
    assert top["id"] == child_id_1
    assert top["score"] > 0.98

    # Clean up
    await pg_store.delete_document(doc_id)


@pytest.mark.asyncio
async def test_postgres_bm25_text_search(pg_store: PostgresStore):
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        tenant_id="bm25_test",
        title="Unique Keyword Guide",
        doc_type="markdown",
        permission_scope=["default"],
    )
    await pg_store.insert_document(doc)

    parent_id = str(uuid.uuid4())
    parent_chunk = Chunk(
        id=parent_id,
        document_id=doc_id,
        level=ChunkLevel.PARENT,
        content="Full text regarding cryptographic salt and pepper hashing.",
        token_count=80,
    )

    child_id = str(uuid.uuid4())
    child_chunk = Chunk(
        id=child_id,
        document_id=doc_id,
        parent_chunk_id=parent_id,
        level=ChunkLevel.CHILD,
        content="The special cryptographic passcode is SECRET_TOKEN_OMEGA_9918 for authorization.",
        token_count=20,
    )

    await pg_store.insert_chunks([parent_chunk, child_chunk])

    filters = RetrievalFilters(tenant_id="bm25_test", permission_scope=["default"])
    results = await pg_store.search_bm25("SECRET_TOKEN_OMEGA_9918", filters=filters)

    assert len(results) >= 1
    assert results[0]["id"] == child_id

    await pg_store.delete_document(doc_id)


@pytest.mark.asyncio
async def test_postgres_typed_memory(pg_store: PostgresStore):
    test_uid = f"user_{uuid.uuid4().hex[:8]}"
    test_tid = f"tenant_{uuid.uuid4().hex[:8]}"

    # Policy
    await pg_store.set_policy(
        tenant_id=test_tid,
        policy_key="max_discount_rate",
        policy_value={"max_percent": 15, "currency": "INR"},
    )
    pol = await pg_store.get_policy(tenant_id=test_tid, policy_key="max_discount_rate")
    assert pol is not None
    assert pol["policy_value"]["max_percent"] == 15

    # Preference
    await pg_store.set_preference(
        user_id=test_uid,
        preference_key="theme",
        preference_value={"mode": "dark", "accent": "cyan"},
    )
    pref = await pg_store.get_preference(user_id=test_uid, preference_key="theme")
    assert pref is not None
    assert pref["preference_value"]["accent"] == "cyan"

    # Semantic Fact with pgvector
    fact_vec = [0.0] * 768
    fact_vec[5] = 1.0
    fact_id = await pg_store.insert_fact(
        tenant_id=test_tid,
        user_id=test_uid,
        content="User has configured PostgreSQL with pgvector for local embeddings.",
        embedding=fact_vec,
        source="user_stated",
        confidence=0.99,
    )
    assert fact_id is not None

    facts = await pg_store.search_facts(
        query="PostgreSQL pgvector embeddings",
        query_embedding=fact_vec,
        tenant_id=test_tid,
        user_id=test_uid,
    )
    assert len(facts) >= 1
    assert any(f["id"] == fact_id for f in facts)

    # Episode with pgvector
    ep_id = await pg_store.insert_episode(
        user_id=test_uid,
        summary="User built a high-performance RAG with pgvector HNSW indexing.",
        task_type="rag_setup",
        outcome="success",
        embedding=fact_vec,
    )
    episodes = await pg_store.search_episodes(user_id=test_uid, query_embedding=fact_vec)
    assert len(episodes) >= 1
    assert any(e["id"] == ep_id for e in episodes)


@pytest.mark.asyncio
async def test_postgres_event_traces(pg_store: PostgresStore):
    sess_id = str(uuid.uuid4())
    # Event trace
    trace_id = await pg_store.insert_event_trace(
        event_type="hybrid_retrieval",
        payload={"query": "pgvector", "hits": 5},
        session_id=sess_id,
        token_cost=150,
        latency_ms=8,
    )
    assert trace_id > 0

    traces = await pg_store.list_event_traces(session_id=sess_id)
    assert len(traces) >= 1
    assert traces[0]["event_type"] == "hybrid_retrieval"


@pytest.mark.asyncio
async def test_postgres_optimized_indexes_and_tsv_trigger(pg_store: PostgresStore):
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        tenant_id="idx_test_tenant",
        title="Optimization Guide",
        doc_type="markdown",
        permission_scope=["default"],
    )
    await pg_store.insert_document(doc)

    parent_id = str(uuid.uuid4())
    parent_chunk = Chunk(
        id=parent_id,
        document_id=doc_id,
        level=ChunkLevel.PARENT,
        content="Parent chunk explaining database architecture and indexing.",
        token_count=100,
    )

    child_id = str(uuid.uuid4())
    child_chunk = Chunk(
        id=child_id,
        document_id=doc_id,
        parent_chunk_id=parent_id,
        level=ChunkLevel.CHILD,
        content="Detailed technical chunk about cache invalidation algorithms.",
        summary_text="High-level overview of deterministic LRU and TTL cache pruning.",
        summary_model="qwen3-0.6b",
        token_count=30,
    )

    await pg_store.insert_chunks([parent_chunk, child_chunk])

    # 1. Test search_tsv trigger automatically populated search_tsv
    pool = pg_store._get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT search_tsv IS NOT NULL as has_search_tsv FROM chunks WHERE id = $1::uuid",
            child_id,
        )
        assert row is not None
        assert row["has_search_tsv"] is True

    # 2. Test search_bm25 matches terms from content AND summary_text via search_tsv
    filters = RetrievalFilters(tenant_id="idx_test_tenant", permission_scope=["default"])

    # Query matching summary_text
    res_summary = await pg_store.search_bm25("deterministic LRU", filters=filters)
    assert len(res_summary) >= 1
    assert res_summary[0]["id"] == child_id

    # Query matching content
    res_content = await pg_store.search_bm25("invalidation algorithms", filters=filters)
    assert len(res_content) >= 1
    assert res_content[0]["id"] == child_id

    await pg_store.delete_document(doc_id)


@pytest.mark.asyncio
async def test_postgres_ef_search_runtime_parameter(pg_store: PostgresStore):
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        tenant_id="ef_test_tenant",
        title="EF Search Guide",
        doc_type="markdown",
        permission_scope=["default"],
    )
    await pg_store.insert_document(doc)

    parent_id = str(uuid.uuid4())
    parent_chunk = Chunk(
        id=parent_id,
        document_id=doc_id,
        level=ChunkLevel.PARENT,
        content="Parent chunk for EF search testing.",
        token_count=50,
    )

    child_id = str(uuid.uuid4())
    vec = [0.0] * 768
    vec[42] = 1.0
    child_chunk = Chunk(
        id=child_id,
        document_id=doc_id,
        parent_chunk_id=parent_id,
        level=ChunkLevel.CHILD,
        content="Child chunk for vector recall tuning with ef_search.",
        token_count=20,
        embedding=vec,
    )

    await pg_store.insert_chunks([parent_chunk, child_chunk])

    filters = RetrievalFilters(tenant_id="ef_test_tenant", permission_scope=["default"])

    # Test with default ef_search (from settings)
    res1 = await pg_store.search_vector(vec, filters=filters, limit=5)
    assert len(res1) >= 1
    assert res1[0]["id"] == child_id

    # Test with custom ef_search override (e.g. 200)
    res2 = await pg_store.search_vector(vec, filters=filters, limit=5, ef_search=200)
    assert len(res2) >= 1
    assert res2[0]["id"] == child_id

    await pg_store.delete_document(doc_id)
