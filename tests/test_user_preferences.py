"""Tests for User Preference management (embedding models, dimensions, rerankers)."""

import pytest
from httpx import ASGITransport, AsyncClient

from deep_context.api.app import app
from deep_context.core.types import IngestRequest
from deep_context.ingestion.pipeline import ingestion_pipeline
from deep_context.memory.stores import MemoryStoreManager
from deep_context.retrieval.engine import retrieval_engine
from deep_context.storage import get_storage


@pytest.mark.asyncio
async def test_memory_store_embedding_preferences() -> None:
    storage = await get_storage()
    mgr = MemoryStoreManager(storage)

    # Set user preference
    await mgr.set_embedding_preferences(
        user_id="engineer_alice",
        embedding_model="gemini-embedding-2",
        embedding_dim=768,
        reranker="ecohash",
        llm_model="gemini-3.7-flash",
    )

    prefs = await mgr.get_embedding_preferences("engineer_alice")
    assert prefs["embedding_model"] == "gemini-embedding-2"
    assert prefs["embedding_dim"] == 768
    assert prefs["reranker"] == "ecohash"
    assert prefs["llm_model"] == "gemini-3.7-flash"


@pytest.mark.asyncio
async def test_preferences_api_endpoints() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # POST preference
        post_res = await client.post(
            "/v1/preferences",
            json={
                "user_id": "user_bob",
                "embedding_model": "gemini-embedding-2",
                "embedding_dim": 1536,
                "reranker": "cross_encoder",
                "llm_model": "gemini-3.7-flash",
            },
        )
        assert post_res.status_code == 200
        data = post_res.json()
        assert data["user_id"] == "user_bob"
        assert data["embedding_model"] == "gemini-embedding-2"
        assert data["embedding_dim"] == 1536

        # GET preference
        get_res = await client.get("/v1/preferences?user_id=user_bob")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["embedding_model"] == "gemini-embedding-2"
        assert get_data["embedding_dim"] == 1536


@pytest.mark.asyncio
async def test_retrieval_resolves_user_preferences() -> None:
    storage = await get_storage()
    mgr = MemoryStoreManager(storage)

    # Ingest document with Gemini embeddings
    doc_req = IngestRequest(
        title="Gemini Preference Guide",
        content="Google Gemini embedding 2 supports Matryoshka Representation Learning with 768 dimensions.",
        doc_type="text",
        embedding_model="gemini-embedding-2",
        embedding_dim=768,
    )
    await ingestion_pipeline.ingest(doc_req)

    # Set user preference for user_carol
    await mgr.set_embedding_preferences(
        user_id="user_carol",
        embedding_model="gemini-embedding-2",
        embedding_dim=768,
        reranker="cross_encoder",
    )

    # Retrieve without passing explicit model - should automatically resolve user_carol preferences
    res = await retrieval_engine.retrieve(
        query="What dimensions does Gemini embedding 2 support?",
        user_id="user_carol",
        top_k=3,
    )
    assert res.sufficient is True
    assert len(res.parent_chunks) > 0
