import pytest

from deep_context.retrieval.ecohash_reranker import EcoHashReranker
from deep_context.retrieval.reranker import Reranker


@pytest.mark.asyncio
async def test_ecohash_reranker_semantic_ranking():
    candidates = [
        {
            "id": "c1",
            "content": "The weather forecast predicts sunny skies and mild temperatures.",
            "chunk_index": 0,
        },
        {
            "id": "c2",
            "content": "Robb Stark was crowned King in the North by Greatjon Umber and the Northern lords at Riverrun.",
            "chunk_index": 1,
        },
        {
            "id": "c3",
            "content": "Baking sourdough bread requires flour, water, salt, and active yeast culture.",
            "chunk_index": 2,
        },
    ]

    reranked = await EcoHashReranker.rerank(
        query="Who was named King in the North by the northern lords?",
        candidates=candidates,
        top_k=3,
    )

    assert len(reranked) == 3
    # c2 must be ranked at rank 1 due to high semantic relevance
    assert reranked[0]["id"] == "c2"


@pytest.mark.asyncio
async def test_ecohash_reranker_fallback_on_empty_key(monkeypatch):
    from deep_context.core.config import settings

    monkeypatch.setattr(settings, "ecohash_api_key", "")

    candidates = [
        {"id": "c1", "content": "Bananas and apples.", "chunk_index": 0},
        {"id": "c2", "content": "Kingsguard knights in white armor.", "chunk_index": 1},
    ]

    reranked = await EcoHashReranker.rerank(
        query="Kingsguard white armor",
        candidates=candidates,
        top_k=2,
    )
    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_ecohash_reranker_dispatcher():
    candidates = [
        {"id": "c1", "content": "Robb Stark at Riverrun.", "chunk_index": 0},
        {"id": "c2", "content": "Bananas on a tree.", "chunk_index": 1},
    ]

    reranked = await Reranker.rerank(
        query="Robb Stark Riverrun",
        candidates=candidates,
        strategy="ecohash",
        top_k=2,
    )
    assert len(reranked) == 2
