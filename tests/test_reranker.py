"""Tests for Multi-Strategy Reranker (Cross-Encoder, Gemini Semantic, Gemini LLM)."""

import pytest

from deep_context.retrieval.reranker import (
    CrossEncoderReranker,
    GeminiSemanticReranker,
    Reranker,
)


@pytest.mark.asyncio
async def test_cross_encoder_reranker_exact_needle_bonus() -> None:
    query = "What is the secret token for the admin vault?"
    candidates = [
        {"id": "1", "content": "General database documentation and user guides.", "score": 0.9},
        {
            "id": "2",
            "content": "Security Notice: secret token for the admin vault is VAULT-9981.",
            "score": 0.4,
        },
        {"id": "3", "content": "Network topology and subnet mask definitions.", "score": 0.8},
    ]

    reranked = await CrossEncoderReranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    # Candidate 2 has exact needle match so must be ranked #1
    assert reranked[0]["id"] == "2"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


@pytest.mark.asyncio
async def test_gemini_semantic_reranker() -> None:
    query = "Explain the RLM subagent message passing protocol"
    candidates = [
        {"id": "1", "content": "Recipe for baking chocolate chip cookies."},
        {
            "id": "2",
            "content": "RLM subagents communicate asynchronously by sending structured messages to parent orchestrator.",
        },
        {"id": "3", "content": "Weather forecast for London and Tokyo."},
    ]

    reranked = await GeminiSemanticReranker.rerank(
        query=query,
        candidates=candidates,
        top_k=2,
        embedding_model="gemini-embedding-2",
        embedding_dim=768,
    )
    assert len(reranked) == 2
    assert reranked[0]["id"] == "2"
    assert "gemini_cos_sim" in reranked[0]


@pytest.mark.asyncio
async def test_unified_reranker_strategy_dispatch() -> None:
    query = "Kafka event bus consumers"
    candidates = [
        {
            "id": "1",
            "content": "Kafka event bus consumers subscribe to topic partitions and process batch records.",
        },
        {"id": "2", "content": "Random unrelated text."},
    ]

    # Test cross_encoder dispatch
    res_ce = await Reranker.rerank(query, candidates, strategy="cross_encoder")
    assert len(res_ce) == 2
    assert res_ce[0]["id"] == "1"

    # Test gemini_semantic dispatch
    res_sem = await Reranker.rerank(query, candidates, strategy="gemini_semantic")
    assert len(res_sem) == 2
    assert res_sem[0]["id"] == "1"
