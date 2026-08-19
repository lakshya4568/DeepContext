"""Tests for LocalCrossEncoderReranker (BGE-reranker-v2-m3)."""

import pytest

from deep_context.retrieval.reranker import LocalCrossEncoderReranker, Reranker


@pytest.mark.asyncio
async def test_local_cross_encoder_reranks_semantically() -> None:
    """BGE reranker should rank a semantically relevant chunk above an irrelevant one."""
    query = "What is the capital of France?"
    candidates = [
        {"id": "1", "content": "Paris is the capital and most populous city of France."},
        {"id": "2", "content": "The recipe calls for two cups of flour and one teaspoon of salt."},
        {"id": "3", "content": "Berlin is the capital of Germany."},
    ]

    reranked = await LocalCrossEncoderReranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["id"] == "1"
    assert "rerank_score" in reranked[0]


@pytest.mark.asyncio
async def test_local_cross_encoder_handles_paraphrase() -> None:
    """BGE reranker should handle paraphrased queries better than lexical overlap."""
    query = "How do I reset my password?"
    candidates = [
        {
            "id": "1",
            "content": "To change your credentials, navigate to Settings > Security and click 'Reset Password'.",
        },
        {"id": "2", "content": "Our office is closed on weekends and public holidays."},
    ]

    reranked = await LocalCrossEncoderReranker.rerank(query, candidates, top_k=1)
    assert reranked[0]["id"] == "1"


@pytest.mark.asyncio
async def test_reranker_dispatches_to_local_cross_encoder() -> None:
    """The unified Reranker should route 'local_cross_encoder' strategy correctly."""
    query = "What is photosynthesis?"
    candidates = [
        {
            "id": "1",
            "content": "Photosynthesis is the process by which plants convert sunlight into energy.",
        },
        {"id": "2", "content": "The stock market closed higher on Tuesday."},
    ]

    reranked = await Reranker.rerank(query, candidates, strategy="local_cross_encoder", top_k=1)
    assert reranked[0]["id"] == "1"
