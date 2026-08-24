"""Unit tests for lazy-loaded ChunkSummarizer and Qwen3 integration."""

from __future__ import annotations

import pytest

from deep_context.core.types import Chunk, ChunkLevel
from deep_context.ingestion.summarizer import ChunkSummarizer


def test_summarizer_lazy_initialization() -> None:
    """Verify that creating a ChunkSummarizer does not immediately load weights into memory."""
    summarizer = ChunkSummarizer(model_name="Qwen/Qwen3-0.6B")
    assert summarizer._is_loaded is False
    assert summarizer._model is None
    assert summarizer._tokenizer is None


def test_summarizer_prompt_construction() -> None:
    """Verify instruct prompt formatting for Qwen."""
    summarizer = ChunkSummarizer()
    prompt = summarizer._build_prompt(
        chunk_text="FastAPI is a modern, fast web framework for building APIs with Python.",
        context_prefix="Architecture > Presentation Layer",
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "Topic / Path: Architecture > Presentation Layer" in prompt
    assert "FastAPI is a modern" in prompt
    assert "<|im_start|>assistant\n<think>\n</think>" in prompt


@pytest.mark.asyncio
async def test_summarize_empty_chunk() -> None:
    summarizer = ChunkSummarizer()
    summary, tokens = await summarizer.summarize_chunk("   ")
    assert summary == ""
    assert tokens == 0


@pytest.mark.asyncio
async def test_summarize_chunk_execution() -> None:
    summarizer = ChunkSummarizer()
    text = (
        "PostgreSQL with the pgvector extension provides transactional storage for both structured metadata "
        "and dense vector embeddings. It uses an HNSW index to achieve sub-millisecond nearest neighbor search."
    )
    summary, tokens = await summarizer.summarize_chunk(text, context_prefix="Database Architecture")
    assert len(summary) > 0
    assert tokens > 0
    # Lazy load flag should now be True
    assert summarizer._is_loaded is True
    assert summarizer._model is not None


@pytest.mark.asyncio
async def test_summarize_chunks_in_place() -> None:
    summarizer = ChunkSummarizer()
    chunks = [
        Chunk(
            id="c1",
            document_id="doc1",
            level=ChunkLevel.CHILD,
            content="Reciprocal Rank Fusion fuses ranked lists from BM25 and vector search with score 1 / (60 + rank).",
            section_path="Retrieval > Hybrid Fusion",
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            level=ChunkLevel.CHILD,
            content="Cross-encoder rerankers evaluate the full cross-attention between query and candidate documents.",
            section_path="Retrieval > Reranking",
        ),
    ]

    updated = await summarizer.summarize_chunks(chunks)
    assert len(updated) == 2
    for c in updated:
        assert c.summary_text is not None and len(c.summary_text) > 0
        assert c.summary_tokens is not None and c.summary_tokens > 0
        assert c.summary_model == summarizer.model_name
        assert c.generated_at is not None
