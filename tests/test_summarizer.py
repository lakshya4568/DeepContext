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
    """Verify instruct prompt formatting for Qwen with Anthropic document and chunk context."""
    summarizer = ChunkSummarizer()
    prompt = summarizer._build_prompt(
        chunk_text="FastAPI is a modern, fast web framework for building APIs with Python.",
        section_path="Architecture > Presentation Layer",
        document_title="Deep Context Architecture Guide",
        parent_context="The system uses FastAPI for all REST API endpoints and SSE streaming.",
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<document>" in prompt
    assert "Document: Deep Context Architecture Guide" in prompt
    assert "Section: Architecture > Presentation Layer" in prompt
    assert "Enclosing Section Context:\nThe system uses FastAPI" in prompt
    assert "<chunk>\nFastAPI is a modern" in prompt
    assert "Please give a short succinct context to situate this chunk" in prompt
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
    summary, tokens = await summarizer.summarize_chunk(
        text,
        section_path="Database Architecture",
        document_title="Storage Specifications",
        parent_context="PostgreSQL 16 serves as the primary relational and vector persistence store.",
    )
    assert len(summary) > 0
    assert tokens > 0
    # Lazy load flag should now be True
    assert summarizer._is_loaded is True
    assert summarizer._model is not None


@pytest.mark.asyncio
async def test_summarize_chunks_in_place() -> None:
    summarizer = ChunkSummarizer()
    parent_chunk = Chunk(
        id="p1",
        document_id="doc1",
        level=ChunkLevel.PARENT,
        content="Retrieval layer uses Reciprocal Rank Fusion and cross-encoder reranking.",
        section_path="Retrieval",
    )
    parent_map = {"p1": parent_chunk}

    chunks = [
        Chunk(
            id="c1",
            document_id="doc1",
            parent_chunk_id="p1",
            level=ChunkLevel.CHILD,
            content="Reciprocal Rank Fusion fuses ranked lists from BM25 and vector search with score 1 / (60 + rank).",
            section_path="Retrieval > Hybrid Fusion",
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            parent_chunk_id="p1",
            level=ChunkLevel.CHILD,
            content="Cross-encoder rerankers evaluate the full cross-attention between query and candidate documents.",
            section_path="Retrieval > Reranking",
        ),
    ]

    updated = await summarizer.summarize_chunks(
        chunks, parent_chunk_map=parent_map, document_title="Platform Architecture"
    )
    assert len(updated) == 2
    for c in updated:
        assert c.summary_text is not None and len(c.summary_text) > 0
        assert c.summary_tokens is not None and c.summary_tokens > 0
        assert c.summary_model == summarizer.model_name
        assert c.generated_at is not None


def test_clean_and_complete_summary_deduplication() -> None:
    """Verify that sentence deduplication eliminates repeated sentences in summary output."""
    raw = (
        "The document discusses the system architecture and database design. "
        "The document discusses the system architecture and database design. "
        "PostgreSQL pgvector is used for dense embeddings."
    )
    cleaned = ChunkSummarizer._clean_and_complete_summary(raw)
    assert cleaned.count("The document discusses the system architecture and database design.") == 1
    assert "PostgreSQL pgvector is used for dense embeddings." in cleaned
