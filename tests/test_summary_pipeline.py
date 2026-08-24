"""Integration tests for SummaryIngestionPipeline and parent-child + summary persistence."""

from __future__ import annotations

import pytest

from deep_context.core.types import IngestRequest, RetrievalMode
from deep_context.ingestion.summary_pipeline import SummaryIngestionPipeline
from deep_context.storage import get_storage


@pytest.mark.asyncio
async def test_summary_ingestion_pipeline_end_to_end() -> None:
    storage = await get_storage()
    pipeline = SummaryIngestionPipeline(storage=storage)

    content = """# Deep Context Platform Architecture

## Ingestion Layer
The ingestion layer extracts structured sections from PDFs, Markdown, and code ASTs.
It partitions documents into parent chunks (1,000 to 2,500 tokens) for generation context
and child chunks (300 to 600 tokens) for high-precision embedding search.

## Summarization Mechanism
Qwen3 local models generate semantic summaries for each child chunk.
These summaries are indexed in PostgreSQL using a dedicated tsvector column (summary_tsv)
and GIN indexes to supercharge keyword and entity matching across abbreviations.

## Hybrid Retrieval
Reciprocal Rank Fusion blends BM25 tsvector scoring with pgvector cosine similarity.
A cross-encoder reranker further refines candidate precision before context assembly.
"""

    req = IngestRequest(
        title="Summary Pipeline Test Document",
        content=content,
        doc_type="markdown",
        generate_summaries=True,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    resp = await pipeline.ingest(req)
    assert resp.document_id is not None
    assert resp.parent_chunks_count >= 1
    assert resp.child_chunks_count >= 1
    assert resp.summaries_generated_count == resp.child_chunks_count

    # Verify chunks in storage
    doc = await storage.get_document(resp.document_id)
    assert doc is not None
    assert doc.title == req.title

    parents = await storage.get_document_chunks(resp.document_id, level="parent")
    assert len(parents) == resp.parent_chunks_count

    child_chunks = await storage.get_document_chunks(resp.document_id, level="child")
    assert len(child_chunks) == resp.child_chunks_count

    for c in child_chunks:
        assert c["parent_chunk_id"] is not None
        assert c.get("summary_text") is not None and len(c["summary_text"]) > 0
        assert c.get("summary_tokens") is not None and c["summary_tokens"] > 0
        assert c.get("summary_model") is not None
