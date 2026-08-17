"""Tests for multi-page streaming PDF parsing and ingestion."""

import io

import pytest
from pypdf import PdfWriter

from deep_context.core.types import IngestRequest
from deep_context.ingestion.parser import DocumentParser
from deep_context.ingestion.pipeline import ingestion_pipeline


def create_synthetic_pdf(num_pages: int = 5) -> bytes:
    """Generate in-memory multi-page PDF."""
    writer = PdfWriter()
    for i in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_pdf_parser_structure() -> None:
    pdf_bytes = create_synthetic_pdf(3)
    sections = DocumentParser.parse_pdf(pdf_bytes)
    assert len(sections) >= 1
    assert sections[0].page_number is not None


@pytest.mark.asyncio
async def test_pdf_ingest_pipeline() -> None:
    # Test ingesting a multi-page document
    content = """# Page 1: Introduction
Distributed computing architecture overview.

# Page 2: Storage
PostgreSQL and Redis caching layers.

# Page 3: RLM Engine
Recursive Language Model authority bridge.
"""
    req = IngestRequest(
        title="Architecture Guide (Simulated PDF)",
        content=content,
        doc_type="markdown",
    )
    res = await ingestion_pipeline.ingest(req)
    assert res.document_id is not None
    assert res.parent_chunks_count >= 1
    assert res.child_chunks_count >= 1
