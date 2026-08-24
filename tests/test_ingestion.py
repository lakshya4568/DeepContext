"""Tests for document parsing, chunking, tree indexing, and ingestion pipeline."""

import pytest

from deep_context.core.types import IngestRequest, RetrievalMode
from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.parser import DocumentParser
from deep_context.ingestion.pipeline import ingestion_pipeline


def test_markdown_parser() -> None:
    content = """# Title Section
This is introductory text.

## Subsection 1
Details about component A.

### Deep Topic
Specific facts about component A sub-feature.

## Subsection 2
Details about component B.
"""
    sections = DocumentParser.parse_markdown(content)
    assert len(sections) >= 3
    assert sections[0].title == "Title Section"
    assert "Subsection 1" in sections[1].section_path


def test_code_parser() -> None:
    code = """
import os

def calculate_score(a: int, b: int) -> int:
    '''Computes sum.'''
    return a + b

class VectorStore:
    def __init__(self):
        self.vectors = []

    def add(self, v):
        self.vectors.append(v)
"""
    sections = DocumentParser.parse_code(code, doc_type="python")
    assert len(sections) >= 2
    assert any("calculate_score" in s.title for s in sections)
    assert any("VectorStore" in s.title for s in sections)


def test_parent_child_chunker() -> None:
    content = "# Main\n" + (
        "This is paragraph text with information about system architecture. " * 30
    )
    sections = DocumentParser.parse_markdown(content)
    chunker = ParentChildChunker(
        child_min_tokens=20,
        child_max_tokens=60,
        parent_min_tokens=100,
        parent_max_tokens=300,
    )
    parents, children = chunker.chunk_sections("doc-123", sections)
    assert len(parents) >= 1
    assert len(children) >= 2
    for c in children:
        assert c.parent_chunk_id == parents[0].id
        assert c.level.value == "child"


@pytest.mark.asyncio
async def test_end_to_end_ingest_pipeline() -> None:
    req = IngestRequest(
        title="Spring Security Guide",
        content="""# Spring Security
Authentication and Authorization architecture in modern web applications.

## JWT Filter Ordering
Ensure authentication filter executes before authorization filter.
JWT token validation occurs in custom OncePerRequestFilter.

## CORS Configuration
CORS origins should be explicitly configured in SecurityFilterChain.
""",
        doc_type="markdown",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    res = await ingestion_pipeline.ingest(req)
    assert res.document_id is not None
    assert res.parent_chunks_count >= 1
    assert res.child_chunks_count >= 1
