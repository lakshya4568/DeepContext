"""Unit tests for IBM Docling parser integration and fallback mechanisms."""

from __future__ import annotations

import io

from pypdf import PdfWriter

from deep_context.ingestion.parser import DocumentParser


def create_synthetic_pdf(num_pages: int = 2) -> bytes:
    """Generate in-memory multi-page PDF."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_docling_markdown_parser_with_tables() -> None:
    """Verify structured parsing of tables and headers produced by Docling."""
    markdown_doc = """# Financial Overview 2024

Here is the operational summary for the fiscal year.

## Quarterly Performance Table

| Quarter | Revenue ($M) | Operating Margin | Growth YoY |
|---------|--------------|------------------|------------|
| Q1      | 112.5        | 28.4%            | +12%       |
| Q2      | 118.2        | 29.1%            | +14%       |
| Q3      | 124.0        | 30.5%            | +18%       |
| Q4      | 131.7        | 31.2%            | +20%       |

## Analysis and Projections
The operating margins showed steady growth due to cloud optimization.
"""
    sections = DocumentParser.parse_markdown(markdown_doc)
    assert len(sections) >= 2

    # Check table preservation
    table_section = next((s for s in sections if "Quarterly Performance Table" in s.title), None)
    assert table_section is not None
    assert "| Quarter | Revenue ($M) |" in table_section.content
    assert "| Q4      | 131.7        |" in table_section.content


def test_pdf_parsing_fallback_and_structure() -> None:
    """Verify PDF parsing works cleanly with fallback support."""
    pdf_bytes = create_synthetic_pdf(2)
    sections = DocumentParser.parse_pdf(pdf_bytes, use_docling=False)
    assert len(sections) >= 1
    assert sections[0].page_number is not None


def test_docling_html_parsing() -> None:
    """Verify HTML documents are converted cleanly via Docling / HTML parser into Markdown sections."""
    html_content = """
    <html>
    <head><title>System Architecture</title></head>
    <body>
        <h1>Storage Layer</h1>
        <p>PostgreSQL with pgvector is used for persistence.</p>
        <h2>Cache Layer</h2>
        <p>Redis is used for caching frequent queries.</p>
    </body>
    </html>
    """
    sections = DocumentParser.parse(html_content, doc_type="html")
    assert len(sections) >= 2
    titles = [s.title for s in sections]
    assert any("Storage Layer" in t for t in titles)
    assert any("Cache Layer" in t for t in titles)


def test_docling_markdown_parser_tagging() -> None:
    """Verify Docling parsing tags sections with metadata."""
    md_content = "# Docling Test\n\nTesting metadata tagging."
    sections = DocumentParser.parse(md_content, doc_type="markdown")
    assert len(sections) >= 1
    assert sections[0].metadata.get("parser") == "ibm_docling"
    assert sections[0].metadata.get("format") == "md"
