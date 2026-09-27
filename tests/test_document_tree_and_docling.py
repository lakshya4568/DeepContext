"""Comprehensive test suite for DocumentTree, Docling parser integration,
structure-aware chunking, text cleaning, multimodal enrichment, storage, and retrieval context expansion.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from deep_context.core.types import (
    Document,
    DocumentElementType,
    DocumentNode,
    DocumentTree,
    FigureDataModel,
    IngestRequest,
    MultimodalEnrichment,
    Provenance,
    TableCellData,
    TableDataModel,
)
from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.cleaner import TextCleaner
from deep_context.ingestion.parser import DocumentParser
from deep_context.ingestion.pipeline import IngestionPipeline
from deep_context.multimodal.enricher import MultimodalEnricher
from deep_context.retrieval.engine import RetrievalEngine
from deep_context.storage.sqlite_store import SQLiteStore

# ---------------------------------------------------------------------------
# 1. Text Cleaner Unit Tests
# ---------------------------------------------------------------------------


def test_text_cleaner_control_chars_and_whitespace():
    cleaner = TextCleaner()
    dirty = "Hello \x00World\x08! \t This is   a   test.\r\nNext line."
    cleaned = cleaner.clean_text(dirty)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Hello World!" in cleaned
    assert "This is a test." in cleaned


def test_text_cleaner_dehyphenation():
    cleaner = TextCleaner()
    hyphenated = "This is a docu-\nmentation pipeline with super-\ncalifragilistic behavior."
    cleaned = cleaner.clean_text(hyphenated)
    assert "documentation" in cleaned
    assert "supercalifragilistic" in cleaned


def test_text_cleaner_header_footer_suppression():
    cleaner = TextCleaner()
    pages = [
        "Confidential Report\nPage 1 of 3\n\nIntroduction to the project.\n\nFooter note\nPage 1",
        "Confidential Report\nPage 2 of 3\n\nMethods and experiments.\n\nFooter note\nPage 2",
        "Confidential Report\nPage 3 of 3\n\nConclusion and findings.\n\nFooter note\nPage 3",
    ]
    suppressed = cleaner.suppress_headers_footers(pages)
    assert len(suppressed) == 3
    for p in suppressed:
        # Repeated "Confidential Report" header should be suppressed
        assert "Confidential Report" not in p
        # "Footer note" should be suppressed
        assert "Footer note" not in p
    assert "Introduction to the project." in suppressed[0]
    assert "Methods and experiments." in suppressed[1]
    assert "Conclusion and findings." in suppressed[2]


# ---------------------------------------------------------------------------
# 2. DocumentTree & Node Hierarchy Tests
# ---------------------------------------------------------------------------


def test_document_tree_construction_and_hierarchy():
    root = DocumentNode(
        id="root-1",
        document_id="doc-123",
        node_type=DocumentElementType.DOCUMENT,
        text="Root Document",
        reading_order=0,
    )
    sec1 = DocumentNode(
        id="sec-1",
        document_id="doc-123",
        parent_id="root-1",
        node_type=DocumentElementType.SECTION,
        text="Executive Summary",
        reading_order=1,
        children_ids=["p-1", "tbl-1"],
    )
    p1 = DocumentNode(
        id="p-1",
        document_id="doc-123",
        parent_id="sec-1",
        node_type=DocumentElementType.PARAGRAPH,
        text="The project exceeded expectations by 45% in Q3.",
        reading_order=2,
    )
    tbl1 = DocumentNode(
        id="tbl-1",
        document_id="doc-123",
        parent_id="sec-1",
        node_type=DocumentElementType.TABLE,
        text="| Metric | Value |\n|---|---|\n| Growth | 45% |",
        reading_order=3,
        table_data=TableDataModel(
            num_rows=2,
            num_cols=2,
            headers=["Metric", "Value"],
            rows=[["Growth", "45%"]],
            markdown="| Metric | Value |\n|---|---|\n| Growth | 45% |",
            caption="Table 1: Financial Metrics",
        ),
    )

    tree = DocumentTree(
        document_id="doc-123",
        title="Quarterly Review",
        nodes=[root, sec1, p1, tbl1],
        root_node_id="root-1",
    )

    assert tree.get_node("sec-1") is sec1
    assert tree.get_parent("p-1") is sec1
    assert tree.get_parent("sec-1") is root
    assert [a.id for a in tree.get_ancestors("p-1")] == ["sec-1", "root-1"]
    assert [c.id for c in tree.get_children("sec-1")] == ["p-1", "tbl-1"]

    # Test canonical text rendering
    rendered = tree.render_canonical_text()
    assert "## Executive Summary" in rendered
    assert "The project exceeded expectations by 45% in Q3." in rendered
    assert "Table 1: Financial Metrics" in rendered
    assert "| Growth | 45% |" in rendered

    # Test backward-compatible section conversion
    sections = tree.to_parsed_sections()
    assert len(sections) >= 1
    assert any(
        "Executive Summary" in s.title or "Executive Summary" in s.section_path for s in sections
    )
    assert any("45%" in s.content for s in sections)


# ---------------------------------------------------------------------------
# 3. Table Data Model & Oversized Splitting Tests
# ---------------------------------------------------------------------------


def test_table_data_model_serialization():
    cells = [
        TableCellData(row_idx=0, col_idx=0, text="Quarter", is_header=True),
        TableCellData(row_idx=0, col_idx=1, text="Revenue", is_header=True),
        TableCellData(row_idx=1, col_idx=0, text="Q1", is_header=False),
        TableCellData(row_idx=1, col_idx=1, text="$1.2M", is_header=False),
    ]
    model = TableDataModel(
        num_rows=2,
        num_cols=2,
        headers=["Quarter", "Revenue"],
        rows=[["Q1", "$1.2M"]],
        cells=cells,
        caption="Quarterly Revenue Table",
        markdown="| Quarter | Revenue |\n|---|---|\n| Q1 | $1.2M |",
    )

    d = model.to_dict()
    assert d["num_rows"] == 2
    assert d["caption"] == "Quarterly Revenue Table"
    assert len(d["cells"]) == 4

    restored = TableDataModel.from_dict(d)
    assert restored is not None
    assert restored.num_cols == 2
    assert restored.cells[0].is_header is True
    assert restored.cells[2].text == "Q1"


def test_chunker_splits_oversized_table_preserving_headers_and_caption():
    chunker = ParentChildChunker(parent_max_tokens=500, child_max_tokens=60)
    doc_id = "doc-tbl-split"

    # Create large table with 30 rows
    headers = ["ID", "Parameter", "Spec Value", "Tolerance"]
    rows = [
        [f"ID-{i}", f"Param_{i}", f"{i * 10.5:.2f} kHz", f"+/- {i * 0.1:.1f}%"] for i in range(30)
    ]
    table_md = "| " + " | ".join(headers) + " |\n|---|---|---|---|\n"
    for r in rows:
        table_md += "| " + " | ".join(r) + " |\n"

    table_node = DocumentNode(
        id="tbl-large-1",
        document_id=doc_id,
        node_type=DocumentElementType.TABLE,
        reading_order=1,
        text=table_md,
        section_path="Hardware Specifications",
        provenance=Provenance(page_number=4),
        table_data=TableDataModel(
            num_rows=len(rows) + 1,
            num_cols=4,
            headers=headers,
            rows=rows,
            caption="Table 4-A: RF Calibration Parameters",
            markdown=table_md,
        ),
    )

    tree = DocumentTree(
        document_id=doc_id,
        title="Hardware Specs",
        nodes=[table_node],
        root_node_id="tbl-large-1",
    )

    parents, children = chunker.chunk_tree(tree)
    assert len(parents) >= 1
    # Because table is large and child_max_tokens is 60, it should produce multiple child chunks
    assert len(children) >= 2

    # Verify that table splits repeat the headers and keep the caption
    for c in children:
        assert "Table 4-A: RF Calibration Parameters" in c.content
        assert "| ID | Parameter | Spec Value | Tolerance |" in c.content
        assert c.metadata.get("table_caption") == "Table 4-A: RF Calibration Parameters"
        assert c.page_number == 4


# ---------------------------------------------------------------------------
# 4. Multimodal Figure & Chart Association Tests
# ---------------------------------------------------------------------------


def test_figure_data_model_and_multimodal_enrichment():
    from datetime import timezone

    enrichment = MultimodalEnrichment(
        status="completed",
        model_name="gemini-2.5-flash",
        description="Bar chart illustrating 40% growth in Latin America and 25% growth in EMEA.",
        extracted_labels=["Latin America", "EMEA", "Growth Rate"],
        extracted_values=[
            {"region": "Latin America", "rate": 0.40},
            {"region": "EMEA", "rate": 0.25},
        ],
        confidence=0.96,
        processed_at=datetime.now(timezone.utc),
    )
    figure = FigureDataModel(
        asset_id="asset-chart-1",
        caption="Figure 3: Regional Expansion Velocity",
        figure_type="chart",
        chart_metadata={"type": "bar", "x_axis": "Region", "y_axis": "Growth"},
        ocr_text="Regional Velocity 2026",
        enrichment=enrichment,
    )

    d = figure.to_dict()
    assert d["figure_type"] == "chart"
    assert d["caption"] == "Figure 3: Regional Expansion Velocity"
    assert d["enrichment"]["status"] == "completed"
    assert "Latin America" in d["enrichment"]["description"]

    restored = FigureDataModel.from_dict(d)
    assert restored is not None
    assert restored.caption == figure.caption
    assert restored.enrichment is not None
    assert restored.enrichment.confidence == 0.96


@pytest.mark.asyncio
async def test_multimodal_enricher_error_isolation():
    # Enricher should fail gracefully without interrupting ingestion
    def failing_vision(fig, text):
        raise RuntimeError("ResourceExhausted: Quota limit exceeded for model")

    enricher = MultimodalEnricher(vision_fn=failing_vision)

    fig_node = DocumentNode(
        id="fig-fail-1",
        document_id="doc-test-enrich",
        node_type=DocumentElementType.FIGURE,
        reading_order=1,
        text="[Figure]",
        figure_data=FigureDataModel(caption="Figure 1: Architectural Topology"),
    )
    tree = DocumentTree(document_id="doc-test-enrich", nodes=[fig_node])

    enriched_tree = await enricher.enrich_tree(tree, enabled=True)
    assert enriched_tree is tree

    # Figure should have enrichment record with status="failed" and limitation recorded
    enrichment = fig_node.figure_data.enrichment
    assert enrichment is not None
    assert enrichment.status == "failed"
    assert "Quota limit exceeded" in enrichment.limitations
    # Ingestion continues unimpeded!


# ---------------------------------------------------------------------------
# 5. Storage Persistence for DocumentTree Nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_storage_tree_nodes(test_db: SQLiteStore):
    store = test_db

    doc = Document(
        id="doc-tree-db-1",
        title="System Specification",
        doc_type="pdf",
        permission_scope=["engineering"],
    )
    await store.insert_document(doc)

    node1 = DocumentNode(
        id="node-h1",
        document_id=doc.id,
        node_type=DocumentElementType.HEADING,
        reading_order=1,
        text="1. Introduction",
        section_path="1. Introduction",
        provenance=Provenance(page_number=1, bbox=(50.0, 700.0, 200.0, 720.0)),
    )
    node2 = DocumentNode(
        id="node-p1",
        document_id=doc.id,
        parent_id="node-h1",
        node_type=DocumentElementType.PARAGRAPH,
        reading_order=2,
        text="The system operates in real-time under sub-50ms latency SLAs.",
        section_path="1. Introduction",
        provenance=Provenance(page_number=1, bbox=(50.0, 650.0, 500.0, 690.0)),
    )
    node3 = DocumentNode(
        id="node-tbl1",
        document_id=doc.id,
        parent_id="node-h1",
        node_type=DocumentElementType.TABLE,
        reading_order=3,
        text="| SLA | Target |\n|---|---|\n| Latency | <50ms |",
        section_path="1. Introduction",
        provenance=Provenance(page_number=2),
        table_data=TableDataModel(
            num_rows=2,
            num_cols=2,
            headers=["SLA", "Target"],
            rows=[["Latency", "<50ms"]],
            caption="Table 1: Latency Targets",
            markdown="| SLA | Target |\n|---|---|\n| Latency | <50ms |",
        ),
    )

    ids = await store.insert_tree_nodes([node1, node2, node3])
    assert len(ids) == 3

    # Query back
    fetched_nodes = await store.get_tree_nodes(doc.id)
    assert len(fetched_nodes) == 3
    assert fetched_nodes[0].id == "node-h1"
    assert fetched_nodes[0].node_type == DocumentElementType.HEADING
    assert fetched_nodes[0].provenance.bbox == (50.0, 700.0, 200.0, 720.0)

    # Check table node
    tbl_fetched = await store.get_tree_node("node-tbl1")
    assert tbl_fetched is not None
    assert tbl_fetched.table_data is not None
    assert tbl_fetched.table_data.caption == "Table 1: Latency Targets"
    assert tbl_fetched.table_data.headers == ["SLA", "Target"]

    # Test cascade deletion
    await store.delete_document(doc.id)
    remaining = await store.get_tree_nodes(doc.id)
    assert len(remaining) == 0


# ---------------------------------------------------------------------------
# 6. End-to-End Pipeline & Retrieval Context Expansion Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_with_tree_chunking_and_retrieval_expansion(test_db: SQLiteStore):
    store = test_db
    pipeline = IngestionPipeline()
    retrieval_engine = RetrievalEngine()

    content = """# Deep Space Communications

The deep space network handles communications with interstellar probes.

| Band | Frequency | Uplink Power |
|---|---|---|
| X-band | 7.145 GHz | 20 kW |
| Ka-band | 34.2 GHz | 800 W |

*Table 1: Deep Space Frequency Bands*

Recent telemetry confirmed signal acquisition with the Voyager probes.
"""

    req = IngestRequest(
        title="Deep Space Communications Guide",
        content=content,
        doc_type="markdown",
        tenant_id="nasa",
        permission_scope=["default"],
    )

    res = await pipeline.ingest(req)
    assert res.document_id is not None
    assert res.parent_chunks_count >= 1
    assert res.child_chunks_count >= 1
    assert res.tree_nodes_count >= 1

    # Check that tree nodes were persisted
    nodes = await store.get_tree_nodes(res.document_id)
    assert len(nodes) >= 3

    # Fetch child chunks
    chunks_detail = await store.get_document_chunks_detail(res.document_id)
    child_chunks = [c for c in chunks_detail if c["level"] == "child"]
    assert len(child_chunks) >= 1

    target_chunk_id = child_chunks[0]["id"]

    # Perform Retrieval Context Expansion!
    expanded = await retrieval_engine.expand_chunk_context(target_chunk_id)
    assert expanded["chunk_id"] == target_chunk_id
    assert expanded["document_id"] == res.document_id
    assert len(expanded["nodes"]) >= 1

    # Check table resolution if table chunk
    tbl_chunk = next((c for c in child_chunks if "X-band" in c["content"]), None)
    if tbl_chunk:
        tbl_expanded = await retrieval_engine.expand_chunk_context(tbl_chunk["id"])
        assert len(tbl_expanded["tables"]) >= 1
        table_info = tbl_expanded["tables"][0]
        assert "X-band" in table_info["markdown"]


# ---------------------------------------------------------------------------
# 7. Resilient Fallbacks & Edge Cases
# ---------------------------------------------------------------------------


def test_parser_empty_and_whitespace_documents():
    parser = DocumentParser()

    empty_tree = parser.parse_tree("", doc_type="markdown", title="Empty Doc")
    assert len(empty_tree.nodes) == 1
    assert empty_tree.nodes[0].node_type == DocumentElementType.DOCUMENT

    ws_tree = parser.parse_tree("   \n\n\t   \n", doc_type="markdown", title="Whitespace Doc")
    assert len(ws_tree.nodes) == 1


def test_parser_code_tree():
    parser = DocumentParser()
    py_code = """
import os

def authenticate_user(token: str) -> bool:
    '''Validates user token.'''
    return token == "secret"

class ServiceWorker:
    def start(self):
        pass
"""
    tree = parser.parse_tree(py_code, doc_type="code")
    assert any(
        n.node_type == DocumentElementType.CODE and n.metadata.get("symbol") == "authenticate_user"
        for n in tree.nodes
    )
    assert any(
        n.node_type == DocumentElementType.CODE and n.metadata.get("symbol") == "ServiceWorker"
        for n in tree.nodes
    )

    sections = tree.to_parsed_sections()
    assert any("authenticate_user" in s.title for s in sections)
    assert any("ServiceWorker" in s.title for s in sections)
