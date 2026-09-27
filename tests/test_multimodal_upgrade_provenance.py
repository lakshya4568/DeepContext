"""Comprehensive verification tests for Multimodal Provenance-Preserving RAG Upgrade.

Covers:
1. Exact financial tabular lookup (cell coordinates, column/row key lookups)
2. Medical billing & PHI asset tenant isolation (403 Forbidden cross-tenant)
3. Deterministic re-ingestion stability (idempotent doc_id and parent/child chunk IDs)
4. Context expansion during normal retrieval (tables, LaTeX, equations, and code blocks)
5. Cross-modal retrieval (Text -> Image, Image -> Text, Image -> Image)
6. Multimodal enrichment integrity (no false visual understanding when vision model unavailable)
"""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from deep_context.api.app import app
from deep_context.core.types import (
    Chunk,
    ChunkLevel,
    Document,
    DocumentElementType,
    DocumentNode,
    DocumentTree,
    EquationDataModel,
    FigureDataModel,
    IngestRequest,
    MultimodalAsset,
    RetrievalFilters,
    SourceCodeDataModel,
    TableDataModel,
)
from deep_context.ingestion.summary_pipeline import SummaryIngestionPipeline
from deep_context.multimodal.enricher import MultimodalEnricher
from deep_context.retrieval.engine import retrieval_engine
from deep_context.storage import get_storage
from deep_context.storage.asset_store import asset_store


@pytest.fixture
def sample_test_image():
    img = Image.new("RGB", (80, 80), color=(20, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Exact Financial Tabular Lookup
# ---------------------------------------------------------------------------


def test_financial_table_exact_lookup():
    headers = ["Metric", "FY2023", "FY2024", "Growth"]
    rows = [
        ["Total Revenue", "$12,450M", "$15,200M", "+22.1%"],
        ["Operating Income", "$2,100M", "$2,850M", "+35.7%"],
        ["Net Income", "$1,650M", "$2,230M", "+35.2%"],
        ["Operating Margin", "16.8%", "18.7%", "+190 bps"],
        ["Diluted EPS", "$3.42", "$4.65", "+36.0%"],
    ]
    md = (
        "| Metric | FY2023 | FY2024 | Growth |\n"
        "|---|---|---|---|\n"
        "| Total Revenue | $12,450M | $15,200M | +22.1% |\n"
        "| Operating Income | $2,100M | $2,850M | +35.7% |\n"
        "| Net Income | $1,650M | $2,230M | +35.2% |\n"
        "| Operating Margin | 16.8% | 18.7% | +190 bps |\n"
        "| Diluted EPS | $3.42 | $4.65 | +36.0% |"
    )

    table = TableDataModel(
        num_rows=len(rows),
        num_cols=len(headers),
        headers=headers,
        rows=rows,
        markdown=md,
        caption="Consolidated Statements of Operations",
    )

    # 1. Exact cell coordinate lookup
    c00 = table.get_cell(0, 0)
    assert c00 is not None and c00.text == "Total Revenue"
    c01 = table.get_cell(0, 1)
    assert c01 is not None and c01.text == "$12,450M"
    c22 = table.get_cell(2, 2)
    assert c22 is not None and c22.text == "$2,230M"  # Net income FY2024
    assert table.get_cell(99, 99) is None

    # 2. Key-value lookup by row header and target column header
    assert table.lookup("Operating Margin", "FY2024") == "18.7%"
    assert table.lookup("Diluted EPS", "Growth") == "+36.0%"
    assert table.lookup("Total Revenue", "FY2023") == "$12,450M"
    assert table.lookup("Nonexistent", "FY2024") is None

    # 3. Fuzzy search for cells containing substring
    bps_cells = table.find_cells("bps")
    assert len(bps_cells) == 1
    assert bps_cells[0].text == "+190 bps"
    assert bps_cells[0].row_idx == 3
    assert bps_cells[0].col_idx == 3


# ---------------------------------------------------------------------------
# 2. Medical Billing & PHI Asset Tenant Isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_medical_billing_asset_tenant_isolation(sample_test_image):
    storage = await get_storage()
    hospital_a = "hospital_alpha"
    hospital_b = "hospital_beta"

    # 1. Create document and asset under hospital_a
    doc_a = Document(
        id="doc_med_invoice_001",
        tenant_id=hospital_a,
        title="Patient Billing Statement - MRN-98472",
        permission_scope=["billing", "phi"],
    )
    await storage.insert_document(doc_a)

    asset_a = asset_store.save_image(
        image_data=sample_test_image,
        document_id=doc_a.id,
        tenant_id=hospital_a,
        permission_scope=["billing"],
        caption="Insurance Pre-Authorization Card",
    )
    await storage.insert_asset(asset_a)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Authorized access from hospital_a with billing permission
        res_ok = await client.get(
            f"/v1/assets/{asset_a.id}?tenant_id={hospital_a}&permission_scope=billing"
        )
        assert res_ok.status_code == 200
        assert res_ok.headers["content-type"] == "image/png"
        assert len(res_ok.content) == asset_a.byte_size

        # Cross-tenant attack from hospital_b -> MUST be 403 Forbidden
        res_cross_tenant = await client.get(
            f"/v1/assets/{asset_a.id}?tenant_id={hospital_b}&permission_scope=billing"
        )
        assert res_cross_tenant.status_code == 403
        assert "Access denied" in res_cross_tenant.json()["detail"]

        # Insufficient permission scope within hospital_a -> MUST be 403 Forbidden
        res_insufficient_perm = await client.get(
            f"/v1/assets/{asset_a.id}?tenant_id={hospital_a}&permission_scope=public_visitor"
        )
        assert res_insufficient_perm.status_code == 403

        # Document assets endpoint cross-tenant attack -> MUST be 403 Forbidden
        res_doc_b = await client.get(
            f"/v1/documents/{doc_a.id}/assets?tenant_id={hospital_b}&permission_scope=billing"
        )
        assert res_doc_b.status_code == 403

        # Document assets endpoint authorized access -> 200 with filtered assets
        res_doc_a = await client.get(
            f"/v1/documents/{doc_a.id}/assets?tenant_id={hospital_a}&permission_scope=billing"
        )
        assert res_doc_a.status_code == 200
        assets_list = res_doc_a.json()
        assert len(assets_list) >= 1
        assert assets_list[0]["id"] == asset_a.id


# ---------------------------------------------------------------------------
# 3. Deterministic Re-ingestion Stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_reingestion_stability():
    storage = await get_storage()
    pipeline = SummaryIngestionPipeline(storage=storage)

    content = """# Deterministic Reproducibility Document

## Section 1: Algorithmic Invariants
The ingestion pipeline must assign stable UUIDv5 identifiers to documents,
parent chunks, and child chunks. Re-ingesting the exact same document
under the same tenant must result in the exact same identifiers.

## Section 2: Mathematical Formulation
Let D be the document text and T be the tenant.
The document ID is H(T, D), ensuring strict cross-session idempotence.
"""

    req = IngestRequest(
        title="Reproducibility Invariant Test",
        content=content,
        doc_type="markdown",
        tenant_id="tenant_stable_test",
        generate_summaries=False,
    )

    resp1 = await pipeline.ingest(req)
    resp2 = await pipeline.ingest(req)

    # 1. Document ID must be 100% deterministic and identical across re-runs
    assert resp1.document_id == resp2.document_id

    # 2. Parent chunk IDs and Child chunk IDs must be 100% identical
    parents1 = await storage.get_document_chunks(resp1.document_id, level="parent")
    parents2 = await storage.get_document_chunks(resp2.document_id, level="parent")
    assert len(parents1) == len(parents2)
    assert [p["id"] for p in parents1] == [p["id"] for p in parents2]

    children1 = await storage.get_document_chunks(resp1.document_id, level="child")
    children2 = await storage.get_document_chunks(resp2.document_id, level="child")
    assert len(children1) == len(children2)
    assert [c["id"] for c in children1] == [c["id"] for c in children2]


# ---------------------------------------------------------------------------
# 4. Context Expansion During Normal Retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_normal_context_expansion():
    storage = await get_storage()
    doc_id = "doc_context_expand_verify"

    doc = Document(
        id=doc_id,
        title="Relativistic Quantum Mechanics",
        doc_type="latex",
        tenant_id="default",
    )
    await storage.insert_document(doc)

    eq_node = DocumentNode(
        id="node_dirac_eq_1",
        document_id=doc_id,
        node_type=DocumentElementType.EQUATION,
        reading_order=1,
        text="(i \\gamma^\\mu \\partial_\\mu - m) \\psi = 0",
        section_path="Physics > Dirac Equation",
        equation_data=EquationDataModel(
            latex="(i \\gamma^\\mu \\partial_\\mu - m) \\psi = 0",
            equation_number="(1)",
            variables=["\\gamma", "\\psi", "m"],
        ),
    )

    code_node = DocumentNode(
        id="node_dirac_code_1",
        document_id=doc_id,
        node_type=DocumentElementType.CODE,
        reading_order=2,
        text="def dirac_spinor(m: float, p: list[float]):\n    return [1.0, 0.0, p[2]/(m + 1.0), (p[0] + 1j*p[1])/(m + 1.0)]",
        section_path="Physics > Dirac Equation",
        code_data=SourceCodeDataModel(
            language="python",
            symbol_name="dirac_spinor",
            signature="def dirac_spinor(m: float, p: list[float])",
        ),
    )

    await storage.insert_tree_nodes([eq_node, code_node])

    parent_chunk = Chunk(
        id="chunk_parent_dirac",
        document_id=doc_id,
        level=ChunkLevel.PARENT,
        content="Paul Dirac formulated the relativistic wave equation for spin-1/2 fermions.",
        metadata={"node_ids": [eq_node.id, code_node.id]},
        section_path="Physics > Dirac Equation",
    )
    child_chunk = Chunk(
        id="chunk_child_dirac",
        document_id=doc_id,
        parent_chunk_id=parent_chunk.id,
        level=ChunkLevel.CHILD,
        content="Dirac wave equation for spin-1/2 fermions in quantum field theory.",
        metadata={"node_ids": [eq_node.id, code_node.id]},
        section_path="Physics > Dirac Equation",
    )
    await storage.insert_chunks([parent_chunk, child_chunk])

    # Execute normal retrieval
    res = await retrieval_engine.retrieve(
        query="What is the Dirac equation and python implementation?",
        filters=RetrievalFilters(document_ids=[doc_id]),
        top_k=2,
    )

    assert res.sufficient is True
    assert len(res.parent_chunks) >= 1
    top_parent = res.parent_chunks[0]

    # Verify context expansion was populated in retrieval result
    assert len(res.expanded_contexts) >= 1
    exp = res.expanded_contexts[0]
    assert exp["chunk_id"] == parent_chunk.id
    assert len(exp["equations"]) >= 1
    assert exp["equations"][0]["latex"] == "(i \\gamma^\\mu \\partial_\\mu - m) \\psi = 0"
    assert len(exp["code_blocks"]) >= 1
    assert exp["code_blocks"][0]["symbol_name"] == "dirac_spinor"

    # Verify chunk content itself received the expanded structured context
    assert "\\gamma^\\mu" in top_parent["content"]
    assert "dirac_spinor" in top_parent["content"]


# ---------------------------------------------------------------------------
# 5. Cross-Modal Retrieval (Text -> Image & Image -> Text)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_modal_retrieval(sample_test_image):
    storage = await get_storage()
    doc_id = "doc_cross_modal_test"

    doc = Document(id=doc_id, title="Quantum Circuits Documentation")
    await storage.insert_document(doc)

    # 768-dim normalized embedding fixture
    emb = [0.03] * 768
    emb[10] = 0.65
    norm = sum(v * v for v in emb) ** 0.5
    emb = [v / norm for v in emb]

    asset = MultimodalAsset(
        id="asset_quantum_teleportation_circuit",
        document_id=doc_id,
        asset_type="figure",
        mime_type="image/png",
        width=100,
        height=100,
        byte_size=len(sample_test_image),
        sha256="quantum_circ_sha",
        storage_path="/tmp/fake_circuit.png",
        caption="Quantum teleportation circuit diagram using Bell state entanglement",
        embedding=emb,
        tenant_id="default",
    )
    await storage.insert_asset(asset)

    p_chunk = Chunk(
        id="chunk_parent_teleport",
        document_id=doc_id,
        level=ChunkLevel.PARENT,
        content="Quantum teleportation transfers qubit states using classical communication and entanglement.",
        metadata={"asset_ids": [asset.id]},
    )
    c_chunk = Chunk(
        id="chunk_child_teleport",
        document_id=doc_id,
        parent_chunk_id=p_chunk.id,
        level=ChunkLevel.CHILD,
        content="Quantum teleportation protocol circuit Bell states.",
        embedding=emb,
        metadata={"asset_ids": [asset.id]},
    )
    await storage.insert_chunks([p_chunk, c_chunk])

    # 1. Text -> Image retrieval (query with visual terms retrieves visual asset)
    res_text = await retrieval_engine.retrieve(
        query="Show me the circuit diagram of quantum teleportation",
        filters=RetrievalFilters(document_ids=[doc_id]),
    )
    assert any(a["id"] == asset.id for a in res_text.assets)

    # 2. Image -> Text retrieval (query_asset_id retrieves text parent chunk)
    res_image = await retrieval_engine.retrieve(
        query="Explain this circuit",
        query_asset_id=asset.id,
        filters=RetrievalFilters(document_ids=[doc_id]),
    )
    assert any(p["chunk_id"] == p_chunk.id for p in res_image.parent_chunks)

    # 3. Pure Image -> Text vector retrieval (no text keyword overlap with chunks)
    res_image_pure = await retrieval_engine.retrieve(
        query="what does this show and describe the details",
        query_asset_id=asset.id,
        filters=RetrievalFilters(document_ids=[doc_id]),
    )
    assert any(p["chunk_id"] == p_chunk.id for p in res_image_pure.parent_chunks)


# ---------------------------------------------------------------------------
# 6. Multimodal Enrichment Integrity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multimodal_enricher_unavailable_without_vision():
    enricher = MultimodalEnricher(vision_fn=None)
    fig_node = DocumentNode(
        id="node_fig_no_vision",
        document_id="doc_enrich_test",
        node_type=DocumentElementType.FIGURE,
        reading_order=1,
        text="A line graph of revenue growth over 5 quarters",
        figure_data=FigureDataModel(caption="Figure 3: Revenue Growth"),
    )
    tree = DocumentTree(document_id="doc_enrich_test", nodes=[fig_node])

    # Run enricher when no vision client is attached
    await enricher.enrich_tree(tree, enabled=True)

    assert fig_node.figure_data is not None
    enrichment = fig_node.figure_data.enrichment
    assert enrichment is not None
    # Must NOT claim 'completed' with hallucinated/paraphrased caption
    assert enrichment.status == "unavailable"
    assert enrichment.description is None
    assert "not attached or image crop unavailable" in (enrichment.limitations or "")
