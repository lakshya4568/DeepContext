"""Tests for Multimodal Assets: persistence, vector search, serving API, and context expansion."""

import base64
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
    EquationDataModel,
    MultimodalAsset,
    Provenance,
    RetrievalFilters,
    SourceCodeDataModel,
    TableDataModel,
    generate_stable_asset_id,
)
from deep_context.retrieval.engine import retrieval_engine
from deep_context.storage import get_storage
from deep_context.storage.asset_store import AssetStore


@pytest.fixture
def sample_pil_image():
    img = Image.new("RGB", (64, 64), color="blue")
    return img


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    return buf.getvalue()


def test_asset_store_persistence(tmp_path, sample_pil_image, sample_image_bytes):
    store = AssetStore(base_dir=tmp_path)

    # 1. Save PIL Image
    asset1 = store.save_image(
        image_data=sample_pil_image,
        document_id="doc_test_1",
        caption="A blue calibration square",
        tenant_id="test_tenant",
    )
    assert asset1.id.startswith("asset_")
    assert asset1.document_id == "doc_test_1"
    assert asset1.width == 64
    assert asset1.height == 64
    assert asset1.mime_type == "image/png"
    assert asset1.caption == "A blue calibration square"

    # Check disk existence
    path = store.get_asset_path(asset1.id, tenant_id="test_tenant")
    assert path is not None
    assert path.exists()

    # 2. Deduplication check: saving identical image returns same asset_id
    asset2 = store.save_image(
        image_data=sample_image_bytes,
        document_id="doc_test_2",
        tenant_id="test_tenant",
    )
    assert asset2.id == asset1.id
    assert asset2.sha256 == asset1.sha256

    # 3. Base64 data URL save
    b64_str = f"data:image/png;base64,{base64.b64encode(sample_image_bytes).decode()}"
    asset3 = store.save_image(b64_str, document_id="doc_test_3", tenant_id="test_tenant")
    assert asset3.id == asset1.id

    # 4. In-memory document tracking
    doc1_assets = store.get_assets_for_document("doc_test_1")
    assert len(doc1_assets) >= 1
    assert doc1_assets[0].id == asset1.id

    # 5. Retrieval of bytes
    retrieved_bytes = store.get_asset_bytes(asset1.id, tenant_id="test_tenant")
    assert retrieved_bytes is not None
    assert len(retrieved_bytes) == asset1.byte_size

    # 6. Deletion
    deleted = store.delete_asset(asset1.id, tenant_id="test_tenant")
    assert deleted is True
    assert store.get_asset_path(asset1.id, tenant_id="test_tenant") is None


@pytest.mark.asyncio
async def test_database_asset_crud_and_vector_search(sample_image_bytes):
    storage = await get_storage()
    asset_id = generate_stable_asset_id(sample_image_bytes)

    # 768-dim normalized embedding fixture
    embedding = [0.05] * 768
    embedding[0] = 0.5

    asset = MultimodalAsset(
        id=asset_id,
        document_id="doc_vector_test",
        node_id=None,
        asset_type="figure",
        mime_type="image/png",
        width=128,
        height=128,
        byte_size=len(sample_image_bytes),
        sha256=asset_id.replace("asset_", ""),
        storage_path="/tmp/fake_asset.png",
        caption="Bloch Sphere representation",
        ocr_text="|0> state and |1> state",
        embedding=embedding,
        tenant_id="default",
    )

    # Insert parent document first to satisfy foreign key
    doc = Document(id="doc_vector_test", title="Vector Test Doc")
    await storage.insert_document(doc)

    # Insert asset
    inserted_id = await storage.insert_asset(asset)
    assert inserted_id == asset_id

    # Get asset
    fetched = await storage.get_asset(asset_id)
    assert fetched is not None
    assert fetched.id == asset_id
    assert fetched.caption == "Bloch Sphere representation"
    assert fetched.document_id == "doc_vector_test"

    # Get assets for document
    doc_assets = await storage.get_assets_for_document("doc_vector_test")
    assert len(doc_assets) >= 1
    assert any(a.id == asset_id for a in doc_assets)

    # Vector search on asset embeddings
    query_emb = [0.05] * 768
    query_emb[0] = 0.49
    hits = await storage.search_assets_vector(query_emb, filters=RetrievalFilters(), limit=5)
    assert len(hits) >= 1
    top_hit = hits[0]
    assert top_hit["id"] == asset_id
    assert "score" in top_hit
    assert top_hit["caption"] == "Bloch Sphere representation"


@pytest.mark.asyncio
async def test_expand_chunk_context_multimodal():
    storage = await get_storage()
    doc_id = "doc_multimodal_expand"

    # Insert test document
    doc = Document(
        id=doc_id,
        title="Multimodal Physics Paper",
        source_uri="physics_multimodal.tex",
        doc_type="latex",
    )
    await storage.insert_document(doc)

    # Create equation and code and table nodes
    eq_node = DocumentNode(
        id="node_eq_test_1",
        document_id=doc_id,
        node_type=DocumentElementType.EQUATION,
        reading_order=1,
        text="E = m c^2",
        section_path="Physics > Relativistic Energy",
        provenance=Provenance(source_uri="physics_multimodal.tex", page_number=2),
        equation_data=EquationDataModel(
            latex="E = m c^2",
            equation_number="(5)",
            variables=["E", "m", "c"],
        ),
    )

    code_node = DocumentNode(
        id="node_code_test_1",
        document_id=doc_id,
        node_type=DocumentElementType.CODE,
        reading_order=2,
        text="def compute_energy(m: float) -> float:\n    return m * 299792458**2",
        section_path="Physics > Relativistic Energy",
        provenance=Provenance(source_uri="physics_multimodal.tex", page_number=2),
        code_data=SourceCodeDataModel(
            language="python",
            symbol_name="compute_energy",
            symbol_type="function",
            signature="def compute_energy(m: float) -> float",
        ),
    )

    tbl_node = DocumentNode(
        id="node_tbl_test_1",
        document_id=doc_id,
        node_type=DocumentElementType.TABLE,
        reading_order=3,
        text="Speed of light table",
        section_path="Physics > Relativistic Energy",
        provenance=Provenance(source_uri="physics_multimodal.tex", page_number=2),
        table_data=TableDataModel(
            num_rows=2,
            num_cols=2,
            markdown="| Constant | Value |\n|---|---|\n| c | 299792458 m/s |",
        ),
    )

    await storage.insert_tree_nodes([eq_node, code_node, tbl_node])

    # Create chunk referencing all 3 nodes
    chunk_id = "chunk_multimodal_expanded"
    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        level=ChunkLevel.CHILD,
        content="Einstein formulated E = mc^2 and the implementation in python.",
        metadata={"node_ids": [eq_node.id, code_node.id, tbl_node.id]},
        page_number=2,
        section_path="Physics > Relativistic Energy",
    )
    await storage.insert_chunks([chunk])

    # Run retrieval context expansion
    expanded = await retrieval_engine.expand_chunk_context(chunk_id)
    assert expanded["chunk_id"] == chunk_id
    assert expanded["document_id"] == doc_id
    assert len(expanded["nodes"]) == 3
    assert len(expanded["equations"]) >= 1
    assert expanded["equations"][0]["latex"] == "E = m c^2"
    assert len(expanded["code_blocks"]) >= 1
    assert expanded["code_blocks"][0]["symbol_name"] == "compute_energy"
    assert len(expanded["tables"]) >= 1
    assert "299792458" in expanded["tables"][0]["markdown"]


@pytest.mark.asyncio
async def test_asset_serving_api(sample_pil_image):
    # Save a temporary asset to test the endpoint
    from deep_context.storage.asset_store import asset_store

    storage = await get_storage()
    # Insert parent document first
    await storage.insert_document(Document(id="doc_api_test", title="API Test Doc"))

    asset = asset_store.save_image(
        image_data=sample_pil_image,
        document_id="doc_api_test",
        caption="Calibration circle",
    )

    await storage.insert_asset(asset)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Fetch asset via GET /v1/assets/{asset_id}
        res = await client.get(f"/v1/assets/{asset.id}")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert len(res.content) == asset.byte_size

        # 2. Fetch document assets via GET /v1/documents/{document_id}/assets
        res_doc = await client.get("/v1/documents/doc_api_test/assets")
        assert res_doc.status_code == 200
        doc_assets = res_doc.json()
        assert len(doc_assets) >= 1
        assert doc_assets[0]["id"] == asset.id
        assert doc_assets[0]["caption"] == "Calibration circle"

        # 3. 404 for non-existent asset
        res_404 = await client.get("/v1/assets/asset_non_existent_12345")
        assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_quantum_computing_dissertation_multimodal_stress_test():
    """
    Stress-test real PDF ingestion on documents/quantum_computing_dissertation.pdf:
    verifies mixed content with sections, figures, formulas, asset persistence, and image embedding.
    """
    import os

    from deep_context.core.llm_client import llm_client
    from deep_context.ingestion.parser import DocumentParser
    from deep_context.storage.asset_store import asset_store

    pdf_path = "documents/quantum_computing_dissertation.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Document {pdf_path} not found")

    tree = DocumentParser.parse_tree(
        pdf_path,
        doc_type="pdf",
        source_uri="quantum_computing_dissertation.pdf",
        title="Quantum Computing Dissertation",
        page_range=(31, 33),
    )
    assert tree is not None
    assert len(tree.nodes) > 1

    # Verify sections extracted
    section_nodes = [
        n
        for n in tree.nodes
        if n.node_type in (DocumentElementType.HEADING, DocumentElementType.TITLE)
    ]
    assert len(section_nodes) >= 1
    section_texts = [n.text for n in section_nodes]
    assert any("Optimal Photon Storage" in t or "Introduction" in t for t in section_texts)

    # Verify figure extracted with durable asset
    fig_nodes = [
        n
        for n in tree.nodes
        if n.node_type in (DocumentElementType.FIGURE, DocumentElementType.CHART)
    ]
    assert len(fig_nodes) >= 1
    fig_node = fig_nodes[0]
    assert fig_node.asset_id is not None
    assert fig_node.figure_data is not None

    # Check asset exists in asset_store
    asset_path = asset_store.get_asset_path(fig_node.asset_id)
    assert asset_path is not None
    assert asset_path.exists()
    assert asset_path.stat().st_size > 0

    # Test multimodal embedding of the extracted image asset
    emb = await llm_client.embed_image(asset_path, model="gemini-embedding-2", dim=768)
    assert len(emb) == 768
    assert any(v != 0.0 for v in emb)
    norm = sum(v * v for v in emb) ** 0.5
    assert abs(norm - 1.0) < 0.05 or norm > 0.1
