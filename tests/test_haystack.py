"""Tests for Needle-in-a-Haystack generator and 5-stage benchmark diagnostics."""

import pytest
from httpx import ASGITransport, AsyncClient

from deep_context.api.app import app


@pytest.mark.asyncio
async def test_haystack_generate_and_benchmark() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Generate Haystack
        gen_payload = {
            "needle": "SECRET_PASSCODE_APOLLO_7722",
            "needle_query": "What is the secret passcode for Apollo?",
            "total_words": 3000,
            "depth_percent": 50.0,
            "topic": "High-Throughput Distributed Storage",
        }
        gen_resp = await ac.post("/v1/haystack/generate", json=gen_payload)
        assert gen_resp.status_code == 200
        gen_data = gen_resp.json()
        doc_id = gen_data["document_id"]
        assert doc_id is not None
        assert gen_data["parent_chunks_count"] >= 1
        assert gen_data["child_chunks_count"] >= 2

        # 2. Run Diagnostic Benchmark
        bench_payload = {
            "document_id": doc_id,
            "needle": "SECRET_PASSCODE_APOLLO_7722",
            "query": "What is the secret passcode for Apollo?",
            "top_k": 4,
        }
        bench_resp = await ac.post("/v1/haystack/benchmark", json=bench_payload)
        assert bench_resp.status_code == 200
        bench_data = bench_resp.json()

        # Check diagnostic stages
        stages = bench_data["stages"]
        assert len(stages) >= 5
        stage_names = [s["stage_name"] for s in stages]
        assert any("BM25" in name for name in stage_names)
        assert any("Dense Vector" in name for name in stage_names)
        assert any("RRF" in name for name in stage_names)
        assert any("Reranker" in name for name in stage_names)
        assert any("Parent Chunk" in name for name in stage_names)

        # Check needle was found and retrieved
        assert bench_data["passed"] is True
        assert bench_data["retrieved_parent_chunk"] is not None
        assert "SECRET_PASSCODE_APOLLO_7722" in bench_data["retrieved_parent_chunk"]["content"]


@pytest.mark.asyncio
async def test_documents_list_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert isinstance(docs, list)
