"""Tests for FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from deep_context.api.app import app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "deep-context-platform"


@pytest.mark.asyncio
async def test_api_ingest_retrieve_query_flow() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Ingest document
        ingest_payload = {
            "title": "FastAPI Deployment Guide",
            "content": """# Deployment
Deploy FastAPI using Uvicorn with Gunicorn process workers.
Run on port 8000 behind NGINX reverse proxy.
""",
            "doc_type": "markdown",
        }
        ingest_resp = await ac.post("/v1/ingest", json=ingest_payload)
        assert ingest_resp.status_code == 200
        doc_data = ingest_resp.json()
        assert doc_data["document_id"] is not None

        # 2. Retrieve
        retrieve_resp = await ac.post(
            "/v1/retrieve", json={"query": "How to deploy FastAPI?", "top_k": 3}
        )
        assert retrieve_resp.status_code == 200
        ret_data = retrieve_resp.json()
        assert len(ret_data["parent_chunks"]) >= 1

        # 3. Query
        query_resp = await ac.post(
            "/v1/query", json={"query": "What process manager is used with Uvicorn?"}
        )
        assert query_resp.status_code == 200
        q_data = query_resp.json()
        assert len(q_data["answer"]) > 0


@pytest.mark.asyncio
async def test_api_memory_observe() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        obs_payload = {
            "text": "User is evaluating Next.js for frontend.",
            "user_id": "user_api_1",
            "source": "inferred",
        }
        resp = await ac.post("/v1/memory/observe", json=obs_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "write"
        assert data["memory_type"] == "fact"


@pytest.mark.asyncio
async def test_api_batch_upload_and_folder_sync() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test batch upload with multiple files (markdown + text)
        files = [
            ("files", ("doc1.md", b"# Doc 1\nThis is document 1 content.\n", "text/markdown")),
            ("files", ("doc2.txt", b"This is plain text document 2.\n", "text/plain")),
        ]
        resp = await ac.post("/v1/upload-batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["document_id"] is not None
        assert data[1]["document_id"] is not None

        # Test sync folder
        sync_resp = await ac.post("/v1/sync-folder", params={"folder_path": "documents"})
        assert sync_resp.status_code == 200
        sync_data = sync_resp.json()
        assert isinstance(sync_data, list)


@pytest.mark.asyncio
async def test_quota_status_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/quota/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "groq" in data
        assert "is_rate_limited" in data["groq"]
        assert "nvidia_nim" in data
