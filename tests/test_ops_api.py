"""Tests for ops endpoints: agentic RAG, scheduler control, and cache status."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deep_context.api.app import create_app
from deep_context.cache import response_cache
from deep_context.core.config import settings


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


class TestCacheStatusEndpoint:
    def test_cache_status_reports_backend(self) -> None:
        with _client() as client:
            res = client.get("/v1/cache/status")
            assert res.status_code == 200
            data = res.json()
            assert data["enabled"] == settings.cache_enabled
            assert data["backend"] in ("redis", "memory")
            assert "default_ttl" in data and "namespace" in data

    def test_invalidate_cache_endpoint(self) -> None:
        with _client() as client:
            res = client.post("/v1/cache/invalidate?namespace=rag")
            assert res.status_code == 200
            assert res.json()["status"] == "ok"


class TestConfigEndpoints:
    def test_get_and_update_config(self) -> None:
        with _client() as client:
            res = client.get("/v1/config")
            assert res.status_code == 200
            data = res.json()
            assert "summary_enabled" in data
            assert "summary_model" in data
            assert "summary_device" in data

            up_res = client.post("/v1/config", json={"summary_max_tokens": 75, "default_top_k": 10})
            assert up_res.status_code == 200
            up_data = up_res.json()
            assert up_data["status"] == "updated"
            assert up_data["config"]["summary_max_tokens"] == 75
            assert up_data["config"]["default_top_k"] == 10


class TestSchedulerEndpoints:
    def test_list_jobs(self) -> None:
        with _client() as client:
            res = client.get("/v1/scheduler/jobs")
            assert res.status_code == 200
            data = res.json()
            assert "registered_tasks" in data
            assert "cleanup_orphaned_docs" in data["registered_tasks"]

    def test_upsert_unknown_job_rejected(self) -> None:
        with _client() as client:
            res = client.post(
                "/v1/scheduler/jobs",
                json={"name": "nope", "schedule_cron": "every:60"},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "error"

    def test_upsert_known_job_ok(self) -> None:
        with _client() as client:
            res = client.post(
                "/v1/scheduler/jobs",
                json={"name": "cleanup_orphaned_docs", "schedule_cron": "every:120"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ok"
            assert data["name"] == "cleanup_orphaned_docs"

    def test_scheduler_tick(self) -> None:
        with _client() as client:
            res = client.post("/v1/scheduler/tick")
            assert res.status_code == 200
            assert res.json()["status"] == "ok"

    def test_install_defaults(self) -> None:
        with _client() as client:
            res = client.post("/v1/scheduler/defaults")
            assert res.status_code == 200


class TestAgenticRagEndpoint:
    def test_agentic_rag_endpoint_runs(self) -> None:
        import asyncio

        from deep_context.core.types import IngestRequest, RetrievalMode
        from deep_context.ingestion.pipeline import ingestion_pipeline
        from deep_context.storage import get_storage

        async def _seed() -> None:
            storage = await get_storage()
            existing = await storage.list_document_summaries(limit=100)
            if any(d["title"] == "Ops Endpoint Corpus" for d in existing):
                return
            await ingestion_pipeline.ingest(
                IngestRequest(
                    title="Ops Endpoint Corpus",
                    content=(
                        "# Vector Databases\n\n"
                        "A vector database stores high-dimensional embeddings and supports "
                        "approximate nearest neighbor search using indexes such as HNSW."
                    ),
                    doc_type="markdown",
                    retrieval_mode=RetrievalMode.HYBRID,
                )
            )

        asyncio.run(_seed())

        with _client() as client:
            res = client.post(
                "/v1/agentic-rag",
                json={"query": "What is a vector database?", "max_rewrites": 1},
            )
            assert res.status_code == 200
            data = res.json()
            assert "answer" in data
            assert data["grade_result"] in ("relevant", "irrelevant")
            assert isinstance(data["trace"], list)


class TestCachedQueryFlow:
    def test_query_response_includes_cache_hit_field(self) -> None:
        with _client() as client:
            res = client.post("/v1/query", json={"query": "cache hit field check"})
            assert res.status_code == 200
            assert "cache_hit" in res.json()

    def test_retrieve_response_includes_cache_hit_field(self) -> None:
        with _client() as client:
            res = client.post("/v1/retrieve", json={"query": "cache hit retrieval"})
            assert res.status_code == 200
            assert "cache_hit" in res.json()

    def test_second_identical_retrieve_is_cached(self) -> None:
        # Ensure a clean cache namespace for this test.
        import asyncio

        asyncio.run(response_cache.invalidate_namespace("rag"))

        with _client() as client:
            body = {"query": "repeatable deterministic retrieval query"}
            first = client.post("/v1/retrieve", json=body).json()
            second = client.post("/v1/retrieve", json=body).json()
            if first.get("sufficient"):
                assert second["cache_hit"] is True
