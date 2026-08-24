"""Routes for the corrective agentic RAG graph, scheduler control, and cache diagnostics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from deep_context.agentic.graph import run_agentic_rag
from deep_context.cache import response_cache
from deep_context.core.config import settings
from deep_context.core.types import RetrievalFilters
from deep_context.scheduler import TASKS, register_default_jobs, run_due_jobs_once
from deep_context.storage import get_storage

router = APIRouter(tags=["Agentic & Ops"])


# ---------------------------------------------------------------------------
# Agentic Corrective RAG
# ---------------------------------------------------------------------------


class AgenticRagRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    user_id: str | None = None
    permission_scope: list[str] = Field(default_factory=lambda: ["default"])
    document_ids: list[str] | None = None
    top_k: int = 6
    max_rewrites: int | None = None
    model: str | None = None


class AgenticRagResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    grade_result: str
    rewrite_count: int
    abstained: bool
    support_passed: bool
    support_confidence: float
    trace: list[dict[str, Any]]


@router.post("/v1/agentic-rag", response_model=AgenticRagResponse)
async def run_agentic_rag_endpoint(req: AgenticRagRequest) -> AgenticRagResponse:
    """Run the corrective RAG state machine: retrieve -> grade -> rewrite loop -> generate."""
    filters = RetrievalFilters(
        tenant_id=req.tenant_id,
        permission_scope=req.permission_scope,
        document_ids=req.document_ids,
    )
    state = await run_agentic_rag(
        query=req.query,
        filters=filters,
        top_k=req.top_k,
        max_rewrites=req.max_rewrites,
        model=req.model,
        user_id=req.user_id,
    )
    return AgenticRagResponse(
        answer=state.answer or "",
        citations=state.citations,
        grade_result=state.grade_result,
        rewrite_count=state.rewrite_count,
        abstained=state.abstained,
        support_passed=state.support_passed,
        support_confidence=state.support_confidence,
        trace=state.trace,
    )


# ---------------------------------------------------------------------------
# Scheduler Control
# ---------------------------------------------------------------------------


class JobUpsertRequest(BaseModel):
    name: str
    schedule_cron: str = "every:3600"
    max_retries: int = 3


@router.get("/v1/scheduler/jobs")
async def list_scheduler_jobs() -> dict[str, Any]:
    """List all registered scheduled jobs and their current state."""
    storage = await get_storage()
    jobs = await storage.list_jobs()
    return {
        "scheduler_enabled": settings.scheduler_enabled,
        "registered_tasks": sorted(TASKS.keys()),
        "jobs": jobs,
    }


@router.post("/v1/scheduler/jobs")
async def upsert_scheduler_job(req: JobUpsertRequest) -> dict[str, Any]:
    """Register or update a scheduled job. Schedule is cron or 'every:<seconds>'."""
    from deep_context.scheduler import next_run_from_cron

    if req.name not in TASKS:
        return {
            "status": "error",
            "message": f"Unknown task '{req.name}'. Registered: {sorted(TASKS.keys())}",
        }

    try:
        next_run = next_run_from_cron(req.schedule_cron)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    storage = await get_storage()
    await storage.upsert_job(
        name=req.name,
        schedule_cron=req.schedule_cron,
        next_run_at=next_run,
        max_retries=req.max_retries,
    )
    return {"status": "ok", "name": req.name, "next_run_at": next_run.isoformat()}


@router.post("/v1/scheduler/tick")
async def run_scheduler_tick() -> dict[str, Any]:
    """Manually execute all due jobs once (useful for testing and cron-wrapped deploys)."""
    executed = await run_due_jobs_once()
    return {"status": "ok", "executed": executed}


@router.post("/v1/scheduler/defaults")
async def install_default_jobs() -> dict[str, Any]:
    """Install the built-in maintenance jobs if they are not yet registered."""
    await register_default_jobs()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Cache Diagnostics
# ---------------------------------------------------------------------------


@router.get("/v1/cache/status")
async def cache_status() -> dict[str, Any]:
    """Report cache configuration and active backend kind."""
    return {
        "enabled": settings.cache_enabled,
        "backend": response_cache.backend_kind,
        "default_ttl": settings.cache_default_ttl,
        "namespace": settings.cache_namespace,
    }


@router.post("/v1/cache/invalidate")
async def invalidate_cache(namespace: str = "rag") -> dict[str, Any]:
    """Invalidate all cached entries under a namespace ('rag' clears ask+retrieve)."""
    total = 0
    for ns in ("rag:ask", "rag:retrieve") if namespace == "rag" else (namespace,):
        total += await response_cache.invalidate_namespace(ns)
    return {"status": "ok", "namespace": namespace, "entries_invalidated": total}


# ---------------------------------------------------------------------------
# Runtime Platform Configuration & Testing
# ---------------------------------------------------------------------------


class ConfigUpdateRequest(BaseModel):
    summary_enabled: bool | None = None
    summary_model: str | None = None
    summary_max_tokens: int | None = None
    summary_batch_size: int | None = None
    summary_device: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    default_retrieval_mode: str | None = None
    default_top_k: int | None = None
    reranker_model: str | None = None
    confidence_threshold: float | None = None
    cache_enabled: bool | None = None
    cache_default_ttl: int | None = None
    agentic_max_rewrites: int | None = None
    agentic_grade_threshold: float | None = None


class TestSummaryRequest(BaseModel):
    text: str
    topic: str | None = None


@router.get("/v1/config")
async def get_runtime_config() -> dict[str, Any]:
    """Retrieve full runtime platform configuration."""
    return {
        "summary_enabled": settings.summary_enabled,
        "summary_model": settings.summary_model,
        "summary_max_tokens": settings.summary_max_tokens,
        "summary_batch_size": settings.summary_batch_size,
        "summary_device": settings.summary_device,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "default_retrieval_mode": settings.default_retrieval_mode,
        "default_top_k": settings.default_top_k,
        "reranker_model": settings.reranker_model,
        "confidence_threshold": settings.confidence_threshold,
        "cache_enabled": settings.cache_enabled,
        "cache_default_ttl": settings.cache_default_ttl,
        "agentic_max_rewrites": settings.agentic_max_rewrites,
        "agentic_grade_threshold": settings.agentic_grade_threshold,
        "llm_model": settings.llm_model,
        "llm_provider": settings.llm_provider,
    }


@router.post("/v1/config")
async def update_runtime_config(req: ConfigUpdateRequest) -> dict[str, Any]:
    """Update runtime configuration dynamically."""
    if req.summary_enabled is not None:
        settings.summary_enabled = req.summary_enabled
    if req.summary_model is not None:
        settings.summary_model = req.summary_model
    if req.summary_max_tokens is not None:
        settings.summary_max_tokens = req.summary_max_tokens
    if req.summary_batch_size is not None:
        settings.summary_batch_size = req.summary_batch_size
    if req.summary_device is not None:
        settings.summary_device = req.summary_device
    if req.embedding_model is not None:
        settings.embedding_model = req.embedding_model
    if req.embedding_dim is not None:
        settings.embedding_dim = req.embedding_dim
    if req.default_retrieval_mode is not None:
        settings.default_retrieval_mode = req.default_retrieval_mode
    if req.default_top_k is not None:
        settings.default_top_k = req.default_top_k
    if req.reranker_model is not None:
        settings.reranker_model = req.reranker_model
    if req.confidence_threshold is not None:
        settings.confidence_threshold = req.confidence_threshold
    if req.cache_enabled is not None:
        settings.cache_enabled = req.cache_enabled
    if req.cache_default_ttl is not None:
        settings.cache_default_ttl = req.cache_default_ttl
    if req.agentic_max_rewrites is not None:
        settings.agentic_max_rewrites = req.agentic_max_rewrites
    if req.agentic_grade_threshold is not None:
        settings.agentic_grade_threshold = req.agentic_grade_threshold

    return {"status": "updated", "config": await get_runtime_config()}


@router.post("/v1/test-summary")
async def test_summary_generation(req: TestSummaryRequest) -> dict[str, Any]:
    """Generate a test summary using local Qwen3 model on MPS/Metal."""
    import time

    from deep_context.ingestion.summarizer import ChunkSummarizer

    summarizer = ChunkSummarizer()
    t0 = time.time()
    summary, tokens = await summarizer.summarize_chunk(req.text, context_prefix=req.topic)
    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "summary": summary,
        "tokens": tokens,
        "latency_ms": elapsed_ms,
        "model": summarizer.model_name,
        "device": summarizer._resolve_device(),
    }
