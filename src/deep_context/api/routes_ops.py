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
