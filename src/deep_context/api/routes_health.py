"""Health and metrics API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from deep_context.core.config import settings
from deep_context.storage import get_storage

router = APIRouter(tags=["Health & Observability"])


@router.get("/v1/health")
async def health_check() -> dict[str, Any]:
    await get_storage()
    db_type = settings.database_type
    return {
        "status": "healthy",
        "service": "deep-context-platform",
        "version": "0.1.0",
        "database": db_type,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "has_nvidia_api_key": settings.has_valid_api_key,
    }


@router.get("/v1/metrics")
async def get_metrics(limit: int = 50) -> dict[str, Any]:
    storage = await get_storage()
    traces = await storage.list_event_traces(limit=limit)
    total_events = len(traces)
    avg_latency = sum(t["latency_ms"] for t in traces) / total_events if total_events > 0 else 0
    return {
        "total_traces_logged": total_events,
        "average_latency_ms": round(avg_latency, 2),
        "recent_traces": traces,
    }
