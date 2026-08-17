"""Typed memory API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from deep_context.core.types import Observation
from deep_context.memory.stores import MemoryStoreManager
from deep_context.storage import get_storage

router = APIRouter(prefix="/v1/memory", tags=["Typed Memory"])


class ObserveRequest(BaseModel):
    text: str
    tenant_id: str = "default"
    user_id: str | None = None
    source: str = "user_stated"  # 'user_stated' | 'tool_output' | 'inferred' | 'operator'


@router.post("/observe")
async def observe_memory(req: ObserveRequest) -> dict[str, Any]:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)
    obs = Observation(
        raw_text=req.text,
        tenant_id=req.tenant_id,
        user_id=req.user_id,
        source=req.source,
    )
    result = await manager.observe_and_promote(obs)
    return {
        "decision": result.decision.value,
        "memory_type": result.memory_type.value,
        "atomic_claim": result.atomic_claim,
        "confidence": result.confidence,
        "expires_at": result.expires_at.isoformat() if result.expires_at else None,
        "superseded_id": result.superseded_id,
        "reject_reason": result.reject_reason,
    }


@router.get("/policies")
async def list_policies(
    tenant_id: str = "default", user_id: str | None = None
) -> list[dict[str, Any]]:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)
    return await manager.get_active_policies(tenant_id=tenant_id, user_id=user_id)


@router.get("/preferences")
async def list_preferences(user_id: str) -> list[dict[str, Any]]:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)
    return await manager.get_user_preferences(user_id=user_id)


@router.get("/facts")
async def search_facts(
    query: str, tenant_id: str = "default", user_id: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)
    return await manager.search_relevant_facts(
        query=query, tenant_id=tenant_id, user_id=user_id, limit=limit
    )
