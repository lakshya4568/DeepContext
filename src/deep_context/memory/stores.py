"""Typed memory stores manager (Policy, Preference, Fact, Episode)."""

from __future__ import annotations

from typing import Any

from deep_context.core.llm_client import llm_client
from deep_context.core.types import Observation, PromotionResult
from deep_context.memory.promotion_gate import PromotionGate
from deep_context.storage.base import StorageInterface


class MemoryStoreManager:
    """Manages the four typed memory stores."""

    def __init__(self, storage: StorageInterface):
        self.storage = storage
        self.promotion_gate = PromotionGate(storage)

    async def observe_and_promote(self, observation: Observation) -> PromotionResult:
        """Processes observation through the promotion gate."""
        return await self.promotion_gate.evaluate_and_promote(observation)

    async def get_active_policies(
        self, tenant_id: str = "default", user_id: str | None = None
    ) -> list[dict[str, Any]]:
        return await self.storage.list_policies(tenant_id=tenant_id, user_id=user_id)

    async def get_user_preferences(self, user_id: str) -> list[dict[str, Any]]:
        return await self.storage.list_preferences(user_id=user_id)

    async def search_relevant_facts(
        self, query: str, tenant_id: str = "default", user_id: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        query_emb = await llm_client.get_embedding(query)
        return await self.storage.search_facts(
            query=query,
            query_embedding=query_emb,
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
        )

    async def save_episode(
        self,
        user_id: str,
        summary: str,
        session_id: str | None = None,
        task_type: str | None = None,
        outcome: str = "success",
    ) -> str:
        summary_emb = await llm_client.get_embedding(summary)
        return await self.storage.insert_episode(
            user_id=user_id,
            session_id=session_id,
            task_type=task_type,
            summary=summary,
            outcome=outcome,
            embedding=summary_emb,
        )
