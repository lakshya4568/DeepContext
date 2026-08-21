"""Typed memory stores manager (Policy, Preference, Fact, Episode)."""

from __future__ import annotations

from typing import Any

from deep_context.core.config import settings
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

    async def get_embedding_preferences(self, user_id: str) -> dict[str, Any]:
        """Fetch saved embedding model and reranker preferences for a specific user."""
        prefs = await self.storage.list_preferences(user_id=user_id)
        result: dict[str, Any] = {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "reranker": settings.reranker_strategy,
            "llm_model": settings.llm_model,
        }
        for p in prefs:
            key = p.get("preference_key")
            val = p.get("preference_value")
            if key == "embedding_model":
                if isinstance(val, dict):
                    result["embedding_model"] = val.get("model", result["embedding_model"])
                    result["embedding_dim"] = val.get("dim", result["embedding_dim"])
                elif isinstance(val, str):
                    result["embedding_model"] = val
            elif key == "reranker":
                if isinstance(val, dict):
                    result["reranker"] = val.get("strategy", result["reranker"])
                elif isinstance(val, str):
                    result["reranker"] = val
            elif key == "llm_model":
                if isinstance(val, dict):
                    result["llm_model"] = val.get("model", val.get("value", result["llm_model"]))
                elif isinstance(val, str):
                    result["llm_model"] = val
        return result

    async def set_embedding_preferences(
        self,
        user_id: str,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        reranker: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Persist user embedding and reranker preferences in durable memory_preference store."""
        if embedding_model:
            dim_val = embedding_dim or (
                768 if "gemini" in embedding_model.lower() else settings.embedding_dim
            )
            await self.storage.set_preference(
                user_id=user_id,
                preference_key="embedding_model",
                preference_value={"model": embedding_model, "dim": dim_val},
                confidence=1.0,
                source="explicit",
            )
        if reranker:
            await self.storage.set_preference(
                user_id=user_id,
                preference_key="reranker",
                preference_value={"strategy": reranker},
                confidence=1.0,
                source="explicit",
            )
        if llm_model:
            await self.storage.set_preference(
                user_id=user_id,
                preference_key="llm_model",
                preference_value={"model": llm_model},
                confidence=1.0,
                source="explicit",
            )

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
