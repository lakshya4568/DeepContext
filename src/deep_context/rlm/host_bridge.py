"""Host/Kernel authority boundary implementing ARCHITECTURE.md §4–5 and FR12–FR13."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Coroutine

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.core.types import (
    AdmissionHandle,
    AgentMessage,
    Observation,
    RetrievalFilters,
    RetrievalResult,
    SessionHandle,
    SessionStatus,
)
from deep_context.memory.promotion_gate import PromotionGate
from deep_context.retrieval.engine import retrieval_engine
from deep_context.storage.base import StorageInterface


class RecursionDepthExceeded(Exception):
    """Raised when an RLM subagent exceeds max_recursion_depth."""

    pass


class HostBridge:
    """
    Authority boundary for RLM engine.
    Controls database writes, provider API calls, budget enforcement, and recursion depth.
    The sandboxed kernel only accesses these through typed host-requests.
    """

    def __init__(
        self,
        storage: StorageInterface,
        child_runner_factory: Callable[[str, str, HostBridge], Coroutine[Any, Any, None]]
        | None = None,
    ):
        self.storage = storage
        self.promotion_gate = PromotionGate(storage)
        self.sessions: dict[str, SessionHandle] = {}
        self.children: dict[str, list[AdmissionHandle]] = {}
        self.inbox: dict[str, list[AgentMessage]] = {}
        self._child_runner_factory = child_runner_factory

    async def register_session(self, session: SessionHandle, user_id: str = "default") -> None:
        self.sessions[session.id] = session
        await self.storage.create_session(session, user_id=user_id)

    # -----------------------------------------------------------------------
    # Typed Host Requests
    # -----------------------------------------------------------------------

    async def retrieve(
        self,
        session_id: str,
        query: str,
        tenant_id: str = "default",
        permission_scope: list[str] | None = None,
        top_k: int = 8,
    ) -> RetrievalResult:
        """Thin host forward for retrieval. Kernel never queries DB directly."""
        self._check_active(session_id)
        filters = RetrievalFilters(
            tenant_id=tenant_id,
            permission_scope=permission_scope or ["default"],
        )
        return await retrieval_engine.retrieve(query=query, filters=filters, top_k=top_k)

    async def memory_save_fact(
        self, session_id: str, observation_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Thin host forward for memory writes. Kernel never writes to DB directly."""
        self._check_active(session_id)
        obs = Observation(
            raw_text=observation_dict.get("text", ""),
            tenant_id=observation_dict.get("tenant_id", "default"),
            user_id=observation_dict.get("user_id"),
            source=observation_dict.get("source", "tool_output"),
        )
        res = await self.promotion_gate.evaluate_and_promote(obs)
        return {
            "decision": res.decision.value,
            "memory_type": res.memory_type.value,
            "atomic_claim": res.atomic_claim,
            "confidence": res.confidence,
            "superseded_id": res.superseded_id,
        }

    async def rlm_spawn(
        self,
        parent_session_id: str,
        name: str,
        task_spec: str,
        model: str | None = None,
    ) -> AdmissionHandle:
        """
        Asynchronously admits an RLM subagent.
        Returns an AdmissionHandle IMMEDIATELY and does not block parent execution.
        """
        parent = self._check_active(parent_session_id)
        child_depth = parent.depth + 1

        # Host-enforced recursion depth check (default 1)
        if child_depth > parent.budgets.max_recursion_depth:
            raise RecursionDepthExceeded(
                f"Recursion depth {child_depth} exceeds max allowed depth {parent.budgets.max_recursion_depth}."
            )

        child_id = str(uuid.uuid4())
        child_model = model or settings.llm_model
        child_session = SessionHandle(
            id=child_id,
            parent_session_id=parent_session_id,
            depth=child_depth,
            budgets=parent.budgets,
            status=SessionStatus.ACTIVE,
        )
        self.sessions[child_id] = child_session
        await self.storage.create_session(child_session)

        handle = AdmissionHandle(
            child_id=child_id,
            name=name,
            session_dir=f"/sessions/{child_id}",
            model=child_model,
        )
        self.children.setdefault(parent_session_id, []).append(handle)

        await self.storage.insert_rlm_child(
            parent_session_id=parent_session_id,
            child_session_id=child_id,
            name=name,
            model=child_model,
            depth=child_depth,
        )

        # Launch child subagent background task
        if self._child_runner_factory:
            asyncio.create_task(self._child_runner_factory(child_id, task_spec, self))

        return handle

    async def agent_message_send(
        self,
        sender_session_id: str,
        receiver_role: str,
        receiver_name: str | None,
        content: str,
    ) -> None:
        """Kernel message passing channel."""
        self._check_active(sender_session_id)
        sender = self.sessions[sender_session_id]

        target_session_id = None
        if receiver_role == "parent":
            target_session_id = sender.parent_session_id
        else:
            for handle in self.children.get(sender_session_id, []):
                if handle.name == receiver_name:
                    target_session_id = handle.child_id
                    break

        if not target_session_id:
            logger.warning(
                "Could not route message from %s to role=%s, name=%s",
                sender_session_id,
                receiver_role,
                receiver_name,
            )
            return

        msg = AgentMessage(
            session_id=sender_session_id,
            receiver_role=receiver_role,
            receiver_name=receiver_name,
            content=content,
        )
        self.inbox.setdefault(target_session_id, []).append(msg)
        await self.storage.insert_agent_message(msg)

    async def collect_messages(self, session_id: str) -> list[AgentMessage]:
        """Parent collects pending messages from children between its turns."""
        in_memory_msgs = self.inbox.pop(session_id, [])
        db_msgs = await self.storage.pop_agent_messages(session_id)
        all_msgs = in_memory_msgs + db_msgs
        # Deduplicate
        seen: set[str] = set()
        deduped = []
        for m in all_msgs:
            key = f"{m.session_id}:{m.content}:{m.created_at}"
            if key not in seen:
                seen.add(key)
                deduped.append(m)
        return deduped

    def _check_active(self, session_id: str) -> SessionHandle:
        session = self.sessions.get(session_id)
        if not session or session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is not active.")
        return session
