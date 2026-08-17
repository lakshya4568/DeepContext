"""
Reference implementation of the RLM host bridge described in SKILL.md and
docs/ARCHITECTURE.md §4-5.

This lives on the HOST side of the host/kernel boundary. It is what `retrieve()`,
`memory.*()`, `rlm_spawn()`, and `agent_message.send()` inside the sandboxed kernel actually
call through to -- the kernel process itself never touches Postgres or a provider API key
directly. Deliberately NOT built on LangGraph (docs/TECH_STACK.md §8): the async subagent /
message-passing model doesn't fit a graph-walk abstraction, so this is a small hand-rolled
async component instead.

Wire your real Postgres client, provider client, and sandbox launcher in where marked.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    DELETED = "deleted"


class ChildStatus(str, Enum):
    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    DELETED = "deleted"
    ERROR = "error"


@dataclass
class Budgets:
    max_turns: int = 50
    max_tokens: int = 2_000_000
    max_wall_clock_seconds: int = 3600
    max_recursion_depth: int = 1  # see SKILL.md correction #2 -- do not raise casually


@dataclass
class SessionHandle:
    id: str
    parent_session_id: str | None
    depth: int
    budgets: Budgets
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AdmissionHandle:
    """What rlm_spawn() returns to the kernel IMMEDIATELY -- never the child's answer."""

    child_id: str
    name: str
    session_dir: str
    model: str


@dataclass
class AgentMessage:
    session_id: str  # sender
    receiver_role: str  # 'parent' | 'child'
    receiver_name: str | None
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RecursionDepthExceeded(Exception):
    pass


class HostBridge:
    """
    One instance per root session. Owns everything a compromised kernel must not be able to
    touch directly: DB writes, provider calls, budget enforcement, depth enforcement.
    """

    def __init__(self, root_session: SessionHandle):
        self.root_session = root_session
        self.sessions: dict[str, SessionHandle] = {root_session.id: root_session}
        self.children: dict[str, list[AdmissionHandle]] = {}  # parent_session_id -> children
        self.inbox: dict[str, list[AgentMessage]] = {}  # session_id -> pending messages

    # -----------------------------------------------------------------
    # Typed host requests -- the ONLY channel the kernel has to authority
    # -----------------------------------------------------------------

    async def retrieve(self, session_id: str, query: str, **filters) -> dict:
        """Thin forward to skills/rag-retrieval/scripts/retrieve.py. The kernel-side stub
        just serializes this call; the host is the one that actually queries Postgres."""
        self._check_active(session_id)
        # TODO: from skills.rag_retrieval.scripts.retrieve import retrieve as rag_retrieve
        raise NotImplementedError("wire up skills/rag-retrieval/scripts/retrieve.py here")

    async def memory_save_fact(self, session_id: str, observation: dict) -> dict:
        """Thin forward to skills/typed-memory/scripts/promotion_gate.py.
        The kernel never writes to memory_fact directly."""
        self._check_active(session_id)
        # TODO: from skills.typed_memory.scripts.promotion_gate import run_promotion_gate
        raise NotImplementedError("wire up skills/typed-memory/scripts/promotion_gate.py here")

    async def rlm_spawn(
        self,
        parent_session_id: str,
        name: str,
        task_spec: str,
        model: str | None = None,
    ) -> AdmissionHandle:
        """
        Returns an admission handle IMMEDIATELY. Does NOT await the child's completion and
        does NOT return its answer -- see SKILL.md correction #1. The child runs as an
        independent asyncio task; replies arrive later via agent_message.send().
        """
        parent = self._check_active(parent_session_id)
        child_depth = parent.depth + 1

        # Depth enforcement lives on the HOST, not the kernel -- a compromised kernel must
        # not be able to raise its own budget (docs/ARCHITECTURE.md §8).
        if child_depth > parent.budgets.max_recursion_depth:
            raise RecursionDepthExceeded(
                f"session {parent_session_id} at depth {parent.depth} cannot spawn a child "
                f"(would be depth {child_depth}); max_recursion_depth="
                f"{parent.budgets.max_recursion_depth}"
            )

        child_id = str(uuid.uuid4())
        child_model = model or self._inherited_model(parent_session_id)
        child_session = SessionHandle(
            id=child_id,
            parent_session_id=parent_session_id,
            depth=child_depth,
            budgets=parent.budgets,  # children inherit parent budgets unless overridden
        )
        self.sessions[child_id] = child_session
        handle = AdmissionHandle(
            child_id=child_id, name=name, session_dir=f"/sessions/{child_id}", model=child_model
        )
        self.children.setdefault(parent_session_id, []).append(handle)

        # TODO: insert into rlm_children (docs/DATA_MODEL.sql) here:
        #   INSERT INTO rlm_children (parent_session_id, child_session_id, name, model, depth)

        # Fire-and-forget: the child runs independently. A stuck/erroring child cannot hang
        # the parent -- see docs/ARCHITECTURE.md §8 ("subagents are not threads").
        asyncio.create_task(self._run_child(child_id, task_spec))

        return handle

    async def agent_message_send(
        self, sender_session_id: str, receiver_role: str, receiver_name: str | None, content: str
    ) -> None:
        self._check_active(sender_session_id)
        target_session_id = self._resolve_receiver(sender_session_id, receiver_role, receiver_name)
        self.inbox.setdefault(target_session_id, []).append(
            AgentMessage(sender_session_id, receiver_role, receiver_name, content)
        )
        # TODO: also persist to agent_messages (docs/DATA_MODEL.sql) for trace/audit.

    def collect_messages(self, session_id: str) -> list[AgentMessage]:
        """Parent calls this between its own turns to pick up whatever children have sent so
        far -- this is the ONLY way results travel back, never a return value from rlm_spawn."""
        pending = self.inbox.pop(session_id, [])
        return pending

    def list_subagents(self, parent_session_id: str) -> list[AdmissionHandle]:
        return self.children.get(parent_session_id, [])

    async def delete_subagent(self, child_id: str) -> None:
        """Frees whatever resources the child held. Nothing about a child is assumed to live
        forever by default (docs/ARCHITECTURE.md §5.5)."""
        if child_id in self.sessions:
            self.sessions[child_id].status = SessionStatus.DELETED
        # TODO: UPDATE rlm_children SET status = 'deleted' WHERE child_session_id = %s
        # TODO: tear down the child's kernel process/container here.

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _check_active(self, session_id: str) -> SessionHandle:
        session = self.sessions.get(session_id)
        if session is None or session.status != SessionStatus.ACTIVE:
            raise ValueError(f"session {session_id} is not active")
        return session

    def _inherited_model(self, session_id: str) -> str:
        # TODO: look up the parent's actual configured model; a child inherits the parent's
        # model unless the rlm_spawn call requests another one (docs/TECH_STACK.md §6).
        return "inherited-from-parent"

    def _resolve_receiver(
        self, sender_session_id: str, receiver_role: str, receiver_name: str | None
    ) -> str:
        sender = self.sessions[sender_session_id]
        if receiver_role == "parent":
            if sender.parent_session_id is None:
                raise ValueError("root session has no parent to message")
            return sender.parent_session_id
        # receiver_role == 'child': look up by name among this session's children
        for handle in self.children.get(sender_session_id, []):
            if handle.name == receiver_name:
                return handle.child_id
        raise ValueError(f"no child named {receiver_name!r} under session {sender_session_id}")

    async def _run_child(self, child_id: str, task_spec: str) -> None:
        """
        Placeholder child execution loop. In a real build this launches the child's own
        sandboxed kernel (per docs/TECH_STACK.md §7 tiering) and runs its own parent-model
        turn loop against task_spec, per workflows/03_rlm_recursion_pipeline.md steps 3-6.
        On completion (or timeout against max_wall_clock_seconds), it must call
        agent_message_send(child_id, 'parent', None, result) -- never return a value directly.
        """
        session = self.sessions[child_id]
        session.status = SessionStatus.ACTIVE
        # TODO: real kernel turn loop goes here.
        # TODO: on timeout, mark session.status = SessionStatus.ERROR and still message the
        #       parent with whatever partial result exists, so the parent's sufficiency gate
        #       can make an honest decision rather than hanging.
