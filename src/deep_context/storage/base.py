"""Abstract base interface for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from deep_context.core.types import (
    Chunk,
    Document,
    ExistingMemory,
    RetrievalFilters,
)


class StorageInterface(ABC):
    """Abstract interface defining persistence operations for Agentic Hybrid RAG."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database schema and tables."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connections."""
        pass

    # -----------------------------------------------------------------------
    # Documents & Chunks
    # -----------------------------------------------------------------------

    @abstractmethod
    async def insert_document(self, document: Document) -> str:
        """Insert a document and return its ID."""
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None:
        """Fetch document by ID."""
        pass

    @abstractmethod
    async def list_documents(
        self, tenant_id: str = "default", permission_scope: list[str] | None = None
    ) -> list[Document]:
        """List documents matching tenant and permissions."""
        pass

    @abstractmethod
    async def list_document_summaries(self, limit: int = 50) -> list[dict[str, Any]]:
        """List ingested documents with parent/child chunk counts for UI and diagnostics."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and cascade to its chunks."""
        pass

    @abstractmethod
    async def delete_all_documents(self) -> int:
        """Delete all ingested documents and reset chunk stores."""
        pass

    @abstractmethod
    async def count_chunks_for_document(self, document_id: str) -> tuple[int, int]:
        """Return (child_count, parent_count) for document."""
        pass

    @abstractmethod
    async def get_document_chunks(
        self,
        document_id: str | None = None,
        level: str = "parent",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch chunks for a specific document or all documents."""
        pass

    @abstractmethod
    async def insert_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Insert a batch of chunks (parents and children)."""
        pass

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get chunk by ID."""
        pass

    @abstractmethod
    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get multiple chunks by IDs."""
        pass

    @abstractmethod
    async def search_bm25(
        self,
        query: str,
        filters: RetrievalFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """BM25 full-text search against child chunks."""
        pass

    @abstractmethod
    async def search_vector(
        self,
        query_embedding: list[float],
        filters: RetrievalFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Cosine similarity vector search against child chunks."""
        pass

    # -----------------------------------------------------------------------
    # Typed Memory (Policy, Preference, Fact, Episode)
    # -----------------------------------------------------------------------

    @abstractmethod
    async def get_policy(
        self, tenant_id: str, policy_key: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        """Exact lookup for policy memory."""
        pass

    @abstractmethod
    async def set_policy(
        self,
        tenant_id: str,
        policy_key: str,
        policy_value: dict[str, Any],
        user_id: str | None = None,
    ) -> None:
        """Set a policy memory (operator only)."""
        pass

    @abstractmethod
    async def list_policies(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List active policies."""
        pass

    @abstractmethod
    async def get_preference(self, user_id: str, preference_key: str) -> dict[str, Any] | None:
        """Exact lookup for user preference."""
        pass

    @abstractmethod
    async def set_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: dict[str, Any],
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> None:
        """Set a user preference."""
        pass

    @abstractmethod
    async def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        """List preferences for user."""
        pass

    @abstractmethod
    async def insert_fact(
        self,
        tenant_id: str,
        content: str,
        embedding: list[float] | None,
        source: str,
        confidence: float,
        user_id: str | None = None,
        expires_at: datetime | None = None,
        superseded_by: str | None = None,
    ) -> str:
        """Insert a promoted semantic fact."""
        pass

    @abstractmethod
    async def search_facts(
        self,
        query: str,
        query_embedding: list[float] | None,
        tenant_id: str = "default",
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search across active, unexpired, non-superseded facts."""
        pass

    @abstractmethod
    async def get_facts_for_scope(
        self, tenant_id: str, user_id: str | None = None
    ) -> list[ExistingMemory]:
        """Fetch active facts for scope (used during promotion gate contradiction check)."""
        pass

    @abstractmethod
    async def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> None:
        """Mark an old fact as superseded by a new fact."""
        pass

    @abstractmethod
    async def insert_episode(
        self,
        user_id: str,
        summary: str,
        session_id: str | None = None,
        task_type: str | None = None,
        outcome: str = "success",
        embedding: list[float] | None = None,
    ) -> str:
        """Insert an episodic memory summary."""
        pass

    @abstractmethod
    async def search_episodes(
        self, user_id: str, query_embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search past episodic memories."""
        pass

    # -----------------------------------------------------------------------
    # Events Trace Log
    # -----------------------------------------------------------------------

    @abstractmethod
    async def insert_event_trace(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_id: str | None = None,
        agent_id: str | None = None,
        token_cost: int = 0,
        latency_ms: int = 0,
    ) -> int:
        """Append an event to events_trace table."""
        pass

    @abstractmethod
    async def list_event_traces(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query event trace logs."""
        pass

    # -----------------------------------------------------------------------
    # Scheduler Jobs
    # -----------------------------------------------------------------------

    @abstractmethod
    async def upsert_job(
        self,
        name: str,
        schedule_cron: str,
        next_run_at: datetime,
        max_retries: int = 3,
    ) -> None:
        """Create or update a scheduled job definition."""
        pass

    @abstractmethod
    async def get_due_jobs(self, now: datetime) -> list[dict[str, Any]]:
        """Return jobs whose next_run_at <= now and status is not 'running'."""
        pass

    @abstractmethod
    async def mark_job_running(self, name: str) -> None:
        """Transition a job to 'running' state."""
        pass

    @abstractmethod
    async def mark_job_done(self, name: str, next_run_at: datetime) -> None:
        """Mark a job successful, reset retries, and schedule its next run."""
        pass

    @abstractmethod
    async def mark_job_failed(self, name: str, error: str) -> None:
        """Record a job failure and increment its retry counter."""
        pass

    @abstractmethod
    async def list_jobs(self) -> list[dict[str, Any]]:
        """List all registered jobs with their current state."""
        pass

    # -----------------------------------------------------------------------
    # Maintenance Operations (scheduler tasks)
    # -----------------------------------------------------------------------

    @abstractmethod
    async def cleanup_orphaned_chunks(self) -> int:
        """Delete chunks whose parent document no longer exists. Returns removed count."""
        pass

    @abstractmethod
    async def rebuild_fts_index(self) -> int:
        """Rebuild full-text index entries for all child chunks. Returns entry count."""
        pass

    @abstractmethod
    async def backfill_missing_embeddings(self) -> int:
        """Re-embed child chunks missing embeddings. Returns backfilled count."""
        pass
