"""Tests for the internal scheduler (deep_context.scheduler)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deep_context.scheduler import (
    TASKS,
    next_run_from_cron,
    register_default_jobs,
    run_due_jobs_once,
)
from deep_context.storage import get_storage


class TestCronParser:
    def test_every_shorthand(self) -> None:
        now = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
        nxt = next_run_from_cron("every:300", after=now)
        assert nxt == now + timedelta(seconds=300)

    def test_invalid_every_raises(self) -> None:
        with pytest.raises(ValueError):
            next_run_from_cron("every:0")

    def test_wildcard_cron_next_minute(self) -> None:
        now = datetime(2026, 8, 21, 12, 30, 15, tzinfo=timezone.utc)
        nxt = next_run_from_cron("* * * * *", after=now)
        assert nxt == now.replace(second=0, microsecond=0) + timedelta(minutes=1)

    def test_specific_hour(self) -> None:
        now = datetime(2026, 8, 21, 12, 30, 0, tzinfo=timezone.utc)
        nxt = next_run_from_cron("0 14 * * *", after=now)
        assert nxt.hour == 14 and nxt.minute == 0
        assert nxt > now

    def test_step_expression(self) -> None:
        now = datetime(2026, 8, 21, 12, 30, 0, tzinfo=timezone.utc)
        nxt = next_run_from_cron("*/15 * * * *", after=now)
        assert nxt.minute == 45

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError):
            next_run_from_cron("not a cron")
        with pytest.raises(ValueError):
            next_run_from_cron("99 * * * *")


class TestJobPersistence:
    async def test_upsert_and_list_jobs(self) -> None:
        storage = await get_storage()
        now = datetime.now(timezone.utc)
        await storage.upsert_job("test_job", "every:60", now, max_retries=2)
        jobs = await storage.list_jobs()
        assert any(j["name"] == "test_job" for j in jobs)

    async def test_due_jobs_and_lifecycle(self) -> None:
        storage = await get_storage()
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        await storage.upsert_job("lifecycle_job", "every:60", past)

        due = await storage.get_due_jobs(datetime.now(timezone.utc))
        assert any(j["name"] == "lifecycle_job" for j in due)

        await storage.mark_job_running("lifecycle_job")
        running = await storage.get_due_jobs(datetime.now(timezone.utc))
        assert not any(j["name"] == "lifecycle_job" for j in running)

        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        await storage.mark_job_done("lifecycle_job", future)
        jobs = {j["name"]: j for j in await storage.list_jobs()}
        assert jobs["lifecycle_job"]["status"] == "idle"
        assert jobs["lifecycle_job"]["retries"] == 0

    async def test_mark_failed_increments_retries(self) -> None:
        storage = await get_storage()
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        await storage.upsert_job("fail_job", "every:60", past)
        await storage.mark_job_failed("fail_job", "boom")
        jobs = {j["name"]: j for j in await storage.list_jobs()}
        assert jobs["fail_job"]["retries"] == 1
        assert jobs["fail_job"]["status"] == "failed"
        assert "boom" in jobs["fail_job"]["last_error"]


class TestSchedulerLoop:
    async def test_run_due_jobs_once_executes_registered_task(self) -> None:
        executed_calls: list[str] = []

        from deep_context.scheduler import job

        @job("test_tick_task")
        async def test_tick_task() -> None:
            executed_calls.append("ran")

        storage = await get_storage()
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        await storage.upsert_job("test_tick_task", "every:60", past)

        executed = await run_due_jobs_once()
        assert "test_tick_task" in executed
        assert executed_calls == ["ran"]

        # Next run should be scheduled in the future; a second tick is a no-op.
        executed2 = await run_due_jobs_once()
        assert "test_tick_task" not in executed2

    async def test_unknown_task_marked_failed(self) -> None:
        storage = await get_storage()
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        await storage.upsert_job("no_such_task", "every:60", past)
        await run_due_jobs_once()
        jobs = {j["name"]: j for j in await storage.list_jobs()}
        assert jobs["no_such_task"]["status"] == "failed"
        assert "No registered task" in jobs["no_such_task"]["last_error"]

    async def test_failing_task_retries(self) -> None:
        from deep_context.scheduler import job

        @job("test_failing_task")
        async def test_failing_task() -> None:
            raise RuntimeError("intentional failure")

        storage = await get_storage()
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        await storage.upsert_job("test_failing_task", "every:60", past, max_retries=3)

        await run_due_jobs_once()
        jobs = {j["name"]: j for j in await storage.list_jobs()}
        assert jobs["test_failing_task"]["retries"] == 1
        # Retry is scheduled soon (not at the next cron slot).
        next_run = datetime.fromisoformat(jobs["test_failing_task"]["next_run_at"])
        assert next_run <= datetime.now(timezone.utc) + timedelta(minutes=5)

    async def test_register_default_jobs(self) -> None:
        await register_default_jobs()
        jobs = await storage_list()
        names = {j["name"] for j in jobs}
        assert {
            "cleanup_orphaned_docs",
            "reindex_corpus",
            "refresh_embeddings",
        } <= names
        assert set(TASKS.keys()) >= names


async def storage_list() -> list[dict]:
    storage = await get_storage()
    return await storage.list_jobs()


class TestMaintenanceTasks:
    async def test_cleanup_orphaned_chunks(self) -> None:
        from deep_context.core.types import Chunk, ChunkLevel, Document
        from deep_context.scheduler import cleanup_orphaned_docs

        storage = await get_storage()
        # Create a document + chunk, then delete the document row directly
        # (bypassing cascade) to simulate an inconsistent legacy state.
        doc = Document(title="Orphan Source")
        doc_id = await storage.insert_document(doc)
        child = Chunk(
            document_id=doc_id,
            content="orphan candidate",
            token_count=2,
            level=ChunkLevel.CHILD,
        )
        await storage.insert_chunks([child])
        conn = storage._get_conn()
        await conn.execute("PRAGMA foreign_keys = OFF;")
        await conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.commit()

        removed = await storage.cleanup_orphaned_chunks()
        assert removed >= 1
        # The registered scheduler task runs the same operation end-to-end.
        await cleanup_orphaned_docs()

    async def test_rebuild_fts_index(self) -> None:
        from deep_context.core.types import Chunk, ChunkLevel, Document
        from deep_context.scheduler import reindex_corpus

        storage = await get_storage()
        doc = Document(title="FTS Doc")
        doc_id = await storage.insert_document(doc)
        child = Chunk(
            document_id=doc_id,
            content="searchable text",
            token_count=2,
            level=ChunkLevel.CHILD,
        )
        await storage.insert_chunks([child])

        count = await storage.rebuild_fts_index()
        assert count >= 1
        await reindex_corpus()
