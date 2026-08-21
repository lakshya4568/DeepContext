"""Internal job scheduler for ingestion and index maintenance.

A lightweight, Airflow-in-spirit scheduler purpose-built for DeepContext:
- Jobs are persisted in the ``jobs`` table (SQLite or Postgres).
- A polling loop dispatches due jobs to registered task callables.
- Failed jobs are retried up to ``max_retries`` before being marked failed.
- Cron-like schedules are supported via a minimal 5-field cron parser
  (minute hour day-of-month month day-of-week) plus an ``every:Ns`` shorthand.

Run standalone with::

    uv run python -m deep_context.scheduler

or embed the loop in the FastAPI lifespan when SCHEDULER_ENABLED=true.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.storage import close_storage, get_storage

TaskFn = Callable[[], Coroutine[Any, Any, None]]

TASKS: dict[str, TaskFn] = {}


def job(name: str) -> Callable[[TaskFn], TaskFn]:
    """Register a coroutine as a named schedulable task."""

    def decorator(fn: TaskFn) -> TaskFn:
        TASKS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Minimal cron support (5-field: minute hour dom month dow)
# ---------------------------------------------------------------------------


def next_run_from_cron(expr: str, after: datetime | None = None) -> datetime:
    """Compute the next run time for a cron expression strictly after ``after``.

    Supports standard 5-field syntax with ``*``, ``,`` lists, ``-`` ranges,
    and ``*/n`` steps. Also accepts the ``every:<seconds>`` shorthand.
    Raises ValueError on malformed expressions so misconfiguration fails fast.
    """
    expr = expr.strip()
    now = after or datetime.now(timezone.utc)

    if expr.startswith("every:"):
        seconds = int(expr.split(":", 1)[1])
        if seconds <= 0:
            raise ValueError(f"Invalid every-interval in cron expression: {expr}")
        return now + timedelta(seconds=seconds)

    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"Cron expression must have 5 fields (minute hour dom month dow): {expr!r}"
        )

    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    parsed: list[set[int]] = []
    for idx, field in enumerate(fields):
        lo, hi = ranges[idx]
        values: set[int] = set()
        for part in field.split(","):
            step = 1
            if "/" in part:
                part, step_s = part.split("/", 1)
                step = int(step_s)
                if step <= 0:
                    raise ValueError(f"Invalid step in cron field {field!r}")
            if part == "*":
                start, end = lo, hi
            elif "-" in part:
                s, e = part.split("-", 1)
                start, end = int(s), int(e)
            else:
                start = end = int(part)
            if start < lo or end > hi or start > end:
                raise ValueError(f"Cron field {field!r} out of range [{lo}, {hi}]")
            values.update(range(start, end + 1, step))
        parsed.append(values)

    candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    # Bounded search: at most ~4 years of minutes; practically finds a match fast.
    for _ in range(366 * 24 * 60 * 4):
        if (
            candidate.minute in parsed[0]
            and candidate.hour in parsed[1]
            and candidate.day in parsed[2]
            and candidate.month in parsed[3]
            and (candidate.weekday() % 7) in parsed[4]  # Monday=0 -> cron 0..6
        ):
            return candidate
        candidate += timedelta(minutes=1)
    raise ValueError(f"No valid next run found for cron expression: {expr!r}")


# ---------------------------------------------------------------------------
# Built-in maintenance tasks
# ---------------------------------------------------------------------------


@job("cleanup_orphaned_docs")
async def cleanup_orphaned_docs() -> None:
    """Remove chunks/FTS rows whose parent document no longer exists."""
    storage = await get_storage()
    removed = await storage.cleanup_orphaned_chunks()
    logger.info(
        "Scheduler cleanup_orphaned_docs removed %d orphaned chunk(s).", removed
    )


@job("reindex_corpus")
async def reindex_corpus() -> None:
    """Rebuild FTS index entries for all stored child chunks (SQLite path)."""
    storage = await get_storage()
    rebuilt = await storage.rebuild_fts_index()
    logger.info("Scheduler reindex_corpus rebuilt %d FTS entr(ies).", rebuilt)


@job("refresh_embeddings")
async def refresh_embeddings() -> None:
    """Re-embed child chunks that are missing embeddings (mock-safe fallback)."""
    storage = await get_storage()
    refreshed = await storage.backfill_missing_embeddings()
    logger.info("Scheduler refresh_embeddings backfilled %d embedding(s).", refreshed)


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


async def run_due_jobs_once(now: datetime | None = None) -> list[str]:
    """Single scheduler tick: execute all due jobs. Returns executed job names."""
    current = now or datetime.now(timezone.utc)
    storage = await get_storage()
    due = await storage.get_due_jobs(current)
    executed: list[str] = []

    for j in due:
        name = str(j["name"])
        task = TASKS.get(name)
        if task is None:
            await storage.mark_job_failed(name, f"No registered task named '{name}'.")
            continue

        await storage.mark_job_running(name)
        try:
            await task()
            next_run = next_run_from_cron(str(j["schedule_cron"]), current)
            await storage.mark_job_done(name, next_run)
            executed.append(name)
            logger.info("Scheduler job '%s' completed; next run at %s.", name, next_run)
        except Exception as e:
            retries = int(j.get("retries", 0)) + 1
            max_retries = int(j.get("max_retries", settings.scheduler_max_retries))
            await storage.mark_job_failed(name, str(e))
            if retries >= max_retries:
                logger.error(
                    "Scheduler job '%s' failed permanently after %d attempts: %s",
                    name,
                    retries,
                    e,
                )
            else:
                # Retry soon rather than waiting for the next scheduled slot.
                retry_at = current + timedelta(
                    seconds=settings.scheduler_poll_interval * 2
                )
                from deep_context.storage.base import StorageInterface  # noqa: F401

                await storage.upsert_job(
                    name=name,
                    schedule_cron=str(j["schedule_cron"]),
                    next_run_at=retry_at,
                    max_retries=max_retries,
                )
                logger.warning(
                    "Scheduler job '%s' failed (attempt %d/%d): %s",
                    name,
                    retries,
                    max_retries,
                    e,
                )

    return executed


async def register_default_jobs() -> None:
    """Ensure built-in maintenance jobs exist with sensible default schedules."""
    storage = await get_storage()
    defaults: dict[str, str] = {
        "cleanup_orphaned_docs": "every:3600",
        "reindex_corpus": "every:86400",
        "refresh_embeddings": "every:21600",
    }
    existing = {j["name"] for j in await storage.list_jobs()}
    now = datetime.now(timezone.utc)
    for name, schedule in defaults.items():
        if name not in existing:
            await storage.upsert_job(
                name=name,
                schedule_cron=schedule,
                next_run_at=next_run_from_cron(schedule, now),
                max_retries=settings.scheduler_max_retries,
            )
            logger.info("Registered default scheduler job '%s' (%s).", name, schedule)


async def scheduler_loop(poll_interval: int | None = None) -> None:
    """Long-running scheduler loop. Cancel gracefully via asyncio cancellation."""
    interval = poll_interval or settings.scheduler_poll_interval
    logger.info("Scheduler loop started (poll every %ds).", interval)
    try:
        while True:
            try:
                await run_due_jobs_once()
            except Exception as e:
                logger.exception("Scheduler tick failed: %s", e)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Scheduler loop cancelled; shutting down.")
        raise
    finally:
        await close_storage()


def main() -> None:
    """CLI entrypoint: uv run python -m deep_context.scheduler"""
    asyncio.run(register_default_jobs())
    asyncio.run(scheduler_loop())


if __name__ == "__main__":
    main()
