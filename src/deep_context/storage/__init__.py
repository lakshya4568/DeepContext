"""Storage layer initialization and store factory."""

from __future__ import annotations

from deep_context.core.config import settings
from deep_context.storage.base import StorageInterface
from deep_context.storage.postgres_store import PostgresStore
from deep_context.storage.sqlite_store import SQLiteStore

_active_store: StorageInterface | None = None


async def get_storage() -> StorageInterface:
    """Get or initialize the global storage instance."""
    global _active_store
    if _active_store is None:
        if settings.database_type.lower() == "postgres":
            _active_store = PostgresStore()
        else:
            _active_store = SQLiteStore()
        await _active_store.initialize()
    return _active_store


async def close_storage() -> None:
    """Close active storage connection."""
    global _active_store
    if _active_store is not None:
        await _active_store.close()
        _active_store = None
