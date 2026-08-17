"""Test configuration and fixtures."""

from __future__ import annotations

import os
import tempfile
from typing import AsyncIterator

import pytest

from deep_context.core.config import settings
from deep_context.storage import close_storage
from deep_context.storage.sqlite_store import SQLiteStore


@pytest.fixture(autouse=True)
def mock_live_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests run deterministic offline mock embeddings and completions."""
    from deep_context.core.llm_client import llm_client

    monkeypatch.setattr(llm_client, "_has_live_client", lambda: False)
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.setattr(llm_client, "_groq_client", None)
    monkeypatch.setattr(llm_client, "_gemini_client", None)
    monkeypatch.setattr(llm_client, "_refresh_gemini_client", lambda: None)


@pytest.fixture(autouse=True)
async def test_db() -> AsyncIterator[SQLiteStore]:
    """Provide a fresh isolated SQLite test database for each test."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_deep_context.db")
    settings.sqlite_db_path = db_path
    settings.database_type = "sqlite"
    settings.allow_mock_fallback = True

    await close_storage()
    store = SQLiteStore(db_path=db_path)
    await store.initialize()

    yield store

    await store.close()
    await close_storage()
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)
    except Exception:
        pass
