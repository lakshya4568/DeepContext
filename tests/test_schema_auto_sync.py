"""Tests for database schema auto-synchronization across platforms."""

import pytest

from deep_context.storage.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_sqlite_schema_auto_sync(tmp_path) -> None:
    db_file = tmp_path / "test_sync.db"
    store = SQLiteStore(db_path=str(db_file))
    await store.initialize()

    # Verify that all expected columns were created/synced
    conn = store._get_conn()
    async with conn.execute("PRAGMA table_info(chunks)") as cursor:
        cols = {row[1] for row in await cursor.fetchall()}
        assert "chunk_index" in cols
        assert "section_path" in cols
        assert "page_number" in cols
        assert "summary_text" in cols
        assert "summary_tokens" in cols
        assert "summary_model" in cols
        assert "generated_at" in cols

    async with conn.execute("PRAGMA table_info(documents)") as cursor:
        doc_cols = {row[1] for row in await cursor.fetchall()}
        assert "metadata" in doc_cols

    await store.close()
