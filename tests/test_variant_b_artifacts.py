"""A deep-ingestion index keeps its glossary, synopsis and summary report in its own Postgres schema.

Database tests use a scratch database on the eval Postgres and are skipped when it is not reachable.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from eval import pg
from eval.variant_b.pipeline import find_artifacts, store_artifacts

SCRATCH_DB = "deepcontext_eval_vbart_test"
SERVER_UP = asyncio.run(pg.server_available())
needs_pg = pytest.mark.skipif(not SERVER_UP, reason="eval Postgres is not reachable")


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# ---- which files belong to a run (pure) ----------------------------------------------------
def test_a_dataset_level_run_uses_the_files_next_to_its_summaries(tmp_path: Path) -> None:
    run = tmp_path / "docsqa"
    for name in ("glossary.json", "synopsis.json", "report.json"):
        _write(run / name, {"f": name})
    (run / "summaries.jsonl").write_text("")
    found = find_artifacts(run / "summaries.jsonl")
    assert set(found) == {"glossary", "synopsis", "summary_report"}
    assert found["summary_report"] == run / "report.json"


def test_a_mode_run_shares_the_datasets_glossary_but_has_its_own_report(tmp_path: Path) -> None:
    dataset = tmp_path / "docsqa"
    _write(dataset / "glossary.json", {"g": 1})
    _write(dataset / "synopsis.json", {"s": 1})
    _write(dataset / "report.json", {"r": "dataset-level"})
    mode = dataset / "modes" / "context_v2"
    _write(mode / "report.json", {"r": "mode-level"})
    (mode / "summaries.jsonl").write_text("")
    found = find_artifacts(mode / "summaries.jsonl")
    assert found["summary_report"] == mode / "report.json"
    assert found["glossary"] == dataset / "glossary.json"
    assert found["synopsis"] == dataset / "synopsis.json"


def test_missing_files_are_simply_not_listed(tmp_path: Path) -> None:
    run = tmp_path / "x"
    _write(run / "report.json", {"r": 1})
    (run / "summaries.jsonl").write_text("")
    assert set(find_artifacts(run / "summaries.jsonl")) == {"summary_report"}


# ---- storing them (Postgres) ---------------------------------------------------------------
async def _recreate_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{SCRATCH_DB}"')
    finally:
        await admin.close()


async def _drop_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture(scope="module", autouse=True)
def scratch_db() -> Any:
    if SERVER_UP:
        asyncio.run(_recreate_db())
    yield
    if SERVER_UP:
        asyncio.run(_drop_db())


@pytest.fixture
async def conn() -> Any:
    if not SERVER_UP:
        pytest.skip("eval Postgres is not reachable")
    schema = f"t_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(pg.dsn_for(SCRATCH_DB))
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await admin.close()
    c = await asyncpg.connect(pg.dsn_for(SCRATCH_DB, schema))
    yield c
    await c.close()


@needs_pg
async def test_artifacts_are_stored_as_json_with_their_source(conn: Any, tmp_path: Path) -> None:
    run = tmp_path / "docsqa"
    _write(run / "glossary.json", {"entries": [{"name": "Alpha", "unicode": "café"}]})
    _write(run / "report.json", {"children": 3})
    (run / "summaries.jsonl").write_text("")
    stored = await store_artifacts(conn, run / "summaries.jsonl")
    assert set(stored) == {"glossary", "summary_report"}
    row = await conn.fetchrow("SELECT data, source FROM vb_artifacts WHERE kind = 'glossary'")
    assert json.loads(row["data"]) == {"entries": [{"name": "Alpha", "unicode": "café"}]}
    assert row["source"].endswith("glossary.json")


@needs_pg
async def test_storing_again_replaces_the_values_and_keeps_one_row_per_kind(
    conn: Any, tmp_path: Path
) -> None:
    run = tmp_path / "d"
    report = _write(run / "report.json", {"children": 1})
    (run / "summaries.jsonl").write_text("")
    await store_artifacts(conn, run / "summaries.jsonl")
    _write(report, {"children": 2})
    await store_artifacts(conn, run / "summaries.jsonl")
    assert await conn.fetchval("SELECT count(*) FROM vb_artifacts") == 1
    assert json.loads(await conn.fetchval("SELECT data FROM vb_artifacts")) == {"children": 2}


@needs_pg
async def test_nothing_to_store_creates_no_rows(conn: Any, tmp_path: Path) -> None:
    run = tmp_path / "empty"
    run.mkdir()
    (run / "summaries.jsonl").write_text("")
    assert await store_artifacts(conn, run / "summaries.jsonl") == []
    assert await conn.fetchval("SELECT count(*) FROM vb_artifacts") == 0
