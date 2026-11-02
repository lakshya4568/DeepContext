"""The eval model-call cache lives in Postgres (it used to be two SQLite files).

Database tests run against a scratch database, never the real cache tables, and are skipped when the
eval Postgres server is not reachable.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest
from eval import config as C
from eval import pg
from eval.migrate_cache_to_pg import migrate_sqlite_kv
from eval.pg_cache import MemoryKV, PgKV

SCRATCH_DB = "deepcontext_eval_pgkv_test"
SERVER_UP = asyncio.run(pg.server_available())
needs_pg = pytest.mark.skipif(not SERVER_UP, reason="eval Postgres is not reachable")


async def _drop_scratch_db() -> None:
    admin = await asyncpg.connect(pg.dsn_for("postgres"))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture(scope="module")
def scratch_db() -> Any:
    if SERVER_UP:
        asyncio.run(_drop_scratch_db())
    yield SCRATCH_DB
    if SERVER_UP:
        asyncio.run(_drop_scratch_db())


def _namespace() -> str:
    return f"t_{uuid.uuid4().hex[:10]}"


@pytest.fixture(params=["memory", "postgres"])
def kv(request: pytest.FixtureRequest, scratch_db: str) -> Any:
    """The same behaviour must hold for the in-memory and the Postgres store."""
    if request.param == "memory":
        yield MemoryKV()
        return
    if not SERVER_UP:
        pytest.skip("eval Postgres is not reachable")
    store = PgKV(_namespace(), database=scratch_db)
    yield store
    store.close()


# ---- behaviour shared by both stores -------------------------------------------------------
def test_missing_key_returns_none(kv: Any) -> None:
    assert kv.get("absent") is None


def test_put_then_get_returns_the_same_bytes(kv: Any) -> None:
    blob = bytes(range(256)) * 12 + b"\x00tail"  # ~3 KB with NUL bytes, like a float32 vector
    kv.put("k", blob)
    assert kv.get("k") == blob


def test_put_replaces_an_existing_value(kv: Any) -> None:
    kv.put("k", b"old")
    kv.put("k", b"new")
    assert kv.get("k") == b"new"
    assert kv.count() == 1


def test_put_many_inserts_only_missing_keys_and_reports_how_many(kv: Any) -> None:
    kv.put("a", b"keep")
    inserted = kv.put_many([("a", b"changed"), ("b", b"2"), ("c", b"3")])
    assert inserted == 2
    assert kv.get("a") == b"keep"
    assert kv.count() == 3


# ---- Postgres-only behaviour ---------------------------------------------------------------
@needs_pg
def test_namespaces_do_not_see_each_other(scratch_db: str) -> None:
    a, b = PgKV(_namespace(), database=scratch_db), PgKV(_namespace(), database=scratch_db)
    try:
        a.put("k", b"1")
        assert b.get("k") is None
        assert (a.count(), b.count()) == (1, 0)
    finally:
        a.close()
        b.close()


@needs_pg
def test_a_new_connection_sees_values_written_earlier(scratch_db: str) -> None:
    ns = _namespace()
    first = PgKV(ns, database=scratch_db)
    first.put("k", b"v")
    first.close()
    second = PgKV(ns, database=scratch_db)
    try:
        assert second.get("k") == b"v"
    finally:
        second.close()


@needs_pg
def test_threads_can_write_and_read_at_the_same_time(scratch_db: str) -> None:
    store = PgKV(_namespace(), database=scratch_db)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(40):
                key, val = f"{n}-{i}", f"{n}:{i}".encode()
                store.put(key, val)
                assert store.get(key) == val
        except BaseException as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        assert store.count() == 8 * 40
    finally:
        store.close()


@needs_pg
def test_usable_from_inside_a_running_event_loop(scratch_db: str) -> None:
    """The harness calls the cache synchronously from coroutines."""
    store = PgKV(_namespace(), database=scratch_db)

    async def inside_loop() -> bytes | None:
        store.put("k", b"v")
        return store.get("k")

    try:
        assert asyncio.run(inside_loop()) == b"v"
    finally:
        store.close()


def test_unreachable_server_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_PG_DSN", "postgresql://nobody:x@127.0.0.1:9/postgres")
    with pytest.raises(RuntimeError, match="not reachable"):
        PgKV(_namespace())


# ---- importing the old SQLite cache files --------------------------------------------------
def _make_sqlite(path: Path, rows: list[tuple[str, bytes | str]], value_type: str) -> None:
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE kv (k TEXT PRIMARY KEY, v {value_type} NOT NULL)")
    con.executemany("INSERT INTO kv VALUES (?, ?)", rows)
    con.commit()
    con.close()


def test_migration_copies_blob_values_unchanged(tmp_path: Path) -> None:
    rows = [(f"k{i}", bytes([i % 256]) * 300) for i in range(2500)]
    _make_sqlite(tmp_path / "c.sqlite", rows, "BLOB")
    target = MemoryKV()
    res = migrate_sqlite_kv(tmp_path / "c.sqlite", target, batch=1000)
    assert (res.read, res.inserted) == (2500, 2500)
    assert all(target.get(k) == v for k, v in rows)


def test_migration_stores_text_values_as_utf8_bytes(tmp_path: Path) -> None:
    rows = [("k", '{"text": "café → ok"}')]
    _make_sqlite(tmp_path / "c.sqlite", rows, "TEXT")
    target = MemoryKV()
    migrate_sqlite_kv(tmp_path / "c.sqlite", target)
    assert target.get("k") == rows[0][1].encode("utf-8")


def test_migration_is_repeatable_and_never_overwrites_newer_values(tmp_path: Path) -> None:
    _make_sqlite(tmp_path / "c.sqlite", [("a", b"old"), ("b", b"2")], "BLOB")
    target = MemoryKV()
    target.put("a", b"newer")
    first = migrate_sqlite_kv(tmp_path / "c.sqlite", target)
    second = migrate_sqlite_kv(tmp_path / "c.sqlite", target)
    assert (first.read, first.inserted, second.inserted) == (2, 1, 0)
    assert target.get("a") == b"newer"


def test_migration_leaves_the_source_file_untouched(tmp_path: Path) -> None:
    src = tmp_path / "c.sqlite"
    _make_sqlite(src, [("a", b"1")], "BLOB")
    before = src.read_bytes()
    migrate_sqlite_kv(src, MemoryKV())
    assert src.read_bytes() == before


# ---- wiring into the two model-call clients ------------------------------------------------
class _Spy(MemoryKV):
    made: list[str] = []

    def __init__(self, namespace: str = "") -> None:
        super().__init__()
        type(self).made.append(namespace)


def test_gemini_runtime_caches_in_the_model_calls_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval.strict_clients as sc

    _Spy.made = []
    monkeypatch.setattr(sc, "PgKV", _Spy)
    runtime = sc.VertexRuntime()
    assert _Spy.made == ["model_calls"]
    assert isinstance(runtime.cache, _Spy)


def test_summariser_client_caches_in_its_own_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    import eval.variant_b.llm as llm

    _Spy.made = []
    monkeypatch.setattr(llm, "PgKV", _Spy)
    llm.SlmClient("http://127.0.0.1:1", "m")
    assert _Spy.made == ["variant_b_llm"]


def test_summariser_client_reuses_a_stored_reply_without_calling_the_server() -> None:
    from eval.variant_b.llm import SlmClient

    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    client = SlmClient("http://127.0.0.1:1", "m", cache=MemoryKV())
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def twice() -> tuple[Any, Any]:
        return await client.chat("sys", "user"), await client.chat("sys", "user")

    first, second = asyncio.run(twice())
    assert (first.text, first.cached, second.text, second.cached) == ("hello", False, "hello", True)
    assert calls == [1]


# ---- the eval DSN -------------------------------------------------------------------------
def test_eval_dsn_is_read_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("EVAL_PG_DSN=postgresql://u:p@db.example:1234/ignored\n")
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("EVAL_PG_DSN", "placeholder")  # records the original state for teardown
    monkeypatch.delenv("EVAL_PG_DSN")
    assert pg.dsn_for("deepcontext_eval").startswith(
        "postgresql://u:p@db.example:1234/deepcontext_eval"
    )
