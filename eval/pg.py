"""PostgreSQL helpers for the benchmark index.

All benchmark data lives in a dedicated database (``EVAL_PG_DATABASE``, default
``deepcontext_eval``) on the server configured by ``POSTGRES_DSN`` - never in the application's own
database. Each index gets its own schema (``<dataset>_<fingerprint>``), selected through the DSN's
``search_path`` so the unmodified ``PostgresStore`` (unqualified table names) writes into it.
Credentials come from the environment / ``.env`` and are never logged.
"""

from __future__ import annotations

import os
import re
import uuid
from urllib.parse import urlparse, urlunparse

import asyncpg

EVAL_DB = os.environ.get("EVAL_PG_DATABASE", "deepcontext_eval")
# Documented production profile (README / DATABASE_INDEXING_STRATEGY): ef_search = 100. The stock
# store never sets it, so pgvector's default of 40 would cap a LIMIT 100 dense query at 40 rows.
HNSW_EF_SEARCH = int(os.environ.get("EVAL_HNSW_EF_SEARCH", "100"))
_IDENT = re.compile(r"^[a-z0-9_]{1,63}$")
_DOC_NS = uuid.UUID("0d1e5c0e-7a52-4a7e-9d1e-0c0de0c0beef")


def doc_uuid(external_id: str) -> str:
    """Deterministic UUID for a benchmark document id (Postgres ids are UUIDs)."""
    return str(uuid.uuid5(_DOC_NS, f"doc|{external_id}"))


def _base_dsn() -> str:
    dsn = os.environ.get("EVAL_PG_DSN")
    if not dsn:
        from deep_context.core.config import settings

        dsn = settings.postgres_dsn
    return dsn


def dsn_for(database: str, schema: str | None = None) -> str:
    """Same server/credentials as the configured DSN, different database (+ optional schema)."""
    if not _IDENT.match(database):
        raise ValueError(f"unsafe database name {database!r}")
    u = urlparse(_base_dsn())
    query = f"search_path={schema},public" if schema else ""
    return urlunparse(u._replace(path=f"/{database}", query=query))


def schema_name(index_name: str) -> str:
    s = re.sub(r"[^a-z0-9_]", "_", index_name.lower())[:60]
    if not _IDENT.match(s):
        raise ValueError(f"cannot derive a safe schema name from {index_name!r}")
    return s


async def ensure_database() -> None:
    """Create the eval database + extensions once; set the documented HNSW search width."""
    admin = await asyncpg.connect(dsn_for("postgres"))
    try:
        if not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", EVAL_DB):
            await admin.execute(f'CREATE DATABASE "{EVAL_DB}"')  # name validated by _IDENT
    finally:
        await admin.close()
    conn = await asyncpg.connect(dsn_for(EVAL_DB))
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        await conn.execute(f'ALTER DATABASE "{EVAL_DB}" SET hnsw.ef_search = {HNSW_EF_SEARCH}')
    finally:
        await conn.close()


async def ensure_schema(schema: str) -> None:
    if not _IDENT.match(schema):
        raise ValueError(f"unsafe schema name {schema!r}")
    conn = await asyncpg.connect(dsn_for(EVAL_DB))
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await conn.close()


async def drop_schema(schema: str) -> None:
    """Destructive; only ever called explicitly (tests, or by the user)."""
    if not _IDENT.match(schema):
        raise ValueError(f"unsafe schema name {schema!r}")
    conn = await asyncpg.connect(dsn_for(EVAL_DB))
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


async def server_available() -> bool:
    try:
        conn = await asyncpg.connect(dsn_for("postgres"), timeout=5)
    except Exception:  # noqa: BLE001 - connectivity probe: any failure means "not available"
        return False
    await conn.close()
    return True


def _stock_schema_ddl() -> str:
    """The stock ``PostgresStore.initialize`` schema DDL, extracted from its source (no copy)."""
    import inspect

    from deep_context.storage.postgres_store import PostgresStore

    src = inspect.getsource(PostgresStore.initialize)
    m = re.search(r'conn\.execute\(\s*"""(.*?)"""\s*\)', src, re.S)
    if not m:
        raise RuntimeError("Could not locate the schema DDL in PostgresStore.initialize.")
    return m.group(1)


def make_store_class():  # type: ignore[no-untyped-def]
    """``PostgresStore`` whose schema works on a fresh database with current pgvector.

    On pgvector 0.8.x the stock DDL fails ("column does not have dimensions") because it builds
    HNSW indexes on untyped ``VECTOR`` columns and also ALTERs the columns back to untyped. This
    subclass runs the *same* DDL with ``VECTOR(<dim>)`` columns and without the ALTER block.
    Queries, indexes (hnsw, vector_cosine_ops, default m/ef_construction) and tables are unchanged.
    """
    from pgvector.asyncpg import register_vector

    from deep_context.storage.postgres_store import PostgresStore
    from eval import config as C

    class EvalPostgresStore(PostgresStore):
        async def initialize(self) -> None:
            async def init_conn(conn: asyncpg.Connection) -> None:
                await register_vector(conn)

            ddl = _stock_schema_ddl()
            ddl = re.sub(r"DO \$\$.*?END \$\$;", "", ddl, flags=re.S)
            ddl = ddl.replace("embedding VECTOR,", f"embedding VECTOR({C.EMBEDDING_DIM}),")
            assert "ALTER COLUMN embedding" not in ddl and f"VECTOR({C.EMBEDDING_DIM})" in ddl
            # The extension must exist before the pool's init hook can register the vector codec.
            boot = await asyncpg.connect(self.dsn)
            try:
                await boot.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            finally:
                await boot.close()
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn, init=init_conn, min_size=2, max_size=10
            )
            async with self._pool.acquire() as conn:
                await conn.execute(ddl)

    return EvalPostgresStore
