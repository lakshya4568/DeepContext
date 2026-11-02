"""Key/value cache of the eval harness, stored in Postgres (database ``deepcontext_eval``).

It replaces the two SQLite files the harness used to keep (Gemini/embedding/rerank calls and the
summariser calls). One table, ``eval_cache.kv``, holds every namespace. The API is synchronous because
the harness calls it from inside coroutines; the asyncpg pool runs on a private event loop in a
background thread, so a call never touches the caller's loop.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Iterable
from typing import Any, Protocol

import asyncpg

from eval import pg

SCHEMA = "eval_cache"
_DDL_LOCK = 7270019  # advisory-lock key: serialise DDL across concurrent runner processes


class KV(Protocol):
    """What the harness needs from a cache."""

    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, value: bytes) -> None: ...
    def put_many(self, pairs: Iterable[tuple[str, bytes]]) -> int: ...
    def count(self) -> int: ...
    def flush(self) -> None: ...


class MemoryKV:
    """Same API as :class:`PgKV`, kept in a dict (tests and throw-away runs)."""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._d.get(key)

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._d[key] = value

    def put_many(self, pairs: Iterable[tuple[str, bytes]]) -> int:
        """Insert the missing keys only; returns how many were new."""
        inserted = 0
        with self._lock:
            for k, v in pairs:
                if k not in self._d:
                    self._d[k] = v
                    inserted += 1
        return inserted

    def count(self) -> int:
        with self._lock:
            return len(self._d)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class PgKV:
    """Namespaced key/value store on Postgres (``eval_cache.kv``); safe across threads and processes."""

    def __init__(
        self, namespace: str, *, database: str | None = None, schema: str = SCHEMA
    ) -> None:
        self.namespace = namespace
        self._schema = pg.schema_name(schema)
        self._database = database or pg.EVAL_DB
        self._dsn = pg.dsn_for(self._database)  # also validates the database name
        self._pool: asyncpg.Pool | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name=f"pgkv-{namespace}", daemon=True
        )
        self._thread.start()
        try:
            self._pool = self._run(self._open())
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    # ---- plumbing -------------------------------------------------------------------
    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _open(self) -> asyncpg.Pool:
        try:
            await self._ensure_database()
            pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=8, timeout=15)
        except (OSError, TimeoutError, asyncpg.PostgresError) as e:
            raise RuntimeError(
                f"Postgres for the eval cache is not reachable ({type(e).__name__}). Start the Docker "
                "container 'deepcontext-db' and its port-55432 forwarder (see eval/DATABASE.md)."
            ) from e
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", _DDL_LOCK)
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')
            await conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{self._schema}".kv ('
                "ns TEXT NOT NULL, k TEXT NOT NULL, v BYTEA NOT NULL, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (ns, k))"
            )
        return pool

    async def _ensure_database(self) -> None:
        admin = await asyncpg.connect(pg.dsn_for("postgres"), timeout=10)
        try:
            exists = await admin.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", self._database
            )
            if not exists:
                try:
                    await admin.execute(f'CREATE DATABASE "{self._database}"')  # name validated
                except asyncpg.DuplicateDatabaseError:
                    pass  # another runner process created it first
        finally:
            await admin.close()

    # ---- cache API ------------------------------------------------------------------
    def get(self, key: str) -> bytes | None:
        return self._run(self._get(key))  # type: ignore[no-any-return]

    def put(self, key: str, value: bytes) -> None:
        self._run(self._put(key, value))

    def put_many(self, pairs: Iterable[tuple[str, bytes]]) -> int:
        """Insert the missing keys only (never overwrites); returns how many were new."""
        items = list(pairs)
        if not items:
            return 0
        return self._run(self._put_many([k for k, _ in items], [v for _, v in items]))  # type: ignore[no-any-return]

    def count(self) -> int:
        return self._run(self._count())  # type: ignore[no-any-return]

    def flush(self) -> None:  # every put is already committed
        return None

    def close(self) -> None:
        if self._loop.is_closed():
            return
        if self._pool is not None:
            try:
                self._run(self._pool.close())
            finally:
                self._pool = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()

    # ---- queries (run on the private loop) ------------------------------------------
    def _pool_or_fail(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("the Postgres cache is closed")
        return self._pool

    async def _get(self, key: str) -> bytes | None:
        value = await self._pool_or_fail().fetchval(
            f'SELECT v FROM "{self._schema}".kv WHERE ns = $1 AND k = $2', self.namespace, key
        )
        return None if value is None else bytes(value)

    async def _put(self, key: str, value: bytes) -> None:
        await self._pool_or_fail().execute(
            f'INSERT INTO "{self._schema}".kv (ns, k, v) VALUES ($1, $2, $3) '
            "ON CONFLICT (ns, k) DO UPDATE SET v = EXCLUDED.v, created_at = now()",
            self.namespace, key, value,
        )  # fmt: skip

    async def _put_many(self, keys: list[str], values: list[bytes]) -> int:
        rows = await self._pool_or_fail().fetch(
            f'INSERT INTO "{self._schema}".kv (ns, k, v) '
            "SELECT $1, t.k, t.v FROM unnest($2::text[], $3::bytea[]) AS t(k, v) "
            "ON CONFLICT (ns, k) DO NOTHING RETURNING 1",
            self.namespace, keys, values,
        )  # fmt: skip
        return len(rows)

    async def _count(self) -> int:
        return int(
            await self._pool_or_fail().fetchval(
                f'SELECT count(*) FROM "{self._schema}".kv WHERE ns = $1', self.namespace
            )
        )
