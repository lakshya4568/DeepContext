"""One-off import of the old SQLite cache files into the Postgres cache (the source is opened read-only).

    uv run --no-sync python -m eval.migrate_cache_to_pg              # both default files
    uv run --no-sync python -m eval.migrate_cache_to_pg PATH NAME    # one file into namespace NAME

Rows that already exist in Postgres are kept as they are, so the import can be repeated safely.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from eval import config as C
from eval.pg_cache import KV, PgKV

# (file name, Postgres namespace) of the caches the harness used before the move to Postgres.
LEGACY_CACHES = (("llm_cache.sqlite", "model_calls"), ("variant_b_llm.sqlite", "variant_b_llm"))


@dataclass(frozen=True)
class MigrationResult:
    read: int
    inserted: int


def migrate_sqlite_kv(path: Path, target: KV, *, batch: int = 1000) -> MigrationResult:
    """Copy every ``kv`` row of a SQLite cache into ``target`` (text values become UTF-8 bytes)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    read = inserted = 0
    try:
        cursor = con.execute("SELECT k, v FROM kv")
        while rows := cursor.fetchmany(batch):
            pairs = [(k, v if isinstance(v, bytes) else v.encode("utf-8")) for k, v in rows]
            read += len(pairs)
            inserted += target.put_many(pairs)
    finally:
        con.close()
    return MigrationResult(read, inserted)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    jobs = (
        [(Path(args[0]), args[1])]
        if len(args) == 2
        else [(C.CACHE_DIR / name, ns) for name, ns in LEGACY_CACHES]
    )
    for path, namespace in jobs:
        if not path.exists():
            print(f"{path}: not found, skipped")
            continue
        store = PgKV(namespace)
        try:
            before = store.count()
            res = migrate_sqlite_kv(path, store)
            print(
                f"{path.name} -> eval_cache.kv[{namespace}]: read {res.read}, inserted {res.inserted}, "
                f"rows before {before}, after {store.count()}"
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
