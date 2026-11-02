# Where the benchmark data lives

One Postgres server, one database for the benchmark. Nothing is kept in SQLite any more.

| | |
|---|---|
| Server | Docker container `deepcontext-db`, image `paradedb/paradedb:0.26.4-pg18` (digest `sha256:4abbafffb4e59e679821c5882faf91c494022bb3242fc73b789e753f43df1361`, built 2026-10-29) |
| Versions | PostgreSQL 18.6, pg_search 0.26.4, pgvector 0.8.6 |
| Address | `127.0.0.1:5432` (container port 5432 published as is) |
| Benchmark database | `deepcontext_eval` (one schema per index: `<dataset>_<corpus fingerprint>`, derived deep-ingestion indexes add `_vb*`) |
| Model-call cache | table `eval_cache.kv` in `deepcontext_eval`; namespaces `model_calls` (embeddings, Gemini and rerank replies) and `variant_b_llm` (summariser replies) |
| Connection | `EVAL_PG_DSN` in the git-ignored `.env`; the harness swaps the database name. Never print or commit it. |
| Application database | `awems` on the same server. The benchmark never writes to it. |

Start it with `docker start deepcontext-db` (the container restarts with Docker; it was recreated with `--restart unless-stopped`
and `--shm-size=1g`). Without the bigger `/dev/shm`, pgvector's parallel HNSW build fails with "could not resize shared memory
segment"; if it ever happens again run `SET max_parallel_maintenance_workers = 0` before `CREATE INDEX`.

## History (2026-11-01)

* Every result published before this date was measured on Homebrew PostgreSQL 15.18 with pgvector 0.8.6 (the DSN `127.0.0.1:5432`
  reached Homebrew first, because Homebrew binds the loopback address and Docker only the wildcard). The Docker ParadeDB server was
  never used. The same pgvector version is installed now, so first-stage results should reproduce; see
  `eval/results_paradedb/` (local) for the parity control (`dense_only` against the published baseline).
* The database was copied with `pg_dump | pg_restore` (345 tables, every row count equal, all indexes rebuilt) and the two SQLite
  caches were imported row by row (185,875 + 6,921 rows, `python -m eval.migrate_cache_to_pg`). The old `.sqlite` files are
  still on disk and unused.
* Homebrew PostgreSQL 15 was stopped (`brew services stop postgresql@15`); its data directory `/opt/homebrew/var/postgresql@15`
  is untouched. Backups of the two `awems` databases are in `eval/cache/backups/` (git-ignored).
* After pulling a newer ParadeDB image run `ALTER EXTENSION pg_search UPDATE; ALTER EXTENSION vector UPDATE;` in **every** database
  and in `template1`; new databases are created from `template1`, which carried the old extension objects.

## Vector index

pgvector HNSW, `vector_cosine_ops`, default build parameters (m = 16, ef_construction = 64) and `hnsw.ef_search = 100`, on 768-d
float32 vectors. The application's own schema builds `m = 16, ef_construction = 200`. For up to a few thousand vectors (docs,
thesis, novel, MultiHop-RAG) the planner ignores the index and scans exactly; HNSW is used for SciFact, HotpotQA and FiQA.

## Lexical channels and what they add to the database

`LEXICAL_RANKING` selects the keyword channel of the eval store; the default stays the Okapi BM25 over `tsvector` of the
published baseline. See `eval/PARADEDB_USECASES.md` for why the others exist.

| value | channel | what it adds to `deepcontext_eval` |
|---|---|---|
| `bm25` (default) | Okapi BM25 over `tsvector` | nothing |
| `pg_search` | ParadeDB BM25 (`eval/pg_search.py`), English stemming and stop words | index `idx_chunks_pg_search` on `chunks` of each schema that was queried (the seven fast schemas have it) |
| `main_fts` | port of the fallback `main` actually serves (`eval/main_fts.py`) | extension `pg_trgm`, installed in `public` |

* A table can hold **only one ParadeDB index**. Drop `idx_chunks_pg_search` on a copy before running the application's own
  store on it (its index is `idx_chunks_bm25`).
* `EVAL_STORE=stock` runs the benchmark on the application's own, unmodified store (extensions, feature flags and indexes set
  up by its own startup), e.g. with `PYTHONPATH` pointing at an extract of `main`. Use a throwaway database
  (`EVAL_PG_DATABASE=...`) holding a copy of the schema: the application's startup adds columns, a trigger and indexes.
* An extension must be created `WITH SCHEMA public`; without it the extension lands in whichever per-index schema is first on
  the search path and is invisible to the others.
