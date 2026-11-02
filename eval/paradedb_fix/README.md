# Fix for the ParadeDB BM25 path of `main`

`main_bm25_fix.patch` is a plain `git` patch against `main` @ `b720085`. Nothing on `main` was changed; apply it on a branch of your choice:

```bash
git switch -c fix/paradedb-bm25 main
git apply eval/paradedb_fix/main_bm25_fix.patch
```

(From a checkout that lacks `eval/`, copy the patch file anywhere and run `git apply /path/to/main_bm25_fix.patch`.)

## What was wrong

ParadeDB BM25 has never been active in `main`. Three independent bugs, all hidden because every failure was logged at `debug`:

| # | Where | Bug | Effect |
|---|---|---|---|
| 1 | `PostgresStore.initialize()` and `migrations/003` | `CALL paradedb.create_bm25(...)` was removed from pg_search | no BM25 index is ever created |
| 2 | `search_bm25()` | `chunks @@@ paradedb.parse($n)` names the table, but the query aliases it as `c` | `column "chunks" does not exist`, even with a correct index |
| 3 | `search_bm25()` | the stock clauses number their parameters for the fallback statement (`$4`, `$5`) but only `params[3:]` is passed | wrong parameter count, even with 1 and 2 fixed |

Search therefore always ran the "enhanced FTS" fallback (ts_rank, trigram similarity, identifier and phrase boosts).

## What the patch does

* `initialize()` creates `idx_chunks_bm25` with `CREATE INDEX ... USING paradedb`, English stemming and stop words (the same analysis as the tsvector channel). If creation fails, `_has_pg_search` is switched off and a **warning** is logged once.
* `search_bm25()` queries with the `|||` match operator on `content` and `summary_text`, its own parameter numbering, and logs a failure as a warning.
* BM25 receives the **whole cleaned query**, not `main`'s `plain_term` (the first 8 words that are longer than 2 characters and not in a small stop-word list; the index tokenizer already removes stop words). Chosen on the two development sets before the held-out novel was run: nDCG@10 against `main` as shipped, keyword-only, +0.122 [+0.006, +0.243] on the docs and +0.105 [+0.040, +0.172] on the thesis, against +0.057 and +0.032 (both not significant) with the 8-word string.
* The identifier extracted from the query is bound as a parameter instead of being interpolated into the SQL text (it was regex-limited, but string-built SQL is against the project rules).
* `migrations/004_paradedb_modern_bm25_index.sql` creates the index on existing databases.
* `tests/test_paradedb_bm25.py` (5 integration tests, skipped without pg_search and a Postgres server).

## How it was verified (2026-11-01, ParadeDB 0.26.4 on PostgreSQL 18.6)

* The new test file **fails 2 of 5 on the original code** (index missing, search not served by ParadeDB) and **passes 5 of 5 with the patch**.
* The patch file was applied to a pristine extract of `main` and the test run there.
* Demonstration on `main`'s own store: original = no index and a fallback line on every search; patched = index created at startup, no warnings, scores equal to a direct `pdb.score` query.

## What it is worth

Measured on the benchmark harness running `main`'s own store (`EVAL_STORE=stock`), nDCG@10 gain over `main` as shipped (paired 95% CI): on the held-out novel, keyword-only +0.080 [+0.009, +0.164] and fused (`main`'s equal votes) +0.068 [-0.000, +0.137]; on the docs (dev) +0.122 and +0.076, on the thesis (dev) +0.105 and +0.035 (not significant). Full tables and caveats: `eval/PARADEDB_USECASES.md`. The BM25 engine on its own matches the Okapi-over-tsvector search of the published baseline; the larger lever is how much the lexical channel counts in the fusion.

## Limits

Verified on ParadeDB 0.26.4. For 0.25.x keep `WITH (key_field = 'id')` (0.26 treats it as a no-op and prints a warning). Not exercised with summaries (`summary_text` is NULL in the benchmark corpora).
