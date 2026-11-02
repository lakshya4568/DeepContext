# ParadeDB in DeepContext: history, use cases, what was broken, what it is worth

Measured on 2026-11-01 with ParadeDB 0.26.4 (PostgreSQL 18.6, pgvector 0.8.6). Retrieval only: no reranker, no LLM.
Every number comes from a run in `eval/results_paradedb/` (local, git-ignored, because the PDF-set rows carry book-derived
queries) or from a test in `tests/`; nothing is estimated. Datasets: the project docs (35 questions; a **development** set),
the thesis (31; development-2), the novel (31; the held-out test), SciFact (300), MultiHop-RAG (265), HotpotQA (300), FiQA
(648). CovidQA was not run (it needs the HuggingFace `datasets` package).

> **Update 2026-11-01:** the numbers below were measured on the older benchmark base (26 commits behind `main`). The same
> questions were then re-run on `main`'s own code (its chunker, store and fusion): see `MAIN_REBASE.md` and
> `MAIN_REBASE_RESULTS.md`. The conclusions agree: shipped fusion is below dense-only, the BM25 patch fixes the keyword
> channel, and a keyword vote of 0.02 brings the fused result back to dense-only level.

## 1. When ParadeDB was adopted, and where its code is

* Commit `684d533`, 2026-08-27: "feat(retrieval): upgrade hybrid search with pg_search, pg_trgm, sibling expansion, and DB
  auto-sync". It integrates pg_search BM25 and pg_trgm in `PostgresStore`, adds schema auto-sync at startup, "sibling window
  expansion" in parent-chunk resolution, query-time section filters in `QueryClassifier`, migration 003, tests and docs. The
  ParadeDB image used locally was built the same day.
* It is in `main`, `feat/contextual-parent-child-rag`, `feature/multimodal-provenance-rag` and `worktree-rag-claude-md`. It is
  **not** in `feat/rag-quality-generation`, the branch the benchmark was built on, nor in `feat/llm-summary-hybrid-chunking`
  or `window-update`.
* What the docs promise (`main`): README, "Dual-Channel Hybrid Retrieval: pg_search (ParadeDB BM25) or native weighted
  `tsvector` with pgvector HNSW via Reciprocal Rank Fusion (k = 60)"; `docs/INDEXING.md`, `idx_chunks_bm25`, "True Okapi BM25
  index over chunk content and contextual summaries" and `idx_chunks_trgm`, a trigram index for identifiers, SKUs and typos.

## 2. Every extension use case in the code, and its verified status

| Feature | Where (`main`) | Intended gain | Status on 2026-11-01 |
|---|---|---|---|
| pg_search BM25 | `PostgresStore.initialize`, `search_bm25`, migration 003 | true BM25 over content and summaries | **never active** (three bugs, section 3); fixed by the patch |
| pg_trgm | `idx_chunks_trgm`; `similarity() x 2` and an exact-identifier `ILIKE` boost (+8) in the fallback | typos, codes, identifiers | active, but inside the fallback; that fallback loses to BM25 (section 4) |
| pgvector HNSW | `m = 16, ef_construction = 200`; `SET LOCAL hnsw.ef_search` per query | fast dense search | active; the benchmark used default build parameters; below about 3.5k vectors Postgres scans exactly anyway |
| Sibling window expansion | `RetrievalEngine._resolve_parent_chunks` | stitched context | only a label: two parents on adjacent pages are renamed "Pages a-b"; no neighbouring text is added |
| Query-time filters | `QueryClassifier.extract_semantic_filters` | scope a question to "chapter 10" / "appendix B" | works only where `section_path` has that form; not exercised by the benchmark corpora |
| Schema auto-sync | `initialize`, migration 003 | easy setup | works |
| pg_ivm, pg_cron, postgis, fuzzystrmatch, pg_stat_statements | none | none (they ship with the image) | unused by any code |

## 3. What was wrong with the BM25 path, and the fix

Three independent bugs, all hidden because each failure is logged at `debug` and search then silently runs its fallback
("enhanced FTS": ts_rank, trigram similarity, identifier and quoted-phrase boosts):

1. `CALL paradedb.create_bm25(...)` (startup and migration 003) was removed from pg_search: **the index is never created**.
2. The query `chunks @@@ paradedb.parse($n)` names the table while the statement aliases it as `c`: `column "chunks" does not
   exist`, **even with a correct index** (shown by creating one by hand).
3. The stock WHERE clauses number their parameters for the fallback (`$4`, `$5`) but only `params[3:]` is passed: **wrong
   parameter count** once 1 and 2 are fixed.

`eval/paradedb_fix/main_bm25_fix.patch` fixes all three (modern `CREATE INDEX ... USING paradedb`, the `|||` match operator,
own parameter numbering), logs failures as warnings, binds the identifier instead of interpolating it into SQL, and adds
migration 004 and five integration tests. The tests **fail 2 of 5 on `main` and pass 5 of 5 with the patch**, which was applied
to a pristine extract of `main` (nothing on `main` was changed). Details: `eval/paradedb_fix/README.md`. A table can hold only
one ParadeDB index.

## 4. What the benchmark says

### 4.1 The keyword channel and the fusion, on the seven test sets

Setups (all first stage, nDCG@10, mean; content-hash dedup): **published** = Okapi BM25 over `tsvector`, lexical vote 0.1, the
configuration of every published result; **B** = ParadeDB BM25, vote 0.1; **C** = ParadeDB BM25, vote 1.0; **D** = **what the
application serves today** (the fallback; `eval/main_fts.py` is a port that returned identical rankings and scores to `main`'s
real code on 72 queries and equals `main`'s own store in the full pipeline on docs, thesis and novel), equal votes as in `main`'s
RRF; **F** = ParadeDB BM25, vote 0.02, the weight chosen on development sets (4.2).

| dataset | n | dense only | published fused (0.1) | B: pg_search 0.1 | C: pg_search 1.0 | **D: app today** | F: pg_search 0.02 |
|---|---|---|---|---|---|---|---|
| docs (dev) | 35 | 0.824 | 0.826 | 0.831 | 0.837 | 0.761 | 0.827 |
| thesis (dev) | 31 | 0.518 | 0.537 | 0.535 | 0.511 | 0.476 | 0.530 |
| novel | 31 | 0.779 | 0.722 | 0.737 | 0.510 | 0.442 | 0.776 |
| SciFact | 300 | 0.900 | 0.867 | 0.870 | 0.783 | 0.723 | 0.898 |
| MultiHop-RAG | 265 | 0.804 | 0.814 | 0.813 | 0.812 | 0.662 | 0.805 |
| HotpotQA | 300 | 0.982 | 0.978 | 0.981 | 0.944 | 0.874 | 0.984 |
| FiQA | 648 | 0.614 | 0.592 | 0.594 | 0.453 | 0.394 | 0.612 |

Keyword search alone:

| dataset | published (Okapi BM25) | ParadeDB BM25 (B) | **app today (D)** |
|---|---|---|---|
| docs (dev) | 0.799 | 0.780 | 0.658 |
| thesis (dev) | 0.405 | 0.409 | 0.304 |
| novel | 0.323 | 0.301 | 0.220 |
| SciFact | 0.688 | 0.684 | 0.551 |
| MultiHop-RAG | 0.751 | 0.758 | 0.454 |
| HotpotQA | 0.861 | 0.866 | 0.710 |
| FiQA | 0.243 | 0.248 | 0.167 |

Fused minus dense-only (paired, 95% CI):

| dataset | published fused (0.1) | F: pg_search 0.02 | **app today (D)** |
|---|---|---|---|
| docs (dev) | +0.003 [-0.004, +0.012] | +0.004 [+0.000, +0.011] | -0.062 [-0.135, +0.005] |
| thesis (dev) | +0.019 [+0.006, +0.035] | +0.012 [+0.001, +0.028] | -0.042 [-0.082, -0.002] |
| novel | -0.057 [-0.105, -0.015] | -0.003 [-0.017, +0.007] | -0.337 [-0.468, -0.210] |
| SciFact | -0.033 [-0.048, -0.019] | -0.002 [-0.007, +0.002] | -0.176 [-0.213, -0.140] |
| MultiHop-RAG | +0.010 [+0.005, +0.015] | +0.001 [-0.000, +0.003] | -0.142 [-0.165, -0.118] |
| HotpotQA | -0.004 [-0.009, +0.001] | +0.002 [+0.000, +0.004] | -0.108 [-0.128, -0.089] |
| FiQA | -0.022 [-0.032, -0.012] | -0.001 [-0.005, +0.002] | -0.219 [-0.242, -0.196] |

What it says:

* **ParadeDB BM25 is as good as the Okapi BM25 already used; it is not better.** B against the published setup: the paired CI
  includes zero in 13 of 14 comparisons (the exception, HotpotQA fused, is +0.003). Its gain is operational (one native index
  query instead of several statistics queries), not accuracy.
* **The keyword channel the application actually runs is much weaker than real BM25** (D, keyword-only): worse than the
  published BM25 on all seven sets, by 0.08 to 0.30 nDCG@10 (every CI excludes zero). This is what the BM25 patch repairs.
* **Equal votes cost more than the engine does.** Even with real BM25 (C), equal votes fall below dense-only on SciFact,
  FiQA, HotpotQA and the novel; the application's setup (D) is below dense-only on all seven sets (significantly on six, by
  0.04 to 0.34; the docs set is -0.062 with a CI that touches zero).
* **A small lexical vote makes the hybrid tie dense-only**: F is within 0.012 nDCG@10 of dense-only on every set (+0.012 on
  the thesis is the largest), and it beats the published fused setup on four sets (FiQA +0.021, SciFact +0.031, novel +0.054,
  HotpotQA +0.006; MultiHop -0.009; docs and thesis unchanged). The lexical channel adds no candidate recall in these corpora
  (Recall@100 of the fused list: identical to dense-only within 0.001 for the published setup and F on every set; D is lower on
  FiQA, 0.882 against 0.918).
* **Limit:** first stage only. A reranker over the fused top 100 would shrink these gaps (the deep-ingestion study found it
  erased most first-stage differences) but cannot repair a candidate list that is missing the answer (D on FiQA). The corpora are
  dense-friendly; identifier-heavy queries, the case trigram and the identifier boost were written for, are not represented here.

### 4.2 The lexical vote, chosen on development sets only

The published 0.1 was picked from {0.1, 0.25, 0.5, 1.0} on SciFact-train and FiQA-dev (300 questions each): nDCG@10 fell with every
step up, so the best value sat at the **edge of the grid** and nothing below 0.1 had been tried. The sweep was repeated for
ParadeDB BM25 with weights down to 0.02, on the same development questions, with the rule written down before any result was
seen: *best mean dev nDCG@10 over the two sets, ties to the smaller weight; if the winner is not 0.1, run it once on the test
sets* (that run is column F above). Dense-only on the same questions: SciFact 0.979, FiQA 0.602.

| lexical vote | SciFact-train, ParadeDB / Okapi | FiQA-dev, ParadeDB / Okapi | mean (ParadeDB) | Recall@100 SciFact / FiQA |
|---|---|---|---|---|
| 0.02 | 0.9735 / - | 0.5996 / - | **0.7866** | 1.000 / 0.893 |
| 0.05 | 0.9565 / - | 0.5957 / - | 0.7761 | 1.000 / 0.893 |
| 0.1 | 0.9367 / 0.9392 | 0.5853 / 0.5772 | 0.7610 | 1.000 / 0.893 |
| 0.25 | 0.9022 / 0.8971 | 0.5461 / 0.5413 | 0.7242 | 1.000 / 0.893 |
| 0.5 | 0.8730 / 0.8657 | 0.4803 / 0.4810 | 0.6767 | 1.000 / 0.890 |
| 1.0 | 0.8160 / 0.8063 | 0.4141 / 0.4129 | 0.6151 | 1.000 / 0.858 |

The winner is again the smallest weight tested (0.02), so the true optimum, for first-stage nDCG@10 on these corpora, is at or
very near dense-only. ParadeDB and Okapi BM25 agree at every weight (within 0.01).

### 4.3 The BM25 patch inside `main`'s own code

The harness ran on the application's own store (`EVAL_STORE=stock`, `main` loaded from a read-only extract, a throwaway
database holding a copy of one corpus; its startup then creates its columns, trigger and indexes, as in production), unpatched and
with `eval/paradedb_fix/main_bm25_fix.patch`. nDCG@10; differences are patched minus `main` as shipped (paired 95% CI). "8-word"
is `main`'s own query string for BM25 (the first 8 words longer than 2 characters outside a small stop-word list); "whole" sends
the cleaned query. The whole query was chosen on the two development sets and then run once on the novel.

| set | n | `main` as shipped (keyword / fused, equal votes) | patch, 8-word query: keyword / fused | patch, whole query: keyword / fused |
|---|---|---|---|---|
| docs (dev) | 35 | 0.658 / 0.761 | +0.057 [-0.040, +0.159] / +0.008 [-0.053, +0.075] | **+0.122 [+0.006, +0.243]** / **+0.076 [+0.005, +0.151]** |
| thesis (dev) | 31 | 0.304 / 0.476 | +0.032 [-0.026, +0.091] / +0.025 [-0.009, +0.062] | **+0.105 [+0.040, +0.172]** / +0.035 [-0.007, +0.079] |
| novel (held out) | 31 | 0.220 / 0.442 | **+0.055 [+0.010, +0.103]** / **+0.090 [+0.041, +0.143]** | **+0.080 [+0.009, +0.164]** / +0.068 [-0.000, +0.137] |

The patch improves `main` on every set, including the held-out one. Sending the whole query is clearly better on the
development sets and for keyword-only search on the novel; for the fused score on the novel the two variants are
indistinguishable (n = 31). Even patched, `main`'s equal-vote fusion stays far below dense-only on the novel (0.51 to 0.53
against 0.779, fused), which is the vote effect of 4.1, not a defect of the index.

## 5. What this does not cover

* **No reranker, no LLM.** The published pipeline ends with a cross-encoder over the fused top 100; the deep-ingestion study
  showed that stage erases most first-stage differences. The end-to-end effect of the lexical channel and the fusion weight is
  therefore not measured here (it needs the GPU reranker).
* **The benchmark code is 26 commits behind `main`.** `main` adds the application's own LLM summary pipeline (Qwen3-0.6B
  summaries; the child vector is built from `summary + content`; weighted `search_tsv` and `summary_tsv`), children of 150-300
  tokens, the tuned HNSW index and the ParadeDB code above. The published fast and deep numbers describe the older retrieval
  plus this harness's own re-implementation of deep ingestion, not `main`'s pipeline.
* CovidQA was not run; the novel is the only held-out set of the document sets; selection of any weight uses dev sets only.

## 6. Recommended next steps, in order

1. Apply the BM25 patch on a branch of your choice (`eval/paradedb_fix/README.md`).
2. Re-base the benchmark on `main`'s code (the harness already runs on it, section 4.3), so the paper describes the current
   system; `main`'s own deep ingestion runs locally (Qwen3-0.6B on the Mac's GPU), so no cloud is needed.
3. Make the lexical vote a setting of the fusion and choose it on dev sets (section 4.2); `main` uses equal votes.
4. Measure the reranked, end-to-end effect (about half an hour of the GPU VM).
5. Build the far-reference test (a character named early, referred to only by pronoun or epithet far later); a first, free look
   at the existing GoT questions found 12 of 79 gold pages that contain none of the names the question uses, and fast and deep
   ingestion both found 10-11 of those 12 in the top 10.
