# DeepContext public-benchmark report

**INTERIM - batches 1+2 (retrieval and answer writing); rubric, RAGAS, latency not yet in** — generated from files in `eval/results`; every number below is read from them.

- Branch `eval/public-benchmarks` @ `cb3575d` (base `feat/rag-quality-generation`), working tree dirty at report time
- Judge: `gemini-3.7-flash` (temperature 0) — generator: `gemini-3.8-flash` (different model; both are Google models, so family bias is possible)
- Embeddings: `gemini-embedding-2` 768-d via Vertex AI (`global` endpoint); reranker: bge-reranker-v2-m3 (hosted, EcoHash API)

## 1. Setup and versions

Python 3.12.11 on macOS-26.7.2-arm64-arm-64bit (arm64); ragas 0.4.3, beir 2.2.0, datasets 5.1.0, pandas 3.0.6, numpy 2.5.2, scipy 1.18.1, google-genai 2.31.0, langchain-core 0.3.86, langchain-community 0.3.31, huggingface-hub 1.33.0, pyarrow 26.0.0.

Storage backend: postgres — PostgreSQL database deepcontext_eval (one schema per index; 8 indexes, see `dataset_meta.json`). On PostgreSQL the lexical channel is the product's Postgres full-text search (`ts_rank` over the first 8 query terms), **not BM25**, and the dense channel is pgvector HNSW (approximate). `bm25_only` rows therefore mean *Postgres full-text only*.

Ingestion: *fast* path only (`DocumentParser` + `ParentChildChunker`, child 300–600 tokens, parent 1000–2500, 15% overlap); no Docling, no local summariser, no model loaded on the host. Title (when present) is prepended to the chunk text and passed in the embedding prefix.

### Harness deviations from the stock code (all inside `eval/`, `src/` untouched)

1. **Strict transport** (`eval/strict_clients.py`): Vertex AI (ADC) instead of API keys; no mock/hash embeddings, no Gemini→Groq→NVIDIA→mock failover, no silent heuristic-reranker fallback — a failure raises and the row is retried on the next run. The stock client would otherwise have produced hash embeddings in this environment.
2. **Models**: Gemini 3.x only. `gemini-embedding-2` is served from the `global` endpoint (404 in `us-central1`) and accepts one content per request; Gemini 3.x chat models are also called via `global`.
3. **Thinking flag**: the stock Gemini branch of `LLMClient.complete` ignores `enable_thinking=False`. The harness honours it (`thinking_budget=0`); the table below quantifies what the stock behaviour does to the two-pass extractor on the generator model. Set `EVAL_HONOR_ENABLE_THINKING=0` to reproduce the stock behaviour.
4. **Timeouts**: per-call caller timeouts (5–15 s) are replaced by 180 s so results do not depend on API load; the whole pipeline (rewriter, classifier, generator) is routed to the generator model by configuration.
5. **PostgreSQL schema**: on pgvector 0.8.x the stock `PostgresStore.initialize()` fails on a fresh database ("column does not have dimensions": HNSW index on an untyped `vector` column). The harness subclass runs the *same* DDL (extracted from the stock source) with `vector(768)` columns and without the ALTER-back-to-untyped block; all queries are the stock ones.
6. **HNSW search width**: the stock store never sets `hnsw.ef_search`, so pgvector's default (40) would cap a `LIMIT 100` dense query at 40 rows; the eval database sets `hnsw.ef_search = 100` (the value the project docs describe).
7. **Isolation**: all benchmark data lives in a dedicated database (`deepcontext_eval`), one schema per index; the application's own database is never written. The model-call cache (`eval/cache/llm_cache.sqlite`) is a local file, not corpus data.
8. **One-pass baseline** = the product's `PromptAssembler` prompt + one Gemini call, with the (empty) memory lookup stubbed to avoid a no-op embedding call.
9. **Latency**: cached model calls replay their recorded first-call latency; **latency percentiles come only from the separate live pass (`raw_latency.jsonl`)**.
10. RAGAS `ResponseRelevancy` uses `strictness=1` (a temperature-0 judge returns identical candidates, so the default 3 only triples cost).

## 2. Datasets

| run | queries used / available | corpus docs used / total | license | source | sampling |
|---|---|---|---|---|---|
| ragbench-covidqa|qall|dfull | 242 / 246 | 902 / 902 | CC BY 4.0 (per HF dataset card) | HF galileo-ai/ragbench (covidqa, test) | all test rows |
| scifact|qall|dfull | 300 / 300 | 5183 / 5183 | claims/annotations CC BY 4.0; abstracts ODC-By 1.0 (S2ORC) — allenai/scifact LICENSE.md | HF mirror BeIR/scifact @ b3b5335604 + BeIR/scifact-qrels @ 2938d17dc3 | full corpus; all queries |
| multihoprag|q300|dfull | 300 / 2556 | 609 / 609 | ODC-By (per HF dataset card) | HF yixuantt/MultiHopRAG (MultiHopRAG.json + corpus.json) | stratified by question_type, seed=42; full 609-article corpus |
| fiqa|qall|dfull | 648 / 648 | 57638 / 57638 | HF BeIR card: cc-by-sa-4.0 (packaging); FiQA-2018 upstream terms not stated — UNVERIFIED | HF mirror BeIR/fiqa @ 979c07a7cb + BeIR/fiqa-qrels @ 252958f2d6 | full corpus; all queries |
| docsqa|qall|dfull | 44 / 44 | 177 / 177 | repository documentation (Apache-2.0 repo) | repo documentation on branch eval/public-benchmarks (README.md, AGENTS.md, RERANKER_UPGRADE.md, Rag eval.md, docs/ARCHITECTURE.md, docs/CACHE_SCHEDULER_AGENTIC.md, docs/CRITICAL_ASSESSMENT_AND_SCOPE.md, docs/PRD.md, docs/RAG_QUALITY_NEXT.md, docs/SYSTEM_HANDOFF.md, docs/TECH_STACK.md, docs/VERIFICATION_AND_SOURCES.md) | all items; sections = markdown H1-H3 blocks >= 80 chars |
| pdf-quantum|qall|dfull | 36 / 36 | 296 / 296 | copyrighted work supplied by the project owner; text is not redistributed | documents/quantum_computing_dissertation.pdf + tests/eval_quantum_dataset.json (project files) | all 36 items (5 unanswerable); one document per PDF page |
| pdf-got|qall|dfull | 36 / 36 | 753 / 755 | copyrighted work supplied by the project owner; text is not redistributed | Eval 1.pdf + tests/eval_dataset.json (project files) | all 36 items (5 unanswerable); one document per PDF page |
| hotpotqa|q300|d30000 | 300 / 7405 | 30000 / 5233329 | CC BY-SA 4.0 — hotpotqa.github.io | HF mirror BeIR/hotpotqa @ a7e8bab212 + BeIR/hotpotqa-qrels @ b15429e924 | seed=42: all 599 relevant docs + 29401 uniformly sampled distractors (Algorithm R); 300/7405 queries sampled (seed=42) |


**Custom-PDF label quality (heuristic, `pdf_label_audit.json`).** The two 36-query PDF sets (`tests/eval_dataset.json`, `tests/eval_quantum_dataset.json`) pre-date this benchmark and their `expected_pages` were not audited by a human. Heuristic: the share of an expected fact's content words (length > 3) that occur on the union of the item's expected pages; a fact counts as supported at ≥ 0.8. It under-counts paraphrases and is only a rough signal. Because page labels are imprecise, retrieval is also reported with relevance widened to ±1 page.

| set | answerable items | expected facts | mean word coverage | facts supported (≥0.8) | items with all facts supported |
|---|---|---|---|---|---|
| pdf-got | 31 | 95 | 0.8 | 0.6 | 9 |
| pdf-quantum | 31 | 107 | 0.739 | 0.561 | 7 |

RGB (generation-only): `en_refine` / `en_int` / `en_fact` from github.com/chen700564/RGB (CC BY-NC-SA 4.0, non-commercial research use).

## 3. Headline table (mean [95% bootstrap CI, 10k resamples, seed 0])

`rrf_bge_two_pass` shares the retrieval of `rrf_bge` (same ranking by construction). † `engine_as_shipped` returns at most 6 parent chunks, so its Hit@5 / nDCG@10 come from a ranking of ≤ 6 documents and are not directly comparable with the depth-100 rankings of the other rows. RAGAS metrics (Faithfulness, Context Recall, Factual F1) were run only on `rrf_bge`, `rrf_bge_two_pass` and `engine_as_shipped` (cost); the rubric-correctness metric (§5b) covers all six configurations. Generation metrics are blank where no generation run exists for that dataset/config. p95 latency is end-to-end from the live pass.

**Public benchmarks**

| dataset | config | n | Hit@5 | nDCG@10 | Faithfulness | Context Recall | Factual F1 | Rubric correctness | p95 latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| fiqa | dense_only | 648 | 0.773 [0.739, 0.806] | 0.614 [0.585, 0.641] | – | – | – | – | – |
| fiqa | bm25_only | 648 | 0.248 [0.216, 0.282] | 0.125 [0.108, 0.143] | – | – | – | – | – |
| fiqa | rrf | 648 | 0.599 [0.562, 0.636] | 0.378 [0.354, 0.403] | – | – | – | – | – |
| fiqa | rrf_bge | 648 | 0.656 [0.619, 0.693] | 0.437 [0.411, 0.463] | – | – | – | – | – |
| fiqa | rrf_bge_two_pass | 648 | 0.656 [0.619, 0.693] | 0.437 [0.411, 0.463] | – | – | – | – | – |
| hotpotqa | dense_only | 300 | 0.997 [0.990, 1.000] | 0.982 [0.972, 0.990] | – | – | – | – | – |
| hotpotqa | bm25_only | 300 | 0.907 [0.873, 0.937] | 0.691 [0.658, 0.722] | – | – | – | – | – |
| hotpotqa | rrf | 300 | 0.993 [0.983, 1.000] | 0.868 [0.848, 0.887] | – | – | – | – | – |
| hotpotqa | rrf_bge | 300 | 0.997 [0.990, 1.000] | 0.952 [0.939, 0.963] | – | – | – | – | – |
| hotpotqa | rrf_bge_two_pass | 300 | 0.997 [0.990, 1.000] | 0.952 [0.939, 0.963] | – | – | – | – | – |
| multihoprag | dense_only | 265 | 0.977 [0.958, 0.992] | 0.804 [0.781, 0.826] | – | – | – | – | – |
| multihoprag | bm25_only | 265 | 0.702 [0.645, 0.755] | 0.395 [0.362, 0.429] | – | – | – | – | – |
| multihoprag | rrf | 265 | 0.947 [0.917, 0.974] | 0.637 [0.610, 0.665] | – | – | – | – | – |
| multihoprag | rrf_bge | 265 | 0.970 [0.947, 0.989] | 0.727 [0.701, 0.752] | – | – | – | – | – |
| multihoprag | rrf_bge_two_pass | 265 | 0.970 [0.947, 0.989] | 0.727 [0.701, 0.752] | – | – | – | – | – |
| multihoprag | engine_as_shipped † | 265 | 0.977 [0.958, 0.992] | 0.736 [0.707, 0.766] | – | – | – | – | – |
| ragbench-covidqa | dense_only | 242 | 0.971 [0.950, 0.992] | 0.850 [0.821, 0.877] | – | – | – | – | – |
| ragbench-covidqa | bm25_only | 242 | 0.740 [0.682, 0.793] | 0.588 [0.542, 0.635] | – | – | – | – | – |
| ragbench-covidqa | rrf | 242 | 0.917 [0.880, 0.950] | 0.712 [0.675, 0.749] | – | – | – | – | – |
| ragbench-covidqa | rrf_bge | 242 | 0.930 [0.897, 0.959] | 0.743 [0.705, 0.778] | – | – | – | – | – |
| ragbench-covidqa | rrf_bge_two_pass | 242 | 0.930 [0.897, 0.959] | 0.743 [0.705, 0.778] | – | – | – | – | – |
| ragbench-covidqa | engine_as_shipped † | 242 | 0.921 [0.884, 0.955] | 0.744 [0.707, 0.780] | – | – | – | – | – |
| scifact | dense_only | 300 | 0.953 [0.930, 0.977] | 0.900 [0.874, 0.923] | – | – | – | – | – |
| scifact | bm25_only | 300 | 0.573 [0.520, 0.630] | 0.487 [0.439, 0.536] | – | – | – | – | – |
| scifact | rrf | 300 | 0.783 [0.737, 0.830] | 0.702 [0.660, 0.744] | – | – | – | – | – |
| scifact | rrf_bge | 300 | 0.820 [0.777, 0.863] | 0.744 [0.705, 0.784] | – | – | – | – | – |
| scifact | rrf_bge_two_pass | 300 | 0.820 [0.777, 0.863] | 0.744 [0.705, 0.784] | – | – | – | – | – |
| scifact | engine_as_shipped † | 300 | 0.840 [0.797, 0.880] | 0.738 [0.695, 0.780] | – | – | – | – | – |


**Custom corpora** (project documentation and the two project PDFs; separate table because the queries and page/section labels were written for this project and are not human-audited)

| dataset | config | n | Hit@5 | nDCG@10 | Faithfulness | Context Recall | Factual F1 | Rubric correctness | p95 latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 35 | 0.914 [0.829, 1.000] | 0.824 [0.742, 0.900] | – | – | – | – | – |
| docsqa | bm25_only | 35 | 0.743 [0.600, 0.886] | 0.635 [0.506, 0.760] | – | – | – | – | – |
| docsqa | rrf | 35 | 0.886 [0.771, 0.971] | 0.760 [0.665, 0.849] | – | – | – | – | – |
| docsqa | rrf_bge | 35 | 0.914 [0.800, 1.000] | 0.794 [0.700, 0.883] | – | – | – | – | – |
| docsqa | rrf_bge_two_pass | 35 | 0.914 [0.800, 1.000] | 0.794 [0.700, 0.883] | – | – | – | – | – |
| docsqa | engine_as_shipped † | 35 | 0.857 [0.743, 0.971] | 0.776 [0.660, 0.878] | – | – | – | – | – |
| pdf-got | dense_only | 31 | 1.000 [1.000, 1.000] | 0.779 [0.694, 0.855] | – | – | – | – | – |
| pdf-got | bm25_only | 31 | 0.323 [0.161, 0.484] | 0.162 [0.091, 0.243] | – | – | – | – | – |
| pdf-got | rrf | 31 | 0.710 [0.548, 0.871] | 0.419 [0.312, 0.525] | – | – | – | – | – |
| pdf-got | rrf_bge | 31 | 0.710 [0.548, 0.871] | 0.419 [0.314, 0.522] | – | – | – | – | – |
| pdf-got | rrf_bge_two_pass | 31 | 0.710 [0.548, 0.871] | 0.419 [0.314, 0.522] | – | – | – | – | – |
| pdf-got | engine_as_shipped † | 31 | 0.839 [0.710, 0.968] | 0.486 [0.382, 0.589] | – | – | – | – | – |
| pdf-got (±1 page) | dense_only | 31 | 1.000 [1.000, 1.000] | 0.659 [0.614, 0.699] | – | – | – | – | – |
| pdf-got (±1 page) | bm25_only | 31 | 0.484 [0.323, 0.645] | 0.188 [0.123, 0.262] | – | – | – | – | – |
| pdf-got (±1 page) | rrf | 31 | 0.806 [0.645, 0.935] | 0.400 [0.319, 0.478] | – | – | – | – | – |
| pdf-got (±1 page) | rrf_bge | 31 | 0.871 [0.742, 0.968] | 0.400 [0.322, 0.474] | – | – | – | – | – |
| pdf-got (±1 page) | rrf_bge_two_pass | 31 | 0.871 [0.742, 0.968] | 0.400 [0.322, 0.474] | – | – | – | – | – |
| pdf-got (±1 page) | engine_as_shipped † | 31 | 0.871 [0.742, 0.968] | 0.406 [0.336, 0.476] | – | – | – | – | – |
| pdf-quantum | dense_only | 31 | 0.935 [0.839, 1.000] | 0.518 [0.452, 0.577] | – | – | – | – | – |
| pdf-quantum | bm25_only | 31 | 0.710 [0.548, 0.871] | 0.292 [0.220, 0.366] | – | – | – | – | – |
| pdf-quantum | rrf | 31 | 0.871 [0.742, 0.968] | 0.481 [0.416, 0.539] | – | – | – | – | – |
| pdf-quantum | rrf_bge | 31 | 0.871 [0.742, 0.968] | 0.460 [0.392, 0.523] | – | – | – | – | – |
| pdf-quantum | rrf_bge_two_pass | 31 | 0.871 [0.742, 0.968] | 0.460 [0.392, 0.523] | – | – | – | – | – |
| pdf-quantum | engine_as_shipped † | 31 | 0.871 [0.742, 0.968] | 0.410 [0.342, 0.477] | – | – | – | – | – |
| pdf-quantum (±1 page) | dense_only | 31 | 0.935 [0.839, 1.000] | 0.505 [0.434, 0.571] | – | – | – | – | – |
| pdf-quantum (±1 page) | bm25_only | 31 | 0.839 [0.710, 0.968] | 0.288 [0.229, 0.346] | – | – | – | – | – |
| pdf-quantum (±1 page) | rrf | 31 | 0.903 [0.774, 1.000] | 0.462 [0.397, 0.523] | – | – | – | – | – |
| pdf-quantum (±1 page) | rrf_bge | 31 | 0.903 [0.774, 1.000] | 0.469 [0.408, 0.527] | – | – | – | – | – |
| pdf-quantum (±1 page) | rrf_bge_two_pass | 31 | 0.903 [0.774, 1.000] | 0.469 [0.408, 0.527] | – | – | – | – | – |
| pdf-quantum (±1 page) | engine_as_shipped † | 31 | 0.903 [0.774, 1.000] | 0.381 [0.322, 0.437] | – | – | – | – | – |

## 4. Retrieval (document level)


### fiqa — `fiqa|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 648 | 0.597 [0.559, 0.636] | 0.724 [0.688, 0.759] | 0.773 [0.739, 0.806] | 0.827 [0.796, 0.856] | 0.588 [0.556, 0.618] | 0.696 [0.667, 0.723] | 0.918 [0.901, 0.934] | 0.283 [0.265, 0.302] | 0.679 [0.647, 0.710] | 0.554 [0.525, 0.582] | 0.580 [0.550, 0.610] | 0.614 [0.585, 0.641] | 0.401 [0.364, 0.438] | 0.514 [0.475, 0.551] |
| bm25_only | 648 | 0.100 [0.077, 0.123] | 0.184 [0.154, 0.213] | 0.248 [0.216, 0.282] | 0.285 [0.252, 0.321] | 0.128 [0.108, 0.149] | 0.170 [0.148, 0.193] | 0.392 [0.361, 0.423] | 0.056 [0.048, 0.064] | 0.170 [0.147, 0.193] | 0.097 [0.082, 0.112] | 0.110 [0.093, 0.128] | 0.125 [0.108, 0.143] | 0.056 [0.039, 0.074] | 0.076 [0.056, 0.097] |
| rrf | 648 | 0.321 [0.285, 0.356] | 0.503 [0.465, 0.542] | 0.599 [0.562, 0.636] | 0.682 [0.647, 0.718] | 0.373 [0.345, 0.403] | 0.500 [0.469, 0.531] | 0.876 [0.855, 0.895] | 0.170 [0.157, 0.183] | 0.452 [0.421, 0.482] | 0.316 [0.295, 0.338] | 0.333 [0.308, 0.359] | 0.378 [0.354, 0.403] | 0.204 [0.173, 0.235] | 0.302 [0.267, 0.338] |
| rrf_bge | 648 | 0.381 [0.344, 0.418] | 0.562 [0.523, 0.600] | 0.656 [0.619, 0.693] | 0.735 [0.701, 0.769] | 0.429 [0.399, 0.460] | 0.561 [0.530, 0.592] | 0.876 [0.856, 0.895] | 0.199 [0.184, 0.215] | 0.506 [0.474, 0.538] | 0.366 [0.342, 0.391] | 0.392 [0.365, 0.419] | 0.437 [0.411, 0.463] | 0.247 [0.215, 0.281] | 0.367 [0.330, 0.404] |
| rrf_bge_two_pass | 648 | 0.381 [0.344, 0.420] | 0.562 [0.523, 0.600] | 0.656 [0.619, 0.691] | 0.735 [0.701, 0.767] | 0.429 [0.398, 0.459] | 0.561 [0.530, 0.592] | 0.876 [0.855, 0.895] | 0.199 [0.184, 0.214] | 0.506 [0.474, 0.538] | 0.366 [0.342, 0.391] | 0.392 [0.364, 0.419] | 0.437 [0.411, 0.463] | 0.247 [0.213, 0.281] | 0.367 [0.329, 0.404] |


### hotpotqa — `hotpotqa|q300|d30000`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 300 | 0.990 [0.977, 1.000] | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | 0.982 [0.970, 0.992] | 0.988 [0.980, 0.997] | 0.993 [0.987, 0.998] | 0.393 [0.388, 0.397] | 0.993 [0.984, 1.000] | 0.974 [0.962, 0.985] | 0.979 [0.968, 0.989] | 0.982 [0.972, 0.990] | 0.967 [0.947, 0.987] | 0.977 [0.960, 0.993] |
| bm25_only | 300 | 0.807 [0.760, 0.850] | 0.890 [0.853, 0.923] | 0.907 [0.873, 0.937] | 0.930 [0.900, 0.957] | 0.658 [0.622, 0.693] | 0.702 [0.667, 0.735] | 0.845 [0.817, 0.872] | 0.263 [0.249, 0.277] | 0.856 [0.820, 0.890] | 0.622 [0.587, 0.655] | 0.674 [0.639, 0.706] | 0.691 [0.658, 0.722] | 0.410 [0.353, 0.467] | 0.463 [0.407, 0.520] |
| rrf | 300 | 0.947 [0.920, 0.970] | 0.987 [0.973, 0.997] | 0.993 [0.983, 1.000] | 1.000 [1.000, 1.000] | 0.835 [0.808, 0.862] | 0.908 [0.887, 0.930] | 0.997 [0.992, 1.000] | 0.334 [0.323, 0.345] | 0.966 [0.949, 0.982] | 0.811 [0.787, 0.836] | 0.838 [0.815, 0.861] | 0.868 [0.848, 0.887] | 0.677 [0.623, 0.727] | 0.817 [0.773, 0.860] |
| rrf_bge | 300 | 0.983 [0.967, 0.997] | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | 0.947 [0.928, 0.963] | 0.978 [0.967, 0.988] | 0.997 [0.992, 1.000] | 0.379 [0.371, 0.385] | 0.991 [0.981, 0.998] | 0.921 [0.903, 0.937] | 0.939 [0.923, 0.953] | 0.952 [0.939, 0.963] | 0.897 [0.860, 0.930] | 0.957 [0.933, 0.977] |
| rrf_bge_two_pass | 300 | 0.983 [0.967, 0.997] | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | 0.947 [0.928, 0.963] | 0.978 [0.967, 0.988] | 0.997 [0.992, 1.000] | 0.379 [0.371, 0.385] | 0.991 [0.981, 0.998] | 0.921 [0.904, 0.937] | 0.939 [0.923, 0.953] | 0.952 [0.939, 0.963] | 0.897 [0.860, 0.930] | 0.957 [0.933, 0.977] |


### multihoprag — `multihoprag|q300|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 265 | 0.721 [0.664, 0.774] | 0.955 [0.928, 0.977] | 0.977 [0.958, 0.992] | 0.996 [0.989, 1.000] | 0.796 [0.765, 0.826] | 0.928 [0.908, 0.947] | 0.994 [0.987, 0.999] | 0.402 [0.386, 0.417] | 0.839 [0.806, 0.870] | 0.713 [0.685, 0.741] | 0.744 [0.714, 0.773] | 0.804 [0.781, 0.826] | 0.555 [0.494, 0.615] | 0.808 [0.758, 0.853] |
| bm25_only | 265 | 0.415 [0.358, 0.472] | 0.626 [0.570, 0.683] | 0.702 [0.645, 0.755] | 0.800 [0.751, 0.845] | 0.356 [0.320, 0.392] | 0.460 [0.423, 0.497] | 0.743 [0.709, 0.777] | 0.172 [0.155, 0.189] | 0.548 [0.500, 0.595] | 0.313 [0.284, 0.343] | 0.351 [0.316, 0.385] | 0.395 [0.362, 0.429] | 0.094 [0.060, 0.132] | 0.158 [0.113, 0.204] |
| rrf | 265 | 0.649 [0.592, 0.706] | 0.883 [0.842, 0.921] | 0.947 [0.917, 0.974] | 0.977 [0.958, 0.992] | 0.587 [0.554, 0.620] | 0.741 [0.710, 0.771] | 0.991 [0.983, 0.997] | 0.295 [0.278, 0.312] | 0.773 [0.734, 0.810] | 0.541 [0.513, 0.570] | 0.570 [0.539, 0.602] | 0.637 [0.610, 0.665] | 0.230 [0.181, 0.283] | 0.430 [0.370, 0.491] |
| rrf_bge | 265 | 0.660 [0.604, 0.717] | 0.932 [0.902, 0.962] | 0.970 [0.947, 0.989] | 0.992 [0.981, 1.000] | 0.705 [0.672, 0.738] | 0.847 [0.821, 0.872] | 0.991 [0.983, 0.997] | 0.356 [0.339, 0.374] | 0.799 [0.764, 0.833] | 0.629 [0.600, 0.658] | 0.662 [0.630, 0.693] | 0.727 [0.701, 0.752] | 0.400 [0.343, 0.460] | 0.623 [0.566, 0.679] |
| rrf_bge_two_pass | 265 | 0.660 [0.604, 0.717] | 0.932 [0.902, 0.958] | 0.970 [0.947, 0.989] | 0.992 [0.981, 1.000] | 0.705 [0.672, 0.738] | 0.847 [0.820, 0.872] | 0.991 [0.983, 0.997] | 0.356 [0.339, 0.374] | 0.799 [0.764, 0.834] | 0.629 [0.600, 0.658] | 0.662 [0.630, 0.693] | 0.726 [0.701, 0.752] | 0.400 [0.340, 0.460] | 0.623 [0.562, 0.679] |
| engine_as_shipped | 265 | 0.725 [0.672, 0.777] | 0.940 [0.909, 0.966] | 0.977 [0.958, 0.992] | 0.989 [0.974, 1.000] | 0.763 [0.730, 0.795] | 0.800 [0.769, 0.830] | 0.800 [0.769, 0.830] | 0.383 [0.367, 0.400] | 0.832 [0.798, 0.866] | 0.639 [0.604, 0.673] | 0.719 [0.687, 0.749] | 0.736 [0.707, 0.766] | 0.513 [0.453, 0.574] | 0.555 [0.494, 0.615] |


### ragbench-covidqa — `ragbench-covidqa|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 242 | 0.773 [0.719, 0.822] | 0.934 [0.901, 0.963] | 0.971 [0.950, 0.992] | 0.983 [0.967, 0.996] | 0.881 [0.850, 0.910] | 0.948 [0.927, 0.968] | 1.000 [1.000, 1.000] | 0.429 [0.402, 0.455] | 0.858 [0.824, 0.891] | 0.790 [0.756, 0.823] | 0.819 [0.786, 0.852] | 0.850 [0.821, 0.877] | 0.777 [0.723, 0.826] | 0.893 [0.851, 0.930] |
| bm25_only | 242 | 0.541 [0.479, 0.603] | 0.711 [0.653, 0.769] | 0.740 [0.682, 0.793] | 0.798 [0.744, 0.847] | 0.572 [0.521, 0.624] | 0.693 [0.643, 0.743] | 0.926 [0.896, 0.952] | 0.271 [0.243, 0.300] | 0.641 [0.589, 0.692] | 0.520 [0.476, 0.565] | 0.535 [0.487, 0.583] | 0.588 [0.542, 0.635] | 0.388 [0.326, 0.450] | 0.566 [0.504, 0.628] |
| rrf | 242 | 0.686 [0.628, 0.744] | 0.864 [0.822, 0.905] | 0.917 [0.880, 0.950] | 0.938 [0.905, 0.967] | 0.713 [0.671, 0.755] | 0.793 [0.753, 0.832] | 0.864 [0.829, 0.897] | 0.337 [0.312, 0.364] | 0.785 [0.743, 0.827] | 0.635 [0.594, 0.675] | 0.677 [0.637, 0.717] | 0.712 [0.675, 0.749] | 0.504 [0.442, 0.566] | 0.636 [0.574, 0.694] |
| rrf_bge | 242 | 0.727 [0.669, 0.781] | 0.888 [0.847, 0.926] | 0.930 [0.897, 0.959] | 0.946 [0.917, 0.971] | 0.740 [0.699, 0.780] | 0.807 [0.768, 0.844] | 0.864 [0.830, 0.897] | 0.349 [0.323, 0.374] | 0.817 [0.777, 0.856] | 0.671 [0.629, 0.711] | 0.712 [0.673, 0.749] | 0.743 [0.705, 0.778] | 0.545 [0.479, 0.607] | 0.669 [0.607, 0.727] |
| rrf_bge_two_pass | 242 | 0.727 [0.669, 0.785] | 0.888 [0.847, 0.930] | 0.930 [0.897, 0.959] | 0.946 [0.917, 0.971] | 0.740 [0.698, 0.781] | 0.807 [0.768, 0.844] | 0.864 [0.830, 0.896] | 0.349 [0.323, 0.374] | 0.817 [0.777, 0.857] | 0.671 [0.630, 0.712] | 0.712 [0.672, 0.751] | 0.743 [0.705, 0.779] | 0.545 [0.483, 0.607] | 0.669 [0.607, 0.727] |
| engine_as_shipped | 242 | 0.769 [0.715, 0.822] | 0.897 [0.860, 0.934] | 0.921 [0.884, 0.955] | 0.942 [0.909, 0.971] | 0.749 [0.707, 0.791] | 0.784 [0.744, 0.823] | 0.785 [0.746, 0.824] | 0.355 [0.329, 0.381] | 0.836 [0.794, 0.874] | 0.673 [0.631, 0.715] | 0.728 [0.689, 0.766] | 0.744 [0.707, 0.780] | 0.562 [0.500, 0.624] | 0.624 [0.562, 0.686] |


### scifact — `scifact|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 300 | 0.807 [0.760, 0.850] | 0.940 [0.913, 0.967] | 0.953 [0.930, 0.977] | 0.970 [0.950, 0.987] | 0.949 [0.923, 0.971] | 0.983 [0.967, 0.997] | 0.993 [0.983, 1.000] | 0.214 [0.203, 0.227] | 0.874 [0.843, 0.903] | 0.871 [0.840, 0.900] | 0.888 [0.859, 0.915] | 0.900 [0.874, 0.923] | 0.940 [0.913, 0.963] | 0.983 [0.967, 0.997] |
| bm25_only | 300 | 0.367 [0.313, 0.423] | 0.503 [0.447, 0.560] | 0.573 [0.520, 0.630] | 0.627 [0.573, 0.680] | 0.549 [0.495, 0.605] | 0.616 [0.562, 0.670] | 0.818 [0.775, 0.859] | 0.118 [0.106, 0.130] | 0.464 [0.415, 0.514] | 0.448 [0.400, 0.498] | 0.464 [0.414, 0.515] | 0.487 [0.439, 0.536] | 0.530 [0.473, 0.587] | 0.600 [0.543, 0.653] |
| rrf | 300 | 0.587 [0.533, 0.640] | 0.713 [0.663, 0.763] | 0.783 [0.737, 0.830] | 0.840 [0.797, 0.883] | 0.757 [0.709, 0.803] | 0.847 [0.807, 0.886] | 0.993 [0.983, 1.000] | 0.166 [0.155, 0.177] | 0.679 [0.636, 0.724] | 0.660 [0.615, 0.704] | 0.672 [0.626, 0.718] | 0.702 [0.660, 0.744] | 0.730 [0.680, 0.780] | 0.827 [0.783, 0.870] |
| rrf_bge | 300 | 0.623 [0.570, 0.680] | 0.740 [0.690, 0.790] | 0.820 [0.777, 0.863] | 0.877 [0.837, 0.913] | 0.793 [0.749, 0.839] | 0.896 [0.862, 0.929] | 0.993 [0.983, 1.000] | 0.175 [0.163, 0.186] | 0.712 [0.668, 0.757] | 0.696 [0.653, 0.741] | 0.709 [0.665, 0.754] | 0.744 [0.705, 0.784] | 0.763 [0.717, 0.813] | 0.880 [0.843, 0.917] |
| rrf_bge_two_pass | 300 | 0.623 [0.570, 0.680] | 0.740 [0.690, 0.790] | 0.820 [0.777, 0.863] | 0.877 [0.840, 0.913] | 0.793 [0.749, 0.836] | 0.896 [0.863, 0.928] | 0.993 [0.983, 1.000] | 0.175 [0.163, 0.186] | 0.712 [0.668, 0.756] | 0.696 [0.653, 0.739] | 0.709 [0.664, 0.752] | 0.744 [0.705, 0.782] | 0.763 [0.713, 0.810] | 0.880 [0.843, 0.917] |
| engine_as_shipped | 300 | 0.640 [0.583, 0.693] | 0.770 [0.720, 0.817] | 0.840 [0.797, 0.880] | 0.847 [0.803, 0.887] | 0.818 [0.775, 0.858] | 0.831 [0.790, 0.871] | 0.832 [0.790, 0.872] | 0.182 [0.170, 0.194] | 0.716 [0.670, 0.761] | 0.702 [0.656, 0.746] | 0.733 [0.689, 0.774] | 0.738 [0.695, 0.780] | 0.787 [0.740, 0.830] | 0.807 [0.760, 0.850] |


#### Custom corpora (not public benchmarks: the project's own documentation and the two project PDFs; queries and labels were written for this project, not human-audited)


### docsqa — `docsqa|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 35 | 0.629 [0.457, 0.771] | 0.914 [0.829, 1.000] | 0.914 [0.829, 1.000] | 1.000 [1.000, 1.000] | 0.914 [0.829, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.194 [0.171, 0.217] | 0.766 [0.660, 0.867] | 0.765 [0.658, 0.867] | 0.794 [0.686, 0.890] | 0.824 [0.742, 0.900] | 0.914 [0.829, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 35 | 0.457 [0.286, 0.629] | 0.629 [0.457, 0.800] | 0.743 [0.600, 0.886] | 0.829 [0.686, 0.943] | 0.729 [0.571, 0.857] | 0.814 [0.686, 0.929] | 0.971 [0.914, 1.000] | 0.154 [0.120, 0.183] | 0.585 [0.451, 0.719] | 0.585 [0.449, 0.719] | 0.607 [0.466, 0.743] | 0.635 [0.506, 0.760] | 0.714 [0.571, 0.857] | 0.800 [0.657, 0.914] |
| rrf | 35 | 0.543 [0.371, 0.714] | 0.857 [0.743, 0.971] | 0.886 [0.771, 0.971] | 0.914 [0.800, 1.000] | 0.871 [0.757, 0.971] | 0.971 [0.914, 1.000] | 1.000 [1.000, 1.000] | 0.183 [0.160, 0.206] | 0.698 [0.580, 0.810] | 0.690 [0.572, 0.799] | 0.729 [0.614, 0.834] | 0.760 [0.665, 0.849] | 0.857 [0.743, 0.971] | 0.971 [0.914, 1.000] |
| rrf_bge | 35 | 0.629 [0.457, 0.800] | 0.857 [0.743, 0.971] | 0.914 [0.800, 1.000] | 0.943 [0.857, 1.000] | 0.914 [0.800, 1.000] | 0.971 [0.914, 1.000] | 1.000 [1.000, 1.000] | 0.194 [0.166, 0.217] | 0.745 [0.630, 0.857] | 0.734 [0.620, 0.843] | 0.777 [0.668, 0.876] | 0.794 [0.700, 0.883] | 0.914 [0.800, 1.000] | 0.971 [0.914, 1.000] |
| rrf_bge_two_pass | 35 | 0.629 [0.457, 0.771] | 0.857 [0.743, 0.971] | 0.914 [0.800, 1.000] | 0.943 [0.857, 1.000] | 0.914 [0.800, 1.000] | 0.971 [0.914, 1.000] | 1.000 [1.000, 1.000] | 0.194 [0.171, 0.217] | 0.745 [0.629, 0.854] | 0.734 [0.618, 0.840] | 0.777 [0.667, 0.874] | 0.794 [0.699, 0.881] | 0.914 [0.800, 1.000] | 0.971 [0.914, 1.000] |
| engine_as_shipped | 35 | 0.657 [0.486, 0.800] | 0.829 [0.686, 0.943] | 0.857 [0.743, 0.971] | 0.914 [0.800, 1.000] | 0.843 [0.714, 0.943] | 0.900 [0.800, 0.986] | 0.900 [0.800, 0.986] | 0.177 [0.149, 0.200] | 0.745 [0.617, 0.862] | 0.731 [0.605, 0.850] | 0.755 [0.630, 0.868] | 0.776 [0.660, 0.878] | 0.829 [0.686, 0.943] | 0.886 [0.771, 0.971] |


### pdf-got — `pdf-got|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 31 | 0.710 [0.548, 0.871] | 0.903 [0.806, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.755 [0.655, 0.852] | 0.871 [0.798, 0.939] | 0.995 [0.984, 1.000] | 0.348 [0.290, 0.413] | 0.825 [0.720, 0.919] | 0.710 [0.611, 0.802] | 0.733 [0.631, 0.828] | 0.779 [0.694, 0.855] | 0.548 [0.387, 0.710] | 0.677 [0.516, 0.839] |
| bm25_only | 31 | 0.097 [0.000, 0.226] | 0.258 [0.097, 0.419] | 0.323 [0.161, 0.484] | 0.419 [0.258, 0.581] | 0.142 [0.061, 0.234] | 0.260 [0.146, 0.388] | 0.599 [0.459, 0.732] | 0.077 [0.039, 0.123] | 0.212 [0.117, 0.325] | 0.123 [0.072, 0.183] | 0.117 [0.054, 0.187] | 0.162 [0.091, 0.243] | 0.032 [0.000, 0.097] | 0.129 [0.032, 0.258] |
| rrf | 31 | 0.355 [0.194, 0.516] | 0.613 [0.452, 0.774] | 0.710 [0.548, 0.871] | 0.710 [0.548, 0.871] | 0.433 [0.305, 0.569] | 0.528 [0.389, 0.669] | 0.989 [0.973, 1.000] | 0.194 [0.142, 0.252] | 0.497 [0.361, 0.639] | 0.354 [0.269, 0.442] | 0.373 [0.271, 0.476] | 0.419 [0.312, 0.525] | 0.226 [0.097, 0.387] | 0.323 [0.161, 0.484] |
| rrf_bge | 31 | 0.355 [0.194, 0.516] | 0.516 [0.323, 0.677] | 0.710 [0.548, 0.871] | 0.742 [0.581, 0.871] | 0.468 [0.332, 0.604] | 0.549 [0.408, 0.684] | 0.989 [0.973, 1.000] | 0.219 [0.155, 0.290] | 0.496 [0.357, 0.636] | 0.354 [0.268, 0.442] | 0.386 [0.281, 0.490] | 0.419 [0.314, 0.522] | 0.258 [0.097, 0.419] | 0.355 [0.194, 0.516] |
| rrf_bge_two_pass | 31 | 0.355 [0.194, 0.516] | 0.516 [0.355, 0.677] | 0.710 [0.548, 0.871] | 0.742 [0.581, 0.871] | 0.468 [0.332, 0.603] | 0.549 [0.410, 0.684] | 0.989 [0.973, 1.000] | 0.219 [0.155, 0.290] | 0.496 [0.359, 0.638] | 0.354 [0.268, 0.443] | 0.386 [0.280, 0.490] | 0.419 [0.312, 0.521] | 0.258 [0.097, 0.419] | 0.355 [0.194, 0.516] |
| engine_as_shipped | 31 | 0.548 [0.387, 0.710] | 0.742 [0.581, 0.871] | 0.839 [0.710, 0.968] | 0.871 [0.742, 0.968] | 0.510 [0.389, 0.634] | 0.563 [0.438, 0.687] | 0.563 [0.438, 0.687] | 0.245 [0.187, 0.303] | 0.650 [0.504, 0.788] | 0.371 [0.270, 0.478] | 0.469 [0.362, 0.576] | 0.486 [0.382, 0.589] | 0.258 [0.097, 0.419] | 0.323 [0.161, 0.484] |


### pdf-got (±1 page) — `pdf-got|qall|dfull|tol1`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 31 | 0.935 [0.839, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.490 [0.428, 0.550] | 0.570 [0.515, 0.623] | 0.832 [0.785, 0.880] | 0.523 [0.458, 0.587] | 0.968 [0.919, 1.000] | 0.550 [0.502, 0.595] | 0.691 [0.642, 0.737] | 0.659 [0.614, 0.699] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| bm25_only | 31 | 0.194 [0.065, 0.355] | 0.323 [0.161, 0.484] | 0.484 [0.323, 0.645] | 0.613 [0.452, 0.774] | 0.106 [0.063, 0.152] | 0.211 [0.145, 0.282] | 0.512 [0.419, 0.602] | 0.129 [0.077, 0.194] | 0.322 [0.201, 0.453] | 0.125 [0.078, 0.181] | 0.153 [0.086, 0.231] | 0.188 [0.123, 0.262] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf | 31 | 0.581 [0.419, 0.742] | 0.774 [0.613, 0.903] | 0.806 [0.645, 0.935] | 0.839 [0.710, 0.968] | 0.279 [0.206, 0.359] | 0.366 [0.289, 0.444] | 0.792 [0.738, 0.845] | 0.310 [0.232, 0.387] | 0.674 [0.535, 0.813] | 0.313 [0.254, 0.374] | 0.397 [0.308, 0.488] | 0.400 [0.319, 0.478] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf_bge | 31 | 0.581 [0.419, 0.742] | 0.742 [0.581, 0.871] | 0.871 [0.742, 0.968] | 0.871 [0.742, 0.968] | 0.296 [0.225, 0.369] | 0.375 [0.296, 0.452] | 0.792 [0.739, 0.846] | 0.323 [0.245, 0.406] | 0.693 [0.558, 0.824] | 0.318 [0.259, 0.380] | 0.408 [0.324, 0.493] | 0.400 [0.322, 0.474] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf_bge_two_pass | 31 | 0.581 [0.419, 0.742] | 0.742 [0.581, 0.903] | 0.871 [0.742, 0.968] | 0.871 [0.742, 0.968] | 0.296 [0.227, 0.370] | 0.375 [0.295, 0.453] | 0.792 [0.739, 0.846] | 0.323 [0.245, 0.406] | 0.693 [0.558, 0.824] | 0.318 [0.259, 0.378] | 0.408 [0.322, 0.492] | 0.400 [0.321, 0.474] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| engine_as_shipped | 31 | 0.645 [0.484, 0.806] | 0.806 [0.677, 0.935] | 0.871 [0.742, 0.968] | 0.935 [0.839, 1.000] | 0.323 [0.248, 0.402] | 0.366 [0.294, 0.439] | 0.366 [0.294, 0.439] | 0.361 [0.284, 0.439] | 0.739 [0.605, 0.861] | 0.251 [0.191, 0.315] | 0.449 [0.363, 0.536] | 0.406 [0.336, 0.476] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |


### pdf-quantum — `pdf-quantum|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 31 | 0.774 [0.613, 0.903] | 0.903 [0.774, 1.000] | 0.935 [0.839, 1.000] | 0.968 [0.903, 1.000] | 0.364 [0.307, 0.418] | 0.448 [0.392, 0.499] | 0.635 [0.578, 0.695] | 0.419 [0.342, 0.503] | 0.847 [0.738, 0.944] | 0.385 [0.326, 0.444] | 0.537 [0.457, 0.613] | 0.518 [0.452, 0.577] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| bm25_only | 31 | 0.355 [0.194, 0.516] | 0.613 [0.452, 0.774] | 0.710 [0.548, 0.871] | 0.839 [0.710, 0.968] | 0.198 [0.140, 0.259] | 0.288 [0.224, 0.351] | 0.625 [0.559, 0.690] | 0.206 [0.148, 0.265] | 0.518 [0.385, 0.655] | 0.208 [0.153, 0.267] | 0.269 [0.191, 0.350] | 0.292 [0.220, 0.366] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf | 31 | 0.710 [0.548, 0.871] | 0.871 [0.742, 0.968] | 0.871 [0.742, 0.968] | 0.871 [0.742, 0.968] | 0.336 [0.272, 0.396] | 0.441 [0.392, 0.484] | 0.665 [0.608, 0.725] | 0.394 [0.310, 0.477] | 0.792 [0.666, 0.905] | 0.341 [0.284, 0.396] | 0.488 [0.396, 0.574] | 0.481 [0.416, 0.539] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf_bge | 31 | 0.613 [0.452, 0.774] | 0.839 [0.710, 0.968] | 0.871 [0.742, 0.968] | 0.935 [0.839, 1.000] | 0.324 [0.263, 0.384] | 0.431 [0.374, 0.480] | 0.665 [0.609, 0.725] | 0.374 [0.297, 0.452] | 0.743 [0.618, 0.856] | 0.321 [0.268, 0.374] | 0.453 [0.368, 0.533] | 0.460 [0.392, 0.523] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf_bge_two_pass | 31 | 0.613 [0.452, 0.774] | 0.839 [0.710, 0.968] | 0.871 [0.742, 0.968] | 0.935 [0.839, 1.000] | 0.324 [0.263, 0.383] | 0.431 [0.373, 0.481] | 0.665 [0.608, 0.725] | 0.374 [0.303, 0.452] | 0.743 [0.620, 0.858] | 0.321 [0.269, 0.374] | 0.453 [0.369, 0.534] | 0.460 [0.392, 0.523] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| engine_as_shipped | 31 | 0.645 [0.484, 0.806] | 0.871 [0.742, 0.968] | 0.871 [0.742, 0.968] | 0.903 [0.806, 1.000] | 0.302 [0.245, 0.358] | 0.345 [0.286, 0.403] | 0.345 [0.286, 0.403] | 0.368 [0.290, 0.452] | 0.753 [0.624, 0.871] | 0.260 [0.208, 0.313] | 0.447 [0.363, 0.532] | 0.410 [0.342, 0.477] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |


### pdf-quantum (±1 page) — `pdf-quantum|qall|dfull|tol1`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 31 | 0.806 [0.645, 0.935] | 0.903 [0.774, 1.000] | 0.935 [0.839, 1.000] | 0.968 [0.903, 1.000] | 0.247 [0.201, 0.293] | 0.338 [0.286, 0.388] | 0.628 [0.569, 0.690] | 0.548 [0.458, 0.639] | 0.869 [0.762, 0.958] | 0.340 [0.286, 0.398] | 0.607 [0.515, 0.694] | 0.505 [0.434, 0.571] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| bm25_only | 31 | 0.387 [0.226, 0.548] | 0.742 [0.581, 0.903] | 0.839 [0.710, 0.968] | 0.871 [0.742, 0.968] | 0.127 [0.098, 0.157] | 0.216 [0.169, 0.264] | 0.589 [0.521, 0.658] | 0.271 [0.213, 0.323] | 0.585 [0.461, 0.711] | 0.191 [0.151, 0.235] | 0.305 [0.237, 0.374] | 0.288 [0.229, 0.346] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf | 31 | 0.742 [0.581, 0.903] | 0.903 [0.774, 1.000] | 0.903 [0.774, 1.000] | 0.903 [0.774, 1.000] | 0.223 [0.181, 0.264] | 0.318 [0.271, 0.363] | 0.642 [0.578, 0.709] | 0.490 [0.406, 0.574] | 0.826 [0.711, 0.927] | 0.310 [0.263, 0.356] | 0.546 [0.457, 0.628] | 0.462 [0.397, 0.523] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf_bge | 31 | 0.710 [0.548, 0.871] | 0.903 [0.774, 1.000] | 0.903 [0.774, 1.000] | 0.968 [0.903, 1.000] | 0.217 [0.175, 0.259] | 0.330 [0.288, 0.370] | 0.642 [0.579, 0.708] | 0.477 [0.394, 0.561] | 0.820 [0.709, 0.919] | 0.307 [0.260, 0.354] | 0.529 [0.441, 0.609] | 0.469 [0.408, 0.527] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rrf_bge_two_pass | 31 | 0.710 [0.548, 0.871] | 0.903 [0.774, 1.000] | 0.903 [0.774, 1.000] | 0.968 [0.903, 1.000] | 0.217 [0.175, 0.259] | 0.330 [0.289, 0.369] | 0.642 [0.577, 0.708] | 0.477 [0.394, 0.561] | 0.820 [0.709, 0.919] | 0.307 [0.261, 0.354] | 0.529 [0.442, 0.609] | 0.469 [0.407, 0.528] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| engine_as_shipped | 31 | 0.742 [0.581, 0.871] | 0.903 [0.774, 1.000] | 0.903 [0.774, 1.000] | 0.935 [0.839, 1.000] | 0.196 [0.159, 0.234] | 0.231 [0.188, 0.274] | 0.231 [0.188, 0.274] | 0.458 [0.368, 0.542] | 0.828 [0.710, 0.935] | 0.182 [0.145, 0.219] | 0.512 [0.426, 0.593] | 0.381 [0.322, 0.437] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |


`rrf_bge_two_pass` rows repeat `rrf_bge` retrieval (they differ only in generation). `engine_as_shipped` is the full shipped retrieval path (query rewriter/classifier, gates, hop retry, rerank) and returns ≤ 6 parent chunks: its R@10, R@100, MAP and nDCG@10 are therefore computed on a list of at most 6 documents (lower bounds, not comparable with depth-100 rows); Hit@k and R@5 are the comparable cells.

See `ablation.md` for per-dataset tables with the best value per column in bold.

## 5. Generation quality (RAGAS, judged on the evidence the generator actually saw)

RAGAS `retrieved_contexts` = the slice of each retrieved parent that fits the prompt's 12,000-character evidence budget (`table_visibility.csv`). Reference-based metrics inherit the quality of the references (RAGBench references are model-generated; MultiHop-RAG references are 1–5 word answers).


### multihoprag — `multihoprag|q300|dfull`

| config | answerable | judged | EM | token F1 | contains-answer |
|---|---|---|---|---|---|
| dense_only | 265 | 0 | 0.000 [0.000, 0.000] | 0.026 [0.021, 0.032] | 0.543 [0.483, 0.600] |
| bm25_only | 265 | 0 | 0.000 [0.000, 0.000] | 0.013 [0.010, 0.016] | 0.317 [0.260, 0.374] |
| rrf | 265 | 0 | 0.000 [0.000, 0.000] | 0.019 [0.016, 0.023] | 0.449 [0.389, 0.509] |
| rrf_bge | 265 | 0 | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.030] | 0.521 [0.460, 0.581] |
| rrf_bge_two_pass | 265 | 0 | 0.000 [0.000, 0.000] | 0.016 [0.014, 0.019] | 0.479 [0.419, 0.540] |
| engine_as_shipped | 265 | 0 | 0.000 [0.000, 0.000] | 0.016 [0.014, 0.018] | 0.494 [0.434, 0.555] |


### ragbench-covidqa — `ragbench-covidqa|qall|dfull`

| config | answerable | judged |
|---|---|---|
| dense_only | 242 | 0 |
| bm25_only | 242 | 0 |
| rrf | 242 | 0 |
| rrf_bge | 242 | 0 |
| rrf_bge_two_pass | 242 | 0 |
| engine_as_shipped | 242 | 0 |


#### Custom corpora (not public benchmarks: the project's own documentation and the two project PDFs; queries and labels were written for this project, not human-audited)


### docsqa — `docsqa|qall|dfull`

| config | answerable | judged |
|---|---|---|
| dense_only | 35 | 0 |
| bm25_only | 35 | 0 |
| rrf | 35 | 0 |
| rrf_bge | 35 | 0 |
| rrf_bge_two_pass | 35 | 0 |
| engine_as_shipped | 35 | 0 |


### pdf-got — `pdf-got|qall|dfull`

| config | answerable | judged |
|---|---|---|
| dense_only | 31 | 0 |
| bm25_only | 31 | 0 |
| rrf | 31 | 0 |
| rrf_bge | 31 | 0 |
| rrf_bge_two_pass | 31 | 0 |
| engine_as_shipped | 31 | 0 |


### pdf-quantum — `pdf-quantum|qall|dfull`

| config | answerable | judged |
|---|---|---|
| dense_only | 31 | 0 |
| bm25_only | 31 | 0 |
| rrf | 31 | 0 |
| rrf_bge | 31 | 0 |
| rrf_bge_two_pass | 31 | 0 |
| engine_as_shipped | 31 | 0 |


## 5b. LLM-rubric answer correctness (all six configurations)

RAGAS `FactualCorrectness` (F1 over decomposed claims) penalises a correct answer for every extra true claim it makes, and disagreed with an independent reading of the answers (§5c). The rubric metric asks the judge one question per answer — does it convey what the reference says? — and scores correct = 1, partial = 0.5, incorrect = 0, abstained = 0 (for answerable questions). The judge sees only question, reference and answer (not the evidence), at temperature 0, with `thinking_budget=0` and the reason generated before the verdict (`eval/judge_correctness.py`). For questions the corpus cannot answer, declining or pointing out the false premise scores 1 and a confident made-up answer scores 0 (last column).


### multihoprag — `multihoprag|q300|dfull`

| config | answerable (scored) | rubric correctness | correct | partial | incorrect | abstained | unanswerable handled correctly |
|---|---|---|---|---|---|---|---|
| dense_only | 265 (0) | – | – | – | – | – | – |
| bm25_only | 265 (0) | – | – | – | – | – | – |
| rrf | 265 (0) | – | – | – | – | – | – |
| rrf_bge | 265 (0) | – | – | – | – | – | – |
| rrf_bge_two_pass | 265 (0) | – | – | – | – | – | – |
| engine_as_shipped | 265 (0) | – | – | – | – | – | – |


### ragbench-covidqa — `ragbench-covidqa|qall|dfull`

| config | answerable (scored) | rubric correctness | correct | partial | incorrect | abstained | unanswerable handled correctly |
|---|---|---|---|---|---|---|---|
| dense_only | 242 (0) | – | – | – | – | – | – |
| bm25_only | 242 (0) | – | – | – | – | – | – |
| rrf | 242 (0) | – | – | – | – | – | – |
| rrf_bge | 242 (0) | – | – | – | – | – | – |
| rrf_bge_two_pass | 242 (0) | – | – | – | – | – | – |
| engine_as_shipped | 242 (0) | – | – | – | – | – | – |


#### Custom corpora (not public benchmarks: the project's own documentation and the two project PDFs; queries and labels were written for this project, not human-audited)


### docsqa — `docsqa|qall|dfull`

| config | answerable (scored) | rubric correctness | correct | partial | incorrect | abstained | unanswerable handled correctly |
|---|---|---|---|---|---|---|---|
| dense_only | 35 (0) | – | – | – | – | – | – |
| bm25_only | 35 (0) | – | – | – | – | – | – |
| rrf | 35 (0) | – | – | – | – | – | – |
| rrf_bge | 35 (0) | – | – | – | – | – | – |
| rrf_bge_two_pass | 35 (0) | – | – | – | – | – | – |
| engine_as_shipped | 35 (0) | – | – | – | – | – | – |


### pdf-got — `pdf-got|qall|dfull`

| config | answerable (scored) | rubric correctness | correct | partial | incorrect | abstained | unanswerable handled correctly |
|---|---|---|---|---|---|---|---|
| dense_only | 31 (0) | – | – | – | – | – | – |
| bm25_only | 31 (0) | – | – | – | – | – | – |
| rrf | 31 (0) | – | – | – | – | – | – |
| rrf_bge | 31 (0) | – | – | – | – | – | – |
| rrf_bge_two_pass | 31 (0) | – | – | – | – | – | – |
| engine_as_shipped | 31 (0) | – | – | – | – | – | – |


### pdf-quantum — `pdf-quantum|qall|dfull`

| config | answerable (scored) | rubric correctness | correct | partial | incorrect | abstained | unanswerable handled correctly |
|---|---|---|---|---|---|---|---|
| dense_only | 31 (0) | – | – | – | – | – | – |
| bm25_only | 31 (0) | – | – | – | – | – | – |
| rrf | 31 (0) | – | – | – | – | – | – |
| rrf_bge | 31 (0) | – | – | – | – | – | – |
| rrf_bge_two_pass | 31 (0) | – | – | – | – | – | – |
| engine_as_shipped | 31 (0) | – | – | – | – | – | – |


## 5c. Validation of the rubric metric against Claude's labels (model labels, not human)

Claude (an Anthropic model) labelled correctness (0 / 0.5 / 1) for 50 answers sampled in the smoke run (`rubric_validation/claude_audit_items.jsonl`, `claude_labels.jsonl`). The rubric prompt was written once and run once on these items — it was **not** tuned on them. The items are from the n=20-per-dataset smoke run, configs `rrf_bge` and `rrf_bge_two_pass` only; they contain no unanswerable questions, so the abstention rule of the rubric is not covered by this validation.

| metric | n | Claude mean | metric mean | mean abs. diff | exact agreement | binary agreement @0.5 | Pearson r | Spearman ρ | weighted κ (linear) |
|---|---|---|---|---|---|---|---|---|---|
| rubric correctness (gemini-3.7-flash@think0) | 50 | 0.88 | 0.9 | 0.06 | 0.88 | 1.0 | 0.797 | 0.616 | 0.686 |
| RAGAS FactualCorrectness F1 (for comparison) | 50 | 0.88 | 0.268 | 0.619 |  |  | 0.09 |  |  |


Confusion (rows = Claude label, columns = rubric score): Claude 0.0 → 0.0: 3, 0.5: 0, 1.0: 0; Claude 0.5 → 0.0: 0, 0.5: 2, 1.0: 4; Claude 1.0 → 0.0: 0, 0.5: 2, 1.0: 39. The rubric is more lenient than Claude on 8% of items and stricter on 4%; every disagreement is listed in `rubric_validation/rubric_validation.json`.


## 6. Robustness

Abstention = the system's refusal sentence, an empty output, or a short answer built on a refusal phrase (heuristic, `eval/metrics.py`).

| dataset | config | unanswerable n | abstention on unanswerable ↑ | answerable n | false abstention ↓ | false abst. | gold in context | anachronism gate fired |
|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 9 | 0.889 [0.667, 1.000] | 35 | 0.029 [0.000, 0.086] | 0.029 | – |
| docsqa | bm25_only | 9 | 1.000 [1.000, 1.000] | 35 | 0.143 [0.029, 0.257] | 0.037 | – |
| docsqa | rrf | 9 | 1.000 [1.000, 1.000] | 35 | 0.029 [0.000, 0.086] | 0.000 | – |
| docsqa | rrf_bge | 9 | 1.000 [1.000, 1.000] | 35 | 0.000 [0.000, 0.000] | 0.000 | – |
| docsqa | rrf_bge_two_pass | 9 | 1.000 [1.000, 1.000] | 35 | 0.029 [0.000, 0.086] | 0.031 | – |
| docsqa | engine_as_shipped | 9 | 0.778 [0.444, 1.000] | 35 | 0.057 [0.000, 0.143] | 0.062 | 0.000 |
| multihoprag | dense_only | 35 | 1.000 [1.000, 1.000] | 265 | 0.321 [0.264, 0.377] | 0.319 | – |
| multihoprag | bm25_only | 35 | 1.000 [1.000, 1.000] | 265 | 0.649 [0.592, 0.706] | 0.537 | – |
| multihoprag | rrf | 35 | 1.000 [1.000, 1.000] | 265 | 0.438 [0.377, 0.498] | 0.419 | – |
| multihoprag | rrf_bge | 35 | 1.000 [1.000, 1.000] | 265 | 0.347 [0.291, 0.404] | 0.344 | – |
| multihoprag | rrf_bge_two_pass | 35 | 1.000 [1.000, 1.000] | 265 | 0.147 [0.106, 0.192] | 0.145 | – |
| multihoprag | engine_as_shipped | 35 | 0.971 [0.914, 1.000] | 265 | 0.132 [0.091, 0.174] | 0.130 | 0.003 |
| pdf-got | dense_only | 5 | 1.000 [1.000, 1.000] | 31 | 0.097 [0.000, 0.226] | 0.097 | – |
| pdf-got | bm25_only | 5 | 1.000 [1.000, 1.000] | 31 | 0.677 [0.516, 0.839] | 0.182 | – |
| pdf-got | rrf | 5 | 1.000 [1.000, 1.000] | 31 | 0.290 [0.129, 0.452] | 0.045 | – |
| pdf-got | rrf_bge | 5 | 1.000 [1.000, 1.000] | 31 | 0.323 [0.161, 0.484] | 0.091 | – |
| pdf-got | rrf_bge_two_pass | 5 | 1.000 [1.000, 1.000] | 31 | 0.194 [0.065, 0.323] | 0.045 | – |
| pdf-got | engine_as_shipped | 5 | 1.000 [1.000, 1.000] | 31 | 0.194 [0.065, 0.355] | 0.111 | 0.139 |
| pdf-quantum | dense_only | 5 | 1.000 [1.000, 1.000] | 31 | 0.032 [0.000, 0.097] | 0.033 | – |
| pdf-quantum | bm25_only | 5 | 1.000 [1.000, 1.000] | 31 | 0.194 [0.065, 0.355] | 0.080 | – |
| pdf-quantum | rrf | 5 | 1.000 [1.000, 1.000] | 31 | 0.032 [0.000, 0.097] | 0.000 | – |
| pdf-quantum | rrf_bge | 5 | 1.000 [1.000, 1.000] | 31 | 0.097 [0.000, 0.194] | 0.071 | – |
| pdf-quantum | rrf_bge_two_pass | 5 | 1.000 [1.000, 1.000] | 31 | 0.161 [0.032, 0.290] | 0.143 | – |
| pdf-quantum | engine_as_shipped | 5 | 1.000 [1.000, 1.000] | 31 | 0.194 [0.065, 0.355] | 0.143 | 0.194 |
| ragbench-covidqa | dense_only | 0 | – | 242 | 0.182 [0.136, 0.231] | 0.173 | – |
| ragbench-covidqa | bm25_only | 0 | – | 242 | 0.273 [0.215, 0.331] | 0.144 | – |
| ragbench-covidqa | rrf | 0 | – | 242 | 0.227 [0.174, 0.281] | 0.203 | – |
| ragbench-covidqa | rrf_bge | 0 | – | 242 | 0.178 [0.132, 0.227] | 0.172 | – |
| ragbench-covidqa | rrf_bge_two_pass | 0 | – | 242 | 0.194 [0.145, 0.248] | 0.185 | – |
| ragbench-covidqa | engine_as_shipped | 0 | – | 242 | 0.190 [0.140, 0.240] | 0.180 | 0.000 |


### Evidence visibility (12,000-character prompt budget)

| dataset | config | n | evidence_chars_total_mean | evidence_chars_seen_mean | fraction_of_evidence_seen | gold_in_context_rate | gold_in_seen_context_rate | gold_docs_seen_fraction |
|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 44 | 5850.068 | 5777.773 | 0.988 | 0.971 | 0.971 | 0.971 |
| docsqa | bm25_only | 44 | 7527.773 | 7351.273 | 0.977 | 0.771 | 0.771 | 0.757 |
| docsqa | rrf | 44 | 6807.795 | 6753.795 | 0.992 | 0.886 | 0.886 | 0.871 |
| docsqa | rrf_bge | 44 | 6894.818 | 6742.659 | 0.978 | 0.914 | 0.914 | 0.914 |
| docsqa | rrf_bge_two_pass | 44 | 6894.818 | 6811.5 | 0.988 | 0.914 | 0.914 | 0.914 |
| docsqa | engine_as_shipped | 44 | 7510.477 | 7399.091 | 0.985 | 0.914 | 0.914 | 0.9 |
| multihoprag | dense_only | 300 | 45113.953 | 11745.303 | 0.26 | 0.992 | 0.909 | 0.529 |
| multihoprag | bm25_only | 300 | 46758.297 | 11752.627 | 0.251 | 0.717 | 0.54 | 0.245 |
| multihoprag | rrf | 300 | 46295.99 | 11753.697 | 0.254 | 0.955 | 0.796 | 0.389 |
| multihoprag | rrf_bge | 300 | 46419.663 | 11753.397 | 0.253 | 0.989 | 0.875 | 0.47 |
| multihoprag | rrf_bge_two_pass | 300 | 46419.663 | 11989.5 | 0.258 | 0.989 | 0.875 | 0.475 |
| multihoprag | engine_as_shipped | 300 | 46766.903 | 11949.503 | 0.256 | 0.989 | 0.875 | 0.512 |
| pdf-got | dense_only | 36 | 13146.361 | 10866.722 | 0.827 | 1.0 | 1.0 | 0.771 |
| pdf-got | bm25_only | 36 | 13570.972 | 11358.0 | 0.837 | 0.355 | 0.323 | 0.148 |
| pdf-got | rrf | 36 | 13509.0 | 11197.694 | 0.829 | 0.71 | 0.71 | 0.455 |
| pdf-got | rrf_bge | 36 | 13399.028 | 11110.722 | 0.829 | 0.71 | 0.71 | 0.475 |
| pdf-got | rrf_bge_two_pass | 36 | 13399.028 | 11637.028 | 0.868 | 0.71 | 0.71 | 0.475 |
| pdf-got | engine_as_shipped | 36 | 11842.833 | 9944.528 | 0.84 | 0.871 | 0.871 | 0.563 |
| pdf-quantum | dense_only | 36 | 15719.75 | 11123.139 | 0.708 | 0.968 | 0.903 | 0.363 |
| pdf-quantum | bm25_only | 36 | 16045.0 | 11157.056 | 0.695 | 0.806 | 0.71 | 0.19 |
| pdf-quantum | rrf | 36 | 15632.361 | 11128.444 | 0.712 | 0.871 | 0.871 | 0.328 |
| pdf-quantum | rrf_bge | 36 | 15858.028 | 11134.167 | 0.702 | 0.903 | 0.871 | 0.315 |
| pdf-quantum | rrf_bge_two_pass | 36 | 15858.028 | 11972.0 | 0.755 | 0.903 | 0.903 | 0.336 |
| pdf-quantum | engine_as_shipped | 36 | 12745.333 | 9643.944 | 0.757 | 0.903 | 0.871 | 0.309 |
| ragbench-covidqa | dense_only | 242 | 3597.471 | 3597.471 | 1.0 | 0.979 | 0.979 | 0.898 |
| ragbench-covidqa | bm25_only | 242 | 3970.025 | 3970.025 | 1.0 | 0.773 | 0.773 | 0.612 |
| ragbench-covidqa | rrf | 242 | 3831.888 | 3831.888 | 1.0 | 0.938 | 0.938 | 0.746 |
| ragbench-covidqa | rrf_bge | 242 | 3852.12 | 3852.12 | 1.0 | 0.938 | 0.938 | 0.756 |
| ragbench-covidqa | rrf_bge_two_pass | 242 | 3852.12 | 3852.12 | 1.0 | 0.938 | 0.938 | 0.756 |
| ragbench-covidqa | engine_as_shipped | 242 | 4331.463 | 4331.463 | 1.0 | 0.942 | 0.942 | 0.784 |


### RGB (generation only; 5 passages per query)

| kind | mode | noise_ratio | n | mentions_true | mentions_fake | abstained | accuracy | abstention_rate | rejection_rate |
|---|---|---|---|---|---|---|---|---|---|
| en_fact | one_pass | 0.0 | 100 | 0.03 | 0.94 | 0.02 |  |  |  |
| en_fact | two_pass | 0.0 | 100 | 0.0 | 0.17 | 0.81 |  |  |  |
| en_int | one_pass | 0.0 | 100 |  |  |  | 0.84 | 0.06 |  |
| en_int | two_pass | 0.0 | 100 |  |  |  | 0.13 | 0.81 |  |
| en_refine | one_pass | 0.0 | 300 |  |  |  | 1.0 | 0.0 |  |
| en_refine | one_pass | 0.2 | 300 |  |  |  | 0.997 | 0.003 |  |
| en_refine | one_pass | 0.4 | 300 |  |  |  | 1.0 | 0.003 |  |
| en_refine | one_pass | 0.6 | 300 |  |  |  | 1.0 | 0.003 |  |
| en_refine | one_pass | 0.8 | 300 |  |  |  | 0.987 | 0.023 |  |
| en_refine | one_pass | 1.0 | 300 |  |  |  |  | 0.787 | 0.787 |
| en_refine | two_pass | 0.0 | 300 |  |  |  | 0.147 | 0.853 |  |
| en_refine | two_pass | 0.2 | 300 |  |  |  | 0.147 | 0.853 |  |
| en_refine | two_pass | 0.4 | 300 |  |  |  | 0.147 | 0.853 |  |
| en_refine | two_pass | 0.6 | 300 |  |  |  | 0.143 | 0.857 |  |
| en_refine | two_pass | 0.8 | 300 |  |  |  | 0.137 | 0.86 |  |
| en_refine | two_pass | 1.0 | 300 |  |  |  |  | 0.957 | 0.957 |


## 7. System: tokens, cost, latency

Tokens are from the main run; $ uses list prices in `eval/config.py` (Gemini API standard tier; Vertex billing may differ). The hosted reranker's price is not published in the repo, so its tokens are reported but excluded from $.

| dataset | config | n_generated | gen_in_tokens_mean | gen_out_tokens_mean | rerank_tokens_mean | generator_cost_per_1k_queries_usd |
|---|---|---|---|---|---|---|
| docsqa | dense_only | 44 | 2081.09 | 63.68 | 0.0 | 1.8 |
| docsqa | bm25_only | 44 | 2387.66 | 55.75 | 0.0 | 2.0 |
| docsqa | rrf | 44 | 2296.91 | 62.09 | 0.0 | 1.96 |
| docsqa | rrf_bge | 44 | 2309.5 | 68.61 | 0.0 | 1.99 |
| docsqa | rrf_bge_two_pass | 44 | 2520.23 | 141.59 | 0.0 | 2.42 |
| docsqa | engine_as_shipped | 44 | 2790.95 | 142.09 | 0.0 | 2.93 |
| multihoprag | dense_only | 300 | 2956.5 | 101.37 | 0.0 | 2.6 |
| multihoprag | bm25_only | 300 | 2976.77 | 61.36 | 0.0 | 2.46 |
| multihoprag | rrf | 300 | 2963.11 | 81.76 | 0.0 | 2.53 |
| multihoprag | rrf_bge | 300 | 2962.51 | 92.34 | 31201.62 | 2.57 |
| multihoprag | rrf_bge_two_pass | 300 | 3682.17 | 251.85 | 24657.04 | 3.71 |
| multihoprag | engine_as_shipped | 300 | 3623.65 | 261.01 | 929.75 | 4.07 |
| pdf-got | dense_only | 36 | 3321.31 | 133.22 | 0.0 | 2.99 |
| pdf-got | bm25_only | 36 | 3459.81 | 89.5 | 0.0 | 2.93 |
| pdf-got | rrf | 36 | 3397.5 | 119.94 | 0.0 | 3.0 |
| pdf-got | rrf_bge | 36 | 3388.81 | 117.11 | 0.0 | 2.98 |
| pdf-got | rrf_bge_two_pass | 36 | 3519.36 | 232.89 | 0.0 | 3.51 |
| pdf-got | engine_as_shipped | 36 | 3340.69 | 215.53 | 0.0 | 3.6 |
| pdf-quantum | dense_only | 36 | 3717.72 | 228.36 | 0.0 | 3.64 |
| pdf-quantum | bm25_only | 36 | 3758.28 | 152.56 | 0.0 | 3.39 |
| pdf-quantum | rrf | 36 | 3753.89 | 219.11 | 0.0 | 3.64 |
| pdf-quantum | rrf_bge | 36 | 3861.83 | 245.78 | 0.0 | 3.82 |
| pdf-quantum | rrf_bge_two_pass | 36 | 3432.11 | 326.44 | 0.0 | 3.8 |
| pdf-quantum | engine_as_shipped | 36 | 3485.39 | 295.5 | 0.0 | 4.04 |
| ragbench-covidqa | dense_only | 242 | 1118.39 | 137.9 | 0.0 | 1.36 |
| ragbench-covidqa | bm25_only | 242 | 1194.37 | 108.91 | 0.0 | 1.3 |
| ragbench-covidqa | rrf | 242 | 1165.99 | 110.44 | 0.0 | 1.29 |
| ragbench-covidqa | rrf_bge | 242 | 1166.17 | 119.87 | 14756.33 | 1.32 |
| ragbench-covidqa | rrf_bge_two_pass | 242 | 1153.44 | 224.17 | 9318.1 | 1.71 |
| ragbench-covidqa | engine_as_shipped | 242 | 1284.09 | 210.57 | 0.0 | 2.07 |


## 8. Statistics

Paired tests on identical queries (baseline → variant; positive = variant better, except abstention rows): paired bootstrap (10k, seed 0) and Wilcoxon signed-rank; effect size = Cohen's d_z.

| dataset | comparison | metric | n | mean_baseline | mean_variant | mean_diff | ci_lo | ci_hi | p_bootstrap | p_wilcoxon | cohens_dz | p_bootstrap_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docsqa | dense vs rrf | hit@5 | 35 | 0.9143 | 0.8857 | -0.0286 | -0.115 | 0.0571 | 0.7882 | 0.5637 | -0.0966 | 1.0000 |
| docsqa | dense vs rrf | mrr | 35 | 0.766 | 0.6983 | -0.0677 | -0.1644 | 0.022 | 0.1492 | 0.1826 | -0.2385 | 1.0000 |
| docsqa | dense vs rrf | ndcg@10 | 35 | 0.8236 | 0.7604 | -0.0632 | -0.1401 | 0.0081 | 0.0858 | 0.1314 | -0.2822 | 1.0000 |
| docsqa | dense vs rrf | recall@100 | 35 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| docsqa | dense vs rrf | map | 35 | 0.7646 | 0.6896 | -0.075 | -0.1725 | 0.0177 | 0.1218 | 0.1466 | -0.258 | 1.0000 |
| docsqa | dense vs rrf+bge | hit@5 | 35 | 0.9143 | 0.9143 | 0.0 | -0.0857 | 0.0857 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| docsqa | dense vs rrf+bge | mrr | 35 | 0.766 | 0.7454 | -0.0205 | -0.1268 | 0.0822 | 0.7134 | 0.6817 | -0.0641 | 1.0000 |
| docsqa | dense vs rrf+bge | ndcg@10 | 35 | 0.8236 | 0.7942 | -0.0293 | -0.1101 | 0.0488 | 0.4672 | 0.4938 | -0.1202 | 1.0000 |
| docsqa | dense vs rrf+bge | recall@100 | 35 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| docsqa | dense vs rrf+bge | map | 35 | 0.7646 | 0.7335 | -0.031 | -0.1332 | 0.0685 | 0.5570 | 0.5305 | -0.0989 | 1.0000 |
| docsqa | rrf vs rrf+bge | hit@5 | 35 | 0.8857 | 0.9143 | 0.0286 | 0.0 | 0.0857 | 0.7080 | 0.3173 | 0.169 | 1.0000 |
| docsqa | rrf vs rrf+bge | mrr | 35 | 0.6983 | 0.7454 | 0.0471 | -0.0212 | 0.1169 | 0.1772 | 0.1497 | 0.2233 | 1.0000 |
| docsqa | rrf vs rrf+bge | ndcg@10 | 35 | 0.7604 | 0.7942 | 0.0339 | -0.0152 | 0.083 | 0.1710 | 0.1084 | 0.2254 | 1.0000 |
| docsqa | rrf vs rrf+bge | recall@100 | 35 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| docsqa | rrf vs rrf+bge | map | 35 | 0.6896 | 0.7335 | 0.044 | -0.0214 | 0.1087 | 0.1792 | 0.0952 | 0.2205 | 1.0000 |
| docsqa | rrf+bge vs engine_as_shipped | hit@5 | 35 | 0.9143 | 0.8571 | -0.0571 | -0.1429 | 0.0 | 0.2480 | 0.1573 | -0.2426 | 1.0000 |
| docsqa | rrf+bge vs engine_as_shipped | recall@5 | 35 | 0.9143 | 0.8429 | -0.0714 | -0.1575 | 0.0 | 0.0860 | 0.1025 | -0.2894 | 1.0000 |
| fiqa | dense vs rrf | hit@5 | 648 | 0.7731 | 0.5988 | -0.1744 | -0.2099 | -0.1389 | 0.0000 | 0.0000 | -0.3705 | 0.0000 |
| fiqa | dense vs rrf | mrr | 648 | 0.6788 | 0.4517 | -0.2271 | -0.2575 | -0.1968 | 0.0000 | 0.0000 | -0.5723 | 0.0000 |
| fiqa | dense vs rrf | ndcg@10 | 648 | 0.6136 | 0.3784 | -0.2352 | -0.2586 | -0.2116 | 0.0000 | 0.0000 | -0.7679 | 0.0000 |
| fiqa | dense vs rrf | recall@100 | 648 | 0.9181 | 0.8761 | -0.042 | -0.055 | -0.0299 | 0.0000 | 0.0000 | -0.2549 | 0.0000 |
| fiqa | dense vs rrf | map | 648 | 0.554 | 0.3162 | -0.2378 | -0.2612 | -0.2142 | 0.0000 | 0.0000 | -0.7684 | 0.0000 |
| fiqa | dense vs rrf+bge | hit@5 | 648 | 0.7731 | 0.6559 | -0.1173 | -0.1497 | -0.0849 | 0.0000 | 0.0000 | -0.2734 | 0.0000 |
| fiqa | dense vs rrf+bge | mrr | 648 | 0.6788 | 0.506 | -0.1728 | -0.2011 | -0.1445 | 0.0000 | 0.0000 | -0.469 | 0.0000 |
| fiqa | dense vs rrf+bge | ndcg@10 | 648 | 0.6136 | 0.4369 | -0.1767 | -0.1985 | -0.1544 | 0.0000 | 0.0000 | -0.6157 | 0.0000 |
| fiqa | dense vs rrf+bge | recall@100 | 648 | 0.9181 | 0.8761 | -0.042 | -0.055 | -0.0299 | 0.0000 | 0.0000 | -0.2549 | 0.0000 |
| fiqa | dense vs rrf+bge | map | 648 | 0.554 | 0.3663 | -0.1877 | -0.2098 | -0.1649 | 0.0000 | 0.0000 | -0.6381 | 0.0000 |
| fiqa | rrf vs rrf+bge | hit@5 | 648 | 0.5988 | 0.6559 | 0.0571 | 0.034 | 0.0802 | 0.0000 | 0.0000 | 0.1893 | 0.0000 |
| fiqa | rrf vs rrf+bge | mrr | 648 | 0.4517 | 0.506 | 0.0543 | 0.0387 | 0.0697 | 0.0000 | 0.0000 | 0.2653 | 0.0000 |
| fiqa | rrf vs rrf+bge | ndcg@10 | 648 | 0.3784 | 0.4369 | 0.0585 | 0.0474 | 0.0697 | 0.0000 | 0.0000 | 0.4021 | 0.0000 |
| fiqa | rrf vs rrf+bge | recall@100 | 648 | 0.8761 | 0.8761 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| fiqa | rrf vs rrf+bge | map | 648 | 0.3162 | 0.3663 | 0.0501 | 0.0386 | 0.0619 | 0.0000 | 0.0000 | 0.3292 | 0.0000 |
| hotpotqa | dense vs rrf | hit@5 | 300 | 0.9967 | 0.9933 | -0.0033 | -0.0167 | 0.0067 | 0.7956 | 0.5637 | -0.0333 | 1.0000 |
| hotpotqa | dense vs rrf | mrr | 300 | 0.9933 | 0.9664 | -0.0269 | -0.0455 | -0.0097 | 0.0022 | 0.0122 | -0.1669 | 0.1760 |
| hotpotqa | dense vs rrf | ndcg@10 | 300 | 0.9821 | 0.8679 | -0.1143 | -0.1335 | -0.0956 | 0.0000 | 0.0000 | -0.6708 | 0.0000 |
| hotpotqa | dense vs rrf | recall@100 | 300 | 0.9933 | 0.9967 | 0.0033 | 0.0 | 0.0083 | 0.2640 | 0.1573 | 0.0818 | 1.0000 |
| hotpotqa | dense vs rrf | map | 300 | 0.9742 | 0.8115 | -0.1627 | -0.1866 | -0.1389 | 0.0000 | 0.0000 | -0.7624 | 0.0000 |
| hotpotqa | dense vs rrf+bge | hit@5 | 300 | 0.9967 | 0.9967 | 0.0 | -0.01 | 0.01 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| hotpotqa | dense vs rrf+bge | mrr | 300 | 0.9933 | 0.9906 | -0.0028 | -0.0144 | 0.0083 | 0.6670 | 0.8292 | -0.0274 | 1.0000 |
| hotpotqa | dense vs rrf+bge | ndcg@10 | 300 | 0.9821 | 0.9516 | -0.0306 | -0.0425 | -0.0187 | 0.0000 | 0.0000 | -0.2917 | 0.0000 |
| hotpotqa | dense vs rrf+bge | recall@100 | 300 | 0.9933 | 0.9967 | 0.0033 | 0.0 | 0.0083 | 0.2640 | 0.1573 | 0.0818 | 1.0000 |
| hotpotqa | dense vs rrf+bge | map | 300 | 0.9742 | 0.9208 | -0.0534 | -0.0696 | -0.0372 | 0.0000 | 0.0000 | -0.3713 | 0.0000 |
| hotpotqa | rrf vs rrf+bge | hit@5 | 300 | 0.9933 | 0.9967 | 0.0033 | 0.0 | 0.01 | 0.7358 | 0.3173 | 0.0577 | 1.0000 |
| hotpotqa | rrf vs rrf+bge | mrr | 300 | 0.9664 | 0.9906 | 0.0241 | 0.0108 | 0.0391 | 0.0000 | 0.0019 | 0.1885 | 0.0000 |
| hotpotqa | rrf vs rrf+bge | ndcg@10 | 300 | 0.8679 | 0.9516 | 0.0837 | 0.0687 | 0.0993 | 0.0000 | 0.0000 | 0.6194 | 0.0000 |
| hotpotqa | rrf vs rrf+bge | recall@100 | 300 | 0.9967 | 0.9967 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| hotpotqa | rrf vs rrf+bge | map | 300 | 0.8115 | 0.9208 | 0.1093 | 0.0907 | 0.1283 | 0.0000 | 0.0000 | 0.6486 | 0.0000 |
| multihoprag | dense vs rrf | hit@5 | 265 | 0.9774 | 0.9472 | -0.0302 | -0.0604 | -0.0038 | 0.0392 | 0.0325 | -0.1322 | 1.0000 |
| multihoprag | dense vs rrf | mrr | 265 | 0.8389 | 0.7726 | -0.0663 | -0.1086 | -0.0234 | 0.0018 | 0.0015 | -0.1865 | 0.1458 |
| multihoprag | dense vs rrf | ndcg@10 | 265 | 0.804 | 0.637 | -0.167 | -0.1924 | -0.1416 | 0.0000 | 0.0000 | -0.783 | 0.0000 |
| multihoprag | dense vs rrf | recall@100 | 265 | 0.9937 | 0.9909 | -0.0028 | -0.0088 | 0.0025 | 0.3404 | 0.3573 | -0.063 | 1.0000 |
| multihoprag | dense vs rrf | map | 265 | 0.7131 | 0.5412 | -0.1719 | -0.1994 | -0.144 | 0.0000 | 0.0000 | -0.7482 | 0.0000 |
| multihoprag | dense vs rrf+bge | hit@5 | 265 | 0.9774 | 0.9698 | -0.0075 | -0.0302 | 0.0113 | 0.5964 | 0.4795 | -0.0434 | 1.0000 |
| multihoprag | dense vs rrf+bge | mrr | 265 | 0.8389 | 0.7993 | -0.0397 | -0.0774 | -0.0011 | 0.0438 | 0.0500 | -0.1254 | 1.0000 |
| multihoprag | dense vs rrf+bge | ndcg@10 | 265 | 0.804 | 0.7266 | -0.0774 | -0.0983 | -0.0558 | 0.0000 | 0.0000 | -0.4311 | 0.0000 |
| multihoprag | dense vs rrf+bge | recall@100 | 265 | 0.9937 | 0.9909 | -0.0028 | -0.0088 | 0.0025 | 0.3404 | 0.3573 | -0.063 | 1.0000 |
| multihoprag | dense vs rrf+bge | map | 265 | 0.7131 | 0.6291 | -0.084 | -0.1087 | -0.0593 | 0.0000 | 0.0000 | -0.4042 | 0.0000 |
| multihoprag | rrf vs rrf+bge | hit@5 | 265 | 0.9472 | 0.9698 | 0.0226 | 0.0 | 0.0453 | 0.0726 | 0.0578 | 0.1171 | 1.0000 |
| multihoprag | rrf vs rrf+bge | mrr | 265 | 0.7726 | 0.7993 | 0.0267 | -0.0002 | 0.0536 | 0.0516 | 0.0369 | 0.1186 | 1.0000 |
| multihoprag | rrf vs rrf+bge | ndcg@10 | 265 | 0.637 | 0.7266 | 0.0896 | 0.0711 | 0.1084 | 0.0000 | 0.0000 | 0.5817 | 0.0000 |
| multihoprag | rrf vs rrf+bge | recall@100 | 265 | 0.9909 | 0.9909 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| multihoprag | rrf vs rrf+bge | map | 265 | 0.5412 | 0.6291 | 0.0879 | 0.0681 | 0.1076 | 0.0000 | 0.0000 | 0.5308 | 0.0000 |
| multihoprag | rrf+bge vs engine_as_shipped | hit@5 | 265 | 0.9698 | 0.9774 | 0.0075 | -0.0113 | 0.0302 | 0.6144 | 0.4795 | 0.0434 | 1.0000 |
| multihoprag | rrf+bge vs engine_as_shipped | recall@5 | 265 | 0.705 | 0.7632 | 0.0582 | 0.0299 | 0.0858 | 0.0002 | 0.0001 | 0.2478 | 0.0178 |
| pdf-got | dense vs rrf | hit@5 | 31 | 1.0 | 0.7097 | -0.2903 | -0.4516 | -0.129 | 0.0000 | 0.0027 | -0.6292 | 0.0000 |
| pdf-got | dense vs rrf | mrr | 31 | 0.8253 | 0.4974 | -0.3278 | -0.4806 | -0.171 | 0.0000 | 0.0015 | -0.7318 | 0.0000 |
| pdf-got | dense vs rrf | ndcg@10 | 31 | 0.7791 | 0.4194 | -0.3597 | -0.484 | -0.2371 | 0.0000 | 0.0000 | -1.0156 | 0.0000 |
| pdf-got | dense vs rrf | recall@100 | 31 | 0.9946 | 0.9892 | -0.0054 | -0.0161 | 0.0 | 0.7106 | 0.3173 | -0.1796 | 1.0000 |
| pdf-got | dense vs rrf | map | 31 | 0.71 | 0.3544 | -0.3557 | -0.4783 | -0.2326 | 0.0000 | 0.0000 | -1.0146 | 0.0000 |
| pdf-got | dense vs rrf+bge | hit@5 | 31 | 1.0 | 0.7097 | -0.2903 | -0.4516 | -0.129 | 0.0000 | 0.0027 | -0.6292 | 0.0000 |
| pdf-got | dense vs rrf+bge | mrr | 31 | 0.8253 | 0.4964 | -0.3288 | -0.4795 | -0.1742 | 0.0000 | 0.0011 | -0.7452 | 0.0000 |
| pdf-got | dense vs rrf+bge | ndcg@10 | 31 | 0.7791 | 0.4186 | -0.3605 | -0.4743 | -0.2456 | 0.0000 | 0.0000 | -1.0776 | 0.0000 |
| pdf-got | dense vs rrf+bge | recall@100 | 31 | 0.9946 | 0.9892 | -0.0054 | -0.0161 | 0.0 | 0.7106 | 0.3173 | -0.1796 | 1.0000 |
| pdf-got | dense vs rrf+bge | map | 31 | 0.71 | 0.3536 | -0.3564 | -0.4741 | -0.2368 | 0.0000 | 0.0000 | -1.0351 | 0.0000 |
| pdf-got | rrf vs rrf+bge | hit@5 | 31 | 0.7097 | 0.7097 | 0.0 | -0.0968 | 0.0968 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-got | rrf vs rrf+bge | mrr | 31 | 0.4974 | 0.4964 | -0.001 | -0.0789 | 0.0848 | 0.9620 | 0.4989 | -0.0043 | 1.0000 |
| pdf-got | rrf vs rrf+bge | ndcg@10 | 31 | 0.4194 | 0.4186 | -0.0008 | -0.0461 | 0.0494 | 0.9376 | 0.3011 | -0.0058 | 1.0000 |
| pdf-got | rrf vs rrf+bge | recall@100 | 31 | 0.9892 | 0.9892 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-got | rrf vs rrf+bge | map | 31 | 0.3544 | 0.3536 | -0.0008 | -0.0476 | 0.0487 | 0.9408 | 0.8506 | -0.0056 | 1.0000 |
| pdf-got | rrf+bge vs engine_as_shipped | hit@5 | 31 | 0.7097 | 0.8387 | 0.129 | 0.0323 | 0.2581 | 0.0282 | 0.0455 | 0.3786 | 1.0000 |
| pdf-got | rrf+bge vs engine_as_shipped | recall@5 | 31 | 0.4683 | 0.5105 | 0.0422 | -0.028 | 0.1183 | 0.2454 | 0.3122 | 0.2008 | 1.0000 |
| pdf-quantum | dense vs rrf | hit@5 | 31 | 0.9355 | 0.871 | -0.0645 | -0.1935 | 0.0645 | 0.4482 | 0.3173 | -0.1796 | 1.0000 |
| pdf-quantum | dense vs rrf | mrr | 31 | 0.8475 | 0.792 | -0.0555 | -0.1434 | 0.0216 | 0.1768 | 0.1829 | -0.2313 | 1.0000 |
| pdf-quantum | dense vs rrf | ndcg@10 | 31 | 0.5181 | 0.4809 | -0.0372 | -0.0788 | 0.0024 | 0.0686 | 0.1068 | -0.318 | 1.0000 |
| pdf-quantum | dense vs rrf | recall@100 | 31 | 0.6351 | 0.6649 | 0.0297 | -0.0058 | 0.0667 | 0.1068 | 0.1181 | 0.286 | 1.0000 |
| pdf-quantum | dense vs rrf | map | 31 | 0.3848 | 0.3409 | -0.0439 | -0.0814 | -0.0082 | 0.0148 | 0.0428 | -0.4183 | 1.0000 |
| pdf-quantum | dense vs rrf+bge | hit@5 | 31 | 0.9355 | 0.871 | -0.0645 | -0.1935 | 0.0645 | 0.4482 | 0.3173 | -0.1796 | 1.0000 |
| pdf-quantum | dense vs rrf+bge | mrr | 31 | 0.8475 | 0.7428 | -0.1047 | -0.2064 | -0.0111 | 0.0292 | 0.0481 | -0.3772 | 1.0000 |
| pdf-quantum | dense vs rrf+bge | ndcg@10 | 31 | 0.5181 | 0.4604 | -0.0577 | -0.1027 | -0.0166 | 0.0056 | 0.0088 | -0.4656 | 0.4368 |
| pdf-quantum | dense vs rrf+bge | recall@100 | 31 | 0.6351 | 0.6649 | 0.0297 | -0.0058 | 0.0667 | 0.1068 | 0.1181 | 0.286 | 1.0000 |
| pdf-quantum | dense vs rrf+bge | map | 31 | 0.3848 | 0.3213 | -0.0636 | -0.1008 | -0.0282 | 0.0006 | 0.0036 | -0.6076 | 0.0516 |
| pdf-quantum | rrf vs rrf+bge | hit@5 | 31 | 0.871 | 0.871 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-quantum | rrf vs rrf+bge | mrr | 31 | 0.792 | 0.7428 | -0.0492 | -0.1344 | 0.0336 | 0.2352 | 0.3054 | -0.209 | 1.0000 |
| pdf-quantum | rrf vs rrf+bge | ndcg@10 | 31 | 0.4809 | 0.4604 | -0.0205 | -0.0514 | 0.0115 | 0.2038 | 0.1615 | -0.2313 | 1.0000 |
| pdf-quantum | rrf vs rrf+bge | recall@100 | 31 | 0.6649 | 0.6649 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-quantum | rrf vs rrf+bge | map | 31 | 0.3409 | 0.3213 | -0.0196 | -0.0449 | 0.004 | 0.1124 | 0.1363 | -0.2831 | 1.0000 |
| pdf-quantum | rrf+bge vs engine_as_shipped | hit@5 | 31 | 0.871 | 0.871 | 0.0 | -0.0968 | 0.0968 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-quantum | rrf+bge vs engine_as_shipped | recall@5 | 31 | 0.3245 | 0.3016 | -0.0228 | -0.0712 | 0.0202 | 0.3208 | 0.2583 | -0.1724 | 1.0000 |
| ragbench-covidqa | dense vs rrf | hit@5 | 242 | 0.9711 | 0.9174 | -0.0537 | -0.0868 | -0.0207 | 0.0008 | 0.0016 | -0.2065 | 0.0672 |
| ragbench-covidqa | dense vs rrf | mrr | 242 | 0.8584 | 0.7854 | -0.073 | -0.1137 | -0.033 | 0.0002 | 0.0004 | -0.2283 | 0.0178 |
| ragbench-covidqa | dense vs rrf | ndcg@10 | 242 | 0.8496 | 0.7124 | -0.1372 | -0.1724 | -0.1025 | 0.0000 | 0.0000 | -0.4974 | 0.0000 |
| ragbench-covidqa | dense vs rrf | recall@100 | 242 | 1.0 | 0.8636 | -0.1364 | -0.1711 | -0.104 | 0.0000 | 0.0000 | -0.5118 | 0.0000 |
| ragbench-covidqa | dense vs rrf | map | 242 | 0.7899 | 0.6348 | -0.1551 | -0.1952 | -0.1159 | 0.0000 | 0.0000 | -0.4971 | 0.0000 |
| ragbench-covidqa | dense vs rrf+bge | hit@5 | 242 | 0.9711 | 0.9298 | -0.0413 | -0.0744 | -0.0124 | 0.0066 | 0.0075 | -0.174 | 0.5082 |
| ragbench-covidqa | dense vs rrf+bge | mrr | 242 | 0.8584 | 0.8173 | -0.0411 | -0.0816 | -0.0019 | 0.0414 | 0.0496 | -0.1288 | 1.0000 |
| ragbench-covidqa | dense vs rrf+bge | ndcg@10 | 242 | 0.8496 | 0.7426 | -0.107 | -0.1441 | -0.0718 | 0.0000 | 0.0000 | -0.3742 | 0.0000 |
| ragbench-covidqa | dense vs rrf+bge | recall@100 | 242 | 1.0 | 0.8636 | -0.1364 | -0.1711 | -0.104 | 0.0000 | 0.0000 | -0.5118 | 0.0000 |
| ragbench-covidqa | dense vs rrf+bge | map | 242 | 0.7899 | 0.6707 | -0.1192 | -0.1614 | -0.0781 | 0.0000 | 0.0000 | -0.365 | 0.0000 |
| ragbench-covidqa | rrf vs rrf+bge | hit@5 | 242 | 0.9174 | 0.9298 | 0.0124 | -0.0041 | 0.0289 | 0.2452 | 0.1797 | 0.0864 | 1.0000 |
| ragbench-covidqa | rrf vs rrf+bge | mrr | 242 | 0.7854 | 0.8173 | 0.0319 | 0.0114 | 0.0536 | 0.0014 | 0.0024 | 0.1898 | 0.1162 |
| ragbench-covidqa | rrf vs rrf+bge | ndcg@10 | 242 | 0.7124 | 0.7426 | 0.0302 | 0.0171 | 0.044 | 0.0000 | 0.0000 | 0.2867 | 0.0000 |
| ragbench-covidqa | rrf vs rrf+bge | recall@100 | 242 | 0.8636 | 0.8636 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| ragbench-covidqa | rrf vs rrf+bge | map | 242 | 0.6348 | 0.6707 | 0.0359 | 0.0203 | 0.0525 | 0.0000 | 0.0000 | 0.2808 | 0.0000 |
| ragbench-covidqa | rrf+bge vs engine_as_shipped | hit@5 | 242 | 0.9298 | 0.9215 | -0.0083 | -0.0289 | 0.0124 | 0.5504 | 0.4142 | -0.0525 | 1.0000 |
| ragbench-covidqa | rrf+bge vs engine_as_shipped | recall@5 | 242 | 0.74 | 0.749 | 0.009 | -0.0127 | 0.0303 | 0.4202 | 0.5447 | 0.0525 | 1.0000 |
| scifact | dense vs rrf | hit@5 | 300 | 0.9533 | 0.7833 | -0.17 | -0.2167 | -0.1233 | 0.0000 | 0.0000 | -0.4144 | 0.0000 |
| scifact | dense vs rrf | mrr | 300 | 0.8738 | 0.6795 | -0.1944 | -0.2371 | -0.1525 | 0.0000 | 0.0000 | -0.5164 | 0.0000 |
| scifact | dense vs rrf | ndcg@10 | 300 | 0.8996 | 0.7022 | -0.1974 | -0.2363 | -0.1602 | 0.0000 | 0.0000 | -0.5845 | 0.0000 |
| scifact | dense vs rrf | recall@100 | 300 | 0.9933 | 0.9933 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| scifact | dense vs rrf | map | 300 | 0.8713 | 0.6596 | -0.2117 | -0.2553 | -0.1698 | 0.0000 | 0.0000 | -0.5644 | 0.0000 |
| scifact | dense vs rrf+bge | hit@5 | 300 | 0.9533 | 0.82 | -0.1333 | -0.1767 | -0.09 | 0.0000 | 0.0000 | -0.353 | 0.0000 |
| scifact | dense vs rrf+bge | mrr | 300 | 0.8738 | 0.7121 | -0.1617 | -0.2014 | -0.1225 | 0.0000 | 0.0000 | -0.461 | 0.0000 |
| scifact | dense vs rrf+bge | ndcg@10 | 300 | 0.8996 | 0.7437 | -0.156 | -0.1902 | -0.1222 | 0.0000 | 0.0000 | -0.5223 | 0.0000 |
| scifact | dense vs rrf+bge | recall@100 | 300 | 0.9933 | 0.9933 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| scifact | dense vs rrf+bge | map | 300 | 0.8713 | 0.6958 | -0.1754 | -0.2151 | -0.1359 | 0.0000 | 0.0000 | -0.5028 | 0.0000 |
| scifact | rrf vs rrf+bge | hit@5 | 300 | 0.7833 | 0.82 | 0.0367 | 0.0133 | 0.0633 | 0.0028 | 0.0045 | 0.166 | 0.2212 |
| scifact | rrf vs rrf+bge | mrr | 300 | 0.6795 | 0.7121 | 0.0327 | 0.0125 | 0.0535 | 0.0016 | 0.0000 | 0.1771 | 0.1312 |
| scifact | rrf vs rrf+bge | ndcg@10 | 300 | 0.7022 | 0.7437 | 0.0414 | 0.0246 | 0.059 | 0.0002 | 0.0000 | 0.2645 | 0.0178 |
| scifact | rrf vs rrf+bge | recall@100 | 300 | 0.9933 | 0.9933 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| scifact | rrf vs rrf+bge | map | 300 | 0.6596 | 0.6958 | 0.0363 | 0.017 | 0.0562 | 0.0006 | 0.0000 | 0.205 | 0.0516 |
| scifact | rrf+bge vs engine_as_shipped | hit@5 | 300 | 0.82 | 0.84 | 0.02 | 0.0 | 0.04 | 0.0746 | 0.0578 | 0.11 | 1.0000 |
| scifact | rrf+bge vs engine_as_shipped | recall@5 | 300 | 0.7934 | 0.8178 | 0.0244 | 0.0035 | 0.0468 | 0.0208 | 0.0254 | 0.1285 | 1.0000 |
| multihoprag | one-pass vs two-pass | false_abstention | 265 | 0.3472 | 0.1472 | -0.2 | -0.2528 | -0.1472 | 0.0000 | 0.0000 | -0.4671 | 0.0000 |
| multihoprag | one-pass vs two-pass | abstention_on_unanswerable | 35 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| ragbench-covidqa | one-pass vs two-pass | false_abstention | 242 | 0.1777 | 0.1942 | 0.0165 | -0.0124 | 0.0455 | 0.3140 | 0.2482 | 0.0743 | 1.0000 |
| docsqa | one-pass vs two-pass | false_abstention | 35 | 0.0 | 0.0286 | 0.0286 | 0.0 | 0.0857 | 0.7190 | 0.3173 | 0.169 | 1.0000 |
| docsqa | one-pass vs two-pass | abstention_on_unanswerable | 9 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-quantum | one-pass vs two-pass | false_abstention | 31 | 0.0968 | 0.1613 | 0.0645 | -0.0645 | 0.1935 | 0.4476 | 0.3173 | 0.1796 | 1.0000 |
| pdf-quantum | one-pass vs two-pass | abstention_on_unanswerable | 5 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |
| pdf-got | one-pass vs two-pass | false_abstention | 31 | 0.3226 | 0.1935 | -0.129 | -0.2581 | -0.0323 | 0.0266 | 0.0455 | -0.3786 | 1.0000 |
| pdf-got | one-pass vs two-pass | abstention_on_unanswerable | 5 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 | 1.0000 |


`p_bootstrap_holm` is the Holm step-down adjustment of `p_bootstrap` over every contrast in this table (a conservative family); raw p-values are descriptive.

## 9. Limitations, leakage and comparability (read before citing)

- **Judge validity.** The primary judge (`gemini-3.7-flash`) is a different model from the generator (`gemini-3.8-flash`) but from the same vendor/family, so family and self-preference bias are possible. The rubric-correctness metric was validated on 50 items against Claude labels (§5c) — agreement is reported there, and the rubric is more lenient than Claude on partially correct answers. An independent cross-check by Claude (Anthropic) on a sample is reported in `claude_audit_summary.json` / `judge_audit.csv` (`claude_label` column) when present. These are **model labels, not human labels**: the `human_label` column is intentionally blank and no human agreement figure is claimed.
- **Data leakage.** BEIR (SciFact/FiQA/HotpotQA), HotpotQA's Wikipedia text, MultiHop-RAG news articles (2023) and RGB are public and may be in the training data of both the generator and the judge; the embedding model and reranker may also have seen them. The documentation-QA set is derived from this repository's own docs and cannot be leaked into any model trained before it existed, but it was authored by an AI assistant (quotes machine-verified; answers not human-reviewed).
- **Comparability with published numbers.** Corpus subsampling (HotpotQA, FiQA/others where capped) makes the task easier than the BEIR leaderboard task; BM25 here is SQLite FTS5 (no stemming/stopwords) not Lucene/Pyserini; chunking splits long documents and scores are max-chunk-per-document; query counts differ from published protocols. Do **not** put these next to leaderboard numbers without these qualifiers.
- **RAGBench** questions are written against their own source passages; pooling passages from many questions makes some questions under-specified, which inflates false abstention and depresses retrieval metrics. References are LLM-generated.
- **MultiHop-RAG / long documents.** Both generators cap the evidence sent to the model at 12,000 characters, so on long articles most retrieved context never reaches the LLM (see §6). This is a property of the shipped system and is measured, not tuned.
- **Hard-coded anachronism gate** (`quality_gates.is_anachronism`) is specific to the GoT corpus; it is only exercised by the `engine_as_shipped` rows (fire rate in §6).
- **Small samples.** `docsqa` has 44 items (9 unanswerable) and RAGBench-covidqa 242; per-category breakdowns would be too thin to interpret. The `smoke/` directory (n=20 per dataset) is a pipeline check, not evidence of differences between configs.
- The SLM contextual-summary ingestion variant (Qwen3-0.6B) was **not run** (not on the base branch; deferred by request). No numbers are reported for it.

## 10. Reproduce

```bash
uv venv --python 3.12 .venv && uv sync --extra dev
uv pip install --python .venv/bin/python -r eval/requirements.txt
gcloud auth application-default login   # Vertex AI ADC
./run_all.sh
```

