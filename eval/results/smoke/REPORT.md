# DeepContext public-benchmark report

**SMOKE TEST (n=20 per dataset; Gemini 3.8 generator, Gemini 3.7 judge, PostgreSQL; not for publication)** — generated from files in `eval/results/smoke`; every number below is read from them.

- Branch `eval/public-benchmarks` @ `0cbfb03` (base `feat/rag-quality-generation`), working tree dirty at report time
- Judge: `gemini-3.7-flash` (temperature 0) — generator: `gemini-3.8-flash` (different model; both are Google models, so family bias is possible)
- Embeddings: `gemini-embedding-2` 768-d via Vertex AI (`global` endpoint); reranker: bge-reranker-v2-m3 (hosted, EcoHash API)

## 1. Setup and versions

Python 3.12.11 on macOS-26.7.2-arm64-arm-64bit (arm64); ragas 0.4.3, beir 2.2.0, datasets 5.1.0, pandas 3.0.6, numpy 2.5.2, scipy 1.18.1, google-genai 2.31.0, langchain-core 0.3.86, langchain-community 0.3.31, huggingface-hub 1.33.0, pyarrow 26.0.0.

Storage backend: postgres — PostgreSQL database deepcontext_eval (one schema per index; 6 indexes, see `dataset_meta.json`). On PostgreSQL the lexical channel is the product's Postgres full-text search (`ts_rank` over the first 8 query terms), **not BM25**, and the dense channel is pgvector HNSW (approximate). `bm25_only` rows therefore mean *Postgres full-text only*.

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


**Thinking-flag check on `gemini-3.8-flash`** (`thinking_flag_check.json`):

| enable_thinking flag | dataset | queries | generation calls | calls hitting MAX_TOKENS | mean thinking tokens | two-pass refusals |
|---|---|---|---|---|---|---|
| honoured (harness default) | docsqa | 20 | 40 | 0 | 3.0 | 1 |
| honoured (harness default) | multihoprag | 12 | 24 | 0 | 0.0 | 4 |
| ignored (stock client) | docsqa | 20 | 40 | 3 | 252.1 | 1 |
| ignored (stock client) | multihoprag | 12 | 24 | 9 | 391.2 | 3 |

## 2. Datasets

| run | queries used / available | corpus docs used / total | license | source | sampling |
|---|---|---|---|---|---|
| scifact|q20|dfull | 20 / 300 | 5183 / 5183 | claims/annotations CC BY 4.0; abstracts ODC-By 1.0 (S2ORC) — allenai/scifact LICENSE.md | HF mirror BeIR/scifact @ b3b5335604 + BeIR/scifact-qrels @ 2938d17dc3 | full corpus; 20/300 queries sampled (seed=42) |
| fiqa|q20|d5000 | 20 / 648 | 5000 / 57638 | HF BeIR card: cc-by-sa-4.0 (packaging); FiQA-2018 upstream terms not stated — UNVERIFIED | HF mirror BeIR/fiqa @ 979c07a7cb + BeIR/fiqa-qrels @ 252958f2d6 | seed=42: all 57 relevant docs + 4943 uniformly sampled distractors (Algorithm R); 20/648 queries sampled (seed=42) |
| hotpotqa|q20|d5000 | 20 / 7405 | 5000 / 5233329 | CC BY-SA 4.0 — hotpotqa.github.io | HF mirror BeIR/hotpotqa @ a7e8bab212 + BeIR/hotpotqa-qrels @ b15429e924 | seed=42: all 40 relevant docs + 4960 uniformly sampled distractors (Algorithm R); 20/7405 queries sampled (seed=42) |
| multihoprag|q20|dfull | 20 / 2556 | 609 / 609 | ODC-By (per HF dataset card) | HF yixuantt/MultiHopRAG (MultiHopRAG.json + corpus.json) | stratified by question_type, seed=42; full 609-article corpus |
| ragbench-covidqa|q20|dfull | 20 / 246 | 78 / 78 | CC BY 4.0 (per HF dataset card) | HF galileo-ai/ragbench (covidqa, test) | seed=42 sample |
| docsqa|qall|dfull | 44 / 44 | 177 / 177 | repository documentation (Apache-2.0 repo) | repo documentation on branch eval/public-benchmarks (README.md, AGENTS.md, RERANKER_UPGRADE.md, Rag eval.md, docs/ARCHITECTURE.md, docs/CACHE_SCHEDULER_AGENTIC.md, docs/CRITICAL_ASSESSMENT_AND_SCOPE.md, docs/PRD.md, docs/RAG_QUALITY_NEXT.md, docs/SYSTEM_HANDOFF.md, docs/TECH_STACK.md, docs/VERIFICATION_AND_SOURCES.md) | all items; sections = markdown H1-H3 blocks >= 80 chars |

RGB (generation-only): `en_refine` / `en_int` / `en_fact` from github.com/chen700564/RGB (CC BY-NC-SA 4.0, non-commercial research use).

## 3. Headline table (mean [95% bootstrap CI, 10k resamples, seed 0])

`rrf_bge_two_pass` shares the retrieval of `rrf_bge`. Generation metrics are blank where no generation run exists for that dataset/config. p95 latency is end-to-end from the live pass.

| dataset | config | n | Hit@5 | nDCG@10 | Faithfulness | Context Recall | Factual F1 | p95 latency (s) |
|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 35 | 0.914 [0.829, 1.000] | 0.821 [0.739, 0.899] | – | – | – | 14.79 |
| docsqa | bm25_only | 35 | 0.771 [0.629, 0.886] | 0.639 [0.521, 0.753] | – | – | – | 2.07 |
| docsqa | rrf | 35 | 0.800 [0.657, 0.914] | 0.680 [0.558, 0.798] | – | – | – | 3.99 |
| docsqa | rrf_bge | 35 | 0.829 [0.686, 0.943] | 0.702 [0.576, 0.822] | 0.957 [0.886, 1.000] | 0.914 [0.800, 1.000] | 0.189 [0.096, 0.291] | 16.20 |
| docsqa | rrf_bge_two_pass | 35 | 0.829 [0.686, 0.943] | 0.702 [0.576, 0.822] | 0.967 [0.905, 1.000] | 0.914 [0.800, 1.000] | 0.242 [0.137, 0.353] | 14.15 |
| fiqa | dense_only | 20 | 0.950 [0.850, 1.000] | 0.902 [0.785, 0.983] | – | – | – | – |
| fiqa | bm25_only | 20 | 0.500 [0.300, 0.700] | 0.365 [0.222, 0.513] | – | – | – | – |
| fiqa | rrf | 20 | 0.900 [0.750, 1.000] | 0.700 [0.563, 0.825] | – | – | – | – |
| fiqa | rrf_bge | 20 | 0.900 [0.750, 1.000] | 0.750 [0.609, 0.873] | – | – | – | – |
| fiqa | rrf_bge_two_pass | 20 | 0.900 [0.750, 1.000] | 0.750 [0.609, 0.873] | – | – | – | – |
| hotpotqa | dense_only | 20 | 1.000 [1.000, 1.000] | 0.983 [0.958, 1.000] | – | – | – | – |
| hotpotqa | bm25_only | 20 | 1.000 [1.000, 1.000] | 0.789 [0.701, 0.877] | – | – | – | – |
| hotpotqa | rrf | 20 | 1.000 [1.000, 1.000] | 0.899 [0.829, 0.960] | – | – | – | – |
| hotpotqa | rrf_bge | 20 | 1.000 [1.000, 1.000] | 0.984 [0.966, 1.000] | – | – | – | – |
| hotpotqa | rrf_bge_two_pass | 20 | 1.000 [1.000, 1.000] | 0.984 [0.966, 1.000] | – | – | – | – |
| multihoprag | dense_only | 18 | 0.944 [0.833, 1.000] | 0.774 [0.697, 0.846] | – | – | – | 6.40 |
| multihoprag | bm25_only | 18 | 0.667 [0.444, 0.889] | 0.385 [0.254, 0.521] | – | – | – | 27.35 |
| multihoprag | rrf | 18 | 0.833 [0.667, 1.000] | 0.562 [0.432, 0.681] | – | – | – | 25.14 |
| multihoprag | rrf_bge | 18 | 0.833 [0.667, 1.000] | 0.661 [0.537, 0.773] | 0.792 [0.630, 0.926] | 0.722 [0.500, 0.889] | 0.272 [0.118, 0.439] | 7.94 |
| multihoprag | rrf_bge_two_pass | 18 | 0.833 [0.667, 1.000] | 0.661 [0.537, 0.773] | 0.763 [0.586, 0.914] | 0.667 [0.444, 0.889] | 0.039 [0.000, 0.118] | 13.00 |
| ragbench-covidqa | dense_only | 20 | 1.000 [1.000, 1.000] | 0.924 [0.848, 0.983] | – | – | – | – |
| ragbench-covidqa | bm25_only | 20 | 0.950 [0.850, 1.000] | 0.805 [0.690, 0.906] | – | – | – | – |
| ragbench-covidqa | rrf | 20 | 1.000 [1.000, 1.000] | 0.847 [0.758, 0.931] | – | – | – | – |
| ragbench-covidqa | rrf_bge | 20 | 1.000 [1.000, 1.000] | 0.861 [0.759, 0.947] | 0.940 [0.830, 1.000] | 0.650 [0.450, 0.850] | 0.315 [0.161, 0.487] | – |
| ragbench-covidqa | rrf_bge_two_pass | 20 | 1.000 [1.000, 1.000] | 0.861 [0.759, 0.947] | 0.954 [0.904, 0.994] | 0.650 [0.450, 0.850] | 0.250 [0.109, 0.408] | – |
| scifact | dense_only | 20 | 0.900 [0.750, 1.000] | 0.838 [0.720, 0.945] | – | – | – | – |
| scifact | bm25_only | 20 | 0.550 [0.350, 0.750] | 0.470 [0.269, 0.670] | – | – | – | – |
| scifact | rrf | 20 | 0.850 [0.700, 1.000] | 0.714 [0.558, 0.860] | – | – | – | – |
| scifact | rrf_bge | 20 | 0.800 [0.600, 0.950] | 0.724 [0.578, 0.857] | – | – | – | – |
| scifact | rrf_bge_two_pass | 20 | 0.800 [0.600, 0.950] | 0.724 [0.578, 0.857] | – | – | – | – |

## 4. Retrieval (document level)


### docsqa — `docsqa|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 35 | 0.629 [0.457, 0.771] | 0.886 [0.771, 0.971] | 0.914 [0.829, 1.000] | 0.943 [0.857, 1.000] | 0.900 [0.800, 0.986] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.189 [0.166, 0.211] | 0.765 [0.658, 0.867] | 0.762 [0.654, 0.864] | 0.789 [0.682, 0.887] | 0.821 [0.739, 0.899] | 0.886 [0.771, 0.971] | 1.000 [1.000, 1.000] |
| bm25_only | 35 | 0.429 [0.257, 0.600] | 0.629 [0.457, 0.771] | 0.771 [0.629, 0.886] | 0.886 [0.771, 0.971] | 0.757 [0.614, 0.886] | 0.871 [0.757, 0.971] | 0.971 [0.914, 1.000] | 0.160 [0.131, 0.189] | 0.570 [0.442, 0.699] | 0.570 [0.443, 0.699] | 0.600 [0.468, 0.728] | 0.639 [0.521, 0.753] | 0.743 [0.600, 0.886] | 0.857 [0.743, 0.971] |
| rrf | 35 | 0.486 [0.314, 0.657] | 0.800 [0.657, 0.914] | 0.800 [0.657, 0.914] | 0.829 [0.686, 0.943] | 0.800 [0.657, 0.914] | 0.829 [0.686, 0.943] | 0.857 [0.743, 0.971] | 0.171 [0.137, 0.200] | 0.633 [0.505, 0.761] | 0.627 [0.499, 0.752] | 0.670 [0.544, 0.792] | 0.680 [0.558, 0.798] | 0.800 [0.657, 0.914] | 0.829 [0.686, 0.943] |
| rrf_bge | 35 | 0.543 [0.371, 0.714] | 0.771 [0.629, 0.914] | 0.829 [0.686, 0.943] | 0.829 [0.686, 0.943] | 0.829 [0.686, 0.943] | 0.829 [0.686, 0.943] | 0.857 [0.743, 0.971] | 0.177 [0.149, 0.206] | 0.666 [0.532, 0.795] | 0.657 [0.526, 0.784] | 0.702 [0.576, 0.822] | 0.702 [0.576, 0.822] | 0.829 [0.686, 0.943] | 0.829 [0.686, 0.943] |


### fiqa — `fiqa|q20|d5000`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0.850 [0.650, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.894 [0.777, 0.977] | 0.933 [0.817, 1.000] | 0.992 [0.975, 1.000] | 0.500 [0.370, 0.630] | 0.904 [0.787, 1.000] | 0.883 [0.768, 0.971] | 0.899 [0.779, 0.988] | 0.902 [0.785, 0.983] | 0.750 [0.550, 0.900] | 0.900 [0.750, 1.000] |
| bm25_only | 20 | 0.350 [0.150, 0.550] | 0.500 [0.300, 0.700] | 0.500 [0.300, 0.700] | 0.600 [0.400, 0.800] | 0.280 [0.142, 0.433] | 0.448 [0.279, 0.617] | 0.771 [0.637, 0.892] | 0.160 [0.080, 0.240] | 0.458 [0.271, 0.650] | 0.302 [0.176, 0.444] | 0.307 [0.161, 0.464] | 0.365 [0.222, 0.513] | 0.100 [0.000, 0.250] | 0.200 [0.050, 0.400] |
| rrf | 20 | 0.650 [0.450, 0.850] | 0.850 [0.700, 1.000] | 0.900 [0.750, 1.000] | 0.900 [0.750, 1.000] | 0.677 [0.540, 0.805] | 0.787 [0.658, 0.904] | 0.992 [0.975, 1.000] | 0.360 [0.250, 0.480] | 0.761 [0.601, 0.904] | 0.632 [0.494, 0.763] | 0.665 [0.525, 0.796] | 0.700 [0.563, 0.825] | 0.350 [0.150, 0.550] | 0.550 [0.350, 0.750] |
| rrf_bge | 20 | 0.700 [0.500, 0.900] | 0.900 [0.750, 1.000] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 0.698 [0.561, 0.823] | 0.796 [0.658, 0.912] | 0.992 [0.975, 1.000] | 0.380 [0.270, 0.510] | 0.810 [0.664, 0.933] | 0.703 [0.561, 0.832] | 0.720 [0.573, 0.852] | 0.750 [0.609, 0.873] | 0.350 [0.150, 0.550] | 0.600 [0.400, 0.800] |


### hotpotqa — `hotpotqa|q20|d5000`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.950 [0.875, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.380 [0.350, 0.400] | 1.000 [1.000, 1.000] | 0.967 [0.917, 1.000] | 0.961 [0.903, 1.000] | 0.983 [0.958, 1.000] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 20 | 0.850 [0.700, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.750 [0.650, 0.850] | 0.800 [0.700, 0.900] | 0.900 [0.800, 0.975] | 0.300 [0.260, 0.340] | 0.925 [0.850, 1.000] | 0.729 [0.618, 0.839] | 0.769 [0.673, 0.865] | 0.789 [0.701, 0.877] | 0.500 [0.300, 0.700] | 0.600 [0.400, 0.800] |
| rrf | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.875 [0.775, 0.950] | 0.900 [0.800, 0.975] | 1.000 [1.000, 1.000] | 0.350 [0.310, 0.380] | 1.000 [1.000, 1.000] | 0.867 [0.788, 0.940] | 0.888 [0.811, 0.954] | 0.899 [0.829, 0.960] | 0.750 [0.550, 0.900] | 0.800 [0.600, 0.950] |
| rrf_bge | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.968 [0.930, 1.000] | 0.984 [0.966, 1.000] | 0.984 [0.966, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |


### multihoprag — `multihoprag|q20|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 18 | 0.722 [0.500, 0.889] | 0.944 [0.833, 1.000] | 0.944 [0.833, 1.000] | 1.000 [1.000, 1.000] | 0.722 [0.593, 0.843] | 0.912 [0.838, 0.972] | 1.000 [1.000, 1.000] | 0.400 [0.333, 0.467] | 0.833 [0.704, 0.944] | 0.665 [0.571, 0.756] | 0.683 [0.566, 0.787] | 0.774 [0.697, 0.846] | 0.389 [0.167, 0.611] | 0.722 [0.500, 0.889] |
| bm25_only | 18 | 0.389 [0.167, 0.611] | 0.556 [0.333, 0.778] | 0.667 [0.444, 0.889] | 0.778 [0.556, 0.944] | 0.324 [0.199, 0.458] | 0.444 [0.306, 0.588] | 0.713 [0.551, 0.861] | 0.167 [0.100, 0.244] | 0.517 [0.331, 0.710] | 0.305 [0.193, 0.432] | 0.330 [0.199, 0.472] | 0.385 [0.254, 0.521] | 0.056 [0.000, 0.167] | 0.111 [0.000, 0.278] |
| rrf | 18 | 0.556 [0.333, 0.778] | 0.778 [0.556, 0.944] | 0.833 [0.667, 1.000] | 0.944 [0.833, 1.000] | 0.468 [0.324, 0.620] | 0.718 [0.560, 0.866] | 1.000 [1.000, 1.000] | 0.256 [0.178, 0.333] | 0.678 [0.502, 0.847] | 0.454 [0.350, 0.556] | 0.452 [0.317, 0.587] | 0.562 [0.432, 0.681] | 0.167 [0.000, 0.333] | 0.500 [0.278, 0.722] |
| rrf_bge | 18 | 0.667 [0.444, 0.889] | 0.722 [0.500, 0.889] | 0.833 [0.667, 1.000] | 0.944 [0.833, 1.000] | 0.593 [0.431, 0.745] | 0.824 [0.690, 0.935] | 1.000 [1.000, 1.000] | 0.322 [0.233, 0.411] | 0.740 [0.560, 0.905] | 0.549 [0.433, 0.667] | 0.554 [0.398, 0.700] | 0.661 [0.537, 0.773] | 0.278 [0.111, 0.500] | 0.611 [0.389, 0.833] |


### ragbench-covidqa — `ragbench-covidqa|q20|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0.850 [0.700, 1.000] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.470 [0.390, 0.550] | 0.912 [0.812, 1.000] | 0.883 [0.780, 0.967] | 0.924 [0.848, 0.983] | 0.924 [0.848, 0.983] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 20 | 0.750 [0.550, 0.950] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] | 0.833 [0.700, 0.950] | 0.908 [0.817, 0.983] | 0.967 [0.917, 1.000] | 0.400 [0.300, 0.500] | 0.835 [0.698, 0.956] | 0.742 [0.606, 0.868] | 0.780 [0.645, 0.898] | 0.805 [0.690, 0.906] | 0.700 [0.500, 0.900] | 0.800 [0.600, 0.950] |
| rrf | 20 | 0.800 [0.600, 0.950] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.883 [0.767, 0.967] | 0.917 [0.817, 1.000] | 0.933 [0.833, 1.000] | 0.400 [0.320, 0.490] | 0.892 [0.792, 0.975] | 0.792 [0.675, 0.900] | 0.831 [0.727, 0.925] | 0.847 [0.758, 0.931] | 0.800 [0.600, 0.950] | 0.850 [0.700, 1.000] |
| rrf_bge | 20 | 0.800 [0.600, 0.950] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.883 [0.767, 0.983] | 0.917 [0.817, 1.000] | 0.933 [0.833, 1.000] | 0.400 [0.310, 0.490] | 0.892 [0.792, 0.975] | 0.814 [0.687, 0.925] | 0.844 [0.728, 0.943] | 0.861 [0.759, 0.947] | 0.800 [0.600, 0.950] | 0.850 [0.700, 1.000] |


### scifact — `scifact|q20|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0.700 [0.500, 0.900] | 0.850 [0.700, 1.000] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.190 [0.160, 0.220] | 0.788 [0.635, 0.925] | 0.788 [0.635, 0.925] | 0.807 [0.653, 0.938] | 0.838 [0.720, 0.945] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 20 | 0.400 [0.200, 0.600] | 0.450 [0.250, 0.650] | 0.550 [0.350, 0.750] | 0.550 [0.350, 0.750] | 0.550 [0.350, 0.750] | 0.550 [0.350, 0.750] | 0.925 [0.800, 1.000] | 0.110 [0.070, 0.150] | 0.460 [0.263, 0.660] | 0.460 [0.262, 0.659] | 0.470 [0.269, 0.670] | 0.470 [0.269, 0.670] | 0.550 [0.350, 0.750] | 0.550 [0.350, 0.750] |
| rrf | 20 | 0.550 [0.350, 0.750] | 0.700 [0.500, 0.900] | 0.850 [0.700, 1.000] | 0.900 [0.750, 1.000] | 0.825 [0.650, 0.950] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] | 0.170 [0.140, 0.200] | 0.662 [0.489, 0.828] | 0.660 [0.487, 0.827] | 0.689 [0.520, 0.848] | 0.714 [0.558, 0.860] | 0.800 [0.600, 0.950] | 0.900 [0.750, 1.000] |
| rrf_bge | 20 | 0.550 [0.350, 0.750] | 0.700 [0.500, 0.900] | 0.800 [0.600, 0.950] | 0.900 [0.750, 1.000] | 0.775 [0.575, 0.950] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] | 0.160 [0.120, 0.190] | 0.670 [0.499, 0.834] | 0.652 [0.484, 0.811] | 0.666 [0.489, 0.828] | 0.724 [0.578, 0.857] | 0.750 [0.550, 0.900] | 0.950 [0.850, 1.000] |


See `ablation.md` for per-dataset tables with the best value per column in bold.

## 5. Generation quality (RAGAS, judged on the evidence the generator actually saw)

RAGAS `retrieved_contexts` = the slice of each retrieved parent that fits the prompt's 12,000-character evidence budget (`table_visibility.csv`). Reference-based metrics inherit the quality of the references (RAGBench references are model-generated; MultiHop-RAG references are 1–5 word answers).


### docsqa — `docsqa|qall|dfull`

| config | answerable | judged | Faithfulness | ResponseRelevancy | CtxPrecision(ref) | CtxRecall | FactualCorr F1 |
|---|---|---|---|---|---|---|---|
| dense_only | 35 | 0 |  |  |  |  |  |
| bm25_only | 35 | 0 |  |  |  |  |  |
| rrf | 35 | 0 |  |  |  |  |  |
| rrf_bge | 35 | 35 | 0.957 [0.886, 1.000] | 0.901 [0.836, 0.946] | 0.859 [0.766, 0.937] | 0.914 [0.800, 1.000] | 0.189 [0.096, 0.291] |
| rrf_bge_two_pass | 35 | 35 | 0.967 [0.905, 1.000] | 0.870 [0.785, 0.935] | 0.859 [0.765, 0.935] | 0.914 [0.800, 1.000] | 0.242 [0.137, 0.353] |
| engine_as_shipped | 35 | 0 |  |  |  |  |  |


### multihoprag — `multihoprag|q20|dfull`

| config | answerable | judged | EM | token F1 | contains-answer | Faithfulness | ResponseRelevancy | CtxPrecision(ref) | CtxRecall | FactualCorr F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 18 | 0 | 0.000 [0.000, 0.000] | 0.030 [0.016, 0.047] | 0.667 [0.444, 0.889] |  |  |  |  |  |
| bm25_only | 18 | 0 | 0.000 [0.000, 0.000] | 0.010 [0.003, 0.019] | 0.333 [0.111, 0.556] |  |  |  |  |  |
| rrf | 18 | 0 | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.038] | 0.611 [0.389, 0.833] |  |  |  |  |  |
| rrf_bge | 18 | 18 | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.041] | 0.611 [0.389, 0.833] | 0.792 [0.630, 0.926] | 0.468 [0.274, 0.664] | 0.546 [0.324, 0.769] | 0.722 [0.500, 0.889] | 0.272 [0.118, 0.439] |
| rrf_bge_two_pass | 18 | 18 | 0.000 [0.000, 0.000] | 0.015 [0.007, 0.023] | 0.500 [0.278, 0.722] | 0.763 [0.586, 0.914] | 0.419 [0.228, 0.610] | 0.574 [0.361, 0.787] | 0.667 [0.444, 0.889] | 0.039 [0.000, 0.118] |
| engine_as_shipped | 18 | 0 | 0.000 [0.000, 0.000] | 0.015 [0.007, 0.023] | 0.500 [0.278, 0.722] |  |  |  |  |  |


### ragbench-covidqa — `ragbench-covidqa|q20|dfull`

| config | answerable | judged | Faithfulness | ResponseRelevancy | CtxPrecision(ref) | CtxRecall | FactualCorr F1 |
|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0 |  |  |  |  |  |
| bm25_only | 20 | 0 |  |  |  |  |  |
| rrf | 20 | 0 |  |  |  |  |  |
| rrf_bge | 20 | 20 | 0.940 [0.830, 1.000] | 0.699 [0.510, 0.864] | 0.668 [0.503, 0.821] | 0.650 [0.450, 0.850] | 0.315 [0.161, 0.487] |
| rrf_bge_two_pass | 20 | 20 | 0.954 [0.904, 0.994] | 0.612 [0.416, 0.803] | 0.668 [0.506, 0.821] | 0.650 [0.450, 0.850] | 0.250 [0.109, 0.408] |
| engine_as_shipped | 20 | 0 |  |  |  |  |  |


## 6. Robustness

Abstention = the system's refusal sentence, an empty output, or a short answer built on a refusal phrase (heuristic, `eval/metrics.py`).

| dataset | config | unanswerable n | abstention on unanswerable ↑ | answerable n | false abstention ↓ | false abst. | gold in context | anachronism gate fired |
|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 9 | 1.000 [1.000, 1.000] | 35 | 0.029 [0.000, 0.086] | 0.000 | – |
| docsqa | bm25_only | 9 | 1.000 [1.000, 1.000] | 35 | 0.143 [0.029, 0.257] | 0.034 | – |
| docsqa | rrf | 9 | 0.889 [0.667, 1.000] | 35 | 0.057 [0.000, 0.143] | 0.000 | – |
| docsqa | rrf_bge | 9 | 1.000 [1.000, 1.000] | 35 | 0.029 [0.000, 0.086] | 0.000 | – |
| docsqa | rrf_bge_two_pass | 9 | 1.000 [1.000, 1.000] | 35 | 0.029 [0.000, 0.086] | 0.000 | – |
| docsqa | engine_as_shipped | 9 | 0.889 [0.667, 1.000] | 35 | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| multihoprag | dense_only | 2 | 1.000 [1.000, 1.000] | 18 | 0.444 [0.222, 0.667] | 0.444 | – |
| multihoprag | bm25_only | 2 | 1.000 [1.000, 1.000] | 18 | 0.667 [0.444, 0.889] | 0.583 | – |
| multihoprag | rrf | 2 | 1.000 [1.000, 1.000] | 18 | 0.500 [0.278, 0.722] | 0.400 | – |
| multihoprag | rrf_bge | 2 | 1.000 [1.000, 1.000] | 18 | 0.500 [0.278, 0.722] | 0.471 | – |
| multihoprag | rrf_bge_two_pass | 2 | 1.000 [1.000, 1.000] | 18 | 0.278 [0.111, 0.500] | 0.235 | – |
| multihoprag | engine_as_shipped | 2 | 1.000 [1.000, 1.000] | 18 | 0.222 [0.056, 0.444] | 0.222 | 0.000 |
| ragbench-covidqa | dense_only | 0 | – | 20 | 0.250 [0.100, 0.450] | 0.250 | – |
| ragbench-covidqa | bm25_only | 0 | – | 20 | 0.300 [0.100, 0.500] | 0.263 | – |
| ragbench-covidqa | rrf | 0 | – | 20 | 0.300 [0.100, 0.500] | 0.300 | – |
| ragbench-covidqa | rrf_bge | 0 | – | 20 | 0.250 [0.050, 0.450] | 0.250 | – |
| ragbench-covidqa | rrf_bge_two_pass | 0 | – | 20 | 0.200 [0.050, 0.400] | 0.200 | – |
| ragbench-covidqa | engine_as_shipped | 0 | – | 20 | 0.300 [0.100, 0.500] | 0.300 | 0.000 |


### Evidence visibility (12,000-character prompt budget)

| dataset | config | n | evidence_chars_total_mean | evidence_chars_seen_mean | fraction_of_evidence_seen | gold_in_context_rate | gold_in_seen_context_rate | gold_docs_seen_fraction |
|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 44 | 6617.25 | 6477.136 | 0.979 | 0.914 | 0.914 | 0.9 |
| docsqa | bm25_only | 44 | 8041.023 | 7777.773 | 0.967 | 0.829 | 0.829 | 0.814 |
| docsqa | rrf | 44 | 7658.818 | 7464.045 | 0.975 | 0.8 | 0.8 | 0.8 |
| docsqa | rrf_bge | 44 | 7644.795 | 7418.955 | 0.97 | 0.829 | 0.829 | 0.829 |
| docsqa | rrf_bge_two_pass | 44 | 7644.795 | 7533.795 | 0.985 | 0.829 | 0.829 | 0.829 |
| docsqa | engine_as_shipped | 44 | 8515.114 | 8196.295 | 0.963 | 0.914 | 0.914 | 0.914 |
| multihoprag | dense_only | 20 | 45097.85 | 11749.15 | 0.261 | 1.0 | 0.889 | 0.417 |
| multihoprag | bm25_only | 20 | 46287.4 | 11763.55 | 0.254 | 0.667 | 0.556 | 0.264 |
| multihoprag | rrf | 20 | 46087.0 | 11771.25 | 0.255 | 0.833 | 0.667 | 0.282 |
| multihoprag | rrf_bge | 20 | 46422.2 | 11760.2 | 0.253 | 0.944 | 0.722 | 0.31 |
| multihoprag | rrf_bge_two_pass | 20 | 46422.2 | 11989.1 | 0.258 | 0.944 | 0.722 | 0.31 |
| multihoprag | engine_as_shipped | 20 | 46831.0 | 11989.7 | 0.256 | 1.0 | 0.722 | 0.431 |
| ragbench-covidqa | dense_only | 20 | 3220.25 | 3220.25 | 1.0 | 1.0 | 1.0 | 1.0 |
| ragbench-covidqa | bm25_only | 20 | 3714.6 | 3714.6 | 1.0 | 0.95 | 0.95 | 0.833 |
| ragbench-covidqa | rrf | 20 | 3558.3 | 3558.3 | 1.0 | 1.0 | 1.0 | 0.9 |
| ragbench-covidqa | rrf_bge | 20 | 3562.65 | 3562.65 | 1.0 | 1.0 | 1.0 | 0.9 |
| ragbench-covidqa | rrf_bge_two_pass | 20 | 3562.65 | 3562.65 | 1.0 | 1.0 | 1.0 | 0.9 |
| ragbench-covidqa | engine_as_shipped | 20 | 4181.0 | 4181.0 | 1.0 | 1.0 | 1.0 | 0.892 |


### RGB (generation only; 5 passages per query)

| kind | mode | noise_ratio | n | mentions_true | mentions_fake | abstained | accuracy | abstention_rate | rejection_rate |
|---|---|---|---|---|---|---|---|---|---|
| en_fact | one_pass | 0.0 | 20 | 0.0 | 0.9 | 0.0 |  |  |  |
| en_fact | two_pass | 0.0 | 20 | 0.0 | 0.85 | 0.05 |  |  |  |
| en_int | one_pass | 0.0 | 20 |  |  |  | 0.7 | 0.05 |  |
| en_int | two_pass | 0.0 | 20 |  |  |  | 0.65 | 0.05 |  |
| en_refine | one_pass | 0.0 | 20 |  |  |  | 1.0 | 0.0 |  |
| en_refine | one_pass | 0.2 | 20 |  |  |  | 0.95 | 0.0 |  |
| en_refine | one_pass | 0.4 | 20 |  |  |  | 1.0 | 0.0 |  |
| en_refine | one_pass | 0.6 | 20 |  |  |  | 1.0 | 0.0 |  |
| en_refine | one_pass | 0.8 | 20 |  |  |  | 0.9 | 0.05 |  |
| en_refine | one_pass | 1.0 | 20 |  |  |  |  | 0.75 | 0.75 |
| en_refine | two_pass | 0.0 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.2 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.4 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.6 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.8 | 20 |  |  |  | 0.85 | 0.1 |  |
| en_refine | two_pass | 1.0 | 20 |  |  |  |  | 0.55 | 0.55 |


## 7. System: tokens, cost, latency

Tokens are from the main run; $ uses list prices in `eval/config.py` (Gemini API standard tier; Vertex billing may differ). The hosted reranker's price is not published in the repo, so its tokens are reported but excluded from $.

| dataset | config | n_generated | live_n | total_p50_s | total_p95_s | retrieval_p50_s | retrieval_p95_s | rerank_p50_s | rerank_p95_s | generation_p50_s | generation_p95_s | gen_in_tokens_mean | gen_out_tokens_mean | rerank_tokens_mean | generator_cost_per_1k_queries_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 44 | 6.0 | 4.49 | 14.79 | 0.41 | 1.48 | 0.0 | 0.0 | 4.07 | 13.9 | 2274.2 | 57.09 | 0.0 | 1.92 |
| docsqa | bm25_only | 44 | 6.0 | 1.64 | 2.07 | 0.0 | 0.0 | 0.0 | 0.0 | 1.61 | 2.06 | 2533.95 | 67.91 | 0.0 | 2.16 |
| docsqa | rrf | 44 | 6.0 | 2.67 | 3.99 | 0.43 | 0.67 | 0.0 | 0.0 | 2.23 | 3.33 | 2510.39 | 59.07 | 0.0 | 2.1 |
| docsqa | rrf_bge | 44 | 6.0 | 6.85 | 16.2 | 0.43 | 0.5 | 2.87 | 5.42 | 2.67 | 12.33 | 2515.0 | 57.61 | 21955.16 | 2.1 |
| docsqa | rrf_bge_two_pass | 44 | 6.0 | 9.99 | 14.15 | 0.45 | 0.59 | 2.7 | 5.77 | 4.84 | 10.93 | 2754.95 | 128.93 | 21955.11 | 2.55 |
| docsqa | engine_as_shipped | 44 | 6.0 | 14.49 | 30.74 | 5.74 | 9.61 | 2.68 | 2.84 | 7.28 | 21.42 | 3094.91 | 133.14 | 21806.66 | 3.13 |
| multihoprag | dense_only | 20 | 5.0 | 2.87 | 6.4 | 0.51 | 1.53 | 0.0 | 0.0 | 2.39 | 5.15 | 2964.15 | 101.25 | 0.0 | 2.6 |
| multihoprag | bm25_only | 20 | 5.0 | 2.43 | 27.35 | 0.0 | 0.0 | 0.0 | 0.0 | 2.35 | 27.28 | 2937.2 | 66.15 | 0.0 | 2.45 |
| multihoprag | rrf | 20 | 5.0 | 3.37 | 25.14 | 0.48 | 0.55 | 0.0 | 0.0 | 2.67 | 24.59 | 2955.2 | 85.35 | 0.0 | 2.54 |
| multihoprag | rrf_bge | 20 | 5.0 | 6.64 | 7.94 | 0.52 | 0.69 | 3.2 | 3.35 | 2.86 | 4.11 | 2943.85 | 86.7 | 31531.4 | 2.53 |
| multihoprag | rrf_bge_two_pass | 20 | 5.0 | 9.5 | 13.0 | 0.45 | 0.51 | 3.47 | 4.25 | 5.13 | 8.99 | 3958.3 | 279.45 | 31531.4 | 4.02 |
| multihoprag | engine_as_shipped | 20 | 5.0 | 10.05 | 10.86 | 3.49 | 4.32 | 3.24 | 3.26 | 4.38 | 5.64 | 3822.0 | 277.2 | 31692.45 | 4.27 |
| ragbench-covidqa | dense_only | 20 |  |  |  |  |  |  |  |  |  | 1050.65 | 116.0 | 0.0 | 1.22 |
| ragbench-covidqa | bm25_only | 20 |  |  |  |  |  |  |  |  |  | 1115.55 | 107.35 | 0.0 | 1.24 |
| ragbench-covidqa | rrf | 20 |  |  |  |  |  |  |  |  |  | 1104.75 | 110.7 | 0.0 | 1.24 |
| ragbench-covidqa | rrf_bge | 20 |  |  |  |  |  |  |  |  |  | 1106.1 | 100.9 | 11054.5 | 1.21 |
| ragbench-covidqa | rrf_bge_two_pass | 20 |  |  |  |  |  |  |  |  |  | 1100.15 | 237.65 | 11054.5 | 1.72 |
| ragbench-covidqa | engine_as_shipped | 20 |  |  |  |  |  |  |  |  |  | 1236.1 | 205.4 | 11057.0 | 2.0 |


## 8. Statistics

Paired tests on identical queries (baseline → variant; positive = variant better, except abstention rows): paired bootstrap (10k, seed 0) and Wilcoxon signed-rank; effect size = Cohen's d_z.

| dataset | comparison | metric | n | mean_baseline | mean_variant | mean_diff | ci_lo | ci_hi | p_bootstrap | p_wilcoxon | cohens_dz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | rrf vs rrf+bge | hit@5 | 20 | 0.85 | 0.8 | -0.05 | -0.15 | 0.0 | 0.7214 | 0.3173 | -0.2236 |
| scifact | rrf vs rrf+bge | mrr | 20 | 0.6625 | 0.6703 | 0.0078 | -0.0765 | 0.1025 | 0.8828 | 0.9441 | 0.0376 |
| scifact | rrf vs rrf+bge | ndcg@10 | 20 | 0.7141 | 0.7245 | 0.0104 | -0.0522 | 0.0717 | 0.7362 | 0.9325 | 0.0709 |
| scifact | rrf vs rrf+bge | recall@100 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| scifact | rrf vs rrf+bge | map | 20 | 0.6604 | 0.6515 | -0.0089 | -0.0807 | 0.0578 | 0.8456 | 0.9441 | -0.0553 |
| fiqa | rrf vs rrf+bge | hit@5 | 20 | 0.9 | 0.9 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| fiqa | rrf vs rrf+bge | mrr | 20 | 0.7606 | 0.8098 | 0.0492 | -0.05 | 0.1617 | 0.3698 | 0.2733 | 0.1946 |
| fiqa | rrf vs rrf+bge | ndcg@10 | 20 | 0.7002 | 0.7496 | 0.0493 | -0.0232 | 0.1297 | 0.1986 | 0.0929 | 0.2729 |
| fiqa | rrf vs rrf+bge | recall@100 | 20 | 0.9917 | 0.9917 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| fiqa | rrf vs rrf+bge | map | 20 | 0.6315 | 0.7031 | 0.0716 | -0.0275 | 0.1792 | 0.1676 | 0.0505 | 0.2923 |
| hotpotqa | rrf vs rrf+bge | hit@5 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| hotpotqa | rrf vs rrf+bge | mrr | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| hotpotqa | rrf vs rrf+bge | ndcg@10 | 20 | 0.8987 | 0.9845 | 0.0858 | 0.0227 | 0.1597 | 0.0034 | 0.0278 | 0.5358 |
| hotpotqa | rrf vs rrf+bge | recall@100 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| hotpotqa | rrf vs rrf+bge | map | 20 | 0.8673 | 0.9683 | 0.101 | 0.0264 | 0.1836 | 0.0062 | 0.0298 | 0.5443 |
| multihoprag | rrf vs rrf+bge | hit@5 | 18 | 0.8333 | 0.8333 | 0.0 | -0.1667 | 0.1667 | 1.0000 | 1.0000 | 0.0 |
| multihoprag | rrf vs rrf+bge | mrr | 18 | 0.6783 | 0.7398 | 0.0615 | -0.0357 | 0.1852 | 0.2834 | 0.3980 | 0.2443 |
| multihoprag | rrf vs rrf+bge | ndcg@10 | 18 | 0.5616 | 0.6611 | 0.0996 | 0.032 | 0.1771 | 0.0022 | 0.0151 | 0.6162 |
| multihoprag | rrf vs rrf+bge | recall@100 | 18 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| multihoprag | rrf vs rrf+bge | map | 18 | 0.4539 | 0.5493 | 0.0955 | 0.0336 | 0.161 | 0.0016 | 0.0168 | 0.6716 |
| ragbench-covidqa | rrf vs rrf+bge | hit@5 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | rrf vs rrf+bge | mrr | 20 | 0.8917 | 0.8917 | 0.0 | -0.075 | 0.075 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | rrf vs rrf+bge | ndcg@10 | 20 | 0.8472 | 0.8605 | 0.0133 | -0.0235 | 0.061 | 0.5702 | 0.8335 | 0.1337 |
| ragbench-covidqa | rrf vs rrf+bge | recall@100 | 20 | 0.9333 | 0.9333 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | rrf vs rrf+bge | map | 20 | 0.7915 | 0.8138 | 0.0222 | -0.0222 | 0.0833 | 0.4514 | 0.5282 | 0.1732 |
| docsqa | rrf vs rrf+bge | hit@5 | 35 | 0.8 | 0.8286 | 0.0286 | 0.0 | 0.0857 | 0.7166 | 0.3173 | 0.169 |
| docsqa | rrf vs rrf+bge | mrr | 35 | 0.6332 | 0.6658 | 0.0326 | -0.0356 | 0.1016 | 0.3412 | 0.3591 | 0.1551 |
| docsqa | rrf vs rrf+bge | ndcg@10 | 35 | 0.6798 | 0.7024 | 0.0226 | -0.0256 | 0.0708 | 0.3472 | 0.3265 | 0.1526 |
| docsqa | rrf vs rrf+bge | recall@100 | 35 | 0.8571 | 0.8571 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | rrf vs rrf+bge | map | 35 | 0.6271 | 0.6573 | 0.0302 | -0.0341 | 0.0945 | 0.3450 | 0.3265 | 0.1518 |
| multihoprag | one-pass vs two-pass | faithfulness | 18 | 0.7923 | 0.7626 | -0.0297 | -0.172 | 0.1095 | 0.6944 | 0.6831 | -0.0937 |
| multihoprag | one-pass vs two-pass | answer_relevancy | 18 | 0.4678 | 0.4193 | -0.0485 | -0.2203 | 0.1041 | 0.5844 | 0.9292 | -0.1307 |
| multihoprag | one-pass vs two-pass | context_precision | 18 | 0.5463 | 0.5741 | 0.0278 | 0.0 | 0.0833 | 0.7090 | 0.3173 | 0.2357 |
| multihoprag | one-pass vs two-pass | context_recall | 18 | 0.7222 | 0.6667 | -0.0556 | -0.1667 | 0.0 | 0.7232 | 0.3173 | -0.2357 |
| multihoprag | one-pass vs two-pass | factual_correctness_f1 | 18 | 0.2722 | 0.0394 | -0.2328 | -0.4089 | -0.0689 | 0.0024 | 0.0277 | -0.608 |
| multihoprag | one-pass vs two-pass | false_abstention | 18 | 0.5 | 0.2778 | -0.2222 | -0.4444 | -0.0556 | 0.0208 | 0.0455 | -0.5195 |
| ragbench-covidqa | one-pass vs two-pass | faithfulness | 20 | 0.94 | 0.9542 | 0.0142 | -0.0625 | 0.1225 | 0.8708 | 0.7150 | 0.0649 |
| ragbench-covidqa | one-pass vs two-pass | answer_relevancy | 20 | 0.6988 | 0.6115 | -0.0873 | -0.2752 | 0.0999 | 0.4128 | 0.7174 | -0.2057 |
| ragbench-covidqa | one-pass vs two-pass | context_precision | 20 | 0.6681 | 0.6681 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | one-pass vs two-pass | context_recall | 20 | 0.65 | 0.65 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | one-pass vs two-pass | factual_correctness_f1 | 20 | 0.315 | 0.25 | -0.065 | -0.1815 | 0.037 | 0.2366 | 0.1731 | -0.2547 |
| ragbench-covidqa | one-pass vs two-pass | false_abstention | 20 | 0.25 | 0.2 | -0.05 | -0.15 | 0.0 | 0.7196 | 0.3173 | -0.2236 |
| docsqa | one-pass vs two-pass | faithfulness | 35 | 0.9571 | 0.9667 | 0.0095 | -0.0143 | 0.0429 | 0.7184 | 0.6547 | 0.106 |
| docsqa | one-pass vs two-pass | answer_relevancy | 35 | 0.9015 | 0.8703 | -0.0312 | -0.0999 | 0.0121 | 0.3034 | 0.9455 | -0.1741 |
| docsqa | one-pass vs two-pass | context_precision | 35 | 0.8589 | 0.8589 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | one-pass vs two-pass | context_recall | 35 | 0.9143 | 0.9143 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | one-pass vs two-pass | factual_correctness_f1 | 35 | 0.1886 | 0.2417 | 0.0531 | 0.0029 | 0.1114 | 0.0360 | 0.0926 | 0.3175 |
| docsqa | one-pass vs two-pass | false_abstention | 35 | 0.0286 | 0.0286 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | one-pass vs two-pass | abstention_on_unanswerable | 9 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |


No multiple-comparison correction is applied; with this many comparisons treat p-values as descriptive.

## 8b. Judge cross-check by Claude (model labels, not human labels)

Claude (an Anthropic model) independently labelled 50 sampled answers (read in full against the evidence the generator saw; labels in `claude_labels.jsonl`, merged into `judge_audit.csv`). `human_label` is still blank.

| metric (Claude label vs Gemini judge score) | n | Claude mean | judge mean | mean |diff| | Pearson r | Spearman ρ | binary agreement @0.5 | judge stricter (>0.25) | judge more lenient (>0.25) |
|---|---|---|---|---|---|---|---|---|---|
| faithfulness | 50 | 0.973 | 0.928 | 0.045 | 0.93 | 0.645 | 0.96 | 0.08 | 0.0 |
| correctness vs RAGAS FactualCorrectness F1 | 50 | 0.88 | 0.268 | 0.619 | 0.09 | 0.009 | 0.38 | 0.7 | 0.0 |



## 9. Limitations, leakage and comparability (read before citing)

- **Judge validity.** The primary judge (`gemini-3.7-flash`) is a different model from the generator (`gemini-3.8-flash`) but from the same vendor/family, so family and self-preference bias are possible. An independent cross-check by Claude (Anthropic) on a sample is reported in `claude_audit_summary.json` / `judge_audit.csv` (`claude_label` column) when present. These are **model labels, not human labels**: the `human_label` column is intentionally blank and no human agreement figure is claimed.
- **Data leakage.** BEIR (SciFact/FiQA/HotpotQA), HotpotQA's Wikipedia text, MultiHop-RAG news articles (2023) and RGB are public and may be in the training data of both the generator and the judge; the embedding model and reranker may also have seen them. The documentation-QA set is derived from this repository's own docs and cannot be leaked into any model trained before it existed, but it was authored by an AI assistant (quotes machine-verified; answers not human-reviewed).
- **Comparability with published numbers.** Corpus subsampling (HotpotQA, FiQA/others where capped) makes the task easier than the BEIR leaderboard task; BM25 here is SQLite FTS5 (no stemming/stopwords) not Lucene/Pyserini; chunking splits long documents and scores are max-chunk-per-document; query counts differ from published protocols. Do **not** put these next to leaderboard numbers without these qualifiers.
- **RAGBench** questions are written against their own source passages; pooling passages from many questions makes some questions under-specified, which inflates false abstention and depresses retrieval metrics. References are LLM-generated.
- **MultiHop-RAG / long documents.** Both generators cap the evidence sent to the model at 12,000 characters, so on long articles most retrieved context never reaches the LLM (see §6). This is a property of the shipped system and is measured, not tuned.
- **Hard-coded anachronism gate** (`quality_gates.is_anachronism`) is specific to the GoT corpus; it is only exercised by the `engine_as_shipped` rows (fire rate in §6).
- **Small samples.** Smoke runs (n=20 per dataset) are pipeline checks, not evidence of differences between configs.
- The SLM contextual-summary ingestion variant (Qwen3-0.6B) was **not run** (not on the base branch; deferred by request). No numbers are reported for it.

## 10. Reproduce

```bash
uv venv --python 3.12 .venv && uv sync --extra dev
uv pip install --python .venv/bin/python -r eval/requirements.txt
gcloud auth application-default login   # Vertex AI ADC
./run_all.sh
```

