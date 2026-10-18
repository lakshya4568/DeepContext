# DeepContext public-benchmark report

**SMOKE TEST (n=20 per dataset; not for publication)** — generated from files in `eval/results/smoke`; every number below is read from them.

- Branch `eval/public-benchmarks` @ `c336002` (base `feat/rag-quality-generation`), working tree dirty at report time
- Judge: `gemini-2.5-pro@think128` (temperature 0) — generator: `gemini-2.5-flash` (different model; both are Google models, so family bias is possible)
- Embeddings: `gemini-embedding-2` 768-d via Vertex AI (`global` endpoint); reranker: bge-reranker-v2-m3 (hosted, EcoHash API)

## 1. Setup and versions

Python 3.12.11 on macOS-26.7.2-arm64-arm-64bit (arm64); ragas 0.4.3, beir 2.2.0, datasets 5.1.0, pandas 3.0.6, numpy 2.5.2, scipy 1.18.1, google-genai 2.31.0, langchain-core 0.3.86, langchain-community 0.3.31, huggingface-hub 1.33.0, pyarrow 26.0.0.

Storage backend: SQLite (FTS5 `bm25()` for the lexical channel; in-memory cosine for the dense channel) — not the Postgres/pgvector/ParadeDB deployment described in the docs, so lexical-channel scores are not comparable to a Postgres `ts_rank_cd` run.

Ingestion: *fast* path only (`DocumentParser` + `ParentChildChunker`, child 300–600 tokens, parent 1000–2500, 15% overlap); no Docling, no local summariser, no model loaded on the host. Title (when present) is prepended to the chunk text and passed in the embedding prefix.

### Harness deviations from the stock code (all inside `eval/`, `src/` untouched)

1. **Strict transport** (`eval/strict_clients.py`): Vertex AI (ADC) instead of API keys; no mock/hash embeddings, no Gemini→Groq→NVIDIA→mock failover, no silent heuristic-reranker fallback — a failure raises and the row is retried on the next run. The stock client would otherwise have produced hash embeddings with this environment.
2. `gemini-embedding-2` is called from the `global` endpoint (404 in `us-central1`), one content per request (batching is rejected by the API).
3. The stock Gemini branch of `LLMClient.complete` ignores `enable_thinking=False`; with `gemini-2.5-flash` default thinking the two-pass extractor (`max_tokens=512`) spent ~490 tokens thinking and returned truncated JSON in a spot check. The harness honours the flag (`thinking_budget=0`); set `EVAL_HONOR_ENABLE_THINKING=0` to reproduce the stock behaviour.
4. Per-call caller timeouts (5–15 s) are replaced by 180 s so results do not depend on API load; the whole pipeline (rewriter, classifier, generator) is routed to the generator model by configuration.
5. `SQLiteStore.search_vector` is replaced by a vectorised equivalent (unit-tested against the stock method).
6. One-pass baseline = the product's `PromptAssembler` prompt + one Gemini call, with the (empty) memory lookup stubbed to avoid a no-op embedding call.
7. Cached model calls replay their recorded first-call latency; **latency percentiles come only from the separate live pass (`raw_latency.jsonl`)**.
8. RAGAS `ResponseRelevancy` uses `strictness=1` (a temperature-0 judge returns identical candidates, so the default 3 only triples cost).

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
| docsqa | dense_only | 35 | 0.914 [0.829, 1.000] | 0.821 [0.738, 0.900] | – | – | – | 4.62 |
| docsqa | bm25_only | 35 | 0.914 [0.800, 1.000] | 0.820 [0.717, 0.912] | – | – | – | 1.42 |
| docsqa | rrf | 35 | 0.886 [0.771, 0.971] | 0.827 [0.722, 0.921] | – | – | – | 2.35 |
| docsqa | rrf_bge | 35 | 0.886 [0.771, 0.971] | 0.786 [0.676, 0.885] | 0.893 [0.793, 0.971] | 0.971 [0.914, 1.000] | 0.269 [0.149, 0.396] | 6.89 |
| docsqa | rrf_bge_two_pass | 35 | 0.886 [0.771, 0.971] | 0.786 [0.676, 0.885] | 0.937 [0.851, 1.000] | 0.971 [0.914, 1.000] | 0.278 [0.162, 0.401] | 5.20 |
| fiqa | dense_only | 20 | 0.950 [0.850, 1.000] | 0.902 [0.783, 0.983] | – | – | – | – |
| fiqa | bm25_only | 20 | 0.750 [0.550, 0.900] | 0.465 [0.331, 0.593] | – | – | – | – |
| fiqa | rrf | 20 | 0.950 [0.850, 1.000] | 0.690 [0.574, 0.796] | – | – | – | – |
| fiqa | rrf_bge | 20 | 0.950 [0.850, 1.000] | 0.738 [0.613, 0.848] | – | – | – | – |
| fiqa | rrf_bge_two_pass | 20 | 0.950 [0.850, 1.000] | 0.738 [0.613, 0.848] | – | – | – | – |
| hotpotqa | dense_only | 20 | 1.000 [1.000, 1.000] | 0.983 [0.958, 1.000] | – | – | – | – |
| hotpotqa | bm25_only | 20 | 1.000 [1.000, 1.000] | 0.942 [0.884, 0.989] | – | – | – | – |
| hotpotqa | rrf | 20 | 1.000 [1.000, 1.000] | 0.976 [0.953, 0.996] | – | – | – | – |
| hotpotqa | rrf_bge | 20 | 1.000 [1.000, 1.000] | 0.992 [0.975, 1.000] | – | – | – | – |
| hotpotqa | rrf_bge_two_pass | 20 | 1.000 [1.000, 1.000] | 0.992 [0.975, 1.000] | – | – | – | – |
| multihoprag | dense_only | 18 | 0.944 [0.833, 1.000] | 0.774 [0.698, 0.846] | – | – | – | 5.05 |
| multihoprag | bm25_only | 18 | 1.000 [1.000, 1.000] | 0.665 [0.566, 0.770] | – | – | – | 2.22 |
| multihoprag | rrf | 18 | 1.000 [1.000, 1.000] | 0.752 [0.661, 0.838] | – | – | – | 2.07 |
| multihoprag | rrf_bge | 18 | 1.000 [1.000, 1.000] | 0.768 [0.684, 0.849] | 0.203 [0.056, 0.378] | 0.556 [0.333, 0.778] | 0.072 [0.000, 0.164] | 5.57 |
| multihoprag | rrf_bge_two_pass | 18 | 1.000 [1.000, 1.000] | 0.768 [0.684, 0.849] | 0.569 [0.381, 0.746] | 0.444 [0.222, 0.667] | 0.046 [0.000, 0.120] | 6.56 |
| ragbench-covidqa | dense_only | 20 | 1.000 [1.000, 1.000] | 0.924 [0.847, 0.982] | – | – | – | – |
| ragbench-covidqa | bm25_only | 20 | 0.900 [0.750, 1.000] | 0.770 [0.629, 0.892] | – | – | – | – |
| ragbench-covidqa | rrf | 20 | 1.000 [1.000, 1.000] | 0.805 [0.692, 0.907] | – | – | – | – |
| ragbench-covidqa | rrf_bge | 20 | 1.000 [1.000, 1.000] | 0.829 [0.710, 0.935] | 0.539 [0.328, 0.750] | 0.717 [0.517, 0.900] | 0.351 [0.177, 0.535] | – |
| ragbench-covidqa | rrf_bge_two_pass | 20 | 1.000 [1.000, 1.000] | 0.829 [0.710, 0.935] | 0.576 [0.375, 0.764] | 0.717 [0.517, 0.900] | 0.373 [0.207, 0.549] | – |
| scifact | dense_only | 20 | 0.900 [0.750, 1.000] | 0.838 [0.720, 0.945] | – | – | – | – |
| scifact | bm25_only | 20 | 0.750 [0.550, 0.900] | 0.671 [0.496, 0.840] | – | – | – | – |
| scifact | rrf | 20 | 0.850 [0.700, 1.000] | 0.699 [0.529, 0.852] | – | – | – | – |
| scifact | rrf_bge | 20 | 0.850 [0.700, 1.000] | 0.719 [0.558, 0.862] | – | – | – | – |
| scifact | rrf_bge_two_pass | 20 | 0.850 [0.700, 1.000] | 0.719 [0.558, 0.862] | – | – | – | – |

## 4. Retrieval (document level)


### docsqa — `docsqa|qall|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 35 | 0.629 [0.457, 0.771] | 0.886 [0.771, 0.971] | 0.914 [0.829, 1.000] | 0.943 [0.857, 1.000] | 0.900 [0.800, 0.986] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.189 [0.166, 0.211] | 0.765 [0.657, 0.868] | 0.762 [0.653, 0.866] | 0.789 [0.681, 0.887] | 0.821 [0.738, 0.900] | 0.886 [0.771, 0.971] | 1.000 [1.000, 1.000] |
| bm25_only | 35 | 0.714 [0.571, 0.857] | 0.857 [0.743, 0.971] | 0.914 [0.800, 1.000] | 0.943 [0.857, 1.000] | 0.900 [0.800, 0.986] | 0.943 [0.857, 1.000] | 1.000 [1.000, 1.000] | 0.189 [0.166, 0.211] | 0.790 [0.673, 0.898] | 0.779 [0.662, 0.886] | 0.806 [0.695, 0.904] | 0.820 [0.717, 0.912] | 0.886 [0.771, 0.971] | 0.943 [0.857, 1.000] |
| rrf | 35 | 0.743 [0.600, 0.886] | 0.857 [0.743, 0.971] | 0.886 [0.771, 0.971] | 0.914 [0.829, 1.000] | 0.886 [0.771, 0.971] | 0.914 [0.829, 1.000] | 0.943 [0.857, 1.000] | 0.189 [0.160, 0.217] | 0.809 [0.694, 0.914] | 0.795 [0.681, 0.899] | 0.817 [0.704, 0.917] | 0.827 [0.722, 0.921] | 0.886 [0.771, 0.971] | 0.914 [0.829, 1.000] |
| rrf_bge | 35 | 0.629 [0.457, 0.800] | 0.857 [0.743, 0.971] | 0.886 [0.771, 0.971] | 0.914 [0.800, 1.000] | 0.886 [0.771, 0.971] | 0.914 [0.800, 1.000] | 0.943 [0.857, 1.000] | 0.189 [0.160, 0.217] | 0.750 [0.631, 0.863] | 0.742 [0.624, 0.854] | 0.776 [0.660, 0.879] | 0.786 [0.676, 0.885] | 0.886 [0.771, 0.971] | 0.914 [0.800, 1.000] |


### fiqa — `fiqa|q20|d5000`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0.850 [0.700, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.894 [0.775, 0.977] | 0.933 [0.817, 1.000] | 0.992 [0.975, 1.000] | 0.500 [0.380, 0.630] | 0.904 [0.787, 1.000] | 0.883 [0.770, 0.972] | 0.899 [0.777, 0.987] | 0.902 [0.783, 0.983] | 0.750 [0.550, 0.950] | 0.900 [0.750, 1.000] |
| bm25_only | 20 | 0.450 [0.250, 0.650] | 0.750 [0.550, 0.900] | 0.750 [0.550, 0.900] | 0.750 [0.550, 0.900] | 0.448 [0.293, 0.606] | 0.529 [0.366, 0.688] | 0.879 [0.783, 0.963] | 0.230 [0.150, 0.320] | 0.594 [0.418, 0.769] | 0.389 [0.280, 0.501] | 0.435 [0.303, 0.567] | 0.465 [0.331, 0.593] | 0.200 [0.050, 0.400] | 0.250 [0.100, 0.450] |
| rrf | 20 | 0.600 [0.400, 0.800] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.744 [0.621, 0.858] | 0.776 [0.651, 0.887] | 0.992 [0.975, 1.000] | 0.410 [0.300, 0.530] | 0.754 [0.611, 0.888] | 0.629 [0.513, 0.744] | 0.694 [0.574, 0.807] | 0.690 [0.574, 0.796] | 0.400 [0.200, 0.600] | 0.450 [0.250, 0.650] |
| rrf_bge | 20 | 0.700 [0.500, 0.900] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 0.732 [0.605, 0.846] | 0.805 [0.675, 0.914] | 0.992 [0.975, 1.000] | 0.400 [0.290, 0.520] | 0.815 [0.673, 0.938] | 0.682 [0.553, 0.803] | 0.721 [0.592, 0.839] | 0.738 [0.613, 0.848] | 0.400 [0.200, 0.600] | 0.550 [0.350, 0.750] |


### hotpotqa — `hotpotqa|q20|d5000`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.950 [0.875, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.380 [0.350, 0.400] | 1.000 [1.000, 1.000] | 0.967 [0.917, 1.000] | 0.961 [0.903, 1.000] | 0.983 [0.958, 1.000] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.950 [0.875, 1.000] | 0.950 [0.875, 1.000] | 1.000 [1.000, 1.000] | 0.380 [0.350, 0.400] | 1.000 [1.000, 1.000] | 0.916 [0.841, 0.977] | 0.942 [0.884, 0.989] | 0.942 [0.884, 0.989] | 0.900 [0.750, 1.000] | 0.900 [0.750, 1.000] |
| rrf | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.952 [0.904, 0.992] | 0.976 [0.953, 0.996] | 0.976 [0.953, 0.996] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| rrf_bge | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.975 [0.925, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.390 [0.370, 0.400] | 1.000 [1.000, 1.000] | 0.983 [0.950, 1.000] | 0.981 [0.942, 1.000] | 0.992 [0.975, 1.000] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] |


### multihoprag — `multihoprag|q20|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 18 | 0.722 [0.500, 0.889] | 0.944 [0.833, 1.000] | 0.944 [0.833, 1.000] | 1.000 [1.000, 1.000] | 0.722 [0.588, 0.843] | 0.912 [0.838, 0.972] | 1.000 [1.000, 1.000] | 0.400 [0.333, 0.467] | 0.833 [0.694, 0.944] | 0.665 [0.571, 0.755] | 0.683 [0.564, 0.786] | 0.774 [0.698, 0.846] | 0.389 [0.167, 0.611] | 0.722 [0.500, 0.889] |
| bm25_only | 18 | 0.556 [0.333, 0.778] | 0.944 [0.833, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.620 [0.500, 0.741] | 0.819 [0.713, 0.917] | 0.986 [0.958, 1.000] | 0.344 [0.267, 0.422] | 0.718 [0.565, 0.861] | 0.562 [0.439, 0.693] | 0.576 [0.449, 0.709] | 0.665 [0.566, 0.770] | 0.278 [0.056, 0.500] | 0.556 [0.333, 0.778] |
| rrf | 18 | 0.556 [0.333, 0.778] | 0.944 [0.833, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.796 [0.699, 0.889] | 0.940 [0.884, 0.986] | 1.000 [1.000, 1.000] | 0.444 [0.378, 0.511] | 0.724 [0.574, 0.870] | 0.645 [0.525, 0.761] | 0.691 [0.580, 0.796] | 0.752 [0.661, 0.838] | 0.444 [0.222, 0.667] | 0.778 [0.556, 0.944] |
| rrf_bge | 18 | 0.667 [0.444, 0.889] | 0.833 [0.667, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.769 [0.685, 0.852] | 0.944 [0.889, 0.986] | 1.000 [1.000, 1.000] | 0.433 [0.367, 0.500] | 0.777 [0.622, 0.919] | 0.656 [0.547, 0.763] | 0.692 [0.581, 0.797] | 0.768 [0.684, 0.849] | 0.333 [0.111, 0.556] | 0.778 [0.556, 0.944] |


### ragbench-covidqa — `ragbench-covidqa|q20|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0.850 [0.700, 1.000] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.470 [0.390, 0.550] | 0.912 [0.812, 1.000] | 0.883 [0.778, 0.967] | 0.924 [0.847, 0.982] | 0.924 [0.847, 0.982] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 20 | 0.650 [0.450, 0.850] | 0.850 [0.700, 1.000] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 0.792 [0.633, 0.925] | 0.900 [0.767, 1.000] | 0.983 [0.950, 1.000] | 0.370 [0.280, 0.460] | 0.744 [0.584, 0.893] | 0.710 [0.558, 0.849] | 0.725 [0.567, 0.866] | 0.770 [0.629, 0.892] | 0.650 [0.450, 0.850] | 0.850 [0.700, 1.000] |
| rrf | 20 | 0.700 [0.500, 0.900] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.875 [0.767, 0.967] | 0.892 [0.783, 0.975] | 0.933 [0.833, 1.000] | 0.400 [0.320, 0.490] | 0.827 [0.703, 0.942] | 0.753 [0.619, 0.877] | 0.796 [0.683, 0.899] | 0.805 [0.692, 0.907] | 0.750 [0.550, 0.900] | 0.800 [0.600, 0.950] |
| rrf_bge | 20 | 0.750 [0.550, 0.900] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.875 [0.767, 0.967] | 0.892 [0.783, 0.983] | 0.933 [0.833, 1.000] | 0.400 [0.310, 0.490] | 0.847 [0.720, 0.950] | 0.789 [0.646, 0.917] | 0.822 [0.698, 0.930] | 0.829 [0.710, 0.935] | 0.750 [0.550, 0.900] | 0.800 [0.600, 0.950] |


### scifact — `scifact|q20|dfull`

| config | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | R@5 | R@10 | R@100 | P@5 | MRR | MAP@100 | nDCG@5 | nDCG@10 | AllEvid@5 | AllEvid@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0.700 [0.500, 0.900] | 0.850 [0.700, 1.000] | 0.900 [0.750, 1.000] | 0.950 [0.850, 1.000] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.190 [0.160, 0.220] | 0.788 [0.636, 0.927] | 0.788 [0.636, 0.927] | 0.807 [0.652, 0.938] | 0.838 [0.720, 0.945] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] |
| bm25_only | 20 | 0.550 [0.350, 0.750] | 0.650 [0.450, 0.850] | 0.750 [0.550, 0.900] | 0.800 [0.600, 0.950] | 0.725 [0.525, 0.900] | 0.825 [0.650, 0.975] | 0.825 [0.650, 0.975] | 0.150 [0.110, 0.180] | 0.627 [0.437, 0.813] | 0.621 [0.430, 0.810] | 0.639 [0.445, 0.825] | 0.671 [0.496, 0.840] | 0.700 [0.500, 0.900] | 0.800 [0.600, 0.950] |
| rrf | 20 | 0.600 [0.400, 0.800] | 0.750 [0.550, 0.900] | 0.850 [0.700, 1.000] | 0.850 [0.700, 1.000] | 0.825 [0.650, 0.975] | 0.825 [0.650, 0.975] | 1.000 [1.000, 1.000] | 0.170 [0.140, 0.200] | 0.682 [0.500, 0.851] | 0.660 [0.482, 0.828] | 0.699 [0.529, 0.852] | 0.699 [0.529, 0.852] | 0.800 [0.600, 0.950] | 0.800 [0.600, 0.950] |
| rrf_bge | 20 | 0.600 [0.400, 0.800] | 0.750 [0.550, 0.900] | 0.850 [0.700, 1.000] | 0.900 [0.750, 1.000] | 0.825 [0.650, 0.950] | 0.875 [0.725, 1.000] | 1.000 [1.000, 1.000] | 0.170 [0.140, 0.200] | 0.691 [0.512, 0.853] | 0.670 [0.496, 0.832] | 0.701 [0.532, 0.856] | 0.719 [0.558, 0.862] | 0.800 [0.600, 0.950] | 0.850 [0.700, 1.000] |


See `ablation.md` for per-dataset tables with the best value per column in bold.

## 5. Generation quality (RAGAS, judged on the evidence the generator actually saw)

RAGAS `retrieved_contexts` = the slice of each retrieved parent that fits the prompt's 12,000-character evidence budget (`table_visibility.csv`). Reference-based metrics inherit the quality of the references (RAGBench references are model-generated; MultiHop-RAG references are 1–5 word answers).


### docsqa — `docsqa|qall|dfull`

| config | answerable | judged | Faithfulness | ResponseRelevancy | CtxPrecision(ref) | CtxRecall | FactualCorr F1 |
|---|---|---|---|---|---|---|---|
| dense_only | 35 | 0 |  |  |  |  |  |
| bm25_only | 35 | 0 |  |  |  |  |  |
| rrf | 35 | 0 |  |  |  |  |  |
| rrf_bge | 35 | 35 | 0.893 [0.793, 0.971] | 0.871 [0.789, 0.934] | 0.860 [0.779, 0.931] | 0.971 [0.914, 1.000] | 0.269 [0.149, 0.396] |
| rrf_bge_two_pass | 35 | 35 | 0.937 [0.851, 1.000] | 0.836 [0.741, 0.915] | 0.860 [0.779, 0.931] | 0.971 [0.914, 1.000] | 0.278 [0.162, 0.401] |
| engine_as_shipped | 35 | 0 |  |  |  |  |  |


### multihoprag — `multihoprag|q20|dfull`

| config | answerable | judged | EM | token F1 | contains-answer | Faithfulness | ResponseRelevancy | CtxPrecision(ref) | CtxRecall | FactualCorr F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense_only | 18 | 0 | 0.000 [0.000, 0.000] | 0.016 [0.002, 0.031] | 0.278 [0.111, 0.500] |  |  |  |  |  |
| bm25_only | 18 | 0 | 0.000 [0.000, 0.000] | 0.012 [0.004, 0.021] | 0.333 [0.111, 0.556] |  |  |  |  |  |
| rrf | 18 | 0 | 0.000 [0.000, 0.000] | 0.018 [0.005, 0.033] | 0.389 [0.167, 0.611] |  |  |  |  |  |
| rrf_bge | 18 | 18 | 0.000 [0.000, 0.000] | 0.016 [0.002, 0.031] | 0.278 [0.111, 0.500] | 0.203 [0.056, 0.378] | 0.147 [0.000, 0.311] | 0.611 [0.389, 0.833] | 0.556 [0.333, 0.778] | 0.072 [0.000, 0.164] |
| rrf_bge_two_pass | 18 | 18 | 0.000 [0.000, 0.000] | 0.022 [0.010, 0.036] | 0.500 [0.278, 0.722] | 0.569 [0.381, 0.746] | 0.426 [0.228, 0.625] | 0.611 [0.389, 0.833] | 0.444 [0.222, 0.667] | 0.046 [0.000, 0.120] |
| engine_as_shipped | 18 | 0 | 0.000 [0.000, 0.000] | 0.021 [0.010, 0.034] | 0.500 [0.278, 0.722] |  |  |  |  |  |


### ragbench-covidqa — `ragbench-covidqa|q20|dfull`

| config | answerable | judged | Faithfulness | ResponseRelevancy | CtxPrecision(ref) | CtxRecall | FactualCorr F1 |
|---|---|---|---|---|---|---|---|
| dense_only | 20 | 0 |  |  |  |  |  |
| bm25_only | 20 | 0 |  |  |  |  |  |
| rrf | 20 | 0 |  |  |  |  |  |
| rrf_bge | 20 | 20 | 0.539 [0.328, 0.750] | 0.543 [0.348, 0.735] | 0.639 [0.462, 0.801] | 0.717 [0.517, 0.900] | 0.351 [0.177, 0.535] |
| rrf_bge_two_pass | 20 | 20 | 0.576 [0.375, 0.764] | 0.627 [0.436, 0.804] | 0.639 [0.462, 0.801] | 0.717 [0.517, 0.900] | 0.373 [0.207, 0.549] |
| engine_as_shipped | 20 | 0 |  |  |  |  |  |


## 6. Robustness

Abstention = the system's refusal sentence, an empty output, or a short answer built on a refusal phrase (heuristic, `eval/metrics.py`).

| dataset | config | unanswerable n | abstention on unanswerable ↑ | answerable n | false abstention ↓ | false abst. | gold in context | anachronism gate fired |
|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 9 | 1.000 [1.000, 1.000] | 35 | 0.114 [0.029, 0.229] | 0.062 | – |
| docsqa | bm25_only | 9 | 1.000 [1.000, 1.000] | 35 | 0.086 [0.000, 0.171] | 0.031 | – |
| docsqa | rrf | 9 | 1.000 [1.000, 1.000] | 35 | 0.057 [0.000, 0.143] | 0.062 | – |
| docsqa | rrf_bge | 9 | 1.000 [1.000, 1.000] | 35 | 0.057 [0.000, 0.143] | 0.062 | – |
| docsqa | rrf_bge_two_pass | 9 | 1.000 [1.000, 1.000] | 35 | 0.086 [0.000, 0.171] | 0.094 | – |
| docsqa | engine_as_shipped | 9 | 0.889 [0.667, 1.000] | 35 | 0.057 [0.000, 0.143] | 0.065 | 0.000 |
| multihoprag | dense_only | 2 | 1.000 [1.000, 1.000] | 18 | 0.778 [0.556, 0.944] | 0.778 | – |
| multihoprag | bm25_only | 2 | 1.000 [1.000, 1.000] | 18 | 0.667 [0.444, 0.889] | 0.667 | – |
| multihoprag | rrf | 2 | 1.000 [1.000, 1.000] | 18 | 0.667 [0.444, 0.889] | 0.667 | – |
| multihoprag | rrf_bge | 2 | 1.000 [1.000, 1.000] | 18 | 0.778 [0.556, 0.944] | 0.778 | – |
| multihoprag | rrf_bge_two_pass | 2 | 1.000 [1.000, 1.000] | 18 | 0.278 [0.111, 0.500] | 0.278 | – |
| multihoprag | engine_as_shipped | 2 | 1.000 [1.000, 1.000] | 18 | 0.333 [0.111, 0.556] | 0.333 | 0.000 |
| ragbench-covidqa | dense_only | 0 | – | 20 | 0.350 [0.150, 0.550] | 0.350 | – |
| ragbench-covidqa | bm25_only | 0 | – | 20 | 0.350 [0.150, 0.550] | 0.278 | – |
| ragbench-covidqa | rrf | 0 | – | 20 | 0.400 [0.200, 0.600] | 0.400 | – |
| ragbench-covidqa | rrf_bge | 0 | – | 20 | 0.400 [0.200, 0.600] | 0.400 | – |
| ragbench-covidqa | rrf_bge_two_pass | 0 | – | 20 | 0.300 [0.100, 0.500] | 0.300 | – |
| ragbench-covidqa | engine_as_shipped | 0 | – | 20 | 0.200 [0.050, 0.400] | 0.200 | 0.000 |


### Evidence visibility (12,000-character prompt budget)

| dataset | config | n | evidence_chars_total_mean | evidence_chars_seen_mean | fraction_of_evidence_seen | gold_in_context_rate | gold_in_seen_context_rate | gold_docs_seen_fraction |
|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 44 | 6617.25 | 6477.136 | 0.979 | 0.914 | 0.914 | 0.9 |
| docsqa | bm25_only | 44 | 7039.682 | 7023.182 | 0.998 | 0.914 | 0.914 | 0.9 |
| docsqa | rrf | 44 | 7110.273 | 6954.682 | 0.978 | 0.914 | 0.914 | 0.914 |
| docsqa | rrf_bge | 44 | 7149.205 | 7003.159 | 0.98 | 0.914 | 0.914 | 0.914 |
| docsqa | rrf_bge_two_pass | 44 | 7149.205 | 7072.682 | 0.989 | 0.914 | 0.914 | 0.914 |
| docsqa | engine_as_shipped | 44 | 7699.795 | 7355.045 | 0.955 | 0.886 | 0.886 | 0.886 |
| multihoprag | dense_only | 20 | 45097.85 | 11749.15 | 0.261 | 1.0 | 0.889 | 0.417 |
| multihoprag | bm25_only | 20 | 45629.8 | 11743.6 | 0.257 | 1.0 | 0.722 | 0.394 |
| multihoprag | rrf | 20 | 46565.55 | 11737.9 | 0.252 | 1.0 | 0.778 | 0.407 |
| multihoprag | rrf_bge | 20 | 46491.9 | 11739.0 | 0.252 | 1.0 | 0.778 | 0.398 |
| multihoprag | rrf_bge_two_pass | 20 | 46491.9 | 11989.1 | 0.258 | 1.0 | 0.778 | 0.426 |
| multihoprag | engine_as_shipped | 20 | 48557.75 | 11988.8 | 0.247 | 1.0 | 0.778 | 0.412 |
| ragbench-covidqa | dense_only | 20 | 3220.25 | 3220.25 | 1.0 | 1.0 | 1.0 | 1.0 |
| ragbench-covidqa | bm25_only | 20 | 3666.65 | 3666.65 | 1.0 | 0.9 | 0.9 | 0.808 |
| ragbench-covidqa | rrf | 20 | 3488.45 | 3488.45 | 1.0 | 1.0 | 1.0 | 0.892 |
| ragbench-covidqa | rrf_bge | 20 | 3468.1 | 3468.1 | 1.0 | 1.0 | 1.0 | 0.875 |
| ragbench-covidqa | rrf_bge_two_pass | 20 | 3468.1 | 3468.1 | 1.0 | 1.0 | 1.0 | 0.875 |
| ragbench-covidqa | engine_as_shipped | 20 | 3884.95 | 3884.95 | 1.0 | 1.0 | 1.0 | 0.867 |


### RGB (generation only; 5 passages per query)

| kind | mode | noise_ratio | n | mentions_true | mentions_fake | abstained | accuracy | abstention_rate | rejection_rate |
|---|---|---|---|---|---|---|---|---|---|
| en_fact | one_pass | 0.0 | 20 | 0.0 | 0.8 | 0.1 |  |  |  |
| en_fact | two_pass | 0.0 | 20 | 0.05 | 0.7 | 0.1 |  |  |  |
| en_int | one_pass | 0.0 | 20 |  |  |  | 0.75 | 0.15 |  |
| en_int | two_pass | 0.0 | 20 |  |  |  | 0.6 | 0.1 |  |
| en_refine | one_pass | 0.0 | 20 |  |  |  | 0.95 | 0.0 |  |
| en_refine | one_pass | 0.2 | 20 |  |  |  | 0.95 | 0.0 |  |
| en_refine | one_pass | 0.4 | 20 |  |  |  | 1.0 | 0.0 |  |
| en_refine | one_pass | 0.6 | 20 |  |  |  | 1.0 | 0.0 |  |
| en_refine | one_pass | 0.8 | 20 |  |  |  | 0.95 | 0.0 |  |
| en_refine | one_pass | 1.0 | 20 |  |  |  |  | 0.8 | 0.8 |
| en_refine | two_pass | 0.0 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.2 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.4 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.6 | 20 |  |  |  | 0.95 | 0.05 |  |
| en_refine | two_pass | 0.8 | 20 |  |  |  | 0.9 | 0.05 |  |
| en_refine | two_pass | 1.0 | 20 |  |  |  |  | 0.8 | 0.8 |


## 7. System: tokens, cost, latency

Tokens are from the main run; $ uses list prices in `eval/config.py` (Gemini API standard tier; Vertex billing may differ). The hosted reranker's price is not published in the repo, so its tokens are reported but excluded from $.

| dataset | config | n_generated | live_n | total_p50_s | total_p95_s | retrieval_p50_s | retrieval_p95_s | rerank_p50_s | rerank_p95_s | generation_p50_s | generation_p95_s | gen_in_tokens_mean | gen_out_tokens_mean | rerank_tokens_mean | generator_cost_per_1k_queries_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 44 | 6.0 | 1.5 | 4.62 | 0.44 | 1.65 | 0.0 | 0.0 | 1.0 | 2.95 | 2274.2 | 32.16 | 0.0 | 0.76 |
| docsqa | bm25_only | 44 | 6.0 | 1.18 | 1.42 | 0.0 | 0.0 | 0.0 | 0.0 | 1.17 | 1.41 | 2347.14 | 32.57 | 0.0 | 0.79 |
| docsqa | rrf | 44 | 6.0 | 1.46 | 2.35 | 0.4 | 0.51 | 0.0 | 0.0 | 1.07 | 1.92 | 2368.2 | 32.05 | 0.0 | 0.79 |
| docsqa | rrf_bge | 44 | 6.0 | 3.92 | 6.89 | 0.44 | 0.59 | 2.56 | 5.15 | 0.96 | 1.23 | 2398.39 | 32.11 | 22055.48 | 0.8 |
| docsqa | rrf_bge_two_pass | 44 | 6.0 | 4.96 | 5.2 | 0.42 | 0.45 | 2.62 | 2.86 | 1.81 | 1.99 | 2591.8 | 113.14 | 22055.48 | 1.06 |
| docsqa | engine_as_shipped | 44 | 6.0 | 6.89 | 28.73 | 2.88 | 3.58 | 2.71 | 3.12 | 2.37 | 24.45 | 2740.66 | 107.68 | 21758.09 | 1.21 |
| multihoprag | dense_only | 20 | 5.0 | 1.85 | 5.05 | 0.43 | 1.81 | 0.0 | 0.0 | 1.44 | 2.85 | 2964.15 | 48.65 | 0.0 | 1.01 |
| multihoprag | bm25_only | 20 | 5.0 | 1.13 | 2.22 | 0.0 | 0.0 | 0.0 | 0.0 | 0.79 | 1.88 | 2911.6 | 68.4 | 0.0 | 1.04 |
| multihoprag | rrf | 20 | 5.0 | 1.9 | 2.07 | 0.43 | 0.47 | 0.0 | 0.0 | 0.86 | 1.39 | 2975.65 | 68.5 | 0.0 | 1.06 |
| multihoprag | rrf_bge | 20 | 5.0 | 5.08 | 5.57 | 0.47 | 0.55 | 3.16 | 3.3 | 0.97 | 1.25 | 2922.85 | 46.7 | 31550.35 | 0.99 |
| multihoprag | rrf_bge_two_pass | 20 | 5.0 | 6.06 | 6.56 | 0.44 | 0.58 | 3.02 | 3.14 | 2.18 | 2.54 | 3932.25 | 239.05 | 31550.35 | 1.78 |
| multihoprag | engine_as_shipped | 20 | 5.0 | 8.11 | 8.69 | 2.74 | 3.71 | 3.03 | 3.49 | 2.19 | 3.48 | 4109.5 | 222.55 | 31582.0 | 1.99 |
| ragbench-covidqa | dense_only | 20 |  |  |  |  |  |  |  |  |  | 1050.65 | 47.45 | 0.0 | 0.43 |
| ragbench-covidqa | bm25_only | 20 |  |  |  |  |  |  |  |  |  | 1110.45 | 50.75 | 0.0 | 0.46 |
| ragbench-covidqa | rrf | 20 |  |  |  |  |  |  |  |  |  | 1097.3 | 39.35 | 0.0 | 0.43 |
| ragbench-covidqa | rrf_bge | 20 |  |  |  |  |  |  |  |  |  | 1092.3 | 43.1 | 11072.15 | 0.44 |
| ragbench-covidqa | rrf_bge_two_pass | 20 |  |  |  |  |  |  |  |  |  | 1120.2 | 143.8 | 11072.15 | 0.7 |
| ragbench-covidqa | engine_as_shipped | 20 |  |  |  |  |  |  |  |  |  | 1135.25 | 177.7 | 11061.6 | 0.88 |


## 8. Statistics

Paired tests on identical queries (baseline → variant; positive = variant better, except abstention rows): paired bootstrap (10k, seed 0) and Wilcoxon signed-rank; effect size = Cohen's d_z.

| dataset | comparison | metric | n | mean_baseline | mean_variant | mean_diff | ci_lo | ci_hi | p_bootstrap | p_wilcoxon | cohens_dz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | rrf vs rrf+bge | hit@5 | 20 | 0.85 | 0.85 | 0.0 | -0.15 | 0.15 | 1.0000 | 1.0000 | 0.0 |
| scifact | rrf vs rrf+bge | mrr | 20 | 0.682 | 0.6907 | 0.0087 | -0.0698 | 0.1004 | 0.8550 | 0.8658 | 0.044 |
| scifact | rrf vs rrf+bge | ndcg@10 | 20 | 0.6987 | 0.7187 | 0.02 | -0.0461 | 0.0967 | 0.6068 | 0.7532 | 0.1173 |
| scifact | rrf vs rrf+bge | recall@100 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| scifact | rrf vs rrf+bge | map | 20 | 0.6604 | 0.6696 | 0.0092 | -0.0693 | 0.1008 | 0.8474 | 0.7794 | 0.0466 |
| fiqa | rrf vs rrf+bge | hit@5 | 20 | 0.95 | 0.95 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| fiqa | rrf vs rrf+bge | mrr | 20 | 0.7543 | 0.8151 | 0.0608 | -0.0342 | 0.1608 | 0.2436 | 0.2809 | 0.2622 |
| fiqa | rrf vs rrf+bge | ndcg@10 | 20 | 0.6904 | 0.7383 | 0.0479 | -0.0173 | 0.1114 | 0.1476 | 0.0774 | 0.317 |
| fiqa | rrf vs rrf+bge | recall@100 | 20 | 0.9917 | 0.9917 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| fiqa | rrf vs rrf+bge | map | 20 | 0.6288 | 0.6817 | 0.0529 | -0.0331 | 0.1381 | 0.2304 | 0.1621 | 0.2629 |
| multihoprag | rrf vs rrf+bge | hit@5 | 18 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| multihoprag | rrf vs rrf+bge | mrr | 18 | 0.7241 | 0.7769 | 0.0528 | -0.0815 | 0.1917 | 0.4568 | 0.5961 | 0.1743 |
| multihoprag | rrf vs rrf+bge | ndcg@10 | 18 | 0.7517 | 0.7683 | 0.0166 | -0.0564 | 0.0889 | 0.6418 | 0.5936 | 0.101 |
| multihoprag | rrf vs rrf+bge | recall@100 | 18 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| multihoprag | rrf vs rrf+bge | map | 18 | 0.645 | 0.6557 | 0.0107 | -0.0772 | 0.0943 | 0.7878 | 0.7776 | 0.055 |
| ragbench-covidqa | rrf vs rrf+bge | hit@5 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | rrf vs rrf+bge | mrr | 20 | 0.8267 | 0.8475 | 0.0208 | -0.0875 | 0.125 | 0.7904 | 1.0000 | 0.0813 |
| ragbench-covidqa | rrf vs rrf+bge | ndcg@10 | 20 | 0.8047 | 0.8295 | 0.0248 | -0.0374 | 0.0979 | 0.5196 | 0.4652 | 0.155 |
| ragbench-covidqa | rrf vs rrf+bge | recall@100 | 20 | 0.9333 | 0.9333 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | rrf vs rrf+bge | map | 20 | 0.753 | 0.789 | 0.036 | -0.0404 | 0.1319 | 0.4596 | 0.4652 | 0.1748 |
| docsqa | rrf vs rrf+bge | hit@5 | 35 | 0.8857 | 0.8857 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | rrf vs rrf+bge | mrr | 35 | 0.8088 | 0.7503 | -0.0586 | -0.1157 | -0.0143 | 0.0088 | 0.0339 | -0.3636 |
| docsqa | rrf vs rrf+bge | ndcg@10 | 35 | 0.8267 | 0.786 | -0.0407 | -0.0844 | -0.0048 | 0.0188 | 0.0583 | -0.334 |
| docsqa | rrf vs rrf+bge | recall@100 | 35 | 0.9429 | 0.9429 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | rrf vs rrf+bge | map | 35 | 0.7945 | 0.7417 | -0.0529 | -0.1115 | -0.0029 | 0.0364 | 0.0583 | -0.313 |
| hotpotqa | rrf vs rrf+bge | hit@5 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| hotpotqa | rrf vs rrf+bge | mrr | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| hotpotqa | rrf vs rrf+bge | ndcg@10 | 20 | 0.9762 | 0.9916 | 0.0154 | -0.0009 | 0.0347 | 0.0822 | 0.1408 | 0.3727 |
| hotpotqa | rrf vs rrf+bge | recall@100 | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| hotpotqa | rrf vs rrf+bge | map | 20 | 0.9517 | 0.9833 | 0.0317 | -0.0017 | 0.0708 | 0.0812 | 0.1408 | 0.3773 |
| multihoprag | one-pass vs two-pass | faithfulness | 18 | 0.2028 | 0.5689 | 0.3661 | 0.1818 | 0.5593 | 0.0000 | 0.0046 | 0.8689 |
| multihoprag | one-pass vs two-pass | answer_relevancy | 18 | 0.1471 | 0.426 | 0.2789 | 0.0897 | 0.4842 | 0.0020 | 0.0357 | 0.6386 |
| multihoprag | one-pass vs two-pass | context_precision | 18 | 0.6111 | 0.6111 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| multihoprag | one-pass vs two-pass | context_recall | 18 | 0.5556 | 0.4444 | -0.1111 | -0.2778 | 0.0 | 0.2372 | 0.1573 | -0.3436 |
| multihoprag | one-pass vs two-pass | factual_correctness_f1 | 18 | 0.0717 | 0.0461 | -0.0256 | -0.1372 | 0.0828 | 0.6752 | 0.7865 | -0.106 |
| multihoprag | one-pass vs two-pass | false_abstention | 18 | 0.7778 | 0.2778 | -0.5 | -0.7222 | -0.2778 | 0.0002 | 0.0027 | -0.9718 |
| ragbench-covidqa | one-pass vs two-pass | faithfulness | 20 | 0.5389 | 0.5764 | 0.0375 | -0.1028 | 0.2111 | 0.6974 | 0.9323 | 0.1007 |
| ragbench-covidqa | one-pass vs two-pass | answer_relevancy | 20 | 0.543 | 0.6274 | 0.0844 | 0.0046 | 0.1959 | 0.0106 | 0.0506 | 0.3696 |
| ragbench-covidqa | one-pass vs two-pass | context_precision | 20 | 0.6389 | 0.6389 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | one-pass vs two-pass | context_recall | 20 | 0.7167 | 0.7167 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| ragbench-covidqa | one-pass vs two-pass | factual_correctness_f1 | 20 | 0.351 | 0.3735 | 0.0225 | -0.042 | 0.09 | 0.5170 | 0.5281 | 0.1469 |
| ragbench-covidqa | one-pass vs two-pass | false_abstention | 20 | 0.4 | 0.3 | -0.1 | -0.25 | 0.0 | 0.2382 | 0.1573 | -0.3249 |
| docsqa | one-pass vs two-pass | faithfulness | 35 | 0.8929 | 0.9371 | 0.0443 | -0.0057 | 0.1157 | 0.1330 | 0.1441 | 0.2279 |
| docsqa | one-pass vs two-pass | answer_relevancy | 35 | 0.8706 | 0.8361 | -0.0344 | -0.1019 | 0.0106 | 0.2170 | 0.3533 | -0.1923 |
| docsqa | one-pass vs two-pass | context_precision | 35 | 0.8599 | 0.8599 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | one-pass vs two-pass | context_recall | 35 | 0.9714 | 0.9714 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |
| docsqa | one-pass vs two-pass | factual_correctness_f1 | 35 | 0.2689 | 0.2777 | 0.0089 | -0.0706 | 0.0826 | 0.8138 | 0.8236 | 0.0383 |
| docsqa | one-pass vs two-pass | false_abstention | 35 | 0.0571 | 0.0857 | 0.0286 | 0.0 | 0.0857 | 0.7416 | 0.3173 | 0.169 |
| docsqa | one-pass vs two-pass | abstention_on_unanswerable | 9 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.0 |


No multiple-comparison correction is applied; with this many comparisons treat p-values as descriptive.

## 9. Limitations, leakage and comparability (read before citing)

- **Judge validity.** The judge is a Google model scoring a Google generator (family/self-preference bias). In the judge comparison on the smoke batch, a different judge changed mean Faithfulness by a large margin (see `judge_comparison.md`). `judge_audit.csv` holds 50 sampled judgments with a blank `human_label` column for manual validation; **no human labels exist yet** and no agreement figure is claimed.
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

