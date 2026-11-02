# Benchmark re-based on `main`

Written 2026-11-01. Numbers are in [MAIN_REBASE_RESULTS.md](MAIN_REBASE_RESULTS.md); this file says what was done and how to repeat it.

## Why

The old benchmark ran on `feat/rag-quality-generation`, which is 26 commits behind `main`. `main` has things the old code never had:
a Docling parse tree, smaller chunks (child 150-300 tokens), a local summariser (Qwen3-0.6B, off by default),
a ParadeDB keyword index, and a keyword fallback (`ts_rank` + trigram similarity + exact-match boosts). A paper about
DeepContext has to measure that code, so the benchmark now runs on `main`.

## Where things are

- Branch `eval/rebase-on-main`, cut from `main`. **Nothing is committed.** `main` is untouched.
- `eval/` is the harness from `eval/variant-b-deep-ingestion` plus the later fixes. The old base is not needed any more.
- Two small patches to the application, kept as files and applied in the working tree only:
  - `eval/paradedb_fix/main_bm25_fix.patch`: main's ParadeDB BM25 never worked (three bugs, all logged at debug level, so it
    silently fell back to the trigram ranking). See `eval/paradedb_fix/README.md`.
  - `eval/rrf_fix/weighted_rrf.patch`: RRF where each list can have its own vote (`RRF_WEIGHT_LEXICAL`, `RRF_WEIGHT_DENSE`).
    The default is 1.0 / 1.0, which is exactly the old behaviour.
- `EVAL_STORE=stock` makes the harness use the application's own `PostgresStore` (its startup, flags and indexes run as in
  production). Without it the harness uses a small subclass that only fixes the schema DDL for pgvector 0.8.

## Runs on main's code (retrieval only)

Same indexes for every run (`EVAL_INDEX_SUFFIX=_main`: main's chunker, one page = one parent, Gemini embeddings 768-d).
Order matters, because the patches were applied to the working tree between runs (file times confirm it):

| run | code | settings |
|---|---|---|
| M0 `M0_stock` | `main` as shipped (before any patch) | `dense_only bm25_only rrf`, equal votes |
| M1 `M1_bm25patch` | main + BM25 patch | `bm25_only rrf`, equal votes |
| M2 `M2_vote0.1`, `M2_vote0.02` | main + BM25 patch + weighted RRF | `rrf`, `RRF_WEIGHT_LEXICAL` = 0.1 or 0.02 |

```bash
env EVAL_STORE=stock EVAL_INDEX_SUFFIX=_main RRF_WEIGHT_LEXICAL=0.02 \
  uv run --no-sync python -m eval.run_retrieval --dataset pdf-got --configs rrf --concurrency 4 \
  --results-dir eval/results_main/M2_vote0.02
```

The vote 0.02 was chosen earlier on the dev sets (docs, thesis) in the old code by a rule written down in advance.
The novel is the held-out test and was run once per setting. Vote 0.1 is shown for reference.

## What the numbers say (short)

1. As shipped, fusing keyword and dense search scores **below dense alone** on all three sets
   (nDCG@10 docs 0.776 vs 0.831, thesis 0.414 vs 0.510, novel 0.449 vs 0.817). The gap is clear on thesis and novel;
   on docs it is within noise.
2. The BM25 patch fixes the keyword channel (keyword-only nDCG@10 +0.149, +0.077, +0.107; all intervals above zero).
   Fused with equal votes it is still below dense alone on thesis and novel.
3. Giving the keyword list a small vote (0.02) brings the fused result back to dense-only level and
   gains +0.055 (docs, not significant), +0.077 (thesis) and +0.356 (novel, held-out) over main as shipped.
4. With both fixes, the keyword channel adds nothing on top of dense search on these three sets
   (vote 0.02 vs dense only: 0.000, -0.020, -0.012 nDCG@10).
5. Far-apart evidence (gold pages at least 100 pages apart): only 5 questions per set, so only large effects can show.
   In the top 10, no setting finds the earliest and the latest gold page together on the thesis (0 of 5). On the novel
   dense only and the small keyword votes find both for 2 of 5, and equal votes find none (1 or 2 of 5 in the top 20).

   **Caution (checked 2026-11-01): item 5 is not a test of the pronoun problem.** None of the 10 questions was written that
   way, and the labels were never audited for necessity: the novel's dying-words question has the same sentence on all three
   gold pages, so one page is enough. The thesis key lists 184 of its 188 gold pages together with a partner exactly 20 pages
   away (printed page number next to PDF page number); the later page holds more of the answer words in 84 of 92 pairs. "Both
   ends" is out of reach there, and every thesis score that counts all gold pages (recall, nDCG) is too low. Nothing has been
   re-scored. Research and plan for a valid test: `reports/Retrieving evidence across distant pages.md`.

## Deep ingestion (main's own summariser)

Main's `SummaryIngestionPipeline` writes a one-to-two sentence context per child chunk with a small local LLM and embeds
`summary + blank line + content`. The summary also feeds the keyword index. It needs a GPU for thousands of chunks (the
Mac has 8 GB RAM; an earlier local test of 16 chunks on the Apple GPU had not finished after 13 minutes and was stopped).

The harness does it in three steps. The code is `eval/main_summaries.py` (cache and job files) and `eval/vm/summarize_prompts.py`
(GPU side, no database, no credentials). Tests: `tests/test_main_summaries.py`.

1. Export the prompts that have no cached summary (done: 3,117 unique prompts, 8.5 MB, in `eval/cache/summaries/jobs.jsonl`;
   docs 244, thesis 945, novel 1,937 child chunks):

   ```bash
   uv run --no-sync python -m eval.main_summaries export --datasets docsqa pdf-quantum pdf-got --out eval/cache/summaries/jobs.jsonl
   ```

2. On a GPU machine (done 2026-11-01: 41 minutes on the L4, 3,117 summaries, none empty), with `src/` and the two files copied over:

   ```bash
   PYTHONPATH=src python summarize_prompts.py jobs.jsonl results.jsonl --batch-size 32 --limit 16   # smoke test
   PYTHONPATH=src python summarize_prompts.py jobs.jsonl results.jsonl --batch-size 32
   ```

   It runs the application's own `ChunkSummarizer` (same model, prompt, greedy decoding, fp16). It appends after every batch, so a
   stopped run resumes. Validate every `gcloud` command with `gcloud help <command>` first.

3. Import and build the deep indexes (a missing summary stops the build; nothing is made up):

   ```bash
   uv run --no-sync python -m eval.main_summaries import eval/cache/summaries/results.jsonl
   env EVAL_INGEST=deep EVAL_STORE=stock EVAL_INDEX_SUFFIX=_maindeep \
     uv run --no-sync python -m eval.run_retrieval --dataset docsqa --configs dense_only bm25_only rrf --concurrency 4 \
     --results-dir eval/results_main/D0_deep
   ```

   The index records `ingest=deep` and the number of empty generations (replaced by the application's own fallback, the first 25
   words). An index built in one mode cannot be reused in the other.

Differences from the application's run, stated so they are not a surprise: prompts of different documents share a batch (32,
longest first) where the application batches inside one document (8). Decoding is greedy, so the text can differ only by fp16
rounding. The summariser sees the first 800 characters of the parent page only, so it cannot resolve a name that appears
far away in the book. That limit is the application's, and the far-evidence slice will show whether it matters.

### Comparison fixed before any deep result exists

- Indexes: `_main` (fast) and `_maindeep` (deep) for docs, thesis and novel, same questions and gold pages.
- Primary: deep vs fast under the settings chosen above (BM25 fix + keyword vote 0.02): nDCG@10, paired bootstrap, per set.
  Reference: deep vs fast with main as shipped (`dense_only`, `bm25_only`, `rrf` with equal votes).
- Secondary: the far-evidence slice (both-ends@10/20, recall@10/20) with the same thresholds (span >= 100 pages).
- No prompt, model, vote or depth is changed after seeing a deep result. If the summariser output looks bad (many empty
  generations), that is reported as a finding, not repaired.
- The novel is the held-out test: one run per setting.

Added 2026-11-01 evening, still before any deep result exists:

- **Settings compared** (fast rows from M0/M1/M2, deep rows run on the same code state): `dense_only`; `bm25_only` (BM25 fix);
  `rrf` with equal votes (BM25 fix, as M1); `rrf` with keyword vote 0.02 (as M2, the primary setting); `rrf_bge` with vote 0.02
  (reranker `bge-reranker-v2-m3` on the cloud GPU, because the hosted reranker answers 402 Payment Required; the fast side is run
  again with the same reranker so that both sides share it).
- **Thesis key.** The stored key lists most pages twice, 20 pages apart (see `eval/pdf_sets.py`). The thesis is reported with the stored key
  and with the corrected key (a gold page that has a partner exactly 20 pages later is dropped; `eval/thesis_key.py`). With the
  corrected key the fast numbers are higher, for example dense only nDCG@10 0.750 instead of 0.510.
- **Far-evidence slice** (the secondary item above) is dropped: its questions were not far-reference questions (see the caution
  under "What the numbers say").
- The tables are produced by `eval/deep_vs_fast.py`; nothing in them is tuned after seeing a deep result.

## Deep ingestion on main: result (2026-11-01)

The application's own contextual summaries (Qwen3-0.6B, `ChunkSummarizer`, summary + blank line + content embedded, summary also in the
keyword index) were generated on the cloud GPU (3,117 chunks, 41 minutes, none empty; 22% reached the 180-token cap) and compared with the fast
indexes on identical questions. Full tables with intervals: [results_main/DEEP_VS_FAST.md](results_main/DEEP_VS_FAST.md)
(`python -m eval.deep_vs_fast`, rules fixed above before any deep result existed).

Change in nDCG@10, deep minus fast (verdict from the 95% paired-bootstrap interval):

| search setting | docs (35 q) | thesis, corrected key (31 q) | novel (31 q) |
|---|---|---|---|
| dense only | -0.002, no clear difference | +0.014, no clear difference | **-0.115, deep worse** |
| keyword only (BM25 fix) | -0.007, no clear difference | +0.047, no clear difference | -0.002, no clear difference |
| mixed, equal votes | +0.015, no clear difference | +0.021, no clear difference | -0.028, no clear difference |
| mixed, keyword vote 0.02 (primary) | -0.002, no clear difference | -0.016, no clear difference | **-0.118, deep worse** |
| mixed + reranker (vote 0.02) | -0.017, no clear difference | +0.006, no clear difference | **-0.076, deep worse** |

* Deep ingestion never beat fast ingestion: 0 of 15 cells better, 12 no clear difference, 3 worse (all on the novel). The thesis gives the same
  verdicts with the stored key.
* On the novel the summaries hurt wherever the dense channel decides the ranking. The cause is not known.
  **Correction (2026-11-02):** an earlier version of this note said that 30% of the novel's summaries name things that are not in the text the
  model was given, and blamed the model's memory of the book. That count was a one-off, was not saved, and cannot be reproduced. The saved and
  tested check (`eval/summary_names.py`, rule in its header) gives 3% for the novel (55 of 1,929), 3% for the thesis (25 of 944) and 1% for the
  docs (3 of 244). So the 0.6B summaries do not add names from memory, and the memory explanation is not supported for them.
* The thesis numbers were too low with the stored key: dense only nDCG@10 is 0.750 with the corrected key and 0.510 with the stored one.
* Not shown by this test: the settings where the literature reports gains (a weak first stage, chunks that look alike). The keyword-only column
  is the closest: thesis +0.047, interval from -0.031 to +0.121.
* Reranker: the self-hosted `bge-reranker-v2-m3` on the L4 (the hosted one answers 402 Payment Required), same server for both sides.
  On these sets it did not improve the fast first stage either (docs 0.831 to 0.842, thesis 0.490 to 0.475, novel 0.805 to 0.741).

## 9B round: rules fixed before any 9B result exists (2026-11-01, late)

The 0.6B summaries gave no gain. This round swaps in a bigger summariser and uses the same open model to answer, on the cloud GPU.

* **Summaries:** Qwen3.5-9B (vLLM on the L4), the application's exact prompt (`ChunkSummarizer._build_prompt`), greedy, 180-token cap, same
  post-processing; only the model changes (`eval/vm/summarize_vllm.py`). Indexes `_maindeep9b`; the 0.6B indexes `_maindeep` and the fast indexes `_main` stay.
* **Retrieval:** the five settings of the 0.6B round, deep-9B against fast and against deep-0.6B (`eval/deep_vs_fast.py`).
* **Answers:** Qwen3.5-9B answers (one-pass `PromptAssembler`, retrieval `rrf_bge` with keyword vote 0.02, six parent pages), for the three
  arms fast / deep-0.6B / deep-9B, on every question of docs (44), thesis (36) and novel (36), unanswerable ones included. Judge: the
  reference-based rubric of `eval/judge_correctness.py` with Gemini 3.7 flash, a different model from the answerer. Reported: rubric
  correctness, correct abstention on unanswerable questions, false abstention, and the paired difference to the fast arm.
* In deep arms the one-pass prompt carries a `Summary:` line above every retrieved page (application behaviour), so summaries can affect answers
  twice: through retrieval and through the prompt.
* Nothing is tuned after seeing a 9B result; an empty or broken generation is reported, not repaired.

## 9B round: result (2026-11-02)

The rules above were not changed. Qwen3.5-9B wrote 3,117 summaries on the cloud GPU (none empty, none cut at the 180-token cap; the 0.6B model hit
the cap in 22%) and then answered the questions. Judge: Gemini 3.7 flash. Tables with intervals:
[DEEP_VS_FAST_9B.md](results_main/DEEP_VS_FAST_9B.md), [DEEP9B_VS_DEEP06.md](results_main/DEEP9B_VS_DEEP06.md),
[ANSWERS_DEEP_VS_FAST.md](results_main/ANSWERS_DEEP_VS_FAST.md).

**Search.** Change in nDCG@10, deep 9B minus fast (95% paired-bootstrap interval decides the verdict):

| search setting | docs (35 q) | thesis, corrected key (31 q) | novel (31 q) |
|---|---|---|---|
| dense only | -0.011, no clear difference | -0.011, no clear difference | -0.007, no clear difference |
| keyword only (BM25 fix) | +0.041, no clear difference | +0.058, no clear difference | **+0.117, deep better** |
| mixed, equal votes | -0.014, no clear difference | +0.010, no clear difference | +0.064, no clear difference |
| mixed, keyword vote 0.02 (primary) | -0.011, no clear difference | -0.031, no clear difference | -0.005, no clear difference |
| mixed + reranker (vote 0.02) | +0.005, no clear difference | -0.024, no clear difference | +0.003, no clear difference |

* 1 of 15 cells better, 14 no clear difference, 0 worse. The thesis gives the same verdicts with the stored key.
* The one better cell is keyword only on the novel. It is not the shipped setting. In the shipped setting (keyword vote 0.02) the novel is
  -0.005, and with the reranker +0.003.
* The novel is not a clean test, because the 9B model knows the book. 70% of its novel summaries (1,347 of 1,929) name something that is not in the
  text it was shown (thesis 9%, docs 2%; `eval/summary_names.py`). Those names come from the model's memory, not from the document. A keyword
  search can match them. A gain like this would not carry over to a document the model has not read, and index text that comes from model memory
  breaks the rule that index text comes only from the document.
* 9B against 0.6B: docs and thesis, no clear difference in all 10 cells. Novel: 9B better in all 5 cells (+0.079 to +0.119), because the 0.6B was
  below fast on the novel and the 9B is about level with fast.

**Answers.** Qwen3.5-9B answered with the same retrieval (`rrf_bge`, keyword vote 0.02, six parent pages). Rubric correctness on answerable
questions, and the difference to fast (95% paired-bootstrap interval):

| set | fast | deep 0.6B | deep 9B |
|---|---|---|---|
| docs (35 answerable) | 0.96 | 0.97, +0.014 [-0.071, +0.100] | 0.97, +0.014 [-0.071, +0.100] |
| thesis (31) | 0.84 | 0.85, +0.016 [-0.081, +0.129] | 0.89, +0.048 [-0.016, +0.129] |
| novel (31) | 0.66 | 0.66, +0.000 [-0.145, +0.161] | 0.77, +0.113 [-0.016, +0.242] |

* Every verdict is "no clear difference". The largest gap is the novel with the 9B summaries: +0.113, interval from -0.016 to +0.242.
* On the novel the evidence page was in the prompt for all 31 answerable questions in the fast and the 9B arm (28 of 31 in the 0.6B arm).
  Deep 9B against fast: 7 questions better, 3 worse, 21 the same, and in all 7 better ones the fast arm also had the evidence page. So finding
  more evidence is not what changed. What changed is the `Summary:` line in the prompt (on the novel it holds names from model memory in 70% of
  cases) or chance. This test did not separate the two.
* False abstention on answerable questions: at most 1 of 31 in any arm. Correct abstention on unanswerable questions (9, 5 and 5 questions per
  set, too few to compare): docs 0.33 / 0.00 / 0.22, thesis 0.20 / 0.40 / 0.60, novel 0.60 / 0.60 / 0.60 (fast / 0.6B / 9B).

**Limits.** The sets are small (31 to 35 answerable questions), so an effect of a few points cannot be excluded either way. The rubric checks
correctness against the reference answer only; it does not check that an answer is supported by the retrieved text, and citations were not
checked. In the deep 9B arm the answering model is the same model that wrote the summaries; that was not separated either.

**Run.** On-demand L4 in Mumbai, from a copy of the disk, because the US zone was out of L4s and the Spot VM was preempted 3 minutes after it
started. It ran from 01:10 to 01:41 IST and is stopped. Storage still costs money: two 120 GB disks and one snapshot (see the end of this file).

## Far-reference pilot (2026-11-01)

A small test of "name here, `he` there" with a local 0.6B model doing the deep ingestion lives in `eval/far_test/`
([README](far_test/README.md), [results](far_test/RESULTS.md)). Three questions on the novel and one on the thesis survived the checks, so it
is a pilot only. It also holds the measurement of how often the local model names the right person on passages that do not
spell the name.

## Not run yet

- Deep ingestion with another summary format, or on a corpus where chunks look alike. (A 9B summariser was run, see above.)
- Answer quality with another answer model (for example a Gemini 3.x model), and a check that answers are supported by the retrieved text
  and that citations point to the right chunks. (Correctness with Qwen3.5-9B was run, see above.)
- Public corpora (SciFact, MultiHop, HotpotQA, FiQA, CovidQA) re-indexed with main's chunker.
- A dedicated far-reference question set. The 5-question slice above is not a valid test of it (see the caution under item 5).

## Disk and memory

The Mac has 8 GB RAM and about 7.8 GB of free disk (97% used). Postgres lives in the Docker volume (about 6.8 GB). Large
new indexes or model downloads need room first.

## Cloud resources that still cost money (2026-11-02)

Both GPU VMs are stopped, so there is no GPU charge. Storage is still billed, and nothing has been deleted:

- VM `rag-slm-gpu` (us-central1-c) with its 120 GB `pd-balanced` disk.
- VM `rag-slm-gpu-b` (asia-south1-a, Mumbai) with its 120 GB `pd-balanced` disk.
- Snapshot `rag-slm-gpu-snap` (120 GB disk, 44.1 GB stored).
