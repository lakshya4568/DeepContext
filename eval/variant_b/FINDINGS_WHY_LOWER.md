# Why the first deep-ingestion run (variant B, v1) scored lower than fast ingestion

Research-paper observation, measured on 2026-11-01. Every number below comes from files produced by the commands at the
end; nothing is estimated. Variant A is parked. "Fast" = child chunk embedded alone. "Deep v1" = child vector made from
`summary + child`, summaries by Qwen3.5-2B (see `PILOT_DOCS.md`).

**Status of the data.** The project-docs set (`docsqa`, 35 answerable questions) is the **development set** for variant B:
it was used to find and fix the problems below, so it must not be quoted as a held-out test. The thesis PDF was a held-out
test once (section 8) and then became a second development set. The novel PDF was never used during development and is the
clean test of the final 2B design (section 10). Public benchmarks (SciFact, CovidQA, MultiHop-RAG, FiQA, HotpotQA) have not
been run with variant B.

**Setup caveats (added 2026-11-01).** (1) All numbers in this file were measured on Homebrew PostgreSQL 15.18 with
pgvector 0.8.6; the benchmark has since moved to the Docker ParadeDB server (`eval/DATABASE.md`), where the dense first
stage reproduces the published baseline to four decimals (parity control in `eval/results_paradedb/`, local). (2) The code
under test is the branch `feat/rag-quality-generation`, which is **26 commits behind `main`**: it predates the application's
own LLM summary pipeline, its ParadeDB search code and its tuned HNSW index. "Deep v1" here is the harness's own
re-implementation of summary-prefixed embedding, not `main`'s pipeline. See `eval/PARADEDB_USECASES.md`.

## 1. The observation

| Setup (docs set, nDCG@10) | fast | deep v1 | change (paired 95% CI) | questions better / worse |
|---|---|---|---|---|
| dense only | 0.824 | 0.753 | -0.070 [-0.132, -0.015] | 4 / 11 |
| RRF hybrid | 0.826 | 0.770 | -0.056 [-0.117, +0.002] | 3 / 10 |
| RRF + bge reranker | 0.825 | 0.813 | -0.011 [-0.048, +0.018] | 2 / 4 |
| lexical only (BM25) | 0.799 | 0.799 | 0 | 0 / 0 |

Exact cosine over the stored vectors reproduces the stored dense run for 35 of 35 questions, so the analysis below
describes the real runs.

## 2. What explains it (measured)

1. **Only the dense vectors changed, and the reranker hides most of it.** Chunks, BM25 column and reranker model are
   identical. RRF follows dense almost exactly (lexical weight 0.1: RRF top-1 equals dense top-1 for 34/35 fast and 33/35
   deep questions). The reranker recovers 80% of the hybrid loss (0.056 to 0.011), but it does not make the result
   independent of the embedding: the reranked top-24 differs for all 35 questions (mean overlap 0.72) because the final
   score blends the reranker score with the first-stage rank. The reranker reads only the chunk text, so it cannot use
   anything that is in the summary.
2. **Extra text hurts even when it carries no information.** Repeating the chunk (`chunk + chunk`) lowers dense nDCG@10 by
   0.040 [-0.082, -0.006]: more than half of the 0.070 loss. Prefixing a *different* chunk's summary lowers it by 0.289, so
   the real summaries are relevant to their chunk; they are simply not additive.
3. **The generated text is large next to the chunk.** Median chunk: 95 words, and 102 of 193 chunks have under 100 words.
   Median summary: 56 words. The summary is longer than half its chunk for 111 of 193 chunks and longer than the chunk
   itself for 36. For over half the chunks the vector is mostly summary.
4. **Summaries borrow from neighbours.** 65 of 189 embedded summaries (34%) contain a name or number that is not in their own
   chunk. Source of those tokens: previous summaries 136, document synopsis 92, glossary 32, parent text 5. The grounding
   check let them through because it accepted anything in the prompt. Six of the 11 worse questions now rank a sibling
   section of the same file first.
5. **Both fixes together remove nearly all of the loss.** Using the summary only when it borrowed nothing and the chunk has
   at least 100 words: -0.005 [-0.044, +0.029] (Hit@5 0.943 vs 0.914). Either fix alone: -0.033 (clean only), -0.039 (long
   only). Selected on the dev set; see section 4.
6. **The summary restates the chunk, so it adds no search words.** The share of a question's content words found in the gold
   chunk is 43%; in the gold summary 34%; in the section title 17% (the fast vector already includes the title). The summary
   alone retrieves nearly as well as the chunk (0.745 vs 0.824), so it is a faithful second view, but a redundant one:
   averaging the two vectors (chunk weight 0.85) gives -0.004 and two-vector rank fusion -0.011 to -0.016.
7. **The corpus is a tight-margin case.** The mean cosine gap between the gold document and the best other document is only
   0.019; deep v1 shrinks it to 0.016 (smaller for 17 of 35 questions). Gold similarity falls by 0.0069 and the best
   competitor's by 0.0036. Nearest-neighbour similarity between chunks rises from 0.857 to 0.867 (more crowding). 39% of the
   summary words are not in the chunk, and many summaries open alike ("The system" 39 times).
8. **No question type benefits.** Change in dense nDCG@10: direct -0.038 (n=15), paraphrased -0.052 (8), multi-fact -0.097
   (9), false-premise -0.198 (3).

## 3. Hypotheses checked

| Hypothesis | Verdict |
|---|---|
| The reranker causes the drop | No: same model, same text. It reduces the drop; the cause is the first stage. |
| The embedding model changes behaviour between fast and deep | Partly: it is the same model, but a longer, mixed input moves the vector and flips near-tied ranks. |
| Extra length alone hurts | Supported (-0.040 for a pure repeat). |
| Neighbour/synopsis leakage hurts | Supported (clean-only recipe recovers about half the loss; with the length rule nearly all). |
| Questions copy the chunk wording, so summaries cannot help | Not supported (only 43% word overlap; paraphrased questions fell too). |
| Weak summariser quality is the cause | Untested: needs a stronger model on the same chunks. |
| Contextual summaries help where chunks lack context | Untested here: docs sections are their own parents (parent text for 19 of 193 chunks). |

## 4. Gaps in the v1 pipeline and the evaluation

Pipeline: no length budget relative to the chunk; previous-summary chain and document synopsis leaked other chunks'
content; grounding check too lenient (any slot counted as support); "Unclear:" comments (9 of 189) and generic openings add
noise; glossary extracted by the 2B model was noisy (15% of entries dropped for unverifiable quotes, unrelated names lumped
together, merges on generic words, broken JSON on 2 passages); the summary asks to restate the chunk, which cannot add
information a chunk already has.

Evaluation: one small set of 35 questions with wide intervals; 16 comparisons, none corrected for multiple testing; the
questions were written by an AI from the docs and are not human-reviewed; the corpus (short self-contained sections whose
title is already in the vector) is the least favourable case for contextual enrichment; the deep-ingestion premise (long
documents) was not exercised; the recipe grid was scored on the same 35 questions it is judged on.

## 5. How to reproduce

```
PYTHONPATH=. uv run --no-sync python -m eval.variant_b.diagnose --dataset docsqa --fast eval/results_final/raw_retrieval.jsonl --vb eval/results_variant_b/retrieval_docsqa/raw_retrieval.jsonl
PYTHONPATH=. uv run --no-sync python -m eval.variant_b.recipes --dataset docsqa --enrich b_v1=eval/results_variant_b/docsqa/summaries.jsonl --flags eval/results_variant_b/docsqa/diagnostics.json --flags-for b_v1
```
Outputs (local, git-ignored): `eval/results_variant_b/docsqa/diagnostics.json`, `recipes.json`, `recipes.md`.

## 6. Recipe grid on the existing summaries (dense only, docs set, 35 questions; change in nDCG@10 vs fast)

| Recipe | nDCG@10 | change (95% CI) | Hit@5 | better / worse |
|---|---|---|---|---|
| fast | 0.824 | 0 | 0.914 | 0 / 0 |
| summary + chunk (deep v1) | 0.753 | -0.070 [-0.132, -0.015] | 0.914 | 4 / 11 |
| chunk + summary | 0.775 | -0.049 [-0.088, -0.015] | 0.914 | 1 / 8 |
| first bullet + chunk | 0.794 | -0.030 [-0.086, +0.024] | 0.971 | 5 / 7 |
| summary + chunk, long chunks only | 0.784 | -0.039 [-0.089, +0.005] | 0.943 | 2 / 7 |
| summary + chunk, clean summaries only | 0.790 | -0.033 [-0.081, +0.010] | 0.914 | 3 / 7 |
| summary + chunk, clean and long only | 0.819 | -0.005 [-0.044, +0.029] | 0.943 | 2 / 3 |
| summary alone | 0.745 | -0.079 [-0.161, -0.001] | 0.886 | 5 / 13 |
| vector average, chunk weight 0.85 / 0.70 / 0.50 | 0.820 / 0.789 / 0.785 | -0.004 / -0.035 / -0.038 | 0.914 | 0-2 / 3-7 |
| two vectors, rank fusion weight 1.0 / 0.5 | 0.808 / 0.813 | -0.016 / -0.011 | 0.943 / 0.971 | 5 / 7, 4 / 5 |
| control: another chunk's summary + chunk | 0.535 | -0.289 [-0.371, -0.212] | 0.743 | 0 / 26 |
| control: chunk repeated | 0.784 | -0.040 [-0.082, -0.006] | 0.943 | 1 / 7 |

No recipe built from the v1 summaries beats fast on nDCG@10. The summaries restate the chunk, so the next step is
enrichment that carries information the chunk lacks (section 7, added after the new runs).

## 7. Enhancements tried on the dev set (docs, 35 questions, 2026-11-01)

What changed (all in `eval/variant_b/`): no previous-summary chain and no synopsis in the strict modes; grounding against the
chunk, title and glossary names only; plain sentences instead of bullets; no comments about what the passage leaves out;
four generation modes instead of one (`summary_v2` restated note, `context` situating line, `questions`, `keywords`);
a recipe for where the generated text goes and a minimum chunk length; contextual BM25 as an option. The 2B model on the
GPU produced each mode for all 193 chunks in 16 to 23 seconds.

**Selection rule, written down before the grid was scored.** From the new modes, take the recipe with the highest dense
nDCG@10 on the dev set whose Hit@5 is at least fast's; if two are within 0.01, take the simpler (no length rule). That recipe
is the only one run on held-out data; everything else is reported as exploratory.

Result of the grid (dense only; full table in `eval/results_variant_b/docsqa/recipes_all.md`; change in nDCG@10 vs fast):

| Generated text | prefix | suffix | other recipes worth noting |
|---|---|---|---|
| v1 summary (bullets, old) | -0.070 | -0.049 | clean and long only: -0.005 |
| `summary_v2` (restated note) | +0.002 | +0.015 | two vectors, rank fusion 0.5: +0.018 |
| `context` (situating line) | **+0.036** [-0.010, +0.085], 7 better / 2 worse, Hit@5 0.971 | +0.018 | long chunks only: +0.040 [+0.003, +0.085]; two vectors 0.5: +0.019 |
| `questions` | -0.030 | -0.040 | first question only, long chunks: +0.012 (Hit@5 1.000) |
| `keywords` (40% of chunks fell back) | -0.028 | -0.055 | vector average 0.85: +0.005 |

Chosen by the rule: **`context`, prefix, all chunks** (the long-chunk variant is within 0.01).

Through the real pipeline (new schema `docsqa_7c20e221ae_vbctx`, four setups, paired against fast):

| Setup | nDCG@10 fast | with context lines | change (95% CI) | MRR change | Hit@5 fast / new |
|---|---|---|---|---|---|
| dense only | 0.824 | 0.859 | +0.036 [-0.010, +0.085] | +0.057 [+0.007, +0.118] | 0.914 / 0.971 |
| RRF hybrid | 0.826 | 0.858 | +0.032 [-0.011, +0.076] | +0.052 [+0.006, +0.106] | 0.914 / 0.943 |
| RRF + bge reranker | 0.825 | 0.828 | +0.003 [-0.030, +0.034] | +0.016 [-0.007, +0.050] | 0.971 / 0.943 |
| lexical only | 0.799 | 0.799 | 0 | 0 | 0.943 / 0.943 |

Seven of 35 questions improve and two get worse in the first stage (a false-premise question about the ONNX BGE switch falls
from rank 6 to 12, as it did under v1); after the reranker the picture is 3 better, 3 worse, 29 unchanged. One question
leaves the top 10 in every setup (Recall@10 1.000 to 0.971).

Further studies, same dev set:

| Study | Result (nDCG@10 vs fast, rerank setup) |
|---|---|
| Reranker also reads the context line, context vectors | 0.831 (+0.006 [-0.046, +0.064]) |
| Reranker also reads the context line, fast vectors | 0.822 (-0.003 [-0.042, +0.033]) |
| Contextual BM25 (context text added to the keyword column) | lexical only 0.799 (+0.001); hybrid 0.860 (+0.033); reranked 0.829 (+0.005) |

**What this says.** The change that matters is what the generated text contains. Restating the chunk (v1, `summary_v2`) is
neutral to harmful; a short line that situates the chunk (`context`) improves the first stage by about 0.03 to 0.04 nDCG@10
and 0.05 MRR on this set. The gain does not survive the reranker in this corpus: the reranker reads the chunk only, and giving
it the context line does not change that (+0.006). The shipped pipeline ends with the reranker, so on the docs set the
end-to-end result is a tie with fast, not an improvement. The interval for the first-stage gain includes zero for nDCG@10,
the grid has about 70 cells scored on 35 questions, and the question set is the one used to pick the recipe, so none of this
counts as evidence of an improvement until the held-out result below.

## 8. First held-out test: the chosen recipe on the thesis PDF (it did not generalise)

The recipe picked on the docs set (`context`, prefix) was run once, unchanged, on the quantum thesis (31 answerable
questions, 472 chunks, one page per document; strict page labels, and labels widened to plus or minus one page because the
page labels are imprecise). The fast baseline is `eval/results_final`.

| Setup | nDCG@10 fast | with `context` | change (95% CI) | MRR change | questions better / worse |
|---|---|---|---|---|---|
| dense only | 0.518 | 0.481 | -0.038 [-0.080, +0.004] | -0.058 | 6 / 18 |
| RRF hybrid | 0.537 | 0.499 | -0.039 [-0.075, -0.004] | -0.031 | |
| RRF + bge reranker | 0.478 | 0.463 | -0.015 [-0.045, +0.017] | +0.028 | 7 / 11 |
| lexical only | 0.405 | 0.405 | 0 | 0 | 0 / 0 |
| dense only, +/-1 page | 0.505 | 0.456 | -0.049 [-0.090, -0.013] | -0.070 | |

Hit@5 is unchanged. One paraphrase question falls from rank 14 to 48 (dense) and from 7 to 38 (reranked). So the first-stage
gain seen on the docs set was a development-set effect (35 questions, about 70 recipes tried); it does not appear on a second
corpus and the direction is the opposite.

**Why it failed there (measured, no tuning on these numbers).**

- *Not* length: thesis chunks are long (median 342 words, context lines 27); the context line is longer than half the chunk for
  6 of 468 chunks.
- *Not* crowding: the vectors became less alike (mean pairwise cosine 0.828 to 0.813; nearest-neighbour 0.945 to 0.935).
- *Document-wide topic painted on every chunk.* The prompt contained the whole-thesis synopsis, and 59% of the words in the
  context lines are not in their chunk. The thesis' topic words appear in most lines: "atomic" and "storage" in 79%,
  "photon" in 77%, "ensembles" in 72%, "thesis" in 44%, "Harvard" in 37%. The chunks themselves use "storage" and "atomic" in
  about 60%. An acknowledgements page was described as covering "quantum communication methods using gradient ascent
  optimization". Wrong context pulls unrelated pages toward topical questions.
- *Formulaic openings.* "This passage describes" opens 186 of 472 lines, "This passage from ..." 173, "This Harvard PhD
  thesis ..." 68: shared words that carry no information about the chunk.

## 9. Second design: `context_v2` (local information only), dev and thesis

Changes: no synopsis in the prompt (title, glossary, surrounding parent text and the passage only), an instruction not to
describe the whole document or to start with "This passage", a rule for front matter and appendices, and a deterministic strip
of formulaic openings. The thesis is no longer a held-out set for this design: it was used to find the problem and to test the fix.

| Setup | docs dev: change in nDCG@10 (95% CI) | thesis: change in nDCG@10 (95% CI) |
|---|---|---|
| dense only | -0.004 [-0.057, +0.048] | -0.024 [-0.055, +0.010] |
| RRF hybrid | +0.009 [-0.035, +0.053] | -0.020 [-0.047, +0.010] |
| RRF + bge reranker | -0.007 [-0.031, +0.008] | +0.004 [-0.015, +0.024] |
| thesis, +/-1 page, dense / RRF / reranked | | -0.021 / -0.019 / +0.001 |

Removing the global topic halves the thesis loss and removes the docs gain: the version is now neutral, not better.
Generation was fast and clean on the 2B model: 188 ok, 2 retried, 3 fallback of 193 chunks (docs); 459, 6, 7 of 472 (thesis);
mean 20 words.

## 10. Clean held-out test of the final 2B design: the novel (never used during development)

`context_v2`, prefix, Qwen3.5-2B, run once on the novel (31 answerable questions, 978 chunks, one page per document).
Generation was harder here: 753 ok, 119 retried, 106 fallback of 978 chunks (10.8% fallback; 77% accepted at the first
try), because the names the model wants to use are often not in the chunk, title or glossary.

| Setup | nDCG@10 fast | with `context_v2` | change (95% CI) | +/-1 page change |
|---|---|---|---|---|
| dense only | 0.779 | 0.737 | -0.042 [-0.094, +0.010] | -0.020 |
| RRF hybrid | 0.722 | 0.701 | -0.021 [-0.062, +0.024] | -0.014 |
| RRF + bge reranker | 0.648 | 0.661 | +0.013 [-0.035, +0.060] | +0.007 |
| lexical only | 0.323 | 0.323 | 0 | 0 |

Hit@5 is unchanged except dense (1.000 to 0.968, one question). The pattern matches the thesis: slightly worse in the first
stage, a tie after the reranker. **Final result for the 2B design: no improvement over fast ingestion on any of the three
corpora.** Across the three datasets the reranked change is -0.007 (docs, dev), +0.004 (thesis) and +0.013 (novel); all
intervals include zero.

## 11. Is the summariser too small? Protocol written before the run

Hypothesis left untested above: a larger model writes better context lines. Test: Qwen3.5-9B (FP8, same server settings,
thinking off), same prompts and checks, `context_v2`, glossary re-extracted with the same 9B model. Run on the docs
(dev) and the thesis. The novel is run with the 9B only if the 9B improves the reranked nDCG@10 on both by at least +0.01 and
the dense nDCG@10 is not lower on either; otherwise the result is reported as negative.

Result (run 2026-11-01; docs: change in nDCG@10 vs fast, 95% CI; thesis: strict labels, then plus or minus one page):

| Setup | docs | thesis strict | thesis +/-1 page |
|---|---|---|---|
| dense only | +0.013 [-0.007, +0.041] | +0.013 [-0.029, +0.059] | -0.009 |
| RRF hybrid | +0.019 [-0.007, +0.053] | +0.011 [-0.023, +0.045] | -0.002 |
| RRF + bge reranker | +0.007 [-0.003, +0.021] | -0.001 [-0.025, +0.021] | -0.005 |

Generation: docs 189 ok, 1 retried, 3 fallback (243 s including a new glossary); thesis 432 ok, 9 retried, 31 fallback (6.6%;
867 s). Reranked Hit@5 falls by one to two questions on both. **The condition for running the 9B on the novel was not met**
(reranked +0.007 and -0.001, needed +0.01 on both), so it was not run. The larger model removes the first-stage harm the
2B caused on the thesis (-0.024 to +0.013) but produces no gain after the reranker. Summariser size matters a little; it
does not change the end result.

## 12. How the embedding model and the reranker shape the result, by setup

Change in nDCG@10 against fast ingestion (95% intervals all include zero except where noted in sections 1, 7 and 8):

| Experiment | dense only | RRF hybrid | RRF + reranker |
|---|---|---|---|
| docs, v1 bullet summary (2B) | -0.070 | -0.056 | -0.011 |
| docs, `context` with synopsis (2B) | +0.036 | +0.032 | +0.003 |
| docs, `context_v2` (2B) | -0.004 | +0.009 | -0.007 |
| docs, `context_v2` (9B) | +0.013 | +0.019 | +0.007 |
| thesis, `context` with synopsis (2B) | -0.038 | -0.039 | -0.015 |
| thesis, `context_v2` (2B) | -0.024 | -0.020 | +0.004 |
| thesis, `context_v2` (9B) | +0.013 | +0.011 | -0.001 |
| novel, `context_v2` (2B), clean test | -0.042 | -0.021 | +0.013 |

- **Embedding model (gemini-embedding-2, one vector per chunk).** It is the only component whose input changes. Any added
  text moves the vector (cosine to the fast vector 0.95 to 0.96 on average, lowest 0.79); when the best and second-best
  documents are only about 0.02 apart in cosine (docs set), a small shift flips ranks. Repeating the chunk with no new
  information already costs 0.040 nDCG@10. A single vector must carry both the chunk and its added context, so extra text
  can only dilute unless it is short and distinguishing.
- **Fusion.** The lexical channel is unchanged (identical BM25 rows in every experiment) and carries weight 0.1 in RRF, so the
  hybrid result follows the dense result (same top-1 as dense for 34 of 35 questions on the docs set). Adding the generated text
  to the keyword column (contextual BM25) changed nothing on the docs set (+0.001).
- **Reranker (bge-reranker-v2-m3, same model for both).** It reads the chunk text only, so it neither sees nor can exploit
  enrichment. It removes most of the first-stage change in both directions: 80% of the hybrid loss on the docs v1 run, 62% on
  the thesis, and the whole first-stage gain of the docs `context` run (+0.036 to +0.003). It does not make the final list
  independent of the embedding: the reranked top-24 differs for every question (mean overlap 0.72, docs v1) because the final
  score blends the reranker with the first-stage rank. Showing it the context line as well did not help (+0.006 and -0.003).
- **Consequence for the architecture.** With a strong reranker and title-prefixed vectors, retrieval is already at Hit@5 of
  0.97 (docs), 0.94 (thesis) and 0.94 (novel); the remaining misses are mostly rank-within-top-5 and label ambiguity, which
  chunk-level enrichment did not move.

## 13. Gaps and what was done about each

| Gap | Evidence | Status |
|---|---|---|
| Generated text too long next to the chunk | median chunk 95 words, summary 56; longer than half the chunk for 111 of 193 | fixed: 35 to 40 word limits, optional minimum-length rule |
| Neighbour summaries chained into the prompt | 136 borrowed tokens | fixed: chain only in `summary_v1` |
| Document synopsis painted onto every chunk | thesis: topic words in 40 to 80% of lines, wrong lines for unrelated pages | fixed in `context_v2` (no synopsis) |
| Grounding check accepted words from any slot | 34% of v1 summaries borrowed | fixed for the v2 modes (chunk, title, glossary names, optional parent text) |
| Comments such as "Unclear:" and "does not specify" | 9 of 189 v1 summaries | fixed: rejected by the check |
| Formulaic openings ("This passage describes") | 186 and 173 of 472 thesis lines | fixed: stripped deterministically |
| Glossary noise from the 2B model | 85% (docs), 55% (thesis), 73% (novel) of entries survive the quote check; lumped names; broken JSON | partly fixed (generic-word merges removed, JSON retry); entries never checked by a human |
| Reranker reads only the chunk | context line given to it: +0.006 and -0.003 | measured, no gain; design left as is |
| One vector per chunk | two-vector rank fusion +0.018 to -0.016 (docs, offline) | not built (needs a multi-vector store) |
| Summariser too small | 9B: +0.013 / -0.001 (docs / thesis, reranked +0.007 / -0.001) | tested, no end-to-end gain |
| Small, AI-written dev questions; uncorrected comparisons; imprecise PDF page labels | 35 questions, about 70 recipes | open; plus or minus one page also reported |
| Spot GPU preempted twice | 43 min and 90 s of run time | fixed: VM switched to on-demand |
| Generation quality (answers, RAGAS) with deep ingestion | not run | open |
| Public benchmarks and the long multi-document sets with deep ingestion | not run | open |

## 14. What the paper can and cannot say from this

Can say (measured): the first deep-ingestion design lost 0.070 nDCG@10 (dense) on the docs set and the loss is explained by
text length, other chunks' words in the summary, and a summary that restates the chunk; a document-level context line paints
the global topic on every chunk and hurts a thesis; local-only context lines of 20 words are neutral on a docs set, a thesis
and a novel, and a 9B summariser is also neutral end to end; the reranker removes most of the first-stage differences (80% and 62% of the loss in the two runs where
that can be computed, all of the gain in a third) and cannot use the enrichment; none of this improves the final retrieval of a pipeline that already has title-prefixed vectors and
a bge reranker.

Cannot say: that deep ingestion improves retrieval (no test showed it); anything about answer quality, RAGAS or latency with
deep ingestion; anything about the public benchmarks; that the result holds for a stronger embedding-free setup, a
multi-vector store or a reranker that reads the context. Treat the docs numbers as development results.

Measured cost and speed of the 2B/9B summariser on one L4 (list prices): generation speed 1.3 to 1.5 thousand tokens per
second at 40 to 47 concurrent requests; the docs set (193 chunks) takes 15 to 23 seconds per mode with a cached glossary;
the thesis (472 chunks, glossary included) 7 minutes with the 2B and 14 minutes with the 9B; the novel (978 chunks) 10
minutes with the 2B. The whole study used about 1.8 hours of VM time (0.72 h and 0.02 h on Spot, then 1.04 h on-demand), about $1.3 at
list prices; the VM is now stopped (disk kept, about $0.40 a day).
