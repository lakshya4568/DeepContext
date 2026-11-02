# Variant B pilot on the project docs set (2026-11-01)

> **Update, same day:** the causes of the loss below were measured and the design was reworked and tested on three corpora;
> see `FINDINGS_WHY_LOWER.md`. The tables here describe the first design only (`summary_v1`).

Everything below was measured in this session; files are in `eval/results_variant_b/` (local, git-ignored). The summariser is
**Qwen3.5-2B (bf16) served by vLLM on one Spot NVIDIA L4**, thinking off. Variant A is parked. This is one small dataset and
a first run: it shows the pipeline works end to end and gives a first, unflattering number. It is not a verdict on variant B.

## 1. GPU smoke test (50 real chunks: 40 project docs, 10 SciFact)

| Check | Result |
|---|---|
| Both models on one L4 | vLLM 12.4 GB + reranker = 14.9 GB of 23 GB used |
| Summary format (2-4 bullets) / length / grounded, temperature 0.2 | 49 / 50, 47 / 50, 48 / 50 |
| Same, model-card sampling (temperature 1.0, presence penalty 2.0) | 43 / 50, 50 / 50, 42 / 50 (worse, so 0.2 is used) |
| Throughput on this small batch (concurrency 32) | about 5,400 input and 590 output tokens per second (includes ramp-up) |
| Reranker, 20 docs questions, 100 candidates each | Hit@1 0.75, Hit@5 0.95, MRR 0.82; scores in [0, 1]; 0.67 s per call (median) |

Not done: score parity against the earlier Colab reranker (the cache stores hashes, not the query and document text).

## 2. Pipeline run on the docs set (193 child chunks, 12 source files, profile `technical_docs`, static)

| Item | Result |
|---|---|
| Summaries | 183 ok, 6 retried once, 4 fell back to the chunk's first sentences (2.1%); the fallbacks are stored but not embedded |
| First try accepted | 94.8% (rejections: 6 too long, 2 wrong format, 2 terms not in the source) |
| Mean summary length | 57 words |
| Glossary entries | 810 proposed, 689 kept (85%): 97 dropped because the quote is not in the text, 15 because no name occurs in it, 9 empty |
| Entries after merge | 335 across 12 files; 185 of 193 chunks received at least one entry |
| Broken JSON | 2 of about 50 passages gave no entries even after a retry |
| Parent context | only 19 of 193 chunks (a docs section is its own parent; there is nothing around it) |
| Cost of the first run | 439 model calls, 238k prompt and 21k completion tokens, 165 s wall |

Bugs found and fixed on the way: the grounding check flagged model names, paths and ratios that the source does contain
(first run: 26 fallbacks); the glossary merged unrelated entries through shared generic words such as "context".

Glossary quality with the 2B model is mixed. Good entries exist, but some are vague or mis-typed (a file name typed as a
"command"), and a few lump unrelated names together; the code now ignores generic words for merging and matching.

## 3. Retrieval, fast ingestion vs variant B (35 answerable questions, same questions, paired)

Variant B changes only the text each child vector embeds (`summary + child`). BM25 and the chunks are untouched, which the
identical `bm25_only` rows confirm. The fast baseline is `eval/results_final`; both use the fixed-search settings.

| Setup | nDCG@10 fast | nDCG@10 variant B | Difference, paired 95% CI | Questions better / worse / same |
|---|---|---|---|---|
| dense only | 0.824 | 0.753 | -0.070 [-0.132, -0.015] | 4 / 11 / 20 |
| RRF hybrid | 0.826 | 0.770 | -0.056 [-0.117, +0.002] | 3 / 10 / 22 |
| RRF + bge rerank | 0.825 | 0.813 | -0.011 [-0.048, +0.018] | 2 / 4 / 29 |
| lexical only | 0.799 | 0.799 | 0 (as expected) | 0 / 0 / 35 |

Hit@5 is unchanged except one question after reranking. Full table: `eval/results_variant_b/retrieval_docsqa/fast_vs_vb.md`
(16 comparisons, none corrected for multiple testing).

**Reading it.** On this set variant B did not help; it slightly hurt the dense channel and the loss mostly disappears after
reranking. Most rank changes are one or two places (one question fell from 6 to 17). Plausible reasons, none tested here:
docs sections are short and self-contained, so a summary adds little; a summary may dilute exact identifiers such as config
keys in the embedding; the 2B summariser and its noisy glossary may be too weak.

## 4. What this does not cover

Long documents (the thesis, the novel, MultiHop-RAG), where context should matter most, have not been run. No ablations yet
(child only, summary only, summary plus child; glossary on or off; a larger summariser). Generation quality, RAGAS and the
rubric were not run. Speed at scale is extrapolated, not measured.
