# Gap analysis (mechanical, from raw result files)

Everything below is computed from `raw_retrieval.jsonl` and `raw_generation.jsonl`; no model was called.

## A. Does merging with the word search help or hurt? (dense_only vs rrf)

Per question: position of the first correct page. 'hurt' = rrf puts it lower than dense-only, 'helped' = higher. 'lexical found nothing useful' = the word-search list had no correct page in its top 100.

| dataset | n | hurt | helped | same | lexical found nothing useful |
|---|---|---|---|---|---|
| ragbench-covidqa | 242 | 23% | 12% | 66% | 4% |
| scifact | 300 | 36% | 5% | 58% | 16% |
| multihoprag | 265 | 27% | 15% | 58% | 2% |
| fiqa | 648 | 56% | 11% | 32% | 41% |
| docsqa | 35 | 26% | 11% | 63% | 3% |
| pdf-quantum | 31 | 19% | 6% | 74% | 0% |
| pdf-got | 31 | 58% | 6% | 35% | 23% |
| hotpotqa | 300 | 5% | 1% | 94% | 1% |

## B. Reranker limits (it only reorders the top 24 candidates)

| dataset | n | correct page in top 24 (reranker can help) | correct page found but beyond rank 24 (reranker cannot help) | not in top 100 at all |
|---|---|---|---|---|
| ragbench-covidqa | 242 | 97% | 1% | 2% |
| scifact | 300 | 97% | 3% | 1% |
| multihoprag | 265 | 100% | 0% | 0% |
| fiqa | 648 | 87% | 8% | 5% |
| docsqa | 35 | 97% | 3% | 0% |
| pdf-quantum | 31 | 100% | 0% | 0% |
| pdf-got | 31 | 97% | 3% | 0% |
| hotpotqa | 300 | 100% | 0% | 0% |

## C. Where does the first correct page land? (share of questions)

| dataset | config | n | rank 1 | 2–5 | 6–24 | 25–100 | not found |
|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 35 | 63% | 29% | 9% | 0% | 0% |
| docsqa | bm25_only | 35 | 46% | 29% | 20% | 3% | 3% |
| docsqa | rrf | 35 | 54% | 34% | 9% | 3% | 0% |
| docsqa | rrf_bge | 35 | 63% | 29% | 6% | 3% | 0% |
| docsqa | engine_as_shipped | 35 | 66% | 20% | 6% | 0% | 9% |
| fiqa | dense_only | 648 | 60% | 18% | 12% | 7% | 3% |
| fiqa | bm25_only | 648 | 10% | 15% | 18% | 17% | 41% |
| fiqa | rrf | 648 | 32% | 28% | 27% | 8% | 5% |
| fiqa | rrf_bge | 648 | 38% | 27% | 22% | 8% | 5% |
| hotpotqa | dense_only | 300 | 99% | 1% | 0% | 0% | 0% |
| hotpotqa | bm25_only | 300 | 81% | 10% | 6% | 2% | 1% |
| hotpotqa | rrf | 300 | 95% | 5% | 1% | 0% | 0% |
| hotpotqa | rrf_bge | 300 | 98% | 1% | 0% | 0% | 0% |
| multihoprag | dense_only | 265 | 72% | 26% | 2% | 0% | 0% |
| multihoprag | bm25_only | 265 | 42% | 29% | 20% | 8% | 2% |
| multihoprag | rrf | 265 | 65% | 30% | 5% | 0% | 0% |
| multihoprag | rrf_bge | 265 | 66% | 31% | 3% | 0% | 0% |
| multihoprag | engine_as_shipped | 265 | 72% | 25% | 1% | 0% | 1% |
| pdf-got | dense_only | 31 | 71% | 29% | 0% | 0% | 0% |
| pdf-got | bm25_only | 31 | 10% | 23% | 39% | 6% | 23% |
| pdf-got | rrf | 31 | 35% | 35% | 26% | 3% | 0% |
| pdf-got | rrf_bge | 31 | 35% | 35% | 29% | 0% | 0% |
| pdf-got | engine_as_shipped | 31 | 55% | 29% | 3% | 0% | 13% |
| pdf-quantum | dense_only | 31 | 77% | 16% | 6% | 0% | 0% |
| pdf-quantum | bm25_only | 31 | 35% | 35% | 16% | 13% | 0% |
| pdf-quantum | rrf | 31 | 71% | 16% | 13% | 0% | 0% |
| pdf-quantum | rrf_bge | 31 | 61% | 26% | 13% | 0% | 0% |
| pdf-quantum | engine_as_shipped | 31 | 65% | 23% | 3% | 0% | 10% |
| ragbench-covidqa | dense_only | 242 | 77% | 20% | 3% | 0% | 0% |
| ragbench-covidqa | bm25_only | 242 | 54% | 20% | 16% | 6% | 4% |
| ragbench-covidqa | rrf | 242 | 69% | 23% | 5% | 1% | 2% |
| ragbench-covidqa | rrf_bge | 242 | 73% | 20% | 4% | 1% | 2% |
| ragbench-covidqa | engine_as_shipped | 242 | 77% | 15% | 2% | 0% | 6% |
| scifact | dense_only | 300 | 81% | 15% | 3% | 1% | 1% |
| scifact | bm25_only | 300 | 37% | 21% | 14% | 12% | 16% |
| scifact | rrf | 300 | 59% | 20% | 18% | 3% | 1% |
| scifact | rrf_bge | 300 | 62% | 20% | 15% | 3% | 1% |
| scifact | engine_as_shipped | 300 | 64% | 20% | 1% | 0% | 15% |

## D. The full system ('engine_as_shipped'): declines and shapes

| dataset | n | declined as insufficient | returned no document | needed a retry | query shapes |
|---|---|---|---|---|---|
| docsqa | 35 | 0% | 0% | 0% | factual_lookup:23, multi_hop:11, how_to:1 |
| multihoprag | 265 | 0% | 0% | 0% | multi_hop:218, factual_lookup:46, aggregation:1 |
| pdf-got | 31 | 0% | 0% | 0% | multi_hop:21, factual_lookup:10 |
| pdf-quantum | 31 | 6% | 6% | 0% | factual_lookup:15, multi_hop:13, how_to:3 |
| ragbench-covidqa | 242 | 0% | 0% | 0% | factual_lookup:208, multi_hop:22, how_to:8, aggregation:4 |
| scifact | 300 | 0% | 0% | 0% | factual_lookup:253, multi_hop:45, aggregation:1, how_to:1 |

## E. PDF case study: how many pages does a question point to?

| set | answerable questions | median pages per question | max | questions with ≥4 pages |
|---|---|---|---|---|
| pdf-got | 31 | 2 | 8 | 29% |
| pdf-quantum | 31 | 6 | 18 | 90% |

A question with many 'correct' pages is hard to score: Recall@k can never reach 1 until all of them are in the top k, and the labels are rough (see the label audit in REPORT.md §2).

## F. Where answers lose points

| dataset | config | answerable | correct page not retrieved | retrieved but cut by the 12k-character limit | correct page visible to the writer | refused to answer | refused although page visible | answered an unanswerable question |
|---|---|---|---|---|---|---|---|---|
| docsqa | bm25_only | 35 | 23% | 0% | 77% | 14% | 3% | 74% | 4% | 0/9 |
| docsqa | dense_only | 35 | 3% | 0% | 97% | 3% | 3% | 97% | 3% | 1/9 |
| docsqa | engine_as_shipped | 35 | 9% | 0% | 91% | 6% | 6% | 89% | 6% | 2/9 |
| docsqa | rrf | 35 | 11% | 0% | 89% | 3% | 0% | 86% | 0% | 0/9 |
| docsqa | rrf_bge | 35 | 9% | 0% | 91% | 0% | 0% | 91% | 0% | 0/9 |
| docsqa | rrf_bge_two_pass | 35 | 9% | 0% | 91% | 3% | 3% | 91% | 3% | 0/9 |
| multihoprag | bm25_only | 265 | 28% | 18% | 54% | 65% | 22% | 2% | 50% | 0/35 |
| multihoprag | dense_only | 265 | 1% | 8% | 91% | 32% | 27% | 20% | 13% | 0/35 |
| multihoprag | engine_as_shipped | 265 | 1% | 11% | 88% | 13% | 9% | 18% | 8% | 1/35 |
| multihoprag | rrf | 265 | 5% | 16% | 80% | 44% | 28% | 8% | 20% | 0/35 |
| multihoprag | rrf_bge | 265 | 1% | 11% | 88% | 35% | 26% | 14% | 27% | 0/35 |
| multihoprag | rrf_bge_two_pass | 265 | 1% | 11% | 88% | 15% | 9% | 15% | 10% | 0/35 |
| pdf-got | bm25_only | 31 | 65% | 3% | 32% | 68% | 3% | 3% | 100% | 0/5 |
| pdf-got | dense_only | 31 | 0% | 0% | 100% | 10% | 10% | 55% | 6% | 0/5 |
| pdf-got | engine_as_shipped | 31 | 13% | 0% | 87% | 19% | 10% | 32% | 0% | 0/5 |
| pdf-got | rrf | 31 | 29% | 0% | 71% | 29% | 3% | 26% | 0% | 0/5 |
| pdf-got | rrf_bge | 31 | 29% | 0% | 71% | 32% | 6% | 29% | 11% | 0/5 |
| pdf-got | rrf_bge_two_pass | 31 | 29% | 0% | 71% | 19% | 3% | 29% | 0% | 0/5 |
| pdf-quantum | bm25_only | 31 | 19% | 10% | 71% | 19% | 0% | 0% | – | 0/5 |
| pdf-quantum | dense_only | 31 | 3% | 6% | 90% | 3% | 0% | 0% | – | 0/5 |
| pdf-quantum | engine_as_shipped | 31 | 10% | 3% | 87% | 19% | 10% | 0% | – | 0/5 |
| pdf-quantum | rrf | 31 | 13% | 0% | 87% | 3% | 0% | 0% | – | 0/5 |
| pdf-quantum | rrf_bge | 31 | 10% | 3% | 87% | 10% | 6% | 0% | – | 0/5 |
| pdf-quantum | rrf_bge_two_pass | 31 | 10% | 0% | 90% | 16% | 13% | 0% | – | 0/5 |
| ragbench-covidqa | bm25_only | 242 | 23% | 0% | 77% | 27% | 11% | 43% | 12% | – |
| ragbench-covidqa | dense_only | 242 | 2% | 0% | 98% | 18% | 17% | 81% | 14% | – |
| ragbench-covidqa | engine_as_shipped | 242 | 6% | 0% | 94% | 19% | 17% | 62% | 16% | – |
| ragbench-covidqa | rrf | 242 | 6% | 0% | 94% | 23% | 19% | 55% | 15% | – |
| ragbench-covidqa | rrf_bge | 242 | 6% | 0% | 94% | 18% | 16% | 57% | 12% | – |
| ragbench-covidqa | rrf_bge_two_pass | 242 | 6% | 0% | 94% | 19% | 17% | 57% | 12% | – |

'Gold' = the correct page(s). 'Visible' means at least one correct page's text reached the writer after the 12,000-character evidence limit both generators apply.

## G. One-pass vs two-pass writing: refusal behaviour

| dataset | config | answerable | refused |
|---|---|---|---|
| docsqa | rrf_bge | 35 | 0% |
| docsqa | rrf_bge_two_pass | 35 | 3% |
| multihoprag | rrf_bge | 265 | 35% |
| multihoprag | rrf_bge_two_pass | 265 | 15% |
| pdf-got | rrf_bge | 31 | 32% |
| pdf-got | rrf_bge_two_pass | 31 | 19% |
| pdf-quantum | rrf_bge | 31 | 10% |
| pdf-quantum | rrf_bge_two_pass | 31 | 16% |
| ragbench-covidqa | rrf_bge | 242 | 18% |
| ragbench-covidqa | rrf_bge_two_pass | 242 | 19% |
