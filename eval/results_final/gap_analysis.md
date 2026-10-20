# Gap analysis (mechanical, from raw result files)

Everything below is computed from `raw_retrieval.jsonl` and `raw_generation.jsonl`; no model was called.

## A. Does merging with the word search help or hurt? (dense_only vs rrf)

Per question: position of the first correct page. 'hurt' = rrf puts it lower than dense-only, 'helped' = higher. 'lexical found nothing useful' = the word-search list had no correct page in its top 100.

| dataset | n | hurt | helped | same | lexical found nothing useful |
|---|---|---|---|---|---|
| ragbench-covidqa | 242 | 5% | 5% | 90% | 3% |
| scifact | 300 | 12% | 4% | 84% | 8% |
| multihoprag | 265 | 1% | 5% | 95% | 0% |
| fiqa | 648 | 22% | 14% | 64% | 26% |
| docsqa | 35 | 3% | 6% | 91% | 0% |
| pdf-quantum | 31 | 3% | 6% | 90% | 0% |
| pdf-got | 31 | 16% | 6% | 77% | 6% |
| hotpotqa | 300 | 0% | 1% | 99% | 0% |

## B. Reranker limits (it only reorders the top 24 candidates)

| dataset | n | correct page in top 24 (reranker can help) | correct page found but beyond rank 24 (reranker cannot help) | not in top 100 at all |
|---|---|---|---|---|
| ragbench-covidqa | 242 | 100% | 0% | 0% |
| scifact | 300 | 99% | 1% | 1% |
| multihoprag | 265 | 100% | 0% | 0% |
| fiqa | 648 | 90% | 7% | 3% |
| docsqa | 35 | 100% | 0% | 0% |
| pdf-quantum | 31 | 100% | 0% | 0% |
| pdf-got | 31 | 100% | 0% | 0% |
| hotpotqa | 300 | 100% | 0% | 0% |

## C. Where does the first correct page land? (share of questions)

| dataset | config | n | rank 1 | 2–5 | 6–24 | 25–100 | not found |
|---|---|---|---|---|---|---|---|
| docsqa | dense_only | 35 | 63% | 29% | 9% | 0% | 0% |
| docsqa | bm25_only | 35 | 63% | 31% | 3% | 3% | 0% |
| docsqa | rrf | 35 | 63% | 29% | 9% | 0% | 0% |
| docsqa | rrf_bge | 35 | 66% | 31% | 3% | 0% | 0% |
| docsqa | engine_as_shipped | 35 | 66% | 26% | 3% | 0% | 6% |
| fiqa | dense_only | 648 | 60% | 18% | 12% | 7% | 3% |
| fiqa | bm25_only | 648 | 23% | 17% | 20% | 14% | 26% |
| fiqa | rrf | 648 | 56% | 22% | 12% | 7% | 3% |
| fiqa | rrf_bge | 648 | 50% | 27% | 13% | 7% | 3% |
| hotpotqa | dense_only | 300 | 99% | 1% | 0% | 0% | 0% |
| hotpotqa | bm25_only | 300 | 95% | 4% | 1% | 1% | 0% |
| hotpotqa | rrf | 300 | 99% | 1% | 0% | 0% | 0% |
| hotpotqa | rrf_bge | 300 | 100% | 0% | 0% | 0% | 0% |
| multihoprag | dense_only | 265 | 72% | 26% | 2% | 0% | 0% |
| multihoprag | bm25_only | 265 | 74% | 23% | 3% | 0% | 0% |
| multihoprag | rrf | 265 | 74% | 25% | 2% | 0% | 0% |
| multihoprag | rrf_bge | 265 | 81% | 18% | 1% | 0% | 0% |
| multihoprag | engine_as_shipped | 265 | 75% | 23% | 1% | 0% | 1% |
| pdf-got | dense_only | 31 | 71% | 29% | 0% | 0% | 0% |
| pdf-got | bm25_only | 31 | 26% | 26% | 19% | 23% | 6% |
| pdf-got | rrf | 31 | 61% | 39% | 0% | 0% | 0% |
| pdf-got | rrf_bge | 31 | 55% | 39% | 6% | 0% | 0% |
| pdf-got | engine_as_shipped | 31 | 58% | 39% | 0% | 0% | 3% |
| pdf-quantum | dense_only | 31 | 77% | 16% | 6% | 0% | 0% |
| pdf-quantum | bm25_only | 31 | 45% | 39% | 13% | 3% | 0% |
| pdf-quantum | rrf | 31 | 77% | 16% | 6% | 0% | 0% |
| pdf-quantum | rrf_bge | 31 | 55% | 39% | 6% | 0% | 0% |
| pdf-quantum | engine_as_shipped | 31 | 55% | 29% | 3% | 0% | 13% |
| ragbench-covidqa | dense_only | 242 | 77% | 20% | 3% | 0% | 0% |
| ragbench-covidqa | bm25_only | 242 | 63% | 20% | 9% | 5% | 3% |
| ragbench-covidqa | rrf | 242 | 78% | 20% | 2% | 0% | 0% |
| ragbench-covidqa | rrf_bge | 242 | 78% | 19% | 3% | 0% | 0% |
| ragbench-covidqa | engine_as_shipped | 242 | 77% | 18% | 2% | 0% | 2% |
| scifact | dense_only | 300 | 81% | 15% | 3% | 1% | 1% |
| scifact | bm25_only | 300 | 55% | 21% | 11% | 5% | 8% |
| scifact | rrf | 300 | 74% | 21% | 4% | 1% | 1% |
| scifact | rrf_bge | 300 | 71% | 22% | 6% | 1% | 1% |
| scifact | engine_as_shipped | 300 | 72% | 20% | 3% | 0% | 6% |

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
| docsqa | bm25_only | 35 | 6% | 0% | 94% | 3% | 0% | 94% | 0% | 1/9 |
| docsqa | dense_only | 35 | 3% | 0% | 97% | 3% | 3% | 97% | 3% | 1/9 |
| docsqa | engine_as_shipped | 35 | 6% | 0% | 94% | 11% | 11% | 94% | 12% | 0/9 |
| docsqa | rrf | 35 | 3% | 0% | 97% | 3% | 3% | 97% | 3% | 1/9 |
| docsqa | rrf_bge | 35 | 3% | 0% | 97% | 3% | 3% | 97% | 3% | 1/9 |
| docsqa | rrf_bge_two_pass | 35 | 3% | 0% | 97% | 11% | 11% | 97% | 12% | 0/9 |
| multihoprag | bm25_only | 100 | 2% | 9% | 89% | 34% | 28% | 16% | 12% | 0/35 |
| multihoprag | dense_only | 100 | 2% | 5% | 93% | 34% | 29% | 18% | 11% | 0/35 |
| multihoprag | engine_as_shipped | 100 | 0% | 11% | 89% | 11% | 8% | 19% | 5% | 2/35 |
| multihoprag | rrf | 100 | 1% | 5% | 94% | 31% | 26% | 18% | 11% | 0/35 |
| multihoprag | rrf_bge | 100 | 1% | 7% | 92% | 30% | 26% | 15% | 20% | 0/35 |
| multihoprag | rrf_bge_two_pass | 100 | 1% | 7% | 92% | 9% | 5% | 15% | 0% | 2/35 |
| pdf-got | bm25_only | 31 | 45% | 3% | 52% | 48% | 6% | 16% | 0% | 0/5 |
| pdf-got | dense_only | 31 | 0% | 0% | 100% | 10% | 10% | 55% | 6% | 0/5 |
| pdf-got | engine_as_shipped | 31 | 3% | 0% | 97% | 10% | 6% | 42% | 0% | 0/5 |
| pdf-got | rrf | 31 | 0% | 0% | 100% | 3% | 3% | 52% | 0% | 0/5 |
| pdf-got | rrf_bge | 31 | 0% | 6% | 94% | 10% | 6% | 45% | 7% | 0/5 |
| pdf-got | rrf_bge_two_pass | 31 | 0% | 3% | 97% | 10% | 6% | 48% | 0% | 0/5 |
| pdf-quantum | bm25_only | 31 | 13% | 3% | 84% | 6% | 0% | 0% | – | 0/5 |
| pdf-quantum | dense_only | 31 | 3% | 6% | 90% | 3% | 0% | 0% | – | 0/5 |
| pdf-quantum | engine_as_shipped | 31 | 13% | 3% | 84% | 10% | 0% | 0% | – | 0/5 |
| pdf-quantum | rrf | 31 | 3% | 3% | 94% | 0% | 0% | 0% | – | 0/5 |
| pdf-quantum | rrf_bge | 31 | 6% | 0% | 94% | 10% | 6% | 0% | – | 0/5 |
| pdf-quantum | rrf_bge_two_pass | 31 | 6% | 0% | 94% | 10% | 6% | 0% | – | 0/5 |
| ragbench-covidqa | bm25_only | 100 | 17% | 0% | 83% | 20% | 12% | 56% | 14% | – |
| ragbench-covidqa | dense_only | 100 | 2% | 0% | 98% | 19% | 19% | 83% | 18% | – |
| ragbench-covidqa | engine_as_shipped | 100 | 2% | 0% | 98% | 20% | 19% | 84% | 18% | – |
| ragbench-covidqa | rrf | 100 | 2% | 0% | 98% | 16% | 16% | 82% | 15% | – |
| ragbench-covidqa | rrf_bge | 100 | 2% | 0% | 98% | 16% | 16% | 85% | 14% | – |
| ragbench-covidqa | rrf_bge_two_pass | 100 | 2% | 0% | 98% | 18% | 18% | 85% | 18% | – |

'Gold' = the correct page(s). 'Visible' means at least one correct page's text reached the writer after the 12,000-character evidence limit both generators apply.

## G. One-pass vs two-pass writing: refusal behaviour

| dataset | config | answerable | refused |
|---|---|---|---|
| docsqa | rrf_bge | 35 | 3% |
| docsqa | rrf_bge_two_pass | 35 | 11% |
| multihoprag | rrf_bge | 100 | 30% |
| multihoprag | rrf_bge_two_pass | 100 | 9% |
| pdf-got | rrf_bge | 31 | 10% |
| pdf-got | rrf_bge_two_pass | 31 | 10% |
| pdf-quantum | rrf_bge | 31 | 10% |
| pdf-quantum | rrf_bge_two_pass | 31 | 10% |
| ragbench-covidqa | rrf_bge | 100 | 16% |
| ragbench-covidqa | rrf_bge_two_pass | 100 | 18% |
