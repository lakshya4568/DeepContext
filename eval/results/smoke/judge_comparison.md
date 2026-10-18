# Judge comparison (same 12 rows scored by every judge)

| judge | est. list-price cost (rows scored) | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness_f1 |
|---|---|---|---|---|---|---|
| gemini-2.5-pro | $1.92 (12 rows, 172,618 out-tokens) | 0.643 (n=12) | 0.633 (n=12) | 0.681 (n=12) | 0.833 (n=12) | 0.355 (n=12) |
| gemini-2.5-pro@think128 | $0.52 (12 rows, 32,272 out-tokens) | 0.619 (n=12) | 0.612 (n=12) | 0.715 (n=12) | 0.833 (n=12) | 0.459 (n=12) |
| gemini-3.7-flash | $0.33 (12 rows, 56,795 out-tokens) | 0.903 (n=12) | 0.619 (n=12) | 0.694 (n=12) | 0.750 (n=12) | 0.429 (n=12) |

**gemini-2.5-pro@think128 vs gemini-2.5-pro** — mean |diff| / Pearson r over rows
- faithfulness: 0.024 / 0.99 (n=12); answer_relevancy: 0.024 / 0.99 (n=12); context_precision: 0.132 / 0.66 (n=12); context_recall: 0.167 / 0.40 (n=12); factual_correctness_f1: 0.136 / 0.76 (n=12)

**gemini-3.7-flash vs gemini-2.5-pro** — mean |diff| / Pearson r over rows
- faithfulness: 0.288 / 0.38 (n=12); answer_relevancy: 0.016 / 1.00 (n=12); context_precision: 0.014 / 0.99 (n=12); context_recall: 0.250 / 0.26 (n=12); factual_correctness_f1: 0.124 / 0.78 (n=12)

Small-n caution: these rows come from one dataset's smoke batch; correlations on ~12 rows are indicative only.
