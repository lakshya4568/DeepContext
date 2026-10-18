# Judge comparison (same 0 rows scored by every judge)

| judge | est. list-price cost (rows scored) | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness_f1 |
|---|---|---|---|---|---|---|
| gemini-2.5-pro | $1.92 (12 rows, 172,618 out-tokens) | n/a | n/a | n/a | n/a | n/a |
| gemini-3.7-flash | $0.33 (12 rows, 56,795 out-tokens) | n/a | n/a | n/a | n/a | n/a |
| gemini-2.5-pro@think128 | $2.26 (146 rows, 136,714 out-tokens) | n/a | n/a | n/a | n/a | n/a |

**gemini-3.7-flash vs gemini-2.5-pro** — mean |diff| / Pearson r over rows
- faithfulness: nan / nan (n=0); answer_relevancy: nan / nan (n=0); context_precision: nan / nan (n=0); context_recall: nan / nan (n=0); factual_correctness_f1: nan / nan (n=0)

**gemini-2.5-pro@think128 vs gemini-2.5-pro** — mean |diff| / Pearson r over rows
- faithfulness: nan / nan (n=0); answer_relevancy: nan / nan (n=0); context_precision: nan / nan (n=0); context_recall: nan / nan (n=0); factual_correctness_f1: nan / nan (n=0)

Small-n caution: these rows come from one dataset's smoke batch; correlations on ~12 rows are indicative only.
