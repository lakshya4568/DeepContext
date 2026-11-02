# Near-duplicate filter and mixed search (deep ingestion)

## Table 1: dense only against mixed search (keyword vote 0.02), same questions

| set | questions | Recall@100 dense | Recall@100 mixed | documents returned, dense | documents returned, mixed |
|---|---|---|---|---|---|
| ragbench-covidqa | 242 | 0.992 | 0.874 | 100.0 | 85.6 |
| scifact | 0 | n/a | n/a | n/a | n/a |
| multihoprag | 265 | 0.980 | 0.980 | 20.4 | 20.3 |
| hotpotqa | 0 | n/a | n/a | n/a | n/a |
| fiqa | 0 | n/a | n/a | n/a | n/a |
| docsqa | 35 | 1.000 | 1.000 | 74.7 | 74.7 |
| pdf-quantum | 31 | 0.619 | 0.619 | 54.7 | 54.7 |
| pdf-got | 31 | 0.995 | 0.995 | 64.0 | 64.0 |

## Table 2: CovidQA, where the lost documents went

- CovidQA has 902 documents; 251 of them (28%) share their first 120 characters with at least one other document.
- Relevant documents that dense search returned and mixed search did not: 85; of these, a twin ranks above them in the dense list: 85.
