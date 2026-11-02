# Near-duplicate filter and mixed search (fast ingestion)

## Table 1: dense only against mixed search (keyword vote 0.02), same questions

| set | questions | Recall@100 dense | Recall@100 mixed | documents returned, dense | documents returned, mixed |
|---|---|---|---|---|---|
| ragbench-covidqa | 242 | 1.000 | 0.875 | 100.0 | 84.9 |
| scifact | 300 | 0.993 | 0.993 | 66.1 | 66.1 |
| multihoprag | 265 | 0.984 | 0.984 | 22.0 | 22.0 |
| hotpotqa | 300 | 0.993 | 0.993 | 99.8 | 99.8 |
| fiqa | 648 | 0.912 | 0.912 | 89.5 | 89.4 |
| docsqa | 35 | 1.000 | 1.000 | 74.7 | 74.7 |
| pdf-quantum | 31 | 0.623 | 0.623 | 55.9 | 55.9 |
| pdf-got | 31 | 0.989 | 0.989 | 66.7 | 66.7 |

## Table 2: CovidQA, where the lost documents went

- CovidQA has 902 documents; 251 of them (28%) share their first 120 characters with at least one other document.
- Relevant documents that dense search returned and mixed search did not: 87; of these, a twin ranks above them in the dense list: 87.
