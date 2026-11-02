# deepkw (summary only in the keyword index) against fast

### deepkw minus fast, nDCG@10 (▲ deepkw clearly better, ▼ clearly worse: the 95% interval excludes 0)

| set | dense only | keyword only | mixed, equal votes | mixed, keyword vote 0.02 | mixed, keyword vote 0.02 + reranker |
|---|---|---|---|---|---|
| ragbench-covidqa | +0.000 | +0.004 | -0.004 | +0.000 | not run |
| scifact | +0.000 | +0.012 | +0.002 | -0.001 | not run |
| multihoprag | +0.000 | -0.018 ▼ | -0.012 ▼ | -0.001 | not run |
| hotpotqa | not run | not run | not run | not run | not run |
| fiqa-sub | not run | not run | not run | not run | not run |
| docsqa | +0.000 | +0.041 | +0.007 | +0.000 | not run |
| pdf-quantum (corrected key) | +0.000 | +0.058 | +0.017 | -0.002 | not run |
| pdf-got | +0.000 | +0.117 ▲ | +0.059 | -0.000 | not run |

24 comparisons: 1 deepkw better, 2 deepkw worse, 21 no clear difference.

### ragbench-covidqa

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 242 | 0.850 | 0.850 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 242 | 0.666 | 0.670 | +0.004 [-0.014, +0.023] | no clear difference | +0.017 [-0.017, +0.050] | -0.005 [-0.017, +0.004] |
| mixed, equal votes | 242 | 0.733 | 0.730 | -0.004 [-0.021, +0.014] | no clear difference | -0.008 [-0.033, +0.012] | -0.008 [-0.019, +0.000] |
| mixed, keyword vote 0.02 | 242 | 0.776 | 0.776 | +0.000 [-0.000, +0.001] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### scifact

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 300 | 0.890 | 0.890 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 300 | 0.683 | 0.695 | +0.012 [-0.007, +0.030] | no clear difference | +0.027 [-0.003, +0.057] | -0.009 [-0.024, +0.005] |
| mixed, equal votes | 300 | 0.793 | 0.794 | +0.002 [-0.011, +0.014] | no clear difference | +0.010 [-0.010, +0.033] | -0.003 [-0.010, +0.000] |
| mixed, keyword vote 0.02 | 300 | 0.889 | 0.888 | -0.001 [-0.006, +0.002] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### multihoprag

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 265 | 0.804 | 0.804 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 265 | 0.758 | 0.740 | -0.018 [-0.035, -0.001] | deepkw worse | -0.011 [-0.038, +0.019] | -0.003 [-0.011, +0.006] |
| mixed, equal votes | 265 | 0.812 | 0.800 | -0.012 [-0.022, -0.002] | deepkw worse | -0.019 [-0.038, -0.004] | -0.006 [-0.012, -0.001] |
| mixed, keyword vote 0.02 | 265 | 0.807 | 0.806 | -0.001 [-0.002, +0.001] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### docsqa

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 35 | 0.831 | 0.831 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 35 | 0.800 | 0.841 | +0.041 [-0.034, +0.117] | no clear difference | +0.029 [+0.000, +0.086] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 35 | 0.857 | 0.864 | +0.007 [-0.052, +0.064] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 | 35 | 0.831 | 0.831 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### pdf-quantum (stored key)

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.510 | 0.510 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 31 | 0.348 | 0.385 | +0.038 [-0.001, +0.073] | no clear difference | +0.032 [-0.065, +0.129] | +0.021 [-0.034, +0.070] |
| mixed, equal votes | 31 | 0.425 | 0.439 | +0.014 [-0.014, +0.042] | no clear difference | +0.032 [+0.000, +0.097] | -0.004 [-0.048, +0.032] |
| mixed, keyword vote 0.02 | 31 | 0.490 | 0.489 | -0.001 [-0.006, +0.003] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### pdf-quantum (corrected key)

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.750 | 0.750 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 31 | 0.511 | 0.569 | +0.058 [-0.004, +0.114] | no clear difference | +0.032 [-0.065, +0.129] | +0.036 [+0.000, +0.082] |
| mixed, equal votes | 31 | 0.628 | 0.644 | +0.017 [-0.028, +0.061] | no clear difference | +0.032 [+0.000, +0.097] | +0.016 [+0.000, +0.048] |
| mixed, keyword vote 0.02 | 31 | 0.715 | 0.714 | -0.002 [-0.008, +0.004] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### pdf-got

| search setting | questions | fast nDCG@10 | deepkw nDCG@10 | deepkw - fast, nDCG@10 | verdict | deepkw - fast, Hit@5 | deepkw - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.817 | 0.817 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 31 | 0.288 | 0.405 | +0.117 [+0.048, +0.188] | deepkw better | +0.097 [+0.000, +0.226] | +0.002 [-0.103, +0.104] |
| mixed, equal votes | 31 | 0.518 | 0.577 | +0.059 [-0.015, +0.138] | no clear difference | +0.032 [-0.097, +0.194] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 | 31 | 0.805 | 0.805 | -0.000 [-0.031, +0.022] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
