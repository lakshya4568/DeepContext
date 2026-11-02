# deepkw (summary only in the keyword index) against deep

### deepkw minus deep, nDCG@10 (▲ deepkw clearly better, ▼ clearly worse: the 95% interval excludes 0)

| set | dense only | keyword only | mixed, equal votes | mixed, keyword vote 0.02 | mixed, keyword vote 0.02 + reranker |
|---|---|---|---|---|---|
| ragbench-covidqa | +0.019 | +0.000 | -0.006 | +0.012 | not run |
| scifact | +0.009 | +0.000 | -0.009 | +0.006 | not run |
| multihoprag | +0.045 ▲ | +0.000 | +0.015 ▲ | +0.045 ▲ | not run |
| hotpotqa | not run | not run | not run | not run | not run |
| fiqa-sub | not run | not run | not run | not run | not run |
| docsqa | +0.011 | +0.000 | +0.021 | +0.011 | not run |
| pdf-quantum (corrected key) | +0.011 | +0.000 | +0.007 | +0.030 | not run |
| pdf-got | +0.007 | +0.000 | -0.005 | +0.005 | not run |

24 comparisons: 3 deepkw better, 0 deepkw worse, 21 no clear difference.

### ragbench-covidqa

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 242 | 0.831 | 0.850 | +0.019 [-0.000, +0.040] | no clear difference | +0.004 [-0.017, +0.025] | +0.008 [+0.001, +0.015] |
| keyword only | 242 | 0.670 | 0.670 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 242 | 0.735 | 0.730 | -0.006 [-0.022, +0.010] | no clear difference | -0.008 [-0.029, +0.012] | -0.011 [-0.027, +0.003] |
| mixed, keyword vote 0.02 | 242 | 0.764 | 0.776 | +0.012 [-0.009, +0.034] | no clear difference | +0.000 [-0.025, +0.025] | +0.001 [-0.015, +0.015] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### scifact

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 300 | 0.881 | 0.890 | +0.009 [-0.006, +0.025] | no clear difference | -0.007 [-0.023, +0.010] | +0.000 [+0.000, +0.000] |
| keyword only | 300 | 0.695 | 0.695 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 300 | 0.804 | 0.794 | -0.009 [-0.022, +0.003] | no clear difference | -0.003 [-0.017, +0.010] | -0.003 [-0.010, +0.000] |
| mixed, keyword vote 0.02 | 300 | 0.882 | 0.888 | +0.006 [-0.009, +0.022] | no clear difference | -0.007 [-0.027, +0.010] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### multihoprag

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 265 | 0.759 | 0.804 | +0.045 [+0.031, +0.060] | deepkw better | +0.015 [-0.004, +0.038] | +0.004 [-0.004, +0.013] |
| keyword only | 265 | 0.740 | 0.740 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 265 | 0.785 | 0.800 | +0.015 [+0.005, +0.026] | deepkw better | +0.000 [-0.011, +0.011] | -0.000 [-0.005, +0.004] |
| mixed, keyword vote 0.02 | 265 | 0.761 | 0.806 | +0.045 [+0.030, +0.060] | deepkw better | +0.011 [-0.008, +0.030] | +0.004 [-0.004, +0.013] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### docsqa

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 35 | 0.820 | 0.831 | +0.011 [-0.031, +0.053] | no clear difference | -0.029 [-0.086, +0.000] | +0.000 [+0.000, +0.000] |
| keyword only | 35 | 0.841 | 0.841 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 35 | 0.843 | 0.864 | +0.021 [+0.000, +0.053] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 | 35 | 0.820 | 0.831 | +0.011 [-0.031, +0.053] | no clear difference | -0.029 [-0.086, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### pdf-quantum (stored key)

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.501 | 0.510 | +0.009 [-0.027, +0.045] | no clear difference | +0.032 [+0.000, +0.097] | +0.003 [-0.038, +0.047] |
| keyword only | 31 | 0.385 | 0.385 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 31 | 0.432 | 0.439 | +0.007 [-0.005, +0.019] | no clear difference | +0.032 [+0.000, +0.097] | +0.003 [-0.032, +0.047] |
| mixed, keyword vote 0.02 | 31 | 0.466 | 0.489 | +0.023 [-0.007, +0.053] | no clear difference | +0.032 [+0.000, +0.097] | +0.003 [-0.038, +0.047] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### pdf-quantum (corrected key)

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.739 | 0.750 | +0.011 [-0.044, +0.065] | no clear difference | +0.000 [-0.097, +0.097] | +0.011 [+0.000, +0.032] |
| keyword only | 31 | 0.569 | 0.569 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 31 | 0.638 | 0.644 | +0.007 [-0.012, +0.027] | no clear difference | +0.032 [+0.000, +0.097] | +0.000 [-0.016, +0.016] |
| mixed, keyword vote 0.02 | 31 | 0.684 | 0.714 | +0.030 [-0.016, +0.074] | no clear difference | +0.000 [-0.097, +0.097] | +0.011 [+0.000, +0.032] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### pdf-got

| search setting | questions | deep nDCG@10 | deepkw nDCG@10 | deepkw - deep, nDCG@10 | verdict | deepkw - deep, Hit@5 | deepkw - deep, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.810 | 0.817 | +0.007 [-0.041, +0.059] | no clear difference | +0.000 [+0.000, +0.000] | -0.005 [-0.016, +0.000] |
| keyword only | 31 | 0.405 | 0.405 | +0.000 [+0.000, +0.000] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 31 | 0.582 | 0.577 | -0.005 [-0.035, +0.027] | no clear difference | -0.065 [-0.161, +0.000] | +0.004 [+0.000, +0.012] |
| mixed, keyword vote 0.02 | 31 | 0.800 | 0.805 | +0.005 [-0.044, +0.057] | no clear difference | +0.000 [+0.000, +0.000] | -0.005 [-0.016, +0.000] |
| mixed, keyword vote 0.02 + reranker | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
