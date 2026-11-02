# Deep ingestion against fast ingestion: retrieval

Deep minus fast, nDCG@10. ▲ deep better, ▼ deep worse (the 95% interval excludes zero); no mark: no clear difference.

| set | dense only | keyword only | mixed, equal votes | mixed, keyword vote 0.02 | mixed, keyword vote 0.02 + reranker |
|---|---|---|---|---|---|
| ragbench-covidqa | -0.019 | +0.004 | +0.002 | -0.012 | -0.008 |
| scifact | -0.009 | +0.012 | +0.011 | -0.008 | +0.009 |
| multihoprag | -0.045 ▼ | -0.018 ▼ | -0.027 ▼ | -0.045 ▼ | -0.032 ▼ |
| hotpotqa | not run | not run | not run | not run | not run |
| fiqa | not run | not run | not run | not run | not run |
| fiqa-sub | -0.068 ▼ | +0.013 | -0.025 ▼ | -0.067 ▼ | -0.041 ▼ |
| docsqa | -0.011 | +0.041 | -0.014 | -0.011 | +0.005 |
| pdf-quantum (stored key) | -0.009 | +0.038 | +0.008 | -0.024 | -0.019 |
| pdf-quantum (corrected key) | -0.011 | +0.058 | +0.010 | -0.031 | -0.024 |
| pdf-got | -0.007 | +0.117 ▲ | +0.064 | -0.005 | +0.003 |

35 comparisons: 1 deep better, 9 deep worse, 25 no clear difference.

## Per dataset

### ragbench-covidqa

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 242 | 0.850 | 0.831 | -0.019 [-0.040, +0.000] | no clear difference | -0.004 [-0.025, +0.017] | -0.008 [-0.015, -0.001] |
| keyword only | 242 | 0.666 | 0.670 | +0.004 [-0.014, +0.023] | no clear difference | +0.017 [-0.017, +0.050] | -0.005 [-0.017, +0.004] |
| mixed, equal votes | 242 | 0.733 | 0.735 | +0.002 [-0.017, +0.021] | no clear difference | +0.000 [-0.025, +0.025] | +0.003 [-0.008, +0.018] |
| mixed, keyword vote 0.02 | 242 | 0.776 | 0.764 | -0.012 [-0.033, +0.009] | no clear difference | +0.000 [-0.025, +0.025] | -0.001 [-0.015, +0.015] |
| mixed, keyword vote 0.02 + reranker | 242 | 0.789 | 0.781 | -0.008 [-0.025, +0.009] | no clear difference | -0.004 [-0.029, +0.021] | -0.001 [-0.015, +0.015] |

### scifact

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 300 | 0.890 | 0.881 | -0.009 [-0.025, +0.006] | no clear difference | +0.007 [-0.010, +0.023] | +0.000 [+0.000, +0.000] |
| keyword only | 300 | 0.683 | 0.695 | +0.012 [-0.007, +0.030] | no clear difference | +0.027 [-0.003, +0.057] | -0.009 [-0.024, +0.005] |
| mixed, equal votes | 300 | 0.793 | 0.804 | +0.011 [-0.006, +0.029] | no clear difference | +0.013 [-0.010, +0.037] | +0.000 [-0.010, +0.010] |
| mixed, keyword vote 0.02 | 300 | 0.889 | 0.882 | -0.008 [-0.024, +0.008] | no clear difference | +0.007 [-0.010, +0.027] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 300 | 0.844 | 0.853 | +0.009 [-0.006, +0.023] | no clear difference | +0.000 [-0.020, +0.020] | +0.000 [+0.000, +0.000] |

### multihoprag

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 265 | 0.804 | 0.759 | -0.045 [-0.060, -0.031] | deep worse | -0.015 [-0.038, +0.004] | -0.004 [-0.013, +0.004] |
| keyword only | 265 | 0.758 | 0.740 | -0.018 [-0.035, -0.001] | deep worse | -0.011 [-0.038, +0.019] | -0.003 [-0.011, +0.006] |
| mixed, equal votes | 265 | 0.812 | 0.785 | -0.027 [-0.040, -0.014] | deep worse | -0.019 [-0.038, -0.004] | -0.005 [-0.013, +0.002] |
| mixed, keyword vote 0.02 | 265 | 0.807 | 0.761 | -0.045 [-0.061, -0.030] | deep worse | -0.011 [-0.030, +0.008] | -0.004 [-0.013, +0.004] |
| mixed, keyword vote 0.02 + reranker | 265 | 0.827 | 0.795 | -0.032 [-0.046, -0.018] | deep worse | +0.000 [-0.011, +0.011] | -0.004 [-0.013, +0.004] |

### hotpotqa: fast only (deep: not run)

| search setting | questions | fast nDCG@10 | fast Hit@5 | fast Recall@100 |
|---|---|---|---|---|
| dense only | 300 | 0.982 | 0.997 | 0.993 |
| keyword only | 300 | 0.866 | 0.987 | 0.960 |
| mixed, equal votes | 300 | 0.943 | 1.000 | 0.997 |
| mixed, keyword vote 0.02 | 300 | 0.985 | 1.000 | 0.993 |
| mixed, keyword vote 0.02 + reranker | 300 | 0.985 | 1.000 | 0.993 |

### fiqa: fast only (deep: not run)

| search setting | questions | fast nDCG@10 | fast Hit@5 | fast Recall@100 |
|---|---|---|---|---|
| dense only | 648 | 0.609 | 0.773 | 0.912 |
| keyword only | 648 | 0.243 | 0.394 | 0.526 |
| mixed, equal votes | 648 | 0.441 | 0.664 | 0.871 |
| mixed, keyword vote 0.02 | 648 | 0.607 | 0.772 | 0.912 |
| mixed, keyword vote 0.02 + reranker | 648 | 0.569 | 0.779 | 0.912 |

### fiqa-sub

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 648 | 0.649 | 0.581 | -0.068 [-0.084, -0.052] | deep worse | -0.049 [-0.073, -0.026] | -0.042 [-0.058, -0.025] |
| keyword only | 648 | 0.274 | 0.287 | +0.013 [-0.001, +0.026] | no clear difference | +0.019 [-0.009, +0.048] | +0.028 [+0.012, +0.045] |
| mixed, equal votes | 648 | 0.481 | 0.456 | -0.025 [-0.040, -0.010] | deep worse | -0.031 [-0.059, -0.005] | -0.049 [-0.065, -0.033] |
| mixed, keyword vote 0.02 | 648 | 0.648 | 0.581 | -0.067 [-0.083, -0.051] | deep worse | -0.048 [-0.071, -0.025] | -0.042 [-0.058, -0.025] |
| mixed, keyword vote 0.02 + reranker | 648 | 0.610 | 0.569 | -0.041 [-0.055, -0.026] | deep worse | -0.042 [-0.065, -0.020] | -0.042 [-0.058, -0.025] |

### docsqa

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 35 | 0.831 | 0.820 | -0.011 [-0.053, +0.031] | no clear difference | +0.029 [+0.000, +0.086] | +0.000 [+0.000, +0.000] |
| keyword only | 35 | 0.800 | 0.841 | +0.041 [-0.034, +0.117] | no clear difference | +0.029 [+0.000, +0.086] | +0.000 [+0.000, +0.000] |
| mixed, equal votes | 35 | 0.857 | 0.843 | -0.014 [-0.078, +0.050] | no clear difference | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 | 35 | 0.831 | 0.820 | -0.011 [-0.053, +0.031] | no clear difference | +0.029 [+0.000, +0.086] | +0.000 [+0.000, +0.000] |
| mixed, keyword vote 0.02 + reranker | 35 | 0.842 | 0.847 | +0.005 [-0.026, +0.036] | no clear difference | -0.029 [-0.086, +0.000] | +0.000 [+0.000, +0.000] |

### pdf-quantum (stored key)

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.510 | 0.501 | -0.009 [-0.045, +0.027] | no clear difference | -0.032 [-0.097, +0.000] | -0.003 [-0.047, +0.038] |
| keyword only | 31 | 0.348 | 0.385 | +0.038 [-0.001, +0.073] | no clear difference | +0.032 [-0.065, +0.129] | +0.021 [-0.034, +0.070] |
| mixed, equal votes | 31 | 0.425 | 0.432 | +0.008 [-0.023, +0.038] | no clear difference | +0.000 [+0.000, +0.000] | -0.008 [-0.068, +0.044] |
| mixed, keyword vote 0.02 | 31 | 0.490 | 0.466 | -0.024 [-0.055, +0.007] | no clear difference | -0.032 [-0.097, +0.000] | -0.003 [-0.047, +0.038] |
| mixed, keyword vote 0.02 + reranker | 31 | 0.475 | 0.455 | -0.019 [-0.047, +0.005] | no clear difference | -0.065 [-0.161, +0.000] | -0.003 [-0.047, +0.038] |

### pdf-quantum (corrected key)

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.750 | 0.739 | -0.011 [-0.065, +0.044] | no clear difference | +0.000 [-0.097, +0.097] | -0.011 [-0.032, +0.000] |
| keyword only | 31 | 0.511 | 0.569 | +0.058 [-0.004, +0.114] | no clear difference | +0.032 [-0.065, +0.129] | +0.036 [+0.000, +0.082] |
| mixed, equal votes | 31 | 0.628 | 0.638 | +0.010 [-0.038, +0.057] | no clear difference | +0.000 [+0.000, +0.000] | +0.016 [-0.011, +0.054] |
| mixed, keyword vote 0.02 | 31 | 0.715 | 0.684 | -0.031 [-0.078, +0.015] | no clear difference | +0.000 [-0.097, +0.097] | -0.011 [-0.032, +0.000] |
| mixed, keyword vote 0.02 + reranker | 31 | 0.697 | 0.672 | -0.024 [-0.065, +0.013] | no clear difference | -0.065 [-0.161, +0.000] | -0.011 [-0.032, +0.000] |

### pdf-got

| search setting | questions | fast nDCG@10 | deep nDCG@10 | deep - fast, nDCG@10 | verdict | deep - fast, Hit@5 | deep - fast, Recall@100 |
|---|---|---|---|---|---|---|---|
| dense only | 31 | 0.817 | 0.810 | -0.007 [-0.059, +0.041] | no clear difference | +0.000 [+0.000, +0.000] | +0.005 [+0.000, +0.016] |
| keyword only | 31 | 0.288 | 0.405 | +0.117 [+0.048, +0.188] | deep better | +0.097 [+0.000, +0.226] | +0.002 [-0.103, +0.104] |
| mixed, equal votes | 31 | 0.518 | 0.582 | +0.064 [-0.013, +0.142] | no clear difference | +0.097 [-0.065, +0.258] | -0.004 [-0.012, +0.000] |
| mixed, keyword vote 0.02 | 31 | 0.805 | 0.800 | -0.005 [-0.064, +0.049] | no clear difference | +0.000 [+0.000, +0.000] | +0.005 [+0.000, +0.016] |
| mixed, keyword vote 0.02 + reranker | 31 | 0.741 | 0.744 | +0.003 [-0.053, +0.057] | no clear difference | +0.000 [+0.000, +0.000] | +0.005 [+0.000, +0.016] |
