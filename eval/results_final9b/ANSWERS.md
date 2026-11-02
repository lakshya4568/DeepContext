# Deep ingestion against fast ingestion: answers, grounding, "I don't know"


### ragbench-covidqa: answers

| arm | answerable | correct (score out of 100) | difference to fast (points) | verdict | wrongly says it does not know (answerable) | unanswerable | rightly says it does not know |
|---|---|---|---|---|---|---|---|
| fast | 242 | 65.7 | - | - | 19% | 0 | n/a |
| deep | 242 | 64.9 | -0.8 [-4.3, +2.7] | no clear difference | 18% | 0 | n/a |

### ragbench-covidqa: grounding and citations

| arm | answers with claims | claims backed by the documents | answers with every claim backed | difference to fast, per answer (points) | verdict | citation precision | answers with a valid citation | invented citation number | cites a relevant document |
|---|---|---|---|---|---|---|---|---|---|
| fast | 201 | 98% | 94% | - | - | 99% | 100% | 0% | 91% |
| deep | 204 | 97% | 91% | -1.3 [-2.8, -0.1] | worse | 97% | 100% | 0% | 88% |

### multihoprag: answers

| arm | answerable | correct (score out of 100) | difference to fast (points) | verdict | wrongly says it does not know (answerable) | unanswerable | rightly says it does not know |
|---|---|---|---|---|---|---|---|
| fast | 265 | 46.0 | - | - | 36% | 35 | 91% |
| deep | 265 | 38.9 | -7.2 [-11.1, -3.2] | worse | 44% | 35 | 91% |

### multihoprag: grounding and citations

| arm | answers with claims | claims backed by the documents | answers with every claim backed | difference to fast, per answer (points) | verdict | citation precision | answers with a valid citation | invented citation number | cites a relevant document |
|---|---|---|---|---|---|---|---|---|---|
| fast | 179 | 98% | 91% | - | - | 98% | 99% | 0% | 94% |
| deep | 159 | 97% | 89% | +0.4 [-0.7, +1.6] | no clear difference | 98% | 100% | 0% | 91% |

### docsqa: answers

| arm | answerable | correct (score out of 100) | difference to fast (points) | verdict | wrongly says it does not know (answerable) | unanswerable | rightly says it does not know |
|---|---|---|---|---|---|---|---|
| fast | 35 | 95.7 | - | - | 3% | 9 | 89% |
| deep | 35 | 97.1 | +1.4 [+0.0, +4.3] | no clear difference | 3% | 9 | 89% |

### docsqa: grounding and citations

| arm | answers with claims | claims backed by the documents | answers with every claim backed | difference to fast, per answer (points) | verdict | citation precision | answers with a valid citation | invented citation number | cites a relevant document |
|---|---|---|---|---|---|---|---|---|---|
| fast | 34 | 98% | 94% | - | - | 98% | 100% | 0% | 94% |
| deep | 35 | 100% | 100% | +1.1 [+0.0, +2.7] | no clear difference | 100% | 100% | 0% | 100% |

### pdf-quantum: answers

| arm | answerable | correct (score out of 100) | difference to fast (points) | verdict | wrongly says it does not know (answerable) | unanswerable | rightly says it does not know |
|---|---|---|---|---|---|---|---|
| fast | 31 | 82.3 | - | - | 3% | 5 | 100% |
| deep | 31 | 83.9 | +1.6 [-6.5, +11.3] | no clear difference | 3% | 5 | 100% |

### pdf-quantum: grounding and citations

| arm | answers with claims | claims backed by the documents | answers with every claim backed | difference to fast, per answer (points) | verdict | citation precision | answers with a valid citation | invented citation number | cites a relevant document |
|---|---|---|---|---|---|---|---|---|---|
| fast | 30 | 99% | 93% | - | - | 99% | 100% | 0% | 93% |
| deep | 31 | 99% | 94% | +0.5 [-0.8, +2.5] | no clear difference | 99% | 100% | 0% | 87% |

### pdf-got: answers

| arm | answerable | correct (score out of 100) | difference to fast (points) | verdict | wrongly says it does not know (answerable) | unanswerable | rightly says it does not know |
|---|---|---|---|---|---|---|---|
| fast | 31 | 74.2 | - | - | 6% | 5 | 100% |
| deep | 31 | 67.7 | -6.5 [-19.4, +6.5] | no clear difference | 10% | 5 | 100% |

### pdf-got: grounding and citations

| arm | answers with claims | claims backed by the documents | answers with every claim backed | difference to fast, per answer (points) | verdict | citation precision | answers with a valid citation | invented citation number | cites a relevant document |
|---|---|---|---|---|---|---|---|---|---|
| fast | 29 | 92% | 59% | - | - | 91% | 100% | 0% | 100% |
| deep | 28 | 90% | 64% | +0.5 [-5.4, +5.9] | no clear difference | 89% | 96% | 0% | 93% |
