# Far-reference test: fast ingestion against deep ingestion (local Qwen)

**Read this first.** This is a pilot. The questions are the few that survived strict checks (written by
gemini-3.7-flash, reviewed by gemini-3.8-flash), so the counts are tiny and no difference below is evidence of
anything. The application's own note-writer (arm `notes`) was not run: it freezes an 8 GB Mac. Search is dense
only, top pages, no reranker, one run.

## pdf-got: pages 6-125, 3 questions

299 chunks; the local model tagged 298 of them with a name or topic. The name route reaches a part-B page for 1 of 3 questions (it is the only way a question that names the entity can reach a page that never spells it).

### pdf-got (top 10 pages)

| arm | answered (both parts) | 95% interval | part A found | part B found | wins / losses vs fast | sign test p |
|---|---|---|---|---|---|---|
| fast | 3/3 | 29%-100% | 3/3 | 3/3 | - | - |
| views | 3/3 | 29%-100% | 3/3 | 3/3 | 0 / 0 | 1.000 |
| fast+views | 3/3 | 29%-100% | 3/3 | 3/3 | 0 / 0 | 1.000 |
| fast+names | 2/3 | 9%-99% | 3/3 | 2/3 | 0 / 1 | 1.000 |
| fast+views+names | 2/3 | 9%-99% | 3/3 | 2/3 | 0 / 1 | 1.000 |

### pdf-got (top 20 pages)

| arm | answered (both parts) | 95% interval | part A found | part B found | wins / losses vs fast | sign test p |
|---|---|---|---|---|---|---|
| fast | 3/3 | 29%-100% | 3/3 | 3/3 | - | - |
| views | 3/3 | 29%-100% | 3/3 | 3/3 | 0 / 0 | 1.000 |
| fast+views | 3/3 | 29%-100% | 3/3 | 3/3 | 0 / 0 | 1.000 |
| fast+names | 2/3 | 9%-99% | 3/3 | 2/3 | 0 / 1 | 1.000 |
| fast+views+names | 2/3 | 9%-99% | 3/3 | 2/3 | 0 / 1 | 1.000 |


## pdf-quantum: pages 21-97, 1 questions

262 chunks; the local model tagged 195 of them with a name or topic. The name route reaches a part-B page for 0 of 1 questions (it is the only way a question that names the entity can reach a page that never spells it).

### pdf-quantum (top 10 pages)

| arm | answered (both parts) | 95% interval | part A found | part B found | wins / losses vs fast | sign test p |
|---|---|---|---|---|---|---|
| fast | 1/1 | 2%-100% | 1/1 | 1/1 | - | - |
| views | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |
| fast+views | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |
| fast+names | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |
| fast+views+names | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |

### pdf-quantum (top 20 pages)

| arm | answered (both parts) | 95% interval | part A found | part B found | wins / losses vs fast | sign test p |
|---|---|---|---|---|---|---|
| fast | 1/1 | 2%-100% | 1/1 | 1/1 | - | - |
| views | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |
| fast+views | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |
| fast+names | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |
| fast+views+names | 1/1 | 2%-100% | 1/1 | 1/1 | 0 / 0 | 1.000 |


## Does the local model name the right person? (novel)

238 passages sit inside a chapter headed by a character; the name is missing from 20 of them.

| predictor | right on all passages | right where the passage does not spell the name |
|---|---|---|
| qwen | 124/238 (52%) | 1/20 (5%) |
| recent | 60/238 (25%) | 5/20 (25%) |
| chunk name or recent | 81/238 (34%) | 3/20 (15%) |

(Most frequent tag: Jon, 49 times.)
