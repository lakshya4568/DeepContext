# Where to find what

| Folder | What it is |
|---|---|
| `eval/results_final/` | **Final results**: fixed search (BM25 + exact dedup + lexical vote 0.1), answers re-written, rubric + RAGAS scores, `REPORT.md`, tables, `GAPS.md`, `run_settings.json`. Start with `REPORT.md`. |
| `eval/results/` (this folder) | Results of the **original code** (before the search fixes): same datasets, same models. Used as the "before" in the comparisons. `GAPS.md` is the gap list. |
| `eval/results_fix/` | Per-variant search runs that isolate each fix (`v1_bm25`, `v2_bm25_dedup`, `v3_weighted`), the dev-question tuning runs (`dev_w*`), and `BEFORE_AFTER.md`. |
| `eval/results/smoke/` | Early n = 20 pipeline checks (not evidence of differences). |
| `eval/results/archive_title_in_content_bug/` | Rows from a harness bug that were discarded; kept for transparency. |

Large raw files (`raw_retrieval.jsonl`, `raw_generation.jsonl`) are committed gzip-compressed (`.jsonl.gz`); `gunzip -k` them to regenerate tables.
