# Notes on the result files (kept next to them)

- 2026-11-02: `dataset_meta.json` in `R_fast_equal` and `R_fast_vote0.02` first carried the chunker label "child 300-600 tok" (a hard-coded text in
  `eval/run_retrieval.py`). The chunker really used is child 150-300, parent 1000-2500, overlap 15%. The label was corrected in all eight entries of both files
  (only the `index.chunker` field); the code now reads the chunker. No result row was changed.
- 2026-11-02: fast retrieval for docsqa, pdf-quantum and pdf-got reused the fast indexes of the earlier round (`reused=True` in the log), so no document was embedded again.
- 2026-11-02 about 13:35 IST, with the owner's OK, the seven index schemas of the old fast-benchmark base (older chunker) were dropped from the eval database to keep the Mac's disk from filling:
  fiqa_9c931e585a 756 MB, hotpotqa_00e3ffeea7 355 MB, hotpotqa_7a10e099be 336 MB, hotpotqa_ea2c6015cf 55 MB, scifact_ff1b0ff0ff 90 MB, multihoprag_03296ea13e 68 MB, fiqa_f93465067e 63 MB
  (1,724 MB in all; the database went from 5,265 MB to 3,541 MB). The results of that benchmark are files (`eval/results_final/`) and are unaffected; the model-call cache (`eval_cache`, which holds the
  embeddings) was not touched, so those indexes should be rebuildable without new embedding calls (not tested). Every `_main`, `_maindeep` and `_maindeep9b` index was kept.
- `NOTES_prefix_twins_fast.md`: the near-duplicate filter and mixed search (made by `eval/scripts_final9b/prefix_twins.py`).
- `RETRIEVAL.md`, `ANSWERS.md`, `table_*.csv` are written by `python -m eval.final_report`; until a run exists they say "not run".
- `NOTES_why_deep.md`: post-hoc analysis of why deep did not help (bug check, vectors, rankings, summary content); made by `eval/scripts_final9b/why_deep.py`.
