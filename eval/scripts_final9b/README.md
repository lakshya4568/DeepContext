# Final round: how the run was done

Protocol and rules: `eval/FINAL_9B.md`. Results go to `eval/results_final9b/`. All commands run from the repo root on macOS or Linux (zsh/bash).
`arm` is `fast` (no summaries, index suffix `_main`) or `deep` (9B summaries, index suffix `_maindeep9b`).
A third arm, `deepkw` (index suffix `_maindeepkw`), was added after the results were known: the plain chunk is embedded (as in fast), the 9B summary goes only into the keyword index
(`retr_nogpu.sh deepkw <dataset>...`, no new embedding: it reuses the fast vectors in the cache). It is a remedy test, not part of the protocol of `eval/FINAL_9B.md`.

| script | what it does | needs |
|---|---|---|
| `retr_nogpu.sh <arm> <dataset>...` | dense, keyword, mixed (equal votes), mixed (keyword vote 0.02) | Gemini embeddings (deep: summaries already imported) |
| `retr_rerank.sh <arm> <dataset>...` | mixed (vote 0.02) + bge-reranker-v2-m3 | GPU tunnel, port 18001 |
| `answers.sh <arm> <dataset>...` | answers by Qwen3.5-9B with the strict documents-only prompt (`LIMIT=3` for a smoke test); one at a time (lock) | GPU tunnel, ports 18000 and 18001 |
| `judge.sh <arm>` | rubric correctness and grounding with Gemini 3.7 flash (resumable) | Gemini |
| `export_jobs.sh <dir>` | the 9B summary prompts, one file per dataset | nothing |
| `fetch_import.sh <dataset>` | copies the summaries written so far from the VM and stores them in the summary cache | `VM_IP`, `VM_ID` |
| `build_index.py --dataset <d>` | embeds and stores every complete window of the deep index, stops at the first unfinished one | summaries in the cache, Gemini |
| `stream_deep.sh <dataset>...` | phase 1, Mac side: every 7 minutes fetch, import, build the finished windows; when a set is done, finish its index and run the first-stage searches | `VM_IP`, `VM_ID`, no GPU |
| `phase2_gpu.sh [<dataset>...]` | phase 2: load the reranker, fast and deep answers, deep reranker runs | tunnel, after the summaries |
| `extra_pass.sh` | after FiQA was stopped at about 45k summaries: store them, find what the FiQA subset (`fiqa-sub`) still lacks, have the 9B write exactly that | `VM_IP`, `VM_ID`, 9B server up |
| `health_watch.sh` | one line per event: set finished, tunnel down, low disk, VM limit near, status every 30 minutes | `VM_IP`, `VM_ID` |
| `prefix_twins.py --arm <arm>` | how much the product's near-duplicate filter changes mixed search | saved rows |
| `phase3_hotpot.sh` | phase 3: after phase 2, the 9B writes the HotpotQA summaries (reranker unloaded), the Mac builds the index and the first-stage searches, then the reranker setting | `VM_IP`, `VM_ID`, tunnel |
| `finish_gpu.sh` | after phase 3: if no answer row failed, stop the VM (the VM and its disk are kept) | gcloud |
| `summary_checks_all.py` | the summary checks for every set (`SUMMARY_CHECKS.md`) | saved rows, summary cache |
| `build_report.py` | assembles `REPORT.md` from `report_template.md` and the generated tables | saved rows |
| `why_deep.py <ranks\|summaries\|vectors\|title\|chunks>` | post-hoc analysis of why deep did not help (`NOTES_why_deep.md`, sections 1 to 5) | saved rows, database, embedding cache; no new embedding |
| `fix_tests.py <ensemble\|ablation\|rrf>` | remedy tests (`NOTES_why_deep.md`, sections 6 to 8): combined dense score, the opening-only / body-only text test (**new Gemini embeddings**, `--count-only` shows the cost first), fusion across arms | database, embedding cache |
| `chain_gpu_a.sh`, `pipeline_deep.sh`, `pipeline_fiqa.sh`, `watch_summaries.sh` | used 12:52-13:41 on 2026-11-02 while the GPU was shared, then replaced by the phase scripts above | |

Report: `uv run --no-sync python -m eval.final_report` writes `RETRIEVAL.md`, `ANSWERS.md`, `table_retrieval.csv`, `table_answers.csv`;
`python -m eval.summary_checks --set "LABEL=JOBS|RESULTS" ...` checks the summaries.

## GPU part, in order

1. Start one VM (never in a retry loop), read its external IP and instance id with `gcloud compute instances describe`, export `VM_IP` and `VM_ID`.
   When the zones are out of stock: `gcloud compute instances bulk create` with `--target-distribution-shape=ANY` and several L4 machine sizes in `--instance-selection-machine-types` found a g2-standard-4 in Singapore.
2. On the VM: `eval/vm/start_9b.sh` (Qwen3.5-9B FP8 on port 8000). Tunnel: `eval/vm/tunnel.sh` (local 18000 and 18001).
3. Upload `jobs9b_<dataset>.jsonl`, `eval/vm/summarize_vllm.py` and `eval/vm/run_summaries.sh`; run `run_summaries.sh ragbench-covidqa multihoprag scifact fiqa` on the VM.
4. Phase 1 (GPU: summaries only): `stream_deep.sh multihoprag scifact fiqa` on the Mac. FiQA was stopped at about 45,000 of 71,000 summaries (the owner's instruction);
   `extra_pass.sh` then wrote the few summaries the FiQA subset still lacked, and `retr_nogpu.sh fast fiqa-sub` / `retr_nogpu.sh deep fiqa-sub` built the subset's two indexes.
5. Phase 2 (after the summaries): `phase2_gpu.sh`, then `retr_rerank.sh fast fiqa-sub` and `retr_rerank.sh deep fiqa-sub`.
6. `judge.sh fast`, `judge.sh deep`, the report, the summary checks, then stop the VM.
