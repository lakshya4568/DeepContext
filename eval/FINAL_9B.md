# Final round: deep ingestion with a 9B model, on every dataset that has a fast-ingestion result

Written 2026-11-02, **before any deep result of this round exists**. The results are added at the end of this file when the run is done.

## What is compared

| | fast (baseline) | deep |
|---|---|---|
| chunks | main's chunker (parent up to 2,500 tokens, child up to 300) | the same chunks |
| text that is embedded | the chunk | a 1-2 sentence summary, a blank line, then the chunk |
| who writes the summary | nobody | Qwen3.5-9B (FP8, vLLM), with the application's own summary prompt (only the model changes) |
| embedding model | gemini-embedding-2, 768-d | the same |
| search | dense, keyword (BM25 fix), mixed, mixed + reranker (bge-reranker-v2-m3) | the same |

The earlier fast report (`results_final/REPORT.md`) used the old code base and Gemini 3.8 flash with the application's prompt. It is **not** reused:
the fast arm is run again here with the same chunker, answer model and prompt as the deep arm, so that only the summary differs.

## Datasets (same corpora and questions as the fast report)

| set | documents | child chunks (9B prompts) | questions | has reference answers |
|---|---|---|---|---|
| ragbench-covidqa | 902 | 902 | 242 | yes (model-written references) |
| scifact | 5,183 | 9,397 | 300 | no |
| multihoprag | 609 | 7,197 | 300 (35 cannot be answered) | yes |
| hotpotqa | 30,000 (sample, all relevant documents kept) | 30,090 | 300 | no |
| fiqa | 57,638 | 72,137 | 648 | no |
| docsqa (our docs) | 177 | 244 | 44 (9 cannot be answered) | yes |
| pdf-quantum (thesis) | 296 pages | 945 | 36 (5 cannot be answered) | yes |
| pdf-got (novel) | 753 pages | 1,937 | 36 (5 cannot be answered) | yes |

Sets without reference answers (SciFact, FiQA, HotpotQA) are retrieval only, as in the fast report.

## What is measured

1. **Retrieval** (all 8 sets, both arms): nDCG@10, Hit@5, Recall@100 for five search settings: dense only; keyword only;
   mixed with equal votes; mixed with keyword vote 0.02; mixed with keyword vote 0.02 + reranker. Difference deep minus fast is a paired bootstrap
   over the questions (95% interval, 10,000 resamples); the verdict is "better" / "worse" only when the interval excludes zero.
2. **Answers** (5 sets with reference answers, both arms, all questions): Qwen3.5-9B with the **strict documents-only prompt**
   (`eval/strict_prompt.py`, version `strict-v1`), search = mixed with keyword vote 0.02 + reranker, six parent chunks, temperature 0.
   The model-written summary is not shown to the answer model.
   - answer quality: rubric correctness (`eval/judge_correctness.py`, Gemini 3.7 flash, reference-based);
   - says "I don't know": wrongly on answerable questions, rightly on the unanswerable ones (set sizes are small, so read these as counts);
   - grounding (`eval/judge_grounding.py`): share of claims the shown documents back, share of answers where every claim is backed;
   - citations: precision (does a cited document back the claim), answers with a valid citation, invented citation numbers, cites a relevant document.
3. **Summary checks** (all 9B summaries): empty, cut at the token cap, and names that the summary writer was not shown (`eval/summary_names.py`; it flags memory of the model, not text of the document).

## Rules fixed in advance

- No prompt, model, vote, depth or threshold is changed after a result is seen. Broken outputs (empty generations, unusable judgements) are counted and reported, not repaired.
- The judge is a different model from the answerer. Gemini is used only as embedding model and judge.
- The thesis (pdf-quantum) is shown with the stored key and with the corrected key (`eval/thesis_key.py`).
- The novel (pdf-got) is the held-out test. The 9B has read that book, so its summaries may contain facts that are not in the shown text; the summary check above measures how often.
- Every number in the report comes from files in `eval/results_final9b/`. Anything not run is written as "not run".
- Costs and run times are written down as measured, never as estimates.

## Found before any deep result (2026-11-02)

- **Fast baseline is re-run, and it reproduces the old one where nothing changed.** Dense only and keyword only match the old fast report within 0.02 nDCG@10 on all five public sets. The old report's mixed search used keyword vote 0.1, which is not one of this round's settings, so mixed has to be run again anyway. The old answers came from Gemini with the application's prompt, so the answers are run again too.
- **The product's near-duplicate filter changes mixed search on CovidQA.** It keeps one chunk per first 120 characters (`src/deep_context/retrieval/hybrid.py`, `deduplicate_candidates`). 28% of CovidQA documents have such a twin. Mixed search (vote 0.02) lost 87 relevant documents that dense search returned, and for all 87 a twin ranked above them: Recall@100 1.000 against 0.875, nDCG@10 0.850 against 0.776 (fast arm). On the other seven sets Recall@100 is the same for dense and mixed (to three decimals). The filter is the same for both arms and is **not** changed; the CovidQA mixed and reranked numbers carry this penalty. Evidence: `results_final9b/NOTES_prefix_twins_fast.md`, `scripts_final9b/prefix_twins.py` (re-run for the deep arm after the deep runs).
- **A stale label was fixed.** `eval/run_retrieval.py` wrote a hard-coded chunker label (child 300-600 tokens) into `dataset_meta.json`; main's chunker is child 150-300, parent 1000-2500, overlap 15%. The code now reads the chunker, and the label was corrected in the eight fast entries. No result row was touched.
- **Fast ingestion was not paid for again for docs, thesis and novel.** Their fast indexes from the earlier round were reused (the index step reports `reused=True`).

## Run plan and limits

- **The GPU VM** (`rag-slm-gpu-c`, Singapore, one NVIDIA L4, 4 vCPU, 15 GB RAM) was started 2026-11-02 12:45 IST and switches itself off after 8 hours (about 20:45). The cap was 6 hours; it was raised to 8 while the VM was off, inside the 6-8 hours the owner had approved. The L4 runs at its 72 W power limit, which sets the summary speed.
- **First hour (12:45-13:40): shared GPU.** The reranker and the answer jobs ran on the GPU while the first summaries were written; the summaries then ran at 7.5-9 kB of prompt text per second.
- **From 13:40: one GPU task at a time (the owner's instruction).** Phase 1: the 9B writes summaries and nothing else is on the GPU (the reranker is not loaded); the Mac copies each finished batch, stores it, and embeds and stores every complete window of the deep index side by side (`scripts_final9b/stream_deep.sh`, a round every 7 minutes, no GPU). Alone, the summaries ran at about 12 kB per second on SciFact. Phase 2 (`scripts_final9b/phase2_gpu.sh`): the reranker and the 9B answers, after the summaries.
- **Order of summaries:** CovidQA, MultiHop-RAG, SciFact, FiQA (stopped at about 45k, see below), and HotpotQA last. HotpotQA's 30,090 summaries (about 1.3-1.5 hours) plus the reranker and answers did not obviously fit before the VM's 8-hour limit, so HotpotQA deep is a last step with hard time limits (`scripts_final9b/phase3_hotpot.sh`): it is not started after 19:10 IST, and its summaries are stopped at 20:15 IST; in either case it is reported as "not run". The owner chose to run it (2026-11-02 about 17:45), after being told it would end about 20:30 against the VM's stop at 20:45. These limits were fixed on time alone, before any HotpotQA deep result existed. HotpotQA is the set where search is already saturated (fast nDCG@10 about 0.98). Fast HotpotQA results are in this round either way.
- **FiQA is cut at about 45,000 summaries (the owner's instruction, 2026-11-02 about 16:00).** By then the 9B had written summaries for the first 45,116 chunks, which are the first 36,000 of FiQA's 57,638 documents. A deep index needs a summary for every chunk of its corpus, so the deep arm cannot cover the whole FiQA corpus. Both arms are therefore compared on **"fiqa-sub": the first 36,000 documents in corpus order plus all 1,706 relevant documents (36,625 documents), with all 648 questions** (`eval/datasets.py`, `FIQA_SUB_FIRST_DOCS`). The 625 relevant documents outside the first 36,000 get their summaries in a short extra pass (about 800 prompts). The fast arm on the subset reuses the embeddings cached by the earlier whole-corpus fast build (checked: 178 of 178 sampled chunks were cache hits), so it needs no embedding calls. The whole-corpus FiQA numbers of the fast arm stay in the tables ("deep: not run"). The subset is not the standard FiQA benchmark; its scores must not be compared with published FiQA numbers.
- Docs, thesis and novel already had deep indexes from the earlier 9B round and were run before phase 1 started.

## Not part of this round

Another answer model, human labels, public corpora with a different chunker, RGB (robustness with supplied passages), end-to-end latency.
