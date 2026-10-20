# Gaps found so far (after batches 1 and 2; updated 2026-10-20)

Plain-language list. Every number comes from the saved result files (`eval/results/gap_analysis.md`, `table_retrieval.csv`, `significance.csv`, `smoke/`).
"Product" = the DeepContext system being measured. "Test" = my benchmark harness. Nothing below has been fixed yet.

Numbers used: nDCG@10 (0 to 1, higher is better; how high the correct page sits in the top 10).

---

## Update 2026-10-20: P1, P2, P3 fixed on branch `fix/search-gaps` (full table: `eval/results_fix/BEFORE_AFTER.md`)

nDCG@10 on the test questions, merged search before → after all three fixes (BM25 + exact-duplicate removal + lexical vote 0.1):

| dataset | before | after | change (paired 95% CI) | dense-only for reference |
|---|---|---|---|---|
| SciFact | 0.702 | 0.867 | +0.164 [+0.132, +0.198] | 0.900 |
| FiQA | 0.378 | 0.592 | +0.213 [+0.192, +0.235] | 0.614 |
| HotpotQA (30k docs) | 0.868 | 0.978 | +0.110 [+0.092, +0.129] | 0.982 |
| MultiHop-RAG | 0.637 | 0.814 | +0.177 [+0.152, +0.202] | 0.804 |
| RAGBench Covid | 0.712 | 0.851 | +0.139 [+0.106, +0.172] | 0.850 |
| Docs set | 0.760 | 0.826 | +0.066 [-0.003, +0.140] | 0.824 |
| GoT PDF | 0.419 | 0.722 | +0.303 [+0.208, +0.396] | 0.779 |
| Quantum PDF | 0.481 | 0.537 | +0.056 [+0.019, +0.097] | 0.518 |

What each fix did on its own:
- **P1 (real BM25)** alone: the word-search half rose from 0.13 to 0.24 on FiQA, 0.49 to 0.69 on SciFact, 0.40 to 0.75 on MultiHop-RAG; merged search rose +0.012 to +0.174 (not significant on RAGBench and the quantum PDF).
- **P3 (duplicate removal)** alone: only matters where passages share an opening line: RAGBench +0.073 (0.725 → 0.798), recall@100 0.855 → 0.996. No effect elsewhere.
- **P2 (lexical vote)**: chosen on separate dev questions (SciFact train, FiQA dev; grid 1, 0.5, 0.25, 0.1 — the best was the edge of the grid, 0.1). It adds +0.003 to +0.171 over BM25+dedup on 7 of 8 datasets (docs set −0.004).

Honest reading (paired test, merged-after minus dense-only, nDCG@10): significantly **below** dense-only on FiQA (−0.022), SciFact (−0.033) and the GoT PDF (−0.057); significantly **above** on MultiHop-RAG (+0.010) and the quantum PDF (+0.019); no difference on the docs set, HotpotQA and RAGBench. So the merge is now roughly as good as meaning-only search, with tiny gains on two sets and losses on three: **not a clear win**. The weight search shows the same pattern: the less the word channel votes, the better, down to the edge of the grid. With this embedding model the word search adds little on these datasets; it may still matter for exact-term queries (names, codes), which these sets contain few of.

Not yet re-measured after the fixes: reranked setups (`rrf_bge`, `rrf_bge_two_pass`) and the full engine, and all answer quality (these need the reranker; Colab is off). The reranked and answer rows in the main results are still the "before" versions.

---

## Update 2026-10-20 (later): reranked setups after the fixes, and batch 3 (answer correctness)

**Reranked search, nDCG@10, before → after all three fixes** (same questions; paired test; reranker = your Colab `bge-reranker-v2-m3`, verified identical to EcoHash earlier):

| dataset | merged+reranker before | after | change | dense-only |
|---|---|---|---|---|
| SciFact | 0.744 | 0.843 | +0.100 | 0.900 |
| FiQA | 0.437 | 0.562 | +0.125 | 0.614 |
| HotpotQA (30k) | 0.952 | 0.983 | +0.031 | 0.982 |
| MultiHop-RAG | 0.727 | 0.837 | +0.111 | 0.804 |
| RAGBench | 0.743 | 0.855 | +0.112 | 0.850 |
| Docs set | 0.794 | 0.825 | +0.031 (n.s.) | 0.824 |
| GoT PDF | 0.419 | 0.648 | +0.229 | 0.779 |
| Quantum PDF | 0.460 | 0.478 | +0.018 (n.s.) | 0.518 |

The full engine also improved (SciFact 0.738 → 0.827, RAGBench 0.744 → 0.836, GoT PDF 0.486 → 0.617; docs, MultiHop-RAG and the quantum PDF not significant). Reranked search is now better than meaning-only on MultiHop-RAG and level on RAGBench, docs and HotpotQA, but still below it on SciFact, FiQA and both PDFs. Full numbers: `eval/results_fix/BEFORE_AFTER.md`.

**Batch 3: answer correctness (LLM-rubric, judge `gemini-3.7-flash`, same questions for every setup; 50 per big dataset, all for docs/PDFs). These are the "before-fix" answers; answers have not been rewritten after the fixes.**

| dataset | meaning-only | words-only | merged | +reranker | +reranker, two-pass | full engine |
|---|---|---|---|---|---|---|
| Docs set (35) | 1.00 | 0.83 | 0.93 | 0.99 | 0.94 | 0.93 |
| MultiHop-RAG (50) | 0.52 | 0.20 | 0.39 | 0.44 | 0.39 | 0.39 |
| GoT PDF (31) | 0.82 | 0.30 | 0.62 | 0.60 | 0.65 | 0.67 |
| Quantum PDF (31) | 0.94 | 0.76 | 0.92 | 0.90 | 0.79 | 0.69 |
| RAGBench (50) | 0.78 | 0.65 | 0.73 | 0.74 | 0.67 | 0.62 |

(1 = fully correct, 0.5 = partly, 0 = wrong or refused.) Findings:
- **C1. Meaning-only search gives the best (or tied-best) answers on every set.** Only the GoT PDF difference to merged (+0.19) is significant before the multiple-comparison correction.
- **C2. Two-pass writing does not improve correctness.** It is lower than one-pass on the quantum PDF (−0.11, p = 0.007) and RAGBench (−0.07, p = 0.046), lower on docs and MultiHop-RAG (not significant), higher on the GoT PDF (+0.05, not significant). None survive the Holm correction across all 161 contrasts.
- **C3. The full engine is not better than simpler setups** (quantum PDF 0.69 vs 0.90 for merged+reranker, RAGBench 0.62 vs 0.74).
- **C4. MultiHop-RAG answers are mostly lost to refusals,** not to wrong answers: the writer refuses 36–69% of questions depending on the setup (see gap analysis F); only 14–20% of questions have all needed articles inside the 12,000-character limit (P5).
- **C5. Declining unanswerable questions works** (rubric: 97–100% handled correctly, small n: 9 docs, 5+5 PDF, 35 MultiHop-RAG).
- Caveat: rubric validated against Claude's labels on 50 items from the smoke run (r = 0.80; slightly lenient on half-correct answers); with 30–50 questions per cell, differences below about 0.1 are within noise.

---

## Part 1 — Gaps in the product (why accuracy is lower than it could be)

| # | Gap | Evidence (from our files) | Why it happens | Possible fix | Size of problem |
|---|---|---|---|---|---|
| P1 | **The "word search" half is weak** | Word-only nDCG@10: FiQA 0.13, SciFact 0.49, MultiHop-RAG 0.40, GoT PDF 0.16. On FiQA it finds no correct page in the top 100 for 41% of questions. | On PostgreSQL it uses `ts_rank`, not real BM25: no weighting of rare words, no length correction, only the first 8 query words, and a short custom stop-word list (`postgres_store.py` `search_bm25`). The published BEIR paper has real BM25 baselines to compare with (https://arxiv.org/pdf/2104.08663); I could not confirm the exact figures here, so check its table before citing. | Use real BM25 (e.g. a BM25 extension or computing IDF), or weight the word channel lower. | **Large** |
| P2 | **Merging makes results worse than meaning-search alone** | Meaning-only beats merged on every dataset. Merged is worse than meaning-only for 56% of FiQA questions, 58% of GoT-PDF, 36% of SciFact, 27% of MultiHop-RAG (and helps only 5–15%). nDCG@10 drops by 0.14 to 0.36. | RRF gives the weak word list the same vote as the strong meaning list (`hybrid.py`, `RRF_K=60`, equal weights). | Weighted fusion, or fix P1 first, then re-measure. | **Large** |
| P3 | **Duplicate-removal throws away real passages** | RAGBench: 27.8% of passages share the same first 120 characters with another passage (they all start "Title: …"), and RRF loses the correct passage for ~14% of questions (recall@100 0.864 vs 1.00 for meaning-only). The GoT PDF has 1.2% (page header). | `deduplicate_candidates` drops any chunk whose first 120 characters match an earlier one (`hybrid.py`). | Compare whole-chunk hashes or similarity, not the first 120 characters. | **Medium** (large on data with shared headers) |
| P4 | **The reranker helps only a little, and only the top 24** | Gain over merged: SciFact +0.04, MultiHop-RAG +0.09, RAGBench +0.03; GoT/quantum PDFs +0.01 or −0.01 (no real change). It still ends below meaning-only everywhere. For FiQA 8% of correct pages sit below rank 24, where it cannot help. | Reranks 24 candidates and blends 60/40 with the merged score, so it inherits P1/P2 errors. | Rerank the meaning-search list too, or a wider window. | Medium |
| P5 | **The 12,000-character evidence limit starves multi-hop questions** | MultiHop-RAG: all the correct articles reach the writer in only 14–20% of questions (dense 20%, rrf+bge 14%). 8–16% of questions have a correct article cut off completely. | Both writers (`generate_grounded_answer` and `PromptAssembler`) cap evidence at 12k characters; news articles are long. | Raise the cap, or send only relevant sentences per article. | **Large for multi-hop** |
| P6 | **The writer refuses too often** | One-pass refuses 35% of MultiHop-RAG and 18% of RAGBench questions. Even when all correct articles are visible it still refuses 27% (MultiHop-RAG) and 12% (RAGBench). Two-pass cuts MultiHop-RAG refusals to 15% but raises quantum-PDF refusals from 10% to 21%. | Strict "answer only from evidence" prompts; partial evidence leads to refusal. | Allow partial answers with a caveat. | Medium |
| P7 | **The "engine as shipped" extras do not clearly pay off** | nDCG@10, engine vs merged+reranker: docs 0.776 vs 0.794, MultiHop-RAG 0.736 vs 0.727, GoT PDF 0.486 vs 0.419, quantum PDF 0.410 vs 0.460, RAGBench 0.744 vs 0.743, SciFact 0.738 vs 0.744: no consistent gain. The retry step never fired (0% of questions); "insufficient evidence" declines fired on 6% of quantum questions and 0% elsewhere. It returns at most 6 documents, so 15% of SciFact and 19% of quantum answers are not among those 6. On answers, the engine wrongly answered 2 of 9 unanswerable docs questions (merged+reranker: 0 of 9). | Rewriter/classifier/retry add cost and rarely change the outcome; the 6-document limit cuts recall. | Test each extra step separately (ablation) before keeping it. | Medium |
| P8 | **A product bug: "no thinking" is ignored for Gemini** | With the flag ignored, 9 of 24 (MultiHop-RAG) and 3 of 40 (docs) two-pass calls ran out of tokens; honoured: 0 of 24 and 0 of 40 (`smoke/thinking_flag_check.json`, small sample). | The stock `LLMClient.complete` ignores `enable_thinking=False` for Gemini. My harness works around it, so all numbers are for the *patched* behaviour. | Fix in `llm_client.py`. | Medium |
| P9 | **Silent fallbacks would hide failures** | Not triggered in our run (disabled on purpose). | Stock code falls back to fake (hash) embeddings, other model providers, and a heuristic reranker without telling the caller. | Fail loudly, or log a visible warning. | Medium (risk, not a measured loss) |
| P10 | **PostgreSQL store problems** | Fresh database fails to initialise with pgvector 0.8; default search width 40 cuts a 100-result query to 40 rows. | Untyped `vector` column with an HNSW index; `hnsw.ef_search` never set. | Typed column; set `ef_search`. | Medium |
| P11 | **Game-of-Thrones-specific rule in general code** | `is_anachronism` gate is hard-coded to one novel (fired 0 times outside it). | Corpus-specific logic in the shared engine. | Make it configuration. | Small |
| P12 | **It sometimes answers questions the documents cannot answer** | Docs set: 1 of 9 unanswerable questions answered by meaning-only, 2 of 9 by the full engine, 1 of 35 on MultiHop-RAG by the engine. Small numbers. | The writer sometimes fills in from general knowledge. | Stricter evidence check before answering. | Small (needs a bigger unanswerable set) |

---

## Part 2 — Gaps in my test setup (these can make the numbers wrong)

| # | Gap | Evidence | Effect | Fix |
|---|---|---|---|---|
| H1 | **I put the document title inside each chunk of text, which triggered P3 artificially** | Quantum PDF: all 296 pages shared the same first-120-characters; docs set: 23% collided with the title in, 0% without. | **Fixed and re-run.** nDCG@10 after the fix: quantum PDF merged 0.354 → 0.481, docs set merged 0.680 → 0.760, merged+reranker 0.702 → 0.794. Old rows archived in `archive_title_in_content_bug/`. | Done. |
| H2 | **The reranker service stopped working** | EcoHash ran out of credits twice (402). You then supplied a self-hosted `bge-reranker-v2-m3` on Colab; rankings were identical to EcoHash on all 242 RAGBench queries. That tunnel has since stopped resolving (01:00 on 20 Oct), which broke the last step of batch 2 (thinking-flag check). | All search and answer runs are complete. The thinking-flag check, the live speed test and any re-runs need the tunnel (or another reranker) back. | Restart the Colab server and send me the new URL. |
| H3 | **Network/sleep interruptions** | The Mac slept and the network dropped; workers died overnight. | Lost time only; rows are saved and resumable. | Keep the Mac awake and on power. |
| H4 | **Scope was reduced** | HotpotQA 300 queries / 30k docs (not 1,000 / 200k); RAGAS on 100 questions per dataset; latency on 15. | Not comparable with published HotpotQA; wider confidence ranges. | Stated in the report. |
| H5 | **SciFact looks too good** | Meaning-only nDCG@10 0.90 (recall@100 0.99). | Possibly a newer, stronger embedding model plus our document-level scoring; not verified. | Do not compare with leaderboards; check against a second embedding model if you want to publish. |
| H6 | **PDF labels are rough** | Only 56–60% of expected facts are supported by their listed pages (heuristic); the quantum questions list a median of 6 pages (90% list 4 or more). | Scores for the PDFs are low partly because of the labels. | Re-label (a human, or a stricter model pass), or use the ±1-page and "answer found" scores. |
| H7 | **Reference answers are weak in places** | Docs-set questions/answers were written by an AI and not human-reviewed; RAGBench references are model-written; MultiHop-RAG answers are 1–5 words. | Correctness scores are proxies. | Human spot-check (the blank `human_label` column). |
| H8 | **Judges are not independent** | Writer and judge are both Google models; the cross-check was done by me (Claude), who also built the harness; only 50 items, only two configs, no unanswerable questions, no PDFs. | Possible bias; these are model labels, not human labels. | Human-check 50 judgments; add unanswerable + PDF items to the cross-check. |
| H9 | **One RAGAS metric is invalid** | FactualCorrectness F1 vs Claude's labels: r = 0.09 (judge mean 0.27 vs Claude 0.88). | Do not use it. | Replaced by the rubric metric (r = 0.80, validated once on 50 items, a bit lenient on half-correct answers). |
| H10 | **Engine rows are not comparable at depth** | The full system returns ≤6 documents. | Recall@10/100 and nDCG@10 of that row are lower bounds. | Flagged in tables; compare Hit@k/R@5 only. |
| H11 | **"Correct page visible" was too generous** | I first counted a question as covered when *one* correct page was visible. | Overstated how often the writer had the evidence (MultiHop-RAG: 88% vs all-pages 14%). | Fixed in the gap analysis (now reports "all visible"). |
| H12 | **HotpotQA is too easy at this size** | With a 30,000-document corpus, meaning-only nDCG@10 is 0.98 (merged 0.87, merged+reranker 0.95). | Small corpus, every correct page included; barely separates the setups. Not comparable with published HotpotQA (millions of documents). | Use a larger corpus if you want it to discriminate. |

---

## Part 3 — Gaps in the earlier evaluations in this session

| # | Earlier evaluation | What was wrong or weak | Status |
|---|---|---|---|
| E1 | **First smoke test (SQLite, Gemini 2.x models)** | Used model families you ruled out; used a different database; later found about 40% of its judge scores were missing (async bug). | Superseded; not used in any report. |
| E2 | **Judge comparison with a larger Gemini model (exploration)** | Used a model outside the Gemini 3.x rule; small sample. | Dropped. |
| E3 | **Second smoke test (PostgreSQL, Gemini 3.x, 20 questions per dataset)** | Too few questions for any conclusion; pipeline check only. | Kept only as `smoke/`. |
| E4 | **RAGAS FactualCorrectness** | Invalid (see H9). | Replaced. |
| E5 | **Claude cross-check of faithfulness** | Good agreement (r = 0.93) but the scores are bunched near 1.0 (Claude mean 0.97), so rank agreement is weaker (ρ = 0.64). | Needs a harder sample (low-faithfulness answers). |
| E6 | **Thinking-flag check** | Small sample (12–20 questions). | Re-run on the full set once batch 2 is complete. |
| E7 | **Your estimated-results request** | I did not produce unlabelled estimates. | Declined by design. |
| E8 | **Sleep/credit failures in this run** | See H2/H3. | Open (credits). |

---

## What I suggest fixing, in order

1. **H1** (test bug): rebuild docs-set and PDF indexes without the title in the chunk text, then re-run them. Needs reranker credits for the reranked rows.
2. **P1 + P2** together: real BM25 and weighted merging, then re-measure. This is the biggest accuracy gain available.
3. **P3**: fix duplicate-removal.
4. **P5 + P6**: evidence limit and refusal behaviour (these decide answer quality on multi-hop questions).
5. **P7**: test the engine extras one at a time.
6. Then continue batch 3 (rubric correctness), batch 4 (RAGAS), batch 5 (speed, report).

Blocked right now: more reranked rows need the EcoHash account to have credits again.
