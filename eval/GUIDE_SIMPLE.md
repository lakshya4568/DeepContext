# The benchmark in simple words

## 1. What are we trying to find out?

Your system (DeepContext) answers questions from documents. Two things can go wrong:

1. **It finds the wrong pages** (search problem).
2. **It finds the right pages but writes a wrong or made-up answer** (writing problem).

So we test both, on questions where *someone else already knows the right answer*. That is all a "benchmark" is:
a list of questions + the correct pages + the correct answer, so we can grade the system like an exam.

## 2. What was done, start to finish

1. **Read your project and its docs**, and set up a separate branch (`eval/public-benchmarks`) so `main` is never touched.
2. **Built a test harness in the `eval/` folder.** It runs the real DeepContext search and answer code, with safety switches so nothing silently falls back to fake or backup models (that would give wrong results).
3. **Downloaded public exams** (see section 5) and added your own (your docs and two PDFs).
4. **Stored every document in PostgreSQL** (as you asked), in a database called `deepcontext_eval`, one section per exam.
5. **Ran search 6 different ways** (see section 4) on every question, and saved each result.
6. **Asked the AI to write answers** for each way, and saved them.
7. **Graded the answers** with a second AI (a different model from the writer) and with Claude on a small sample, to check the grader itself.
8. **Made tables, charts of numbers and a report** with error ranges, so we know what is real and what is luck.

Current state: most search results are done; the reranker service ran out of credits, so the reranked ways are incomplete; the biggest dataset (HotpotQA) is still being loaded.

## 3. The architecture (how the pieces connect)

```
 Question
    |
    v
 [Search]  --> two searchers run side by side:
              - "meaning search" (turns text into numbers; finds similar meaning)
              - "word search"   (finds matching words)
              then their lists are merged (called RRF)
    |
    v
 [Reranker, optional] a smarter model re-sorts the top results (BGE, a hosted service)
    |
    v
 [Writer AI: Gemini 3.8] reads the best pages and writes the answer
    |   (one-pass = writes directly; two-pass = first pulls out facts, then writes from them)
    v
 Answer  --> [Grader AI: Gemini 3.7] scores it --> tables + report

 Storage under everything: PostgreSQL (documents, pieces of documents, number-fingerprints)
 Cache: every AI call is saved so re-running is free and gives the same answers
```

## 4. The 6 "ways" we compare

| Name | Plain meaning |
|---|---|
| dense_only | search by meaning only |
| bm25_only | search by matching words only |
| rrf | meaning + words, merged |
| rrf_bge | the merged list, re-sorted by a smarter model |
| rrf_bge_two_pass | same search, but the answer is written in two careful steps |
| engine_as_shipped | your whole real system as it is (rewrites the question, retries, etc.) |

## 5. The datasets (the "exams")

| Exam | What it is | Where it comes from |
|---|---|---|
| SciFact | check scientific claims against paper abstracts | https://huggingface.co/datasets/BeIR/scifact |
| FiQA | finance questions from forums | https://huggingface.co/datasets/BeIR/fiqa |
| HotpotQA | questions that need two Wikipedia pages | https://huggingface.co/datasets/BeIR/hotpotqa |
| MultiHop-RAG | news questions needing several articles | https://huggingface.co/datasets/yixuantt/MultiHopRAG |
| RAGBench (CovidQA) | covid research questions with reference answers | https://huggingface.co/datasets/galileo-ai/ragbench |
| RGB | tests ignoring noise and refusing when there is no answer | https://github.com/chen700564/RGB |
| docsqa | 44 questions about *your* project docs | made here: `eval/docs_qa_items.py` |
| Game of Thrones PDF, quantum thesis PDF | your 36 questions each | `tests/eval_dataset.json`, `tests/eval_quantum_dataset.json` |

**Where the files are on your Mac** (the `.claude` folder is hidden in Finder; press Cmd+Shift+. to show it, or I can open it for you):

- Folder: `/Users/proximus/Documents/antigravity/RAG/.claude/worktrees/eval-public-benchmarks/`
- Readable samples of every dataset: `eval/dataset_samples/*.md`  (open these first)
- Full downloaded data: `eval/cache/data/` (e.g. `hotpotqa_sub_q1000_d200000_s42.json`)
- Summary of every dataset used: `eval/results/dataset_meta.json`
- Every question's result rows: `eval/results/raw_retrieval.jsonl`, `raw_generation.jsonl`

## 6. What the scores mean (no jargon)

Think of searching as "the system hands me a ranked list of pages; where is the correct page in the list?"

| Score | Plain meaning | Good / bad |
|---|---|---|
| **Hit@5** (hit rate) | In what share of questions is the correct page somewhere in the top 5? | 0.9 = great. 0.5 = misses half the time. |
| **MRR** | How high is the *first* correct page, on average? (1st place = 1, 2nd = 0.5, 3rd = 0.33...) | 0.8+ good, below 0.4 weak |
| **Recall@100** | Is the correct page anywhere in the top 100? (the "was it even found" test) | If this is low, nothing later can fix it |
| **nDCG@10** | Score for the whole top 10: right pages near the top give more points | 0.7+ good, 0.4 so-so, 0.2 poor |
| **Faithfulness** | Does every claim in the answer come from the pages it was shown? (low = made-up facts) | 0.9+ good |
| **Correctness (rubric)** | Does the answer say what the reference answer says? 1 / half / 0 | 0.8+ good |
| **Abstention** | Does it say "I don't know" when the answer isn't in the documents? Good for unanswerable questions, bad for answerable ones | |

All of these run from 0 (worst) to 1 (best). The `[ ]` ranges next to numbers are 95% confidence ranges: if two ranges overlap a lot, the difference may be just luck.

## 7. A first peek at real results so far (from our files, partial)

| Exam | Meaning-only nDCG@10 | Words-only | Merged (rrf) |
|---|---|---|---|
| SciFact | 0.90 | 0.49 | 0.70 |
| FiQA | 0.61 | 0.13 | 0.38 |
| MultiHop-RAG | 0.80 | 0.40 | 0.64 |
| RAGBench-Covid | 0.85 | 0.59 | 0.71 |

What this suggests (not final):
- Meaning search is strong. The word search used on PostgreSQL is weak (it is not true BM25; it is a simpler PostgreSQL ranking).
- Merging with that weak word search *pulled the score down* versus meaning-only. That is a real finding about your setup, not a bug in the test.
- The reranker helps a little over the merged list (e.g. SciFact 0.70 to 0.74), but is still below meaning-only.

Caution: these are partial numbers and SciFact looks high versus published leaderboards; the final report will explain the differences (we use different chunking, and the embedding model is newer).
