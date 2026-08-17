# Deep Context RAG Improvement Plan

## 0. Purpose

This document captures the **known issues** in the current RAG architecture (Gemini embeddings + cross-encoder reranker + agentic layer) and defines a **step-by-step plan** to fix them, with explicit re‑evaluation targets after each change.

The goal is **higher factual consistency, better ranking, safe abstention, and lower latency** without adding unnecessary new agents or memory stores.

---

## 1. Current Architecture Snapshot

**Retrieval:**

- Parent–child hierarchical chunking over a book corpus and related docs.
- Dual sparse/dense retrieval:
  - BM25 full-text.
  - Gemini dense embeddings (e.g. `text-embedding-005` or similar).
- Hybrid fusion via **Reciprocal Rank Fusion (RRF)**.
- Cross-encoder reranker on top of hybrid candidates.
- Parent chunk resolution with deterministic citations.

**Memory:**

- 4-store typed memory:
  - System policies.
  - User preferences.
  - Semantic facts.
  - Episodic summaries.
- Promotion gate, TTL, confidence decay.

**Generation & Agentic:**

- LLM (Gemini) for answer generation.
- Agentic router deciding:
  - Direct generation.
  - Hybrid RAG.
  - Multi-hop decomposition.
  - Recursive RLM subagents for complex queries.
- Existing planner for multi-hop and paraphrased queries, not fully wired into RAG yet.

---

## 2. Evaluation Recap (Baseline)

On a 36-query benchmark (direct, multi-hop, paraphrased, ambiguous/noise, citation-seeking, unanswerable/adversarial):

**Retrieval quality:**

- Context recall: **88.17%** (facts are usually in retrieved chunks).
- Hybrid RRF Hit@5: **45.16%**.
- Full pipeline Hit@5 (with reranker): **16.13%**.
- BM25-only Hit@5: **32.26%**.
- Dense-only Hit@5: **38.71%**.

**Generation & reasoning:**

- Citation precision: **100%**.
- Citation recall: **98.79%**.
- Answer relevancy: **76.16%**.
- Faithfulness: **62.08%**.
- Factual correctness F1: **0.4445**.
- Answer completeness: **42.79%**.
- Abstention accuracy (on unanswerable): **0%**.

**Latency:**

- Retrieval median: **368 ms**, p95: **633 ms**.
- Generation median: **1.34 s**, p95: **~28 s**.
- Total turn median: **1.79 s**, p95: **~28 s**.

**Key category failures:**

- Direct factual: good faithfulness, incomplete extraction.
- Multi-hop: one hop answered, others dropped.
- Paraphrased: Hit@5 ≈ 0%, lexical mismatch.
- Citation/appendix: correct section hinted, wrong page ranked.
- Unanswerable: model hallucinates instead of refusing.

---

## 3. Root Causes (Current Issues)

### 3.1 Ranking is worse than retrieval

- Hybrid RRF alone reaches **45% Hit@5**.
- Introducing cross-encoder reranking **drops** Hit@5 to **16%**.
- This indicates:
  - Reranker is miscalibrated for this domain.
  - Candidate pool is too narrow before rerank.
  - Reranker scores are **overwriting** rather than **blending** RRF scores.

### 3.2 Grounding and abstention are weak

- High context recall + moderate faithfulness + low completeness.
- Model has enough evidence but:
  - Doesn’t extract all factual bullets.
  - Adds unsupported claims instead of **refusing** on unanswerable queries.
- No hard retrieval-score gate before generation.
- Refusal is handled via prompt only, not enforced by logic.

### 3.3 Paraphrased queries break BM25 and dense

- Queries deliberately avoid key book terms.
- BM25 fails because lexical overlap is low.
- Dense retrieval still struggles because:
  - Embedding generalization is imperfect for small, specialized corpora.
- No query expansion / rewriting step tailored to in-corpus vocabulary.

### 3.4 Multi-hop questions are treated as single-hop

- Multi-entity questions:
  - Bran’s assassination.
  - Littlefinger’s dagger lie.
  - Catelyn’s seizure.
- Single-shot retrieval grabs only one entity’s chunks.
- Planner exists but isn’t consistently used to:
  - Split queries.
  - Retrieve per sub-question.
  - Union evidence.

### 3.5 Appendix / citation queries ignore section metadata

- Queries explicitly target “appendix”, “house words”, “sigil”, indices.
- Index has `section_path` (appendix vs chapters) but:
  - Retrieval doesn’t boost or filter by these patterns.
- Result: correct facts exist in appendix pages but rarely rank top.

### 3.6 Gemini-specific considerations

- Embedding model is strong, but:
  - Reranker may not be optimized for your corpus (fiction, narrative, references).
  - Embeddings may be used without:
    - Normalization.
    - Proper chunk metadata.
    - Hybrid tuning per domain.
- Vector space may be slightly misaligned across:
  - Narrative text.
  - Tables/appendices.
  - Evaluation reference answers.

### 3.7 Latency spikes tied to recursive behavior

- p95 at ~28s indicates:
  - Deep recursion in agent/RLM loop.
  - Multi-hop attempts with no strict hop or time budget.
  - Long context windows being used across multiple turns.

---

## 4. Step-by-Step Fix Plan

> **Important:** Change **one dimension at a time**, re-run the same 36-query eval after each phase, and log metrics: Hit@5, context recall, faithfulness, F1, abstention, p95 latency.

### Phase 1 – Fix Ranking

**Goal:** Match or exceed hybrid RRF Hit@5 with reranking, or bypass reranker entirely if it cannot outperform RRF.

**Steps:**

1. **Bypass reranker initially:**

   - Set `use_reranker=False` in retrieval pipeline.
   - Use **hybrid RRF** as the production ranker:
     - BM25 top‑50 + dense top‑50 → RRF → top‑60 candidates.
   - Expand parents and siblings:
     - For each child hit, include parent + previous/next sibling to avoid page-boundary fragmentation.

2. **Log rank trajectories:**

   - For each query:
     - Log gold pages (from eval labels).
     - Log their ranks in:
       - BM25-only.
       - Dense-only.
       - RRF-only.
   - This becomes your “pre-rerank” baseline.

3. **Introduce blended rerank (optional, only if it helps):**

   - Rerank a **wider pool** (top‑60 candidates).
   - Blend scores:

     ```python
     rrf_scores = minmax(rrf_rank_scores)
     rerank_scores = minmax(cross_encoder_scores)
     final_score = 0.6 * rrf_scores + 0.4 * rerank_scores
     ```

   - Never allow reranker to **drop** a candidate that both BM25 and dense consider highly relevant.

4. **Re-evaluate:**

   - Target:
     - Hit@5 ≈ hybrid baseline (~45%).
     - Context recall ≥ 88%.
   - If reranked Hit@5 does **not** exceed RRF-only, keep reranker **off** for this corpus.

### Phase 2 – Grounded Generation & Refusal Gate

**Goal:** Use retrieved context strictly; avoid hallucinations; refuse safely when evidence is insufficient.

**Steps:**

1. **Hard retrieval-score gate:**

   - Aggregate relevance scores of final chunks (per query).
   - If:
     - Mean similarity < threshold (e.g. 0.2–0.3, calibrated).
     - OR no chunk above threshold.
   - Then **skip generation** and respond with refusal template.

2. **Refusal template (hard contract):**

   - Enforce:

     > “Based on the provided context, there is insufficient evidence to answer.”

   - Do not allow:
     - Outside knowledge.
     - Guesses.
   - Implement as:
     - Pre-generation check in code (not only in prompt).
     - Model prompt telling it to **respect this gate**.

3. **Two-pass grounded answer generation:**

   - Pass 1: **Extract bullets** of claims supported by context.
   - Pass 2: **Write final answer only from those bullets**, adding citations you already track.
   - Example structure:

     ```text
     1. List each factual statement the context supports.
     2. Mark unsupported requested facts explicitly.
     3. Write a concise answer that:
        - Mentions only supported statements.
        - Says what cannot be answered from context.
     ```

4. **Re-evaluate:**

   - Targets:
     - Faithfulness ≥ 75%.
     - Factual F1 ≥ 0.6.
     - Abstention accuracy ≥ 80% on unanswerable/adversarial.
   - Keep citation precision ≈ 100%; citation recall ≈ 100%.

### Phase 3 – Paraphrase-Aware Query Rewriting

**Goal:** Make paraphrased queries recover book vocabulary and concepts.

**Steps:**

1. **Add a lightweight query expansion stage:**

   - Before retrieval, run an LLM prompt:

     > “Rewrite the question into 2–3 search queries using likely in‑corpus names, objects, and places. Keep the original query too.”

   - Example:
     - Input: “bizarre beast frozen in snow with horn fragments”.
     - Rewrites:
       - “dead direwolf found in snow shattered antler throat pups”.
       - “antler in direwolf neck pups separated from mother”.

2. **Hybrid retrieval per query variant:**

   - Run BM25 + dense + RRF for:
     - Original query.
     - Each rewrite.
   - Fuse all candidates with another RRF step.

3. **Limit cost:**

   - Cap:
     - Number of rewrites (e.g. 2).
     - Combined candidate pool (e.g. ≤ 80).
   - Focus rewriting on:
     - Paraphrase category queries.
     - Optional heuristic: low lexical overlap with corpus hints.

4. **Re-evaluate paraphrased category:**

   - Target:
     - Paraphrased Hit@5 ≥ 30%.
     - Factual F1 in that category ≥ 0.45.

### Phase 4 – Multi-Hop Retrieval via Planner

**Goal:** Cover each hop in multi-entity queries before generation.

**Steps:**

1. **Use the existing planner for multi-hop:**

   - Detect multi-hop queries (multiple entities, events, or conditions).
   - Split into sub-questions (planner already exists in `agentic/planner.py`).

2. **Retrieve per sub-question:**

   - Run hybrid RRF for each sub-question independently.
   - For each sub-question, ensure:
     - At least one high-relevance chunk is retrieved.

3. **Union and dedupe parents:**

   - Combine all parent chunks across sub-questions.
   - Deduplicate by:
     - Document id.
     - Page or section.

4. **Coverage check:**

   - Only pass to generation when:
     - Each sub-question has at least one supporting chunk.
   - Otherwise:
     - Retrieve again with wider k.
     - Or abstain.

5. **Re-evaluate multi-hop category:**

   - Target:
     - Multi-hop factual F1 ≥ 0.55 without lowering single-hop F1.
     - Faithfulness remains ≥ 80% in multi-hop queries.

### Phase 5 – Appendix / Section-Aware Retrieval

**Goal:** Rank appendix and reference sections correctly when queries mention them.

**Steps:**

1. **Enhance metadata:**

   - Ensure chunks have:
     - `section_path` (e.g. `chapter/17`, `appendix/house_tully`, `appendix/house_stark`).
     - `doc_type` (chapter, appendix, index, glossary).

2. **Appendix-aware query classifier:**

   - If query contains:
     - “appendix”, “index”, “sigil”, “words of House”, “sworn houses”.
   - Then:
     - Filter or boost `section_path` containing appendix markers.

3. **Boost strategy:**

   - For such queries:
     - Multiply relevance for appendix chunks by a factor (e.g. 1.5–2).
     - Keep non-appendix candidates, but lower their rank.

4. **Re-evaluate citation category:**

   - Target:
     - Citation Hit@5 ≥ 50%.
     - Factual F1 ≥ 0.7 in citation-seeking category.

### Phase 6 – Gemini Embeddings & Reranker Tuning

**Goal:** Make Gemini embeddings and reranker fit the corpus and retrieval stack.

**Steps:**

1. **Confirm embedding model choice:**

   - Use the **recommended Gemini embedding model for RAG**, e.g. `text-embedding-005` for general text corpora.
   - Ensure:
     - Consistent normalization.
     - Same model used for all embedding runs (no mixing models).

2. **Rebuild embeddings with improved chunking:**

   - After adjusting parent-child relationships and metadata:
     - Regenerate embeddings for child chunks.
   - Optionally:
     - Use section-aware or late chunking for particularly long documents.

3. **Evaluate alternative rerankers on your eval set:**

   - Try:
     - Cross-encoder variants better suited for narrative text or your language.
     - Possibly Gemini-based reranker if available.
   - For each candidate reranker:
     - Measure:
       - Hit@5 on your 36-query benchmark.
       - MRR, nDCG.
       - Faithfulness and F1.

4. **Adopt reranker only if it beats RRF on this benchmark:**

   - If reranker + blend > RRF-only on:
     - Hit@5.
     - Factual F1.
   - Then enable it; otherwise keep RRF-only.

5. **Consider a simple Gemini-only dense retrieval baseline:**

   - As a sanity check, use pure Gemini embeddings for dense retrieval and compare to hybrid RRF and BM25-only.
   - Use these results to justify hybrid’s complexity or adjust weights.

### Phase 7 – Latency & RLM Budgeting

**Goal:** Keep interactive latency within a few seconds, prevent runaway recursion.

**Steps:**

1. **Introduce hop and time budgets:**

   - For each query:
     - Max recursion depth: 2.
     - Max wall-clock time per turn: ~6 seconds.
   - If budgets are exceeded:
     - Answer with current evidence.
     - Or abstain.

2. **Prefer “local” RAG over full RLM for simple queries:**

   - Use:
     - Direct hybrid RAG + single generation.
   - Reserve RLM-only flows for:
     - Very long tasks.
     - Global analysis over many documents.

3. **Metric tracking:**

   - Record:
     - p50, p95 latency per stage (retrieval, generation).
     - Number of recursive hops.
   - Aim:
     - Total p95 latency < 6 seconds for the 36-query eval.

---

## 5. Architectural Additions & Refinements

Beyond the fixes above, several architectural improvements will make the system more robust:

### 5.1 Separate Retrieval Service

- Move retrieval into its own service (microservice or module) with its **own logs, metrics, and config**.
- Benefits:
  - Independent tuning of BM25, dense, hybrid, rerank.
  - Easier to A/B test retrieval strategies.
  - Clear boundary between RAG and agent orchestration.

### 5.2 Query & Context Observability

- Integrate RAG evaluation into continuous monitoring.
- Store per-query:
  - Retrieval configs (k, filters, hybrid weights).
  - Top 10 candidate ranks before/after rerank.
  - Metrics: faithfulness, answer relevancy, context precision/recall.
- Use dashboards to watch:
  - Drift over time.
  - Impact of new chunks and new models.

### 5.3 Domain-Aware Chunking Profiles

- Maintain **profiles** per corpus type:
  - Narrative book.
  - Technical docs.
  - Appendices/indexes.
- For each profile:
  - Tune:
    - Child/parent sizes.
    - Overlaps.
    - Metadata fields.

### 5.4 RLM Integration Point

- Treat RLM as a **special mode**:
  - For “read everything and aggregate” tasks.
  - For research over large static corpora.
- RLM can:
  - Call the retrieval service as a tool.
  - Use typed memory for state.
  - Store intermediate results in the REPL environment.
- Do not let RLM bypass the retrieval metrics; log its usage and compare success rate vs normal RAG flows.

---

## 6. Re-Evaluation Checklist

For each phase, re-run the same 36-query benchmark and log:

- Retrieval:
  - Hit@5, Hit@1.
  - MRR, nDCG.
  - Context precision & recall.
- Generation:
  - Faithfulness.
  - Answer relevancy.
  - Factual F1.
  - Completeness.
  - Citation precision & recall.
  - Abstention accuracy.
- Latency:
  - p50/p95 per stage and total.

**Targets summary:**

- After Phase 1: Hit@5 near hybrid RRF baseline (~45%).
- After Phase 2: Faithfulness ≥ 75%, Abstention ≥ 80%.
- After Phase 3: Paraphrase Hit@5 ≥ 30%, F1 ≥ 0.45.
- After Phase 4: Multi-hop F1 ≥ 0.55.
- After Phase 5: Citation Hit@5 ≥ 50%, F1 ≥ 0.7.
- After Phase 6: Hybrid + tuned rerank (if adopted) beats RRF-only.
- After Phase 7: p95 latency < 6 seconds.
