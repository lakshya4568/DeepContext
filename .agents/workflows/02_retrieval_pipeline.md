---
description: Retrivel Pipeline
---

# Workflow: Retrieval Pipeline

The default path for the majority of queries. Implements FR1, FR3, FR4, FR5, FR6.

## Steps

```text
1. Classify the query
     → factual lookup / how-to / multi-hop / aggregation / navigation
     (this classification also feeds the router in
      03_rlm_recursion_pipeline.md — it's one classifier call, not two)

2. Rewrite/decompose if needed
     → vague or multi-part queries get rewritten into 1–3 focused
       sub-queries before retrieval, not answered against the raw
       user phrasing

3. Apply filters
     → tenant_id, permission_scope, document date range, doc_type
     → these run as SQL WHERE clauses against chunks/documents,
       never as a "please only use permitted docs" instruction to the model

4. First-stage recall (run BOTH, in parallel)
     BM25 leg:    SELECT ... WHERE tsv @@ websearch_to_tsquery(...)
                  ORDER BY ts_rank(...) LIMIT 50-100
     Vector leg:  SELECT ... ORDER BY embedding <=> query_embedding
                  LIMIT 50-100

     If document.retrieval_mode = 'vectorless' for the target document(s):
     skip BM25/vector entirely and use tree navigation instead —
     see "Vectorless navigation" below.

5. Fuse with Reciprocal Rank Fusion (RRF)
     score(chunk) = Σ 1 / (k + rank_in_list)   for each list the chunk
                                                 appears in (k≈60 is a
                                                 reasonable default)
     → produces one ranked candidate list from the two legs

6. Deduplicate
     → collapse near-identical children (e.g. overlapping windows that
       both matched) before spending reranker budget on near-duplicates

7. Rerank
     → cross-encoder reranker scores the top 50-100 candidates against
       the (rewritten) query
     → keep top 5–10

8. Resolve child → parent
     → for each surviving child chunk, fetch its parent_chunk_id's
       content — THIS is what goes to the model, not the child text
       that was actually searched

9. Evidence sufficiency check (shared gate — see ARCHITECTURE.md §7)
     sufficient?
       Yes → generate answer with citations back to source chunk IDs
       No  → go to step 2 with a rewritten query (max 1 retry)
             still insufficient after retry →
               a) if the query looked like "read everything" /
                  aggregation in step 1 → escalate to RLM
                  (03_rlm_recursion_pipeline.md)
               b) otherwise → return "insufficient evidence" as the
                  answer, not a guess
```

## Vectorless navigation (for `retrieval_mode = 'vectorless'` documents)

```text
1. Start at the document's root document_tree_nodes row
2. Feed the model: [question] + [node summaries of immediate children]
3. Model picks which child node(s) look relevant
4. Descend into the picked node(s); repeat step 2-3 until reaching a
   leaf node (which has chunk_id set)
5. Fetch the leaf chunk's parent content, same as step 8 above
6. Same evidence-sufficiency gate applies before answering
```

This trades "one hop, top-k similarity" for "several hops, LLM-judged relevance at each level" — slower per query,
but it respects document hierarchy instead of imposing arbitrary chunk boundaries, which is the point for
contracts, filings, and manuals (see `PRD.md` FR6).

## What NOT to do (the source verdict's warnings, restated as pipeline rules)

- **Don't skip reranking to save a call.** First-stage recall is optimized for recall, not precision — the
  reranking step is usually the single highest-leverage addition once basic retrieval works (source brief,
  reiterated here because it's easy to cut under time pressure and it's the wrong thing to cut).
- **Don't retrieve 50 chunks and send all of them to the model "to be safe."** More context is not automatically a
  better answer — it increases cost, dilutes attention, and can introduce contradictions between chunks the model
  then has to silently arbitrate. Top 5–10 after reranking is the target, not a floor to pad past.
- **Don't let the corrective-retry loop run unbounded.** One rewrite-and-retry, then escalate or admit
  insufficiency — an unbounded "keep trying different queries" loop is how a retrieval pipeline turns into an
  expensive infinite loop with nothing to show for it.
