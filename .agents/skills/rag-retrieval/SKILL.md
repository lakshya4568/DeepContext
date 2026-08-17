---
name: rag-retrieval
description: Implements the Deep Context Platform's hybrid retrieval pipeline — BM25 full-text + dense vector recall, Reciprocal Rank Fusion, cross-encoder reranking, parent-child chunk resolution, and the corrective-retrieval / evidence-sufficiency gate. Use this skill whenever a question needs to be answered from ingested documents or a codebase, whenever you're implementing, calling, or debugging the retrieve() interface, whenever retrieval quality (recall, precision, reranking, hybrid search) is in question, or whenever a document needs to be ingested and chunked before it's searchable. Also use it to decide whether a query should stay on the hybrid path or be routed elsewhere (RLM, vectorless navigation) per FR1–FR6.
---

# RAG Retrieval Skill

Implements FR1–FR6 of `docs/PRD.md`. This is the **default path** for the large majority of queries — read
`docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §1 for why this skill alone, without the RLM engine, is a complete and
useful system on its own.

## When to reach for the two workflows this skill wraps

- **Ingesting a new or changed document** → follow `workflows/01_ingestion_pipeline.md` exactly. Don't
  freelance the chunking token budgets or the retrieval_mode decision (step 3 of that workflow) — they encode
  real tradeoffs (fact-heavy vs. narrative bias, structured vs. default documents).
- **Answering a query against already-ingested documents** → follow `workflows/02_retrieval_pipeline.md`
  exactly, including the RRF fusion formula, the reranking step, and the evidence-sufficiency gate. See
  `references/retrieval_algorithm.md` for the condensed algorithm if you don't need the full workflow doc's
  discussion of *why* each step exists.

## The `retrieve()` interface contract

Every caller (the agentic planner, the RLM kernel's `retrieve()` stub, a direct skill invocation) goes through
one function signature so the retrieval backend (pgvector today, per `docs/TECH_STACK.md` §3) can be swapped
without touching callers:

```python
def retrieve(
    query: str,
    *,
    tenant_id: str,
    permission_scope: list[str],
    document_ids: list[str] | None = None,  # None = search the whole permitted corpus
    date_range: tuple[str, str] | None = None,
    top_k: int = 8,  # post-rerank result count
) -> RetrievalResult:
    """Returns ranked (parent) chunks with citations, or an insufficient-evidence signal.
    See scripts/retrieve.py for the reference implementation of every step below.
    """
```

`permission_scope` and `tenant_id` are **never** "please only use permitted docs" text in a prompt — they
compile to real SQL `WHERE` clauses against `chunks`/`documents` (FR5). This is non-negotiable; a permission
leak here is a data leak, not a UX bug.

## Core rules (violating any of these is the most common way this pipeline degrades)

1. **Don't skip reranking to save a call.** First-stage recall (BM25 + vector, top 50–100) is optimized for
   recall, not precision. The cross-encoder rerank step is usually the single highest-leverage addition once
   basic retrieval works — it's also the easiest thing to cut under time pressure, and it's the wrong thing to
   cut.
2. **Resolve child → parent before generation.** The child chunk (300–600 tokens) is what was *searched*; the
   parent chunk (1,000–2,500 tokens) is what goes to the model. Never send raw search-matched child text to the
   generator.
3. **Top 5–10 after reranking is the target, not a floor.** Sending 50 "raw" chunks to be safe increases cost,
   dilutes attention, and forces the model to silently arbitrate contradictions between chunks — this is a
   regression, not a safety margin.
4. **The corrective-retry loop is bounded at 1.** One rewrite-and-retry, then escalate (to RLM, if the query
   looks like aggregation) or return "insufficient evidence." An unbounded retry loop is an expensive infinite
   loop with nothing to show for it — see `workflows/02_retrieval_pipeline.md`'s explicit warning on this.
5. **Vectorless documents skip BM25/vector entirely.** If `documents.retrieval_mode = 'vectorless'` for the
   target document, use tree navigation (`document_tree_nodes`) instead — see the "Vectorless navigation"
   section of `workflows/02_retrieval_pipeline.md`. Don't run both; the retrieval_mode decision was already made
   at ingestion time (step 3 of the ingestion workflow) precisely so you don't have to re-decide per query.

## Files in this skill

- `references/retrieval_algorithm.md` — condensed RRF formula, reranker notes, and the switch criteria for
  moving off pgvector/Postgres FTS (mirrors `docs/TECH_STACK.md` §3–§5 so you don't have to open the full tech
  stack doc for a quick check).
- `scripts/retrieve.py` — reference implementation of the `retrieve()` contract above: query classification,
  rewrite/decompose, filtered first-stage recall (BM25 + vector), RRF fusion, dedup, rerank stub, parent
  resolution, and the sufficiency gate. It's written as a clear reference to adapt, not a drop-in production
  module — the SQL and reranker calls are stubbed with clear `# TODO` markers where your actual pgvector
  connection and reranker model go.

## What NOT to do

See `workflows/02_retrieval_pipeline.md`'s "What NOT to do" section — it's restated there deliberately because
these are the three mistakes most likely to happen under time pressure: skipping the reranker, over-retrieving
"to be safe," and letting the corrective loop run unbounded.
