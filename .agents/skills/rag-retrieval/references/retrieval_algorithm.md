# Retrieval Algorithm — Condensed Reference

Full discussion and rationale: `workflows/02_retrieval_pipeline.md`. This file is the quick-lookup version —
formulas and defaults, not the "why."

## Reciprocal Rank Fusion (RRF)

```text
score(chunk) = Σ 1 / (k + rank_in_list)     for each first-stage list the chunk appears in
```

- `k ≈ 60` is a reasonable default (standard RRF constant; dampens the impact of rank-1 dominance from either
  leg).
- Run BM25 and vector recall **in parallel**, each returning top 50–100, before fusing. Fusing after a smaller
  top-k throws away recall the fusion step needs to work with.

## First-stage recall

| Leg | Query | Default limit |
|---|---|---|
| BM25 | `SELECT ... WHERE tsv @@ websearch_to_tsquery(...) ORDER BY ts_rank(...)` | 50–100 |
| Vector | `SELECT ... ORDER BY embedding <=> query_embedding` | 50–100 |

Both legs run against `chunks` (child-level rows only — parents are not embedded by default; see
`docs/DATA_MODEL.sql`).

## Reranking

- Cross-encoder reranker, self-hosted by default (`docs/TECH_STACK.md` §5): BGE- or Qwen3-Reranker-family,
  ~0.5–4B params.
- Scores the top 50–100 fused candidates against the (rewritten) query; keep top 5–10.
- **Measure Hit@k on your own labeled subset before trusting any vendor benchmark number.** The source brief's
  cited 62.67%→83% Hit@1 jump is a single-benchmark, single-reranker, single-dataset result — treat it as "this
  plausibly matters a lot," not as a number that reproduces on your corpus (`docs/TECH_STACK.md` §5).

## Parent resolution

For each surviving (post-rerank) child chunk, fetch `chunks.parent_chunk_id`'s row. The **parent's** `content`
is what goes into the generation prompt. The child's `content` was only ever used for the search match itself.

## Evidence sufficiency gate

Shared with the RLM path (`docs/ARCHITECTURE.md` §7). A draft answer passes if:

1. Every non-trivial claim traces to a retrieved chunk, a computed value, or an explicit "I'm inferring this."
2. For aggregation-style queries, evidence spans the *full* retrieved set, not a sampled subset.
3. The verifier's own confidence score (see `skills/verification/`) clears its threshold.

On failure: rewrite the query, retry retrieval once. Still insufficient →
- looks like "read everything" / aggregation → escalate to `skills/rlm-orchestrator/`
  (`workflows/03_rlm_recursion_pipeline.md`)
- otherwise → return "insufficient evidence" as the answer, not a guess

## Switch criteria (vector store / full-text engine)

Don't switch preemptively. Concrete triggers, from `docs/TECH_STACK.md` §3–§4:

- **pgvector → Qdrant/Milvus/Weaviate:** P95 vector-search latency exceeds budget *after* confirming the index
  is tuned for your actual row count, OR corpus reaches tens of millions of vectors with high write throughput,
  OR you need multi-region replication semantics.
- **Postgres tsvector → Meilisearch/Elasticsearch:** you need faceted search, typo tolerance, or multi-language
  stemming Postgres FTS handles poorly, or FTS query latency becomes the bottleneck at scale.

If/when you do switch, only this skill's internals change — the `retrieve()` interface contract in `SKILL.md`
doesn't leak the backend to callers.
