"""
Reference implementation of the retrieve() contract described in SKILL.md.

This is a clear, readable reference to adapt — not a drop-in production module. Database access,
the embedding call, and the reranker call are stubbed with TODO markers where your actual
connections go. Follow workflows/02_retrieval_pipeline.md step-by-step; this file mirrors that
workflow's numbered steps in its function names/comments.

Requires (when wired up for real): asyncpg or psycopg (Postgres + pgvector), an embedding client,
and a cross-encoder reranker client (self-hosted per docs/TECH_STACK.md §5, or a hosted API).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryShape(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    HOW_TO = "how_to"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    NAVIGATION = "navigation"


@dataclass
class RetrievalFilters:
    tenant_id: str
    permission_scope: list[str]
    document_ids: list[str] | None = None
    date_range: tuple[str, str] | None = None


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    section_path: str | None
    page_number: int | None


@dataclass
class RetrievalResult:
    sufficient: bool
    parent_chunks: list[dict] = field(default_factory=list)  # [{content, citation}, ...]
    citations: list[Citation] = field(default_factory=list)
    query_shape: QueryShape | None = None
    retry_count: int = 0
    insufficiency_reason: str | None = None


# ---------------------------------------------------------------------------
# Step 1 — classify the query (shared with the RLM router; one call, two consumers)
# ---------------------------------------------------------------------------


def classify_query(query: str) -> QueryShape:
    """TODO: replace with a real LLM classifier call. This heuristic stub exists only so the
    rest of the pipeline is runnable/testable without a model call."""
    q = query.lower()
    if any(w in q for w in ("every", "all of", "each of", "don't miss", "none of")):
        return QueryShape.AGGREGATION
    if any(w in q for w in ("how do i", "how to", "steps to")):
        return QueryShape.HOW_TO
    if q.count("?") > 1 or " and " in q:
        return QueryShape.MULTI_HOP
    if any(w in q for w in ("where is", "find", "show me", "list")):
        return QueryShape.NAVIGATION
    return QueryShape.FACTUAL_LOOKUP


# ---------------------------------------------------------------------------
# Step 2 — rewrite / decompose
# ---------------------------------------------------------------------------


def rewrite_query(query: str, shape: QueryShape) -> list[str]:
    """Vague or multi-part queries get rewritten into 1-3 focused sub-queries.
    TODO: replace with an LLM rewrite call for MULTI_HOP / AGGREGATION shapes."""
    if shape in (QueryShape.MULTI_HOP,) and " and " in query:
        parts = [p.strip() for p in re.split(r"\band\b", query) if p.strip()]
        return parts[:3]
    return [query]


# ---------------------------------------------------------------------------
# Step 3 — filters compile to SQL WHERE clauses, never prompt text
# ---------------------------------------------------------------------------


def build_where_clause(filters: RetrievalFilters) -> tuple[str, list]:
    """Returns (sql_fragment, params). This is the ONLY place permission scoping happens —
    never trust a prompt instruction like 'only use permitted docs' instead of this."""
    clauses = ["d.tenant_id = %s", "d.permission_scope && %s"]
    params: list = [filters.tenant_id, filters.permission_scope]
    if filters.document_ids:
        clauses.append("d.id = ANY(%s)")
        params.append(filters.document_ids)
    if filters.date_range:
        clauses.append("d.ingested_at BETWEEN %s AND %s")
        params.extend(filters.date_range)
    return " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# Step 4 — first-stage recall (BM25 + vector, run in parallel)
# ---------------------------------------------------------------------------


async def bm25_recall(query: str, where_sql: str, params: list, limit: int = 100) -> list[dict]:
    """TODO: real asyncpg call:
    SELECT c.id, c.content, c.document_id, c.section_path, c.page_number,
           ts_rank(c.tsv, websearch_to_tsquery('english', %s)) AS score
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.level = 'child' AND {where_sql}
      AND c.tsv @@ websearch_to_tsquery('english', %s)
    ORDER BY score DESC LIMIT {limit}
    """
    raise NotImplementedError("wire up asyncpg/psycopg here")


async def vector_recall(
    query_embedding: list[float], where_sql: str, params: list, limit: int = 100
) -> list[dict]:
    """TODO: real asyncpg call:
    SELECT c.id, c.content, c.document_id, c.section_path, c.page_number,
           c.embedding <=> %s AS distance
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.level = 'child' AND {where_sql}
    ORDER BY distance ASC LIMIT {limit}
    """
    raise NotImplementedError("wire up asyncpg/psycopg + embedding client here")


# ---------------------------------------------------------------------------
# Step 5 — fuse with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],  # each list already sorted best-first, items have "id"
    k: int = 60,
) -> list[tuple[str, float]]:
    """Returns [(chunk_id, fused_score), ...] sorted descending by fused_score."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Step 6 — deduplicate near-identical children (overlapping windows)
# ---------------------------------------------------------------------------


def dedupe_candidates(candidates: list[dict], overlap_threshold: float = 0.85) -> list[dict]:
    """TODO: replace naive exact-prefix check with a real near-duplicate check (e.g. Jaccard on
    shingles, or cosine similarity between candidate embeddings) before spending reranker budget."""
    seen_prefixes: set[str] = set()
    deduped = []
    for c in candidates:
        prefix = c["content"][:120]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Step 7 — rerank
# ---------------------------------------------------------------------------


async def rerank(query: str, candidates: list[dict], top_k: int = 8) -> list[dict]:
    """TODO: call a self-hosted cross-encoder (BGE/Qwen3-Reranker) or hosted API (Cohere/Voyage)
    per docs/TECH_STACK.md §5. Never skip this step to save a call — see SKILL.md rule 1."""
    raise NotImplementedError("wire up a cross-encoder reranker here")


# ---------------------------------------------------------------------------
# Step 8 — resolve child -> parent
# ---------------------------------------------------------------------------


async def resolve_parents(reranked_children: list[dict]) -> list[dict]:
    """TODO: SELECT content FROM chunks WHERE id IN (parent_chunk_id list). The PARENT content is
    what goes to the generator — never the child text that was actually searched."""
    raise NotImplementedError("wire up parent-chunk fetch here")


# ---------------------------------------------------------------------------
# Step 9 — evidence sufficiency gate (shared with the RLM path)
# ---------------------------------------------------------------------------


def sufficient(
    evidence: list[dict], draft_answer: str, query_shape: QueryShape
) -> tuple[bool, str | None]:
    """See skills/verification/ for the full check_answer_support implementation this should call.
    This stub only checks the cheap, structural conditions:
      1. Non-empty evidence.
      2. For AGGREGATION queries, evidence must span the full candidate set, not a sample.
    Returns (is_sufficient, reason_if_not).
    """
    if not evidence:
        return False, "no evidence retrieved"
    if query_shape == QueryShape.AGGREGATION and len(evidence) < 3:
        return False, "aggregation query with too few evidence items — likely a partial answer"
    return True, None


# ---------------------------------------------------------------------------
# Orchestration — retrieve()
# ---------------------------------------------------------------------------

MAX_RETRIES = 1  # see SKILL.md rule 4 — this is a hard cap, not a starting point to raise


async def retrieve(
    query: str,
    *,
    filters: RetrievalFilters,
    top_k: int = 8,
) -> RetrievalResult:
    shape = classify_query(query)
    retry_count = 0
    current_query = query

    while True:
        _sub_queries = rewrite_query(current_query, shape)
        where_sql, params = build_where_clause(filters)

        # NOTE: real implementation embeds `sub_queries` and calls bm25_recall/vector_recall
        # concurrently (asyncio.gather) per sub-query, then merges before RRF.
        bm25_lists: list[list[dict]] = []  # TODO: await bm25_recall(...) per sub_query
        vector_lists: list[list[dict]] = []  # TODO: await vector_recall(...) per sub_query

        fused = reciprocal_rank_fusion(bm25_lists + vector_lists)
        _candidate_ids = [cid for cid, _ in fused[:100]]
        # TODO: fetch full candidate rows for candidate_ids, then:
        candidates: list[dict] = []
        deduped = dedupe_candidates(candidates)
        reranked = await rerank(current_query, deduped, top_k=top_k)
        parents = await resolve_parents(reranked)

        draft_answer = ""  # TODO: generation call goes here, downstream of this skill
        is_sufficient, reason = sufficient(parents, draft_answer, shape)

        if is_sufficient:
            return RetrievalResult(
                sufficient=True,
                parent_chunks=parents,
                citations=[],  # TODO: build Citation objects from parents
                query_shape=shape,
                retry_count=retry_count,
            )

        if retry_count >= MAX_RETRIES:
            if shape == QueryShape.AGGREGATION:
                # Escalate — see skills/rlm-orchestrator/ and workflows/03_rlm_recursion_pipeline.md
                return RetrievalResult(
                    sufficient=False,
                    query_shape=shape,
                    retry_count=retry_count,
                    insufficiency_reason=f"escalate_to_rlm: {reason}",
                )
            return RetrievalResult(
                sufficient=False,
                query_shape=shape,
                retry_count=retry_count,
                insufficiency_reason=reason,
            )

        # One rewrite-and-retry, per SKILL.md rule 4 / workflows/02_retrieval_pipeline.md step 9.
        current_query = query  # TODO: real rewrite based on `reason`
        retry_count += 1
