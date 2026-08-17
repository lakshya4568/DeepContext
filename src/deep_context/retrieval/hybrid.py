"""Hybrid search (BM25 + Vector), Reciprocal Rank Fusion (RRF), and candidate deduplication."""

from __future__ import annotations

import asyncio
from typing import Any

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.types import RetrievalFilters
from deep_context.storage.base import StorageInterface


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combines multiple ranked lists into a single ranking using Reciprocal Rank Fusion.
    Score = sum(1.0 / (k + rank)) across all lists where the item appears.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def deduplicate_candidates(
    candidates: list[dict[str, Any]], max_candidates: int = 100
) -> list[dict[str, Any]]:
    """Deduplicate candidates with identical chunk id or near-identical content prefixes."""
    seen_ids: set[str] = set()
    seen_prefixes: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for c in candidates:
        cid = c["id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        prefix = c["content"][:120].strip().lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        deduped.append(c)
        if len(deduped) >= max_candidates:
            break

    return deduped


class HybridRetriever:
    """Executes parallel BM25 and Dense Vector search, applies RRF, and deduplicates candidates."""

    def __init__(self, storage: StorageInterface):
        self.storage = storage

    async def retrieve_candidates(
        self,
        sub_queries: list[str],
        filters: RetrievalFilters,
        limit: int = 100,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Runs BM25 and vector search in parallel across all sub-queries and returns fused candidate chunks.
        """
        # 1. Generate query embeddings for all sub-queries (asymmetric query format for Gemini)
        query_embeddings = await llm_client.get_embeddings(
            sub_queries,
            model=embedding_model,
            dim=embedding_dim,
            is_query=True,
        )

        bm25_tasks = [
            self.storage.search_bm25(sq, filters=filters, limit=limit) for sq in sub_queries
        ]
        vector_tasks = [
            self.storage.search_vector(q_emb, filters=filters, limit=limit)
            for q_emb in query_embeddings
        ]

        # 2. Run all recall legs in parallel
        all_results = await asyncio.gather(*(bm25_tasks + vector_tasks))
        ranked_lists: list[list[dict[str, Any]]] = list(all_results)

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
        fused_score_map = {cid: score for cid, score in fused}
        fused_ids = [cid for cid, _ in fused[:limit]]

        # Map candidate details
        id_to_candidate: dict[str, dict[str, Any]] = {}
        for r_list in ranked_lists:
            for item in r_list:
                cid = item["id"]
                if cid in fused_ids and cid not in id_to_candidate:
                    cand = dict(item)
                    cand["rrf_score"] = fused_score_map.get(cid, 0.0)
                    cand["score"] = cand["rrf_score"]
                    id_to_candidate[cid] = cand

        ordered_candidates = [id_to_candidate[cid] for cid in fused_ids if cid in id_to_candidate]

        # 4. Deduplicate
        return deduplicate_candidates(ordered_candidates, max_candidates=limit)
