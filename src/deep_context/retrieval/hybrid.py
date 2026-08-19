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


def _best_rank(lists: list[list[dict[str, Any]]], cid: str) -> int | None:
    best: int | None = None
    for ranked_list in lists:
        for rank, item in enumerate(ranked_list, start=1):
            if item.get("id") == cid:
                if best is None or rank < best:
                    best = rank
                break
    return best


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

        all_results = await asyncio.gather(*(bm25_tasks + vector_tasks))
        ranked_lists: list[list[dict[str, Any]]] = list(all_results)
        n_queries = len(sub_queries)
        bm25_lists = ranked_lists[:n_queries]
        vector_lists = ranked_lists[n_queries:]

        fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
        fused_score_map = {cid: score for cid, score in fused}
        fused_ids = [cid for cid, _ in fused[:limit]]

        id_to_candidate: dict[str, dict[str, Any]] = {}
        for r_list in ranked_lists:
            for item in r_list:
                cid = item["id"]
                if cid in fused_ids and cid not in id_to_candidate:
                    cand = dict(item)
                    cand["rrf_score"] = fused_score_map.get(cid, 0.0)
                    cand["score"] = cand["rrf_score"]
                    cand["bm25_rank"] = _best_rank(bm25_lists, cid)
                    cand["dense_rank"] = _best_rank(vector_lists, cid)
                    id_to_candidate[cid] = cand

        # Document-agnostic appendix and section intent matching
        for cid, cand in id_to_candidate.items():
            sec = (cand.get("section_path") or "").lower()
            doc_type = str(cand.get("doc_type") or "").lower()
            content_head = (cand.get("content") or "")[:250].lower()

            boost = 0.0
            for sq in sub_queries:
                sq_lower = sq.lower()
                for app_label in (
                    "appendix a",
                    "appendix b",
                    "appendix c",
                    "appendix d",
                    "appendix e",
                    "appendix f",
                    "appendix g",
                ):
                    if app_label in sq_lower and (app_label in sec or app_label in content_head):
                        boost += 0.12
                for ch_label in (
                    "chapter 1",
                    "chapter 2",
                    "chapter 3",
                    "chapter 4",
                    "chapter 5",
                    "chapter 6",
                    "chapter 7",
                    "chapter 8",
                ):
                    if ch_label in sq_lower and (ch_label in sec or ch_label in content_head):
                        boost += 0.08
                if "appendix" in sq_lower and (
                    "appendix" in sec or doc_type == "appendix" or "appendix" in content_head
                ):
                    boost += 0.05

            if boost > 0:
                fused_score_map[cid] = fused_score_map.get(cid, 0.0) + boost
                cand["rrf_score"] = fused_score_map[cid]
                cand["score"] = cand["rrf_score"]

        fused_ids = sorted(fused_ids, key=lambda cid: fused_score_map.get(cid, 0.0), reverse=True)
        ordered_candidates = [id_to_candidate[cid] for cid in fused_ids if cid in id_to_candidate]
        return deduplicate_candidates(ordered_candidates, max_candidates=limit)
