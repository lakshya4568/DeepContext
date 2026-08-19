"""Retrieval and abstention helpers shared by the engine and generator."""

from __future__ import annotations

import re
from typing import Any

REFUSAL_TEMPLATE = "Based on the provided context, there is insufficient evidence to answer."

ANACHRONISM_MARKERS = (
    "smartphone",
    "iphone",
    "android phone",
    "text message",
    "airplane",
    "aeroplane",
    "nuclear",
    "submarine",
    "harry potter",
    "hermione",
    "presidential election",
    "united states presidential",
    "registration number",
)


def is_anachronism(query: str) -> bool:
    q = query.lower()
    return any(marker in q for marker in ANACHRONISM_MARKERS)


def hop_coverage(sub_queries: list[str], parents: list[dict[str, Any]]) -> list[str]:
    """Return sub-queries whose key terms are missing from retrieved parents."""
    if len(sub_queries) <= 1 or not parents:
        return []
    blob = " ".join(str(parent.get("content", "")) for parent in parents).lower()
    missing: list[str] = []
    for sub_query in sub_queries[1:]:
        terms = [w for w in re.findall(r"\w+", sub_query.lower()) if len(w) > 3]
        if not terms:
            continue
        if sum(term in blob for term in terms) / len(terms) < 0.4:
            missing.append(sub_query)
    return missing


def protect_consensus(
    original: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep candidates that both BM25 and dense ranked in their top 10."""
    merged = list(reranked)
    seen = {str(item.get("id") or item.get("chunk_id")) for item in merged}
    for candidate in original:
        bm25_rank = candidate.get("bm25_rank", 99)
        dense_rank = candidate.get("dense_rank", 99)
        cid = str(candidate.get("id") or candidate.get("chunk_id") or "")
        if bm25_rank < 10 and dense_rank < 10 and cid and cid not in seen:
            merged.append(candidate)
            seen.add(cid)
    return merged[:top_k]
