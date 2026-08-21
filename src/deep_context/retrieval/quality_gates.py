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


def is_top_consensus_candidate(candidate: dict[str, Any]) -> bool:
    """Return True if candidate was ranked in top 10 by both BM25 and dense retrieval."""
    bm25_rank = candidate.get("bm25_rank")
    dense_rank = candidate.get("dense_rank")

    if bm25_rank is None or dense_rank is None:
        return False

    try:
        return int(bm25_rank) <= 10 and int(dense_rank) <= 10
    except (TypeError, ValueError):
        return False


def protect_consensus(
    original: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Guarantee retention of top-consensus candidates supported by both BM25 and dense retrieval."""
    protected = [candidate for candidate in original if is_top_consensus_candidate(candidate)]

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in reranked:
        cid = str(candidate.get("id") or candidate.get("chunk_id") or "")
        if cid and cid not in seen:
            merged.append(candidate)
            seen.add(cid)

    for candidate in protected:
        cid = str(candidate.get("id") or candidate.get("chunk_id") or "")
        if not cid or cid in seen:
            continue

        if len(merged) < top_k:
            merged.append(candidate)
            seen.add(cid)
            continue

        # Replace the weakest non-protected item if possible.
        replacement_index = next(
            (
                index
                for index in range(len(merged) - 1, -1, -1)
                if not is_top_consensus_candidate(merged[index])
            ),
            None,
        )

        if replacement_index is not None:
            merged[replacement_index] = candidate
            seen.add(cid)

    return merged[:top_k]
