"""Cross-encoder and BGE-based Rerankers implementing FR3."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from nltk.corpus import stopwords  # type: ignore[import-untyped]

from deep_context.core.config import settings
from deep_context.retrieval.quality_gates import protect_consensus

RerankerScoreType = Literal["logit", "probability"]

STOPWORDS = set(stopwords.words("english")) | {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
}


def _sigmoid(x: float) -> float:
    """Map a finite reranker logit to a bounded relevance score in [0, 1]."""
    if not math.isfinite(x):
        if math.isnan(x):
            return 0.0
        return 1.0 if x > 0 else 0.0

    if x >= 0.0:
        z = math.exp(-x) if x < 709.0 else 0.0
        return 1.0 / (1.0 + z)

    z = math.exp(x) if x > -709.0 else 0.0
    return z / (1.0 + z)


def _normalize_reranker_scores(
    raw_scores: list[float],
    score_type: RerankerScoreType,
) -> list[float]:
    """Normalize reranker scores without guessing their provider semantics.

    Local cross-encoders commonly return logits. Hosted reranker APIs may
    already return bounded relevance scores. Applying sigmoid based only on
    numeric range can accidentally transform valid bounded scores twice.
    """
    normalized: list[float] = []

    for raw_score in raw_scores:
        score = float(raw_score)

        if score_type == "logit":
            normalized.append(_sigmoid(score))
            continue

        if score_type == "probability":
            if not math.isfinite(score):
                normalized.append(0.0)
            else:
                normalized.append(min(1.0, max(0.0, score)))
            continue

        raise ValueError(f"Unsupported reranker score type: {score_type}")

    return normalized


def _consensus_boost_for_candidate(
    candidate: dict[str, Any],
    candidate_index: int,
) -> float:
    """Return a bounded consensus boost based on retrieval metadata.

    Retrieval ranks are preferred. The index fallback preserves compatibility
    with older candidate objects that do not expose BM25/vector ranks.
    """
    has_explicit_rank_keys = "bm25_rank" in candidate or "dense_rank" in candidate
    bm25_rank = candidate.get("bm25_rank")
    dense_rank = candidate.get("dense_rank")

    if has_explicit_rank_keys:
        if bm25_rank is not None and dense_rank is not None:
            try:
                bm25_rank = int(bm25_rank)
                dense_rank = int(dense_rank)
                if bm25_rank <= 10 and dense_rank <= 10:
                    return min(
                        0.15,
                        max(0.0, settings.reranker_consensus_boost_tier1),
                    )

                if bm25_rank <= 20 and dense_rank <= 20:
                    return min(
                        0.10,
                        max(0.0, settings.reranker_consensus_boost_tier2),
                    )
            except (TypeError, ValueError):
                pass
        # Candidate has explicit rank info but only appeared in one retriever
        return 0.0

    # Compatibility fallback only for legacy candidates that lack rank keys
    if candidate_index < settings.reranker_consensus_top1_count:
        return min(
            0.15,
            max(0.0, settings.reranker_consensus_boost_tier1),
        )

    if candidate_index < settings.reranker_consensus_top2_count:
        return min(
            0.10,
            max(0.0, settings.reranker_consensus_boost_tier2),
        )

    return 0.0


def _blend_with_rrf(
    candidates: list[dict[str, Any]],
    raw_scores: list[float],
    *,
    score_type: RerankerScoreType,
    blend_rrf_weight: float | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    """Blend a secondary reranker signal with normalized RRF consensus."""
    if len(candidates) != len(raw_scores):
        raise ValueError(
            "Candidate count must match reranker score count: "
            f"{len(candidates)} != {len(raw_scores)}"
        )

    rrf_weight = (
        blend_rrf_weight if blend_rrf_weight is not None else settings.reranker_blend_rrf_weight
    )

    rrf_scores = [float(candidate.get("rrf_score", 0.0)) for candidate in candidates]

    min_rrf = min(rrf_scores, default=0.0)
    max_rrf = max(rrf_scores, default=0.0)
    rrf_range = max_rrf - min_rrf

    # If all candidates have identical RRF support, RRF provides no
    # ordering signal and therefore contributes zero to the blend.
    if rrf_range <= 1e-12:
        normalized_rrf = [0.0] * len(rrf_scores)
    else:
        normalized_rrf = [(score - min_rrf) / rrf_range for score in rrf_scores]

    normalized_scores = _normalize_reranker_scores(
        raw_scores,
        score_type,
    )

    scored: list[tuple[float, dict[str, Any]]] = []

    for idx, (candidate, norm_raw, norm_rrf) in enumerate(
        zip(candidates, normalized_scores, normalized_rrf, strict=True)
    ):
        consensus_boost = _consensus_boost_for_candidate(
            candidate,
            idx,
        )

        final_score = rrf_weight * norm_rrf + (1.0 - rrf_weight) * norm_raw + consensus_boost

        copy = dict(candidate)
        copy["rerank_score"] = float(final_score)
        scored.append((final_score, copy))

    return sorted(
        scored,
        key=lambda item: (
            -item[0],
            -float(item[1].get("rrf_score", 0.0)),
            str(item[1].get("id", "")),
        ),
    )


class CrossEncoderReranker:
    """Scores query-document pairs using n-grams, overlap, and blended RRF consensus."""

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        clean_query = query.strip().strip('"').strip("'").strip("\u201c").strip("\u201d")
        q_lower = clean_query.lower()
        all_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 1]
        content_words = [w for w in all_words if w not in STOPWORDS and len(w) > 2] or all_words
        q_word_set = set(content_words)
        n_grams = (
            [f"{content_words[i]} {content_words[i + 1]}" for i in range(len(content_words) - 1)]
            if len(content_words) >= 2
            else []
        )

        raw_scores: list[float] = []
        for idx, candidate in enumerate(candidates):
            content_lower = candidate.get("content", "").lower()
            content_tokens = set(re.findall(r"\w+", content_lower))
            exact_bonus = 0.0
            if clean_query.lower() in content_lower:
                exact_bonus = 1.0
            elif n_grams:
                hits = sum(1 for ng in n_grams if ng in content_lower)
                exact_bonus = min(1.0, (hits / len(n_grams)) * 0.9)
            overlap_ratio = sum(1 for w in q_word_set if w in content_tokens) / max(
                1, len(q_word_set)
            )
            position_score = 1.0 / (1.0 + idx * 0.05)
            raw_scores.append(0.40 * exact_bonus + 0.40 * overlap_ratio + 0.20 * position_score)

        scored = _blend_with_rrf(candidates, raw_scores, score_type="probability")
        return [item for _, item in scored[:top_k]]


class Reranker:
    """Unified entry point for reranking candidates across multiple strategies."""

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        strategy: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> list[dict[str, Any]]:
        active_strategy = (
            (strategy or settings.reranker_strategy or "cross_encoder").lower().replace("-", "_")
        )
        if active_strategy in ("none", "bypass", "rrf", "hybrid", "disabled"):
            ranked = candidates[:top_k]
        elif active_strategy in (
            "ecohash",
            "ecohash_reranker",
            "hosted_bge",
            # Legacy aliases for the removed local BGE-M3 cross-encoder now
            # resolve to the hosted EcoHash reranker exclusively.
            "local_cross_encoder",
            "bge",
            "bge_reranker",
            "bge_m3",
        ):
            from deep_context.retrieval.ecohash_reranker import EcoHashReranker

            ranked = await EcoHashReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            ranked = await CrossEncoderReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        return protect_consensus(candidates, ranked, top_k=top_k)
