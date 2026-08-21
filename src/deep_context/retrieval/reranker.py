"""Cross-encoder and BGE-based Rerankers implementing FR3."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from deep_context.core.config import settings
from deep_context.retrieval.quality_gates import protect_consensus

RerankerScoreType = Literal["logit", "probability"]

STOPWORDS = {
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
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
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
    bm25_rank = candidate.get("bm25_rank")
    dense_rank = candidate.get("dense_rank")

    if bm25_rank is not None and dense_rank is not None:
        try:
            bm25_rank = int(bm25_rank)
            dense_rank = int(dense_rank)
        except (TypeError, ValueError):
            bm25_rank = None
            dense_rank = None

    if bm25_rank is not None and dense_rank is not None:
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

        return 0.0

    # Compatibility fallback for candidates that only contain RRF ordering.
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
) -> list[tuple[float, dict[str, Any]]]:
    """Blend a secondary reranker signal with normalized RRF consensus.

    IMPORTANT TUNING NOTE (regression history):
    A weighting of 0.70 * norm_rrf + 0.30 * norm_raw with a 0.20/0.08 tiered
    consensus boost was tried and *dropped* full-pipeline Hit@5 on the GoT
    eval set from 87.1% to 61.3%, and also degraded Direct Factual, Citation,
    and Multi-Hop Hit@5 by ~25 points each versus the 0.60/0.40 weighting.
    The higher RRF weight combined with the larger consensus boost caused the
    blend to over-anchor on first-stage rank order and under-weight the
    lexical/semantic reranker signal that corrects RRF mistakes.

    Default weights are restored to the values that produced the best
    validated Hit@5 (87.1%) on the same corpus and eval set. Do not change
    these defaults without running the full 36-query benchmark before and
    after, on a frozen scorer version, and confirming Hit@5/nDCG@5 improve
    together (not just one of them).
    """
    if len(candidates) != len(raw_scores):
        raise ValueError(
            "Candidate count must match reranker score count: "
            f"{len(candidates)} != {len(raw_scores)}"
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

        final_score = (
            settings.reranker_blend_rrf_weight * norm_rrf
            + (1.0 - settings.reranker_blend_rrf_weight) * norm_raw
            + consensus_boost
        )

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


_bge_session: Any | None = None
_bge_tokenizer: Any | None = None


def _get_bge_reranker() -> tuple[Any, Any]:
    """Lazy-load quantized INT8 BGE-reranker-v2-m3 ONNX model and tokenizer."""
    global _bge_session, _bge_tokenizer
    if _bge_session is None or _bge_tokenizer is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        repo_id = "tss-deposium/bge-reranker-v2-m3-onnx-int8"
        model_path = hf_hub_download(repo_id=repo_id, filename="model_quantized.onnx")
        _bge_tokenizer = AutoTokenizer.from_pretrained(repo_id)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _bge_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _bge_session, _bge_tokenizer


class LocalCrossEncoderReranker:
    """Real trained cross-encoder using quantized INT8 BGE-reranker-v2-m3 (ONNX).

    Unlike the heuristic CrossEncoderReranker, this model reads the query and
    each candidate document together in a single transformer forward pass,
    producing a learned relevance score that handles paraphrases, synonyms,
    and semantic equivalence.

    Quantized to INT8 for ~75% smaller memory footprint and fast inference.
    """

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates

        session, tokenizer = _get_bge_reranker()
        pairs = [[query, str(c.get("content", ""))[:1000]] for c in candidates]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        ort_inputs = {
            k: v for k, v in inputs.items() if k in [inp.name for inp in session.get_inputs()]
        }
        logits = session.run(None, ort_inputs)[0]
        raw_scores: list[float] = logits.reshape(-1).tolist()

        # Reuse the fixed RRF blend with explicit logit score type
        scored = _blend_with_rrf(candidates, raw_scores, score_type="logit")
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
        elif active_strategy in ("ecohash", "ecohash_reranker", "hosted_bge"):
            from deep_context.retrieval.ecohash_reranker import EcoHashReranker

            ranked = await EcoHashReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        elif active_strategy in ("local_cross_encoder", "bge", "bge_reranker", "bge_m3"):
            if settings.has_ecohash_key:
                from deep_context.retrieval.ecohash_reranker import EcoHashReranker

                ranked = await EcoHashReranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=top_k,
                )
            else:
                ranked = await LocalCrossEncoderReranker.rerank(
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
