"""Cross-encoder, Gemini Semantic, and LLM-based Rerankers implementing FR3."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.retrieval.quality_gates import protect_consensus


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def _blend_with_rrf(
    candidates: list[dict[str, Any]],
    raw_scores: list[float],
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
    rrf_weight = settings.reranker_blend_rrf_weight
    raw_weight = 1.0 - rrf_weight

    rrf_scores = [float(c.get("rrf_score", 0.0)) for c in candidates]
    min_rrf = min(rrf_scores) if rrf_scores else 0.0
    max_rrf = max(rrf_scores) if rrf_scores else 1.0
    rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0
    min_raw = min(raw_scores) if raw_scores else 0.0
    max_raw = max(raw_scores) if raw_scores else 1.0
    raw_range = max_raw - min_raw if max_raw > min_raw else 1.0

    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, (candidate, raw) in enumerate(zip(candidates, raw_scores)):
        norm_rrf = (float(candidate.get("rrf_score", 0.0)) - min_rrf) / rrf_range
        norm_raw = (raw - min_raw) / raw_range
        if idx < settings.reranker_consensus_top1_count:
            consensus_boost = settings.reranker_consensus_boost_tier1
        elif idx < settings.reranker_consensus_top2_count:
            consensus_boost = settings.reranker_consensus_boost_tier2
        else:
            consensus_boost = 0.0
        blended = rrf_weight * norm_rrf + raw_weight * norm_raw + consensus_boost
        copy = dict(candidate)
        copy["rerank_score"] = float(blended)
        scored.append((blended, copy))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


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
        n_grams = [
            f"{content_words[i]} {content_words[i + 1]}"
            for i in range(len(content_words) - 1)
        ] if len(content_words) >= 2 else []

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
            overlap_ratio = sum(1 for w in q_word_set if w in content_tokens) / max(1, len(q_word_set))
            position_score = 1.0 / (1.0 + idx * 0.05)
            raw_scores.append(0.40 * exact_bonus + 0.40 * overlap_ratio + 0.20 * position_score)

        scored = _blend_with_rrf(candidates, raw_scores)
        return [item for _, item in scored[:top_k]]


class GeminiSemanticReranker:
    """Reranker that blends retrieval-task embeddings with RRF instead of replacing ranks."""

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        model = embedding_model or "gemini-embedding-2"
        dim = embedding_dim or 768
        try:
            q_vec = await llm_client.get_embedding(
                query, model=model, dim=dim, task_type="search result", is_query=True
            )
            q_arr = np.array(q_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q_arr)
            if q_norm > 0:
                q_arr = q_arr / q_norm

            c_vecs = await llm_client.get_embeddings(
                [c.get("content", "")[:1000] for c in candidates],
                model=model,
                dim=dim,
                is_query=False,
            )
            candidate_copies: list[dict[str, Any]] = []
            raw_scores: list[float] = []
            for candidate, c_vec in zip(candidates, c_vecs):
                c_arr = np.array(c_vec, dtype=np.float32)
                c_norm = np.linalg.norm(c_arr)
                if c_norm > 0:
                    c_arr = c_arr / c_norm
                cos_sim = max(0.0, float(np.dot(q_arr, c_arr)))
                raw_scores.append(cos_sim)
                cand_copy = dict(candidate)
                cand_copy["gemini_cos_sim"] = cos_sim
                candidate_copies.append(cand_copy)
            scored = _blend_with_rrf(candidate_copies, raw_scores)
            return [item for _, item in scored[:top_k]]
        except Exception as e:
            logger.warning(
                "GeminiSemanticReranker failed (%s). Falling back to CrossEncoderReranker.", e
            )
            return await CrossEncoderReranker.rerank(query, candidates, top_k=top_k)


class GeminiLLMReranker:
    """Reranker that uses LLM reasoning, then blends scores with RRF."""

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates
        if len(candidates) > 15:
            candidates = await CrossEncoderReranker.rerank(query, candidates, top_k=15)

        llm_target_model = model or (
            "gemini-2.5-flash" if settings.has_gemini_key else settings.llm_model
        )
        try:
            snippets_text = "\n\n".join(
                f"[Chunk {idx}]: {c.get('content', '')[:400]}" for idx, c in enumerate(candidates)
            )
            prompt = (
                f"You are a precision search relevance evaluator.\n"
                f'Query: "{query}"\n\n'
                f"Evaluate each chunk's direct relevance to answering the query.\n"
                f"Candidate Chunks:\n{snippets_text}\n\n"
                f"Return a JSON array of objects with 'chunk_index' (int) and 'relevance_score' (0.0 to 1.0).\n"
                f"Respond ONLY with valid JSON."
            )
            answer, _ = await llm_client.complete(
                [
                    {"role": "system", "content": "You are a ranking assistant. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=llm_target_model,
                temperature=0.0,
                timeout=10.0,
            )
            clean_json = answer.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            scores_data = json.loads(clean_json)
            score_map = {
                int(item.get("chunk_index")): float(item.get("relevance_score", 0.5))
                for item in scores_data
                if isinstance(item.get("chunk_index"), int)
            }
            raw_scores = [score_map.get(idx, 0.5) for idx in range(len(candidates))]
            scored = _blend_with_rrf(candidates, raw_scores)
            return [item for _, item in scored[:top_k]]
        except Exception as e:
            logger.warning("GeminiLLMReranker failed (%s). Falling back to CrossEncoderReranker.", e)
            return await CrossEncoderReranker.rerank(query, candidates, top_k=top_k)


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
        elif active_strategy in ("gemini", "gemini_semantic", "gemini_embeddings"):
            ranked = await GeminiSemanticReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
            )
        elif active_strategy in ("gemini_llm", "llm_reranker", "llm"):
            ranked = await GeminiLLMReranker.rerank(
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
