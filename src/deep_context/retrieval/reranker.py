"""Cross-encoder, Gemini Semantic, and LLM-based Rerankers implementing FR3."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger


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
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


class CrossEncoderReranker:
    """Scores query-document pairs to produce high-precision top-k candidates using multi-factor heuristics."""

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using multi-factor relevance scoring:
        1. Exact content phrases & n-gram containment
        2. Filtered lexical token overlap ratio (ignoring stopwords)
        3. RRF / first-stage consensus rank
        4. Density and position preservation
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # Normalize and extract meaningful content words (excluding stopwords)
        clean_query = query.strip().strip('"').strip("'").strip("“").strip("”")
        q_lower = clean_query.lower()
        all_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 1]
        content_words = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
        if not content_words:
            content_words = all_words
        q_word_set = set(content_words)

        # Build n-grams from meaningful content words
        n_grams: list[str] = []
        if len(content_words) >= 2:
            for i in range(len(content_words) - 1):
                n_grams.append(f"{content_words[i]} {content_words[i+1]}")

        scored_candidates: list[tuple[float, dict[str, Any]]] = []

        for idx, c in enumerate(candidates):
            content = c.get("content", "")
            content_lower = content.lower()
            content_words_in_doc = set(re.findall(r"\w+", content_lower))

            # 1. Exact phrase / n-gram match bonus (0 to 1)
            exact_bonus = 0.0
            if clean_query.lower() in content_lower:
                exact_bonus = 1.0
            elif n_grams:
                ngram_hits = sum(1 for ng in n_grams if ng in content_lower)
                exact_bonus = min(1.0, (ngram_hits / len(n_grams)) * 0.9)

            # 2. Meaningful token overlap ratio (0 to 1)
            overlap_count = sum(1 for w in q_word_set if w in content_words_in_doc)
            overlap_ratio = overlap_count / max(1, len(q_word_set))

            # 3. First-stage position & RRF consensus score (0 to 1)
            rrf_score = float(c.get("rrf_score", 0.0))
            position_score = 1.0 / (1.0 + idx * 0.05)

            # Combined weighted score (balances high-recall vector consensus with high-precision lexical alignment)
            relevance = (
                0.30 * exact_bonus
                + 0.35 * overlap_ratio
                + 0.25 * min(1.0, rrf_score * 40.0)
                + 0.10 * position_score
            )

            c_copy = dict(c)
            c_copy["rerank_score"] = float(relevance)
            scored_candidates.append((relevance, c_copy))

        # Sort descending by relevance
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:top_k]]


class GeminiSemanticReranker:
    """Reranker that scores candidates using Gemini embedding cosine similarity combined with phrase signals."""

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
            # 1. Embed query in symmetric / similarity mode
            q_vec = await llm_client.get_embedding(
                query, model=model, dim=dim, task_type="sentence similarity", is_query=True
            )
            q_arr = np.array(q_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q_arr)
            if q_norm > 0:
                q_arr = q_arr / q_norm

            # 2. Embed candidates
            candidate_texts = [c.get("content", "")[:1000] for c in candidates]
            c_vecs = await llm_client.get_embeddings(
                candidate_texts, model=model, dim=dim, is_query=False
            )

            clean_query = query.strip().strip('"').strip("'").strip("“").strip("”").lower()
            q_words = [w for w in re.findall(r"\w+", clean_query) if len(w) > 1]
            q_word_set = set(q_words)

            scored_candidates: list[tuple[float, dict[str, Any]]] = []

            for idx, (c, c_vec) in enumerate(zip(candidates, c_vecs)):
                c_arr = np.array(c_vec, dtype=np.float32)
                c_norm = np.linalg.norm(c_arr)
                if c_norm > 0:
                    c_arr = c_arr / c_norm

                cos_sim = float(np.dot(q_arr, c_arr))

                # Boost with exact match and overlap
                content_lower = c.get("content", "").lower()
                content_words = set(re.findall(r"\w+", content_lower))
                exact_bonus = 1.0 if clean_query and clean_query in content_lower else 0.0
                overlap_ratio = sum(1 for w in q_word_set if w in content_words) / max(
                    1, len(q_word_set)
                )

                # Weighted hybrid score (70% Gemini semantic similarity + 20% exact match + 10% overlap)
                final_score = 0.70 * max(0.0, cos_sim) + 0.20 * exact_bonus + 0.10 * overlap_ratio

                c_copy = dict(c)
                c_copy["rerank_score"] = float(final_score)
                c_copy["gemini_cos_sim"] = float(cos_sim)
                scored_candidates.append((final_score, c_copy))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored_candidates[:top_k]]
        except Exception as e:
            logger.warning(
                "GeminiSemanticReranker failed (%s). Falling back to CrossEncoderReranker.", e
            )
            return await CrossEncoderReranker.rerank(query, candidates, top_k=top_k)


class GeminiLLMReranker:
    """Reranker that uses LLM reasoning to evaluate candidate relevance."""

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

        # Fast heuristic pass if pool is very large to narrow to top 15
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
                f"Return a JSON array of objects with 'chunk_index' (int) and 'relevance_score' (0.0 to 1.0, where 1.0 is direct answer).\n"
                f'Example: [{{"chunk_index": 0, "relevance_score": 0.95}}, ...]\n'
                f"Respond ONLY with valid JSON."
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a ranking assistant. Respond with JSON only.",
                },
                {"role": "user", "content": prompt},
            ]

            answer, _ = await llm_client.complete(
                messages, model=llm_target_model, temperature=0.0, timeout=10.0
            )

            clean_json = answer.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()

            scores_data = json.loads(clean_json)
            score_map: dict[int, float] = {}
            for item in scores_data:
                idx = item.get("chunk_index")
                score = float(item.get("relevance_score", 0.5))
                if idx is not None and isinstance(idx, int):
                    score_map[idx] = score

            scored_candidates: list[tuple[float, dict[str, Any]]] = []
            for idx, c in enumerate(candidates):
                llm_score = score_map.get(idx, 0.5)
                c_copy = dict(c)
                c_copy["rerank_score"] = llm_score
                scored_candidates.append((llm_score, c_copy))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored_candidates[:top_k]]
        except Exception as e:
            logger.warning(
                "GeminiLLMReranker failed (%s). Falling back to CrossEncoderReranker.", e
            )
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

        if active_strategy in ("gemini", "gemini_semantic", "gemini_embeddings"):
            return await GeminiSemanticReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
            )
        elif active_strategy in ("gemini_llm", "llm_reranker", "llm"):
            return await GeminiLLMReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            return await CrossEncoderReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
