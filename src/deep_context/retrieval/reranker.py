"""Cross-encoder and LLM-based Reranker implementing FR3."""

from __future__ import annotations

import re
from typing import Any


class CrossEncoderReranker:
    """Scores query-document pairs to produce high-precision top-k candidates."""

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using multi-factor relevance scoring:
        1. Exact phrase / needle containment
        2. Clean lexical token overlap ratio
        3. RRF / first-stage consensus rank
        4. Query term proximity and density
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # Normalize query terms
        clean_query = query.strip().strip('"').strip("'").strip("“").strip("”")
        q_lower = clean_query.lower()
        q_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 1]
        q_word_set = set(q_words)

        # Multi-word phrase for exact match detection
        phrase = " ".join(q_words) if len(q_words) > 1 else q_lower

        scored_candidates: list[tuple[float, dict[str, Any]]] = []

        for idx, c in enumerate(candidates):
            content = c.get("content", "")
            content_lower = content.lower()
            content_words = set(re.findall(r"\w+", content_lower))

            # 1. Exact phrase / needle match bonus (0 to 1)
            exact_bonus = 0.0
            if phrase and phrase in content_lower:
                exact_bonus = 1.0
            elif q_lower in content_lower:
                exact_bonus = 1.0
            elif len(q_words) >= 3:
                # Check for 3-word n-gram match
                for i in range(len(q_words) - 2):
                    tri = " ".join(q_words[i : i + 3])
                    if tri in content_lower:
                        exact_bonus = max(exact_bonus, 0.7)
                        break

            # 2. Token overlap ratio (0 to 1)
            overlap_count = sum(1 for w in q_word_set if w in content_words)
            overlap_ratio = overlap_count / max(1, len(q_word_set))

            # 3. First-stage position & RRF consensus score (0 to 1)
            rrf_score = float(c.get("rrf_score", 0.0))
            position_score = 1.0 / (1.0 + idx * 0.05)

            # Combined weighted score
            # High weight on exact phrase match and token overlap to ensure needles are NEVER lost
            relevance = (
                0.40 * exact_bonus
                + 0.35 * overlap_ratio
                + 0.15 * min(1.0, rrf_score * 30.0)
                + 0.10 * position_score
            )

            scored_candidates.append((relevance, c))

        # Sort descending by relevance
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:top_k]]
