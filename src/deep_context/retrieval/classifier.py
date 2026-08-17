"""Query classifier implementing Step 1 of retrieval & RLM routing."""

from __future__ import annotations

import json

from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import QueryShape


class QueryClassifier:
    """Classifies incoming queries into QueryShape (shared by RAG & RLM router)."""

    @classmethod
    async def classify(cls, query: str) -> QueryShape:
        q = query.strip().lower()

        # Fast heuristic classification first
        if any(
            w in q
            for w in (
                "every",
                "all of",
                "each of",
                "don't miss",
                "none of",
                "aggregate",
                "summarize everything",
                "read all",
            )
        ):
            return QueryShape.AGGREGATION
        if any(w in q for w in ("how do i", "how to", "steps to", "guide on", "how can")):
            return QueryShape.HOW_TO
        if q.count("?") > 1 or " and " in q or " vs " in q or "compare " in q:
            return QueryShape.MULTI_HOP
        if any(
            w in q
            for w in (
                "where is",
                "find in document",
                "navigate to",
                "table of contents",
                "section about",
            )
        ):
            return QueryShape.NAVIGATION
        if any(
            q.startswith(w)
            for w in (
                "who",
                "what",
                "when",
                "which",
                "why",
                "where",
                "is ",
                "are ",
                "did ",
                "do ",
                "does ",
                "tell me",
                "describe",
                "explain",
            )
        ):
            return QueryShape.FACTUAL_LOOKUP

        # If LLM is available, perform refinement classification
        try:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a fast query classifier for a RAG system. Return ONLY valid JSON with key 'shape' "
                        "which must be one of: 'factual_lookup', 'how_to', 'multi_hop', 'aggregation', 'navigation'."
                    ),
                },
                {"role": "user", "content": f"Query: {query}"},
            ]
            content, _ = await llm_client.complete(
                prompt,
                max_tokens=64,
                temperature=0.0,
                enable_thinking=False,
                timeout=8.0,
                max_retries=1,
            )
            data = json.loads(content.strip().strip("```json").strip("```"))
            shape_str = data.get("shape", "").lower()
            for s in QueryShape:
                if s.value == shape_str:
                    return s
        except Exception as e:
            logger.debug("LLM query classification fallback to heuristic: %s", e)

        return QueryShape.FACTUAL_LOOKUP
