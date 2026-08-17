"""Query rewriter & sub-query decomposer."""

from __future__ import annotations

import json
import re

from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import QueryShape


class QueryRewriter:
    """Decomposes or rewrites complex or multi-part queries into focused search queries."""

    @classmethod
    async def rewrite_or_decompose(cls, query: str, shape: QueryShape) -> list[str]:
        if shape not in (QueryShape.MULTI_HOP, QueryShape.AGGREGATION):
            return [query]

        # Quick heuristic split for compound 'and' queries
        if " and " in query and query.count("?") <= 1:
            parts = [p.strip() for p in re.split(r"\band\b", query) if len(p.strip()) > 3]
            if len(parts) in (2, 3):
                return parts

        # LLM query decomposition
        try:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Decompose the complex user query into 1 to 3 distinct, focused sub-queries for hybrid search. "
                        'Return ONLY a JSON array of strings, e.g. ["sub-query 1", "sub-query 2"].'
                    ),
                },
                {"role": "user", "content": f"Query: {query}"},
            ]
            content, _ = await llm_client.complete(
                prompt,
                max_tokens=128,
                temperature=0.0,
                enable_thinking=False,
                timeout=8.0,
                max_retries=1,
            )
            data = json.loads(content.strip().strip("```json").strip("```"))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data[:3]
        except Exception as e:
            logger.debug("LLM query rewrite fallback to original query: %s", e)

        return [query]
