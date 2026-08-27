"""Query classifier implementing Step 1 of retrieval and routing."""

from __future__ import annotations

import json

from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import QueryShape, RetrievalFilters


class QueryClassifier:
    """Classifies incoming queries into QueryShape (shared by RAG & agentic router)."""

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
                "entire document",
                "whole document",
                "all documents",
                "everything in",
                "go through",
                "scan all",
                "full corpus",
                "across all",
                "comprehensive overview",
                "complete list",
                "all instances",
                "every mention",
                "total count",
                "how many times",
                "exhaustive",
            )
        ):
            return QueryShape.AGGREGATION
        if any(
            w in q
            for w in (
                "how do i",
                "how to",
                "steps to",
                "guide on",
                "how can",
                "tutorial",
                "walk me through",
                "instructions for",
                "set up",
                "configure",
                "install",
                "getting started",
                "best way to",
                "procedure",
            )
        ):
            return QueryShape.HOW_TO
        if (
            q.count("?") > 1
            or " and " in q
            or " vs " in q
            or "compare " in q
            or " versus " in q
            or " difference between " in q
            or " relationship between " in q
            or " connect " in q
            or " relate " in q
        ):
            return QueryShape.MULTI_HOP
        if any(
            w in q
            for w in (
                "where is",
                "find in document",
                "navigate to",
                "table of contents",
                "section about",
                "locate",
                "which section",
                "which page",
                "what page",
                "index of",
                "chapter on",
                "jump to",
                "look up",
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

    @classmethod
    def extract_semantic_filters(
        cls, query: str, base_filters: RetrievalFilters | None = None
    ) -> RetrievalFilters:
        """Extract structured document types and section prefixes directly from query text."""
        filters = base_filters or RetrievalFilters()
        q = query.lower()

        # Detect appendix or chapter references
        if not filters.section_prefix:
            if "appendix" in q:
                for letter in ("a", "b", "c", "d", "e", "f", "g"):
                    if f"appendix {letter}" in q:
                        filters.section_prefix = f"appendix/{letter}"
                        break
            elif "chapter" in q:
                for ch in range(1, 50):
                    if f"chapter {ch}" in q:
                        filters.section_prefix = f"chapter/{ch}"
                        break

        # Detect document type hints if not already specified
        if not filters.doc_types:
            if "appendix" in q and not filters.section_prefix:
                filters.doc_types = ["appendix"]
            elif any(k in q for k in ("python code", "source code", "repo code", "in the code")):
                filters.doc_types = ["code"]

        return filters

    @classmethod
    async def classify_and_filter(
        cls, query: str, filters: RetrievalFilters | None = None
    ) -> tuple[QueryShape, RetrievalFilters]:
        """Classify query shape and extract structured semantic filters."""
        shape = await cls.classify(query)
        enriched_filters = cls.extract_semantic_filters(query, filters)
        return shape, enriched_filters
