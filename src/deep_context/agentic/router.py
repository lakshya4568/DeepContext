"""Query Classifier & Multi-Path Router implementing ARCHITECTURE.md §1, §5.4 and FR16."""

from __future__ import annotations

from deep_context.core.types import (
    QueryShape,
    RouterDecision,
    RoutingPath,
)
from deep_context.retrieval.classifier import QueryClassifier


class QueryRouter:
    """
    Decides routing path based on query shape and complexity:
    - Simple QA / Factual / How-To / Aggregation -> Hybrid RAG (default)
    - Multi-Hop / Complex Tasks -> Agentic Planner (Corrective state machine)
    """

    @classmethod
    async def route(
        cls,
        query: str,
        *,
        estimated_corpus_tokens: int = 0,
        context_budget: int = 128_000,
        forced_path: RoutingPath | None = None,
        retrieval_failed_twice: bool = False,
    ) -> RouterDecision:
        if forced_path:
            shape = await QueryClassifier.classify(query)
            return RouterDecision(
                path=forced_path,
                query_shape=shape,
                reason=f"Forced path override: {forced_path.value}",
                estimated_tokens=estimated_corpus_tokens,
            )

        shape = await QueryClassifier.classify(query)

        # Multi-hop or complex decomposition queries route to Agentic Planner
        if shape == QueryShape.MULTI_HOP or retrieval_failed_twice:
            return RouterDecision(
                path=RoutingPath.AGENTIC_PLANNER,
                query_shape=shape,
                reason="Multi-hop or hard query requiring iterative decomposition and corrective rewriting.",
                estimated_tokens=estimated_corpus_tokens,
            )

        # Default path: Hybrid RAG
        return RouterDecision(
            path=RoutingPath.HYBRID_RAG,
            query_shape=shape,
            reason="Standard query routed to fast, high-precision Hybrid RAG.",
            estimated_tokens=estimated_corpus_tokens,
        )
