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
    Decides routing path based on query shape, corpus size, and aggregation requirement:
    - Simple QA / Factual / How-To -> Hybrid RAG (default)
    - Multi-Hop / Complex Tasks -> Agentic Planner
    - Massive Corpus / 'Read Everything' Global Aggregation -> RLM Engine
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

        # Condition 1: Exceeds context window budget
        if estimated_corpus_tokens > context_budget:
            return RouterDecision(
                path=RoutingPath.RLM_ENGINE,
                query_shape=shape,
                reason=f"Corpus token count ({estimated_corpus_tokens}) exceeds context budget ({context_budget}).",
                estimated_tokens=estimated_corpus_tokens,
                requires_aggregation=True,
            )

        # Condition 2: Global aggregation ("every single file", "read all papers", "don't miss anything")
        q_lower = query.lower()
        if shape == QueryShape.AGGREGATION and any(
            w in q_lower
            for w in (
                "every file",
                "all documents",
                "read everything",
                "aggregate across all",
                "exhaustive",
            )
        ):
            return RouterDecision(
                path=RoutingPath.RLM_ENGINE,
                query_shape=shape,
                reason="Task requires exhaustive global aggregation over all documents.",
                estimated_tokens=estimated_corpus_tokens,
                requires_aggregation=True,
            )

        # Condition 3: Hybrid retrieval has already failed twice
        if retrieval_failed_twice:
            return RouterDecision(
                path=RoutingPath.RLM_ENGINE,
                query_shape=shape,
                reason="Hybrid retrieval failed twice; escalating to RLM engine.",
                estimated_tokens=estimated_corpus_tokens,
            )

        # Multi-hop queries route to Agentic Planner
        if shape == QueryShape.MULTI_HOP:
            return RouterDecision(
                path=RoutingPath.AGENTIC_PLANNER,
                query_shape=shape,
                reason="Multi-hop query requiring iterative decomposition and tool calls.",
                estimated_tokens=estimated_corpus_tokens,
            )

        # Default path: Hybrid RAG
        return RouterDecision(
            path=RoutingPath.HYBRID_RAG,
            query_shape=shape,
            reason="Standard query routed to fast, high-precision Hybrid RAG.",
            estimated_tokens=estimated_corpus_tokens,
        )
