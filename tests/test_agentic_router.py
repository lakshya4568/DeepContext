"""Tests for QueryRouter and AgenticPlanner."""

import pytest

from deep_context.agentic.planner import AgenticPlanner
from deep_context.agentic.router import QueryRouter
from deep_context.core.types import IngestRequest, RetrievalFilters, RoutingPath
from deep_context.ingestion.pipeline import ingestion_pipeline
from deep_context.storage import get_storage


@pytest.mark.asyncio
async def test_query_router_decisions() -> None:
    # 1. Standard query -> HYBRID_RAG
    d1 = await QueryRouter.route("How to configure JWT tokens in Spring?")
    assert d1.path == RoutingPath.HYBRID_RAG

    # 2. Multi-hop query -> AGENTIC_PLANNER
    d2 = await QueryRouter.route(
        "Compare PostgreSQL with Redis and explain their caching tradeoffs?"
    )
    assert d2.path in (RoutingPath.AGENTIC_PLANNER, RoutingPath.HYBRID_RAG)

    # 3. Massive corpus or global aggregation -> RLM_ENGINE
    d3 = await QueryRouter.route(
        "Read every file in the codebase and check for memory leaks",
        estimated_corpus_tokens=200_000,
    )
    assert d3.path == RoutingPath.RLM_ENGINE


@pytest.mark.asyncio
async def test_agentic_planner_execution() -> None:
    storage = await get_storage()
    # Ingest mock doc
    await ingestion_pipeline.ingest(
        IngestRequest(
            title="Comparison Guide",
            content="""# DB Comparisons
Redis provides sub-millisecond in-memory caching.
Postgres provides durable transactional persistence.
""",
        )
    )

    planner = AgenticPlanner(storage)
    answer, citations, reasoning = await planner.execute_plan(
        query="What are the differences between Redis and Postgres?",
        filters=RetrievalFilters(),
    )
    assert len(answer) > 0
    assert isinstance(citations, list)
