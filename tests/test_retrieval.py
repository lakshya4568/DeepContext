"""Tests for hybrid retrieval, BM25, Vector search, RRF fusion, and parent resolution."""

import pytest

from deep_context.core.types import IngestRequest, RetrievalFilters
from deep_context.ingestion.pipeline import ingestion_pipeline
from deep_context.retrieval.classifier import QueryClassifier
from deep_context.retrieval.engine import retrieval_engine
from deep_context.retrieval.hybrid import reciprocal_rank_fusion


@pytest.mark.asyncio
async def test_query_classifier() -> None:
    shape1 = await QueryClassifier.classify("What is the JWT filter order?")
    assert shape1.value in ("factual_lookup", "how_to")

    shape2 = await QueryClassifier.classify(
        "Read every single file in the repository and list all endpoints"
    )
    assert shape2.value == "aggregation"


def test_reciprocal_rank_fusion() -> None:
    list1 = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
    list2 = [{"id": "doc2"}, {"id": "doc1"}, {"id": "doc4"}]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 4
    # doc1 and doc2 appear in both lists, so their fused scores should be highest
    top_ids = [item[0] for item in fused[:2]]
    assert "doc1" in top_ids
    assert "doc2" in top_ids


@pytest.mark.asyncio
async def test_hybrid_retrieve_and_parent_resolution() -> None:
    # 1. Ingest test document
    doc_req = IngestRequest(
        title="Microservice Architecture Guide",
        content="""# Distributed Systems
Overview of microservices communication patterns.

## Message Queues
We use RabbitMQ and Kafka for asynchronous message publishing.
Producer services emit domain events to Kafka topics.
Consumer services subscribe to events and update read models.

## Caching Strategy
Redis is used for caching session tokens and rate limits.
TTL on auth tokens is set to 15 minutes.
""",
        doc_type="markdown",
    )
    ingest_res = await ingestion_pipeline.ingest(doc_req)

    # 2. Retrieve knowledge
    filters = RetrievalFilters(tenant_id="default")
    retrieval_res = await retrieval_engine.retrieve(
        query="What message queue is used for asynchronous publishing?",
        filters=filters,
        top_k=3,
    )

    assert retrieval_res.sufficient is True
    assert len(retrieval_res.parent_chunks) >= 1
    # Check parent content has full section context
    top_content = retrieval_res.parent_chunks[0]["content"]
    assert "Message Queues" in top_content or "Kafka" in top_content
    # Check citations
    assert len(retrieval_res.citations) >= 1
    assert retrieval_res.citations[0].document_id == ingest_res.document_id
