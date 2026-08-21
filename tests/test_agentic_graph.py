"""Tests for the corrective agentic RAG state machine (deep_context.agentic.graph)."""

from __future__ import annotations

import pytest

from deep_context.agentic.graph import (
    ABSTENTION_ANSWER,
    RAGState,
    _deterministic_rewrite,
    _term_overlap_ratio,
    node_grade_documents,
    run_agentic_rag,
)
from deep_context.core.config import settings
from deep_context.storage import get_storage


@pytest.fixture(autouse=True)
def seeded_corpus() -> None:
    """Ingest a small deterministic corpus for retrieval tests."""
    import asyncio

    from deep_context.core.types import IngestRequest, RetrievalMode
    from deep_context.ingestion.pipeline import ingestion_pipeline

    async def _seed() -> None:
        storage = await get_storage()
        existing = await storage.list_document_summaries(limit=100)
        if any(d["title"] == "Graph Test Corpus" for d in existing):
            return
        req = IngestRequest(
            title="Graph Test Corpus",
            content=(
                "# Reward Hacking\n\n"
                "Reward hacking occurs when an AI agent exploits flaws or ambiguities "
                "in its reward function to achieve high rewards without performing the "
                "intended behaviors. There are two main types: environment or goal "
                "misspecification, and reward tampering.\n\n"
                "# Diffusion Models\n\n"
                "Diffusion models generate images by iteratively denoising random noise "
                "guided by a learned gradient field. They are widely used for text-to-image "
                "synthesis.\n"
            ),
            doc_type="markdown",
            retrieval_mode=RetrievalMode.HYBRID,
        )
        await ingestion_pipeline.ingest(req)

    asyncio.run(_seed())


class TestGrading:
    def test_term_overlap_ratio(self) -> None:
        ratio = _term_overlap_ratio(
            "reward hacking types", "reward hacking has two types"
        )
        assert ratio > 0.5

    def test_term_overlap_zero_for_unrelated(self) -> None:
        assert (
            _term_overlap_ratio("quantum entanglement", "recipe for banana bread")
            == 0.0
        )

    async def test_grade_relevant(self) -> None:
        state = RAGState(query="reward hacking")
        state.retrieved_docs = [
            {"content": "Reward hacking exploits flaws in the reward function."}
        ]
        result = await node_grade_documents(state)
        assert result.grade_result == "relevant"
        assert len(result.relevant_docs) == 1

    async def test_grade_irrelevant(self) -> None:
        state = RAGState(query="quantum entanglement experiments")
        state.retrieved_docs = [{"content": "Banana bread requires ripe bananas."}]
        result = await node_grade_documents(state)
        assert result.grade_result == "irrelevant"
        assert result.relevant_docs == []


class TestRewrite:
    def test_deterministic_rewrite_expands_keywords(self) -> None:
        rewritten = _deterministic_rewrite("What is reward hacking?")
        assert "reward" in rewritten.lower()
        assert rewritten != "What is reward hacking?"


class TestAgenticLoop:
    async def test_relevant_docs_skip_rewrite_and_answer(self) -> None:
        state = await run_agentic_rag(
            query="What is reward hacking and what are its types?",
            max_rewrites=2,
        )
        assert state.grade_result == "relevant"
        assert state.rewrite_count == 0
        assert not state.abstained
        assert state.answer
        assert state.trace[-1]["node"] in ("generate_answer", "done")

    async def test_irrelevant_docs_trigger_rewrite_then_abstain(self) -> None:
        # A query with zero lexical overlap against the seeded corpus forces
        # the grade gate to fail, exercising the rewrite loop and abstention.
        state = await run_agentic_rag(
            query="xylophone theremin kazoo orchestration",
            max_rewrites=1,
        )
        assert state.abstained
        assert state.answer == ABSTENTION_ANSWER
        assert state.rewrite_count >= 1
        assert state.trace[0]["node"] == "retrieve"
        assert any(t["node"] == "rewrite_query" for t in state.trace)
        assert state.trace[-1]["node"] == "abstain"

    async def test_max_rewrites_respected(self) -> None:
        settings.agentic_max_rewrites = 5
        state = await run_agentic_rag(
            query="xylophone theremin kazoo orchestration",
            max_rewrites=0,
        )
        # With zero rewrites allowed, abstain immediately after first grading.
        assert state.rewrite_count == 0
        assert state.abstained

    async def test_trace_records_all_nodes(self) -> None:
        state = await run_agentic_rag(
            query="What is reward hacking?",
            max_rewrites=2,
        )
        node_names = [t["node"] for t in state.trace]
        assert "retrieve" in node_names
        assert "grade_documents" in node_names
