"""Corrective agentic RAG state machine (hand-built, LangGraph-equivalent).

Implements the canonical corrective RAG loop — retrieve -> grade documents ->
(rewrite question & retry) | generate answer -> verify support — as plain
Python functions over an explicit state dataclass, following the CRAG paper
(arXiv:2401.15884) and LangGraph's agentic RAG reference pattern:

    START -> retrieve -> grade_documents --relevant--> generate_answer -> END
                                  |
                             irrelevant
                                  |
                          rewrite_question -> retrieve  (bounded by max_rewrites)
                                  |
                        exhausted -> abstain (safe fallback)

Nodes reuse the existing DeepContext retrieval engine, evidence verifier, and
grounded generator so this graph is a corrective wrapper around the production
pipeline rather than a parallel implementation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.core.types import RetrievalFilters
from deep_context.generation.grounded_answer import generate_grounded_answer
from deep_context.retrieval.engine import retrieval_engine
from deep_context.storage import get_storage

ABSTENTION_ANSWER = "Based on the provided context, there is insufficient evidence to answer."


@dataclass
class RAGState:
    """Shared mutable state flowing through the corrective RAG graph."""

    query: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    top_k: int = 6
    model: str | None = None
    user_id: str | None = None

    rewritten_query: str | None = None
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    relevant_docs: list[dict[str, Any]] = field(default_factory=list)
    grade_result: Literal["unknown", "relevant", "irrelevant"] = "unknown"
    rewrite_count: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    support_passed: bool = True
    support_confidence: float = 1.0
    abstained: bool = False


# ---------------------------------------------------------------------------
# Nodes (plain functions over state)
# ---------------------------------------------------------------------------


async def node_retrieve(state: RAGState) -> RAGState:
    """Retrieve parent chunks for the current (possibly rewritten) query."""
    effective_query = state.rewritten_query or state.query
    result = await retrieval_engine.retrieve(
        query=effective_query,
        filters=state.filters,
        top_k=state.top_k,
        user_id=state.user_id,
    )
    state.retrieved_docs = result.parent_chunks
    state.trace.append(
        {
            "node": "retrieve",
            "query": effective_query,
            "retrieved": len(state.retrieved_docs),
            "sufficient": result.sufficient,
        }
    )
    return state


def _term_overlap_ratio(query: str, doc_content: str) -> float:
    q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
    if not q_terms:
        return 0.0
    blob = doc_content.lower()
    hits = sum(1 for t in q_terms if t in blob)
    return hits / len(q_terms)


async def node_grade_documents(state: RAGState) -> RAGState:
    """Grade each retrieved document for relevance to the original question.

    Uses deterministic term-overlap scoring against AGENTIC_GRADE_THRESHOLD.
    This mirrors LangGraph's binary GradeDocuments gate but stays fully
    offline/deterministic; an LLM grader can be swapped in behind the same
    interface without changing the graph topology.
    """
    threshold = settings.agentic_grade_threshold
    relevant = [
        doc
        for doc in state.retrieved_docs
        if _term_overlap_ratio(state.query, str(doc.get("content", ""))) >= threshold
    ]
    state.relevant_docs = relevant
    state.grade_result = "relevant" if relevant else "irrelevant"
    state.trace.append(
        {
            "node": "grade_documents",
            "threshold": threshold,
            "relevant": len(relevant),
            "total": len(state.retrieved_docs),
            "result": state.grade_result,
        }
    )
    return state


async def node_rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the query to improve retrieval odds; bounded by max_rewrites."""
    from deep_context.core.types import QueryShape
    from deep_context.retrieval.rewriter import QueryRewriter

    state.rewrite_count += 1
    rewritten: str | None = None
    try:
        variants = await QueryRewriter.rewrite_or_decompose(state.query, QueryShape.FACTUAL_LOOKUP)
        # Pick the first variant that differs from the original query.
        for v in variants:
            if v and v.strip() != state.query.strip():
                rewritten = v
                break
    except Exception as e:
        logger.debug("Query rewrite failed: %s", e)
    # Fall back to a deterministic keyword-expansion rewrite when the LLM
    # rewriter is unavailable (offline/test mode).
    if not rewritten or rewritten.strip() == state.query.strip():
        rewritten = _deterministic_rewrite(state.query)

    state.rewritten_query = rewritten
    state.trace.append(
        {
            "node": "rewrite_query",
            "rewrite_count": state.rewrite_count,
            "rewritten": rewritten,
        }
    )
    return state


def _deterministic_rewrite(query: str) -> str:
    """Expand a query with generic retrieval-friendly phrasing when LLM unavailable."""
    stop = {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "does",
        "do",
        "did",
        "can",
        "could",
        "tell",
        "about",
    }
    keywords = [w for w in re.findall(r"\w+", query.lower()) if w not in stop and len(w) > 2]
    if not keywords:
        return f"{query} details explanation"
    return f"{query} ({' '.join(keywords)})"


async def node_generate_answer(state: RAGState) -> RAGState:
    """Generate a grounded answer from the graded-relevant documents only."""
    grounded = await generate_grounded_answer(
        query=state.query,
        retrieved_chunks=state.relevant_docs,
        model=state.model,
    )
    state.answer = grounded.answer
    state.support_passed = grounded.support_passed
    state.support_confidence = grounded.support_confidence

    citations: list[dict[str, Any]] = []
    for doc in state.relevant_docs:
        citation = doc.get("citation")
        if isinstance(citation, dict):
            citations.append(citation)
        elif doc.get("chunk_id") or doc.get("id"):
            citations.append(
                {
                    "chunk_id": doc.get("chunk_id") or doc.get("id"),
                    "document_id": doc.get("document_id"),
                    "title": doc.get("document_title") or doc.get("title"),
                    "section_path": doc.get("section_path"),
                    "page_number": doc.get("page_number"),
                }
            )
    state.citations = citations
    state.trace.append(
        {
            "node": "generate_answer",
            "refused": grounded.refused,
            "support_passed": grounded.support_passed,
            "citations": len(citations),
        }
    )
    return state


# ---------------------------------------------------------------------------
# Graph runner with explicit conditional edges
# ---------------------------------------------------------------------------


async def run_agentic_rag(
    query: str,
    *,
    filters: RetrievalFilters | None = None,
    top_k: int = 6,
    max_rewrites: int | None = None,
    model: str | None = None,
    user_id: str | None = None,
) -> RAGState:
    """Execute the corrective RAG loop and return the final state.

    Conditional edges:
    - grade_documents == relevant      -> generate_answer
    - grade_documents == irrelevant and rewrite_count < max_rewrites
                                       -> rewrite_query -> retrieve -> grade
    - exhausted                        -> safe abstention fallback
    """
    state = RAGState(
        query=query,
        filters=filters or RetrievalFilters(),
        top_k=top_k,
        model=model,
        user_id=user_id,
    )
    max_loops = max_rewrites if max_rewrites is not None else settings.agentic_max_rewrites
    t0 = time.time()

    await node_retrieve(state)
    await node_grade_documents(state)

    while state.grade_result == "irrelevant" and state.rewrite_count < max_loops:
        await node_rewrite_query(state)
        await node_retrieve(state)
        await node_grade_documents(state)

    if state.grade_result == "irrelevant":
        # Safe fallback per grounding policy: never answer from weak context.
        state.abstained = True
        state.answer = ABSTENTION_ANSWER
        state.support_passed = True
        state.support_confidence = 1.0
        state.trace.append({"node": "abstain", "reason": "no_relevant_documents"})

        storage = await get_storage()
        await storage.insert_event_trace(
            event_type="agentic_rag",
            payload={
                "query_shape": "corrective_loop_exhausted",
                "rewrite_count": state.rewrite_count,
                "retrieved_total": len(state.retrieved_docs),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )
        return state

    await node_generate_answer(state)
    state.trace.append({"node": "done", "latency_ms": int((time.time() - t0) * 1000)})
    return state
