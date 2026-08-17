"""Core Retrieval Engine implementing the retrieve() contract."""

from __future__ import annotations

import time
from typing import Any

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.core.types import (
    Citation,
    QueryShape,
    RetrievalFilters,
    RetrievalMode,
    RetrievalResult,
)
from deep_context.retrieval.classifier import QueryClassifier
from deep_context.retrieval.hybrid import HybridRetriever
from deep_context.retrieval.reranker import CrossEncoderReranker
from deep_context.retrieval.rewriter import QueryRewriter
from deep_context.retrieval.tree_navigator import TreeNavigator
from deep_context.storage import get_storage


class RetrievalEngine:
    """The central retrieval engine implementing FR1–FR6 of PRD.md."""

    def __init__(self) -> None:
        self.classifier = QueryClassifier()
        self.rewriter = QueryRewriter()

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Executes the full retrieval pipeline:
        1. Classify query shape
        2. Rewrite / decompose into sub-queries
        3. Parallel BM25 + Vector recall (or tree navigation if vectorless)
        4. Reciprocal Rank Fusion & deduplication
        5. Cross-encoder reranking
        6. Child -> Parent chunk resolution
        7. Evidence sufficiency check (max 1 retry)
        """
        t0 = time.time()
        storage = await get_storage()
        filters = filters or RetrievalFilters()
        target_top_k = top_k or settings.default_top_k
        max_retries = settings.max_retrieval_retries

        shape = await self.classifier.classify(query)
        current_query = query
        retry_count = 0

        while True:
            sub_queries = await self.rewriter.rewrite_or_decompose(current_query, shape)

            # Check if target documents specify vectorless mode
            is_vectorless = False
            if filters.document_ids and len(filters.document_ids) == 1:
                doc = await storage.get_document(filters.document_ids[0])
                if doc and doc.retrieval_mode == RetrievalMode.VECTORLESS:
                    is_vectorless = True

            candidates: list[dict[str, Any]] = []

            if is_vectorless and filters.document_ids:
                # Tree navigation path
                tree_nav = TreeNavigator(storage)
                leaf_ids = await tree_nav.navigate(current_query, filters.document_ids[0])
                leaf_chunks = await storage.get_chunks_by_ids(leaf_ids)
                doc = await storage.get_document(filters.document_ids[0])
                for lc in leaf_chunks:
                    candidates.append(
                        {
                            "id": lc.id,
                            "document_id": lc.document_id,
                            "parent_chunk_id": lc.parent_chunk_id or lc.id,
                            "content": lc.content,
                            "section_path": lc.section_path,
                            "page_number": lc.page_number,
                            "document_title": doc.title if doc else "",
                            "source_uri": doc.source_uri if doc else None,
                            "score": 1.0,
                        }
                    )
            else:
                # Standard Hybrid RAG path (BM25 + Vector + RRF)
                hybrid_retriever = HybridRetriever(storage)
                candidates = await hybrid_retriever.retrieve_candidates(
                    sub_queries=sub_queries,
                    filters=filters,
                    limit=settings.first_stage_limit,
                )

            # Rerank candidates
            reranked_children = await CrossEncoderReranker.rerank(
                query=current_query,
                candidates=candidates,
                top_k=target_top_k,
            )

            # Resolve Child -> Parent chunks (FR2)
            parents = await self._resolve_parent_chunks(storage, reranked_children)

            # Build Citation objects
            citations = [
                Citation(
                    chunk_id=p["chunk_id"],
                    document_id=p["document_id"],
                    title=p.get("document_title", ""),
                    source_uri=p.get("source_uri"),
                    section_path=p.get("section_path"),
                    page_number=p.get("page_number"),
                )
                for p in parents
            ]

            # Evidence sufficiency check
            is_sufficient, insufficiency_reason = self._check_evidence_sufficiency(parents, shape)

            if is_sufficient:
                latency_ms = int((time.time() - t0) * 1000)
                await storage.insert_event_trace(
                    event_type="retrieval",
                    payload={
                        "query": query,
                        "query_shape": shape.value,
                        "candidates_found": len(candidates),
                        "parents_returned": len(parents),
                        "retries": retry_count,
                        "sufficient": True,
                    },
                    latency_ms=latency_ms,
                )
                return RetrievalResult(
                    sufficient=True,
                    parent_chunks=parents,
                    citations=citations,
                    query_shape=shape,
                    retry_count=retry_count,
                )

            # Insufficient evidence handling
            if retry_count >= max_retries:
                latency_ms = int((time.time() - t0) * 1000)
                await storage.insert_event_trace(
                    event_type="retrieval",
                    payload={
                        "query": query,
                        "query_shape": shape.value,
                        "candidates_found": len(candidates),
                        "parents_returned": len(parents),
                        "retries": retry_count,
                        "sufficient": False,
                        "reason": insufficiency_reason,
                    },
                    latency_ms=latency_ms,
                )
                return RetrievalResult(
                    sufficient=False,
                    parent_chunks=parents,
                    citations=citations,
                    query_shape=shape,
                    retry_count=retry_count,
                    insufficiency_reason=insufficiency_reason,
                )

            # Corrective rewrite retry (FR4: max 1 retry)
            current_query = f"{query} relevant details specifications context"
            retry_count += 1
            logger.info("Corrective retrieval retry %d for query: %s", retry_count, query)

    async def _resolve_parent_chunks(
        self, storage: Any, children: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fetch parent chunk contents so model receives full 1000-2500 token context."""
        if not children:
            return []

        parent_ids = list({c["parent_chunk_id"] for c in children if c.get("parent_chunk_id")})
        parent_chunk_map = {}
        if parent_ids:
            parent_chunks = await storage.get_chunks_by_ids(parent_ids)
            for p in parent_chunks:
                parent_chunk_map[p.id] = p

        resolved: list[dict[str, Any]] = []
        seen_parent_ids: set[str] = set()

        for c in children:
            pid = c.get("parent_chunk_id")
            if pid and pid in parent_chunk_map:
                if pid in seen_parent_ids:
                    continue
                seen_parent_ids.add(pid)
                p = parent_chunk_map[pid]
                resolved.append(
                    {
                        "chunk_id": p.id,
                        "document_id": p.document_id,
                        "content": p.content,
                        "section_path": p.section_path or c.get("section_path"),
                        "page_number": p.page_number or c.get("page_number"),
                        "document_title": c.get("document_title", ""),
                        "source_uri": c.get("source_uri"),
                    }
                )
            else:
                # If child had no parent or parent not found, use child itself
                if c["id"] in seen_parent_ids:
                    continue
                seen_parent_ids.add(c["id"])
                resolved.append(
                    {
                        "chunk_id": c["id"],
                        "document_id": c["document_id"],
                        "content": c["content"],
                        "section_path": c.get("section_path"),
                        "page_number": c.get("page_number"),
                        "document_title": c.get("document_title", ""),
                        "source_uri": c.get("source_uri"),
                    }
                )

        return resolved

    def _check_evidence_sufficiency(
        self, parents: list[dict[str, Any]], shape: QueryShape
    ) -> tuple[bool, str | None]:
        """Initial structural evidence sufficiency check before generation."""
        if not parents:
            return False, "No relevant evidence chunks retrieved."
        if shape == QueryShape.AGGREGATION and len(parents) < 2:
            return False, "Aggregation query requires broader context coverage across documents."
        return True, None


retrieval_engine = RetrievalEngine()
