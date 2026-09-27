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
    RetrievalResult,
)
from deep_context.retrieval.classifier import QueryClassifier
from deep_context.retrieval.hybrid import HybridRetriever
from deep_context.retrieval.quality_gates import hop_coverage, is_anachronism, protect_consensus
from deep_context.retrieval.reranker import Reranker
from deep_context.retrieval.rewriter import QueryRewriter
from deep_context.storage import get_storage


class RetrievalEngine:
    """The central retrieval engine implementing hybrid RAG with dual BM25 + dense search."""

    def __init__(self) -> None:
        self.classifier = QueryClassifier()
        self.rewriter = QueryRewriter()

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        reranker: str | None = None,
        user_id: str | None = None,
    ) -> RetrievalResult:
        """
        Executes the full retrieval pipeline:
        1. Classify query shape (factoid, multi-hop, how-to, etc.)
        2. Resolve stored user preferences for embedding model / reranker
        3. Check anachronisms
        4. Rewrite / decompose into sub-queries
        5. Dual-stage hybrid retrieval (BM25 content + summary_tsv + Dense Vector HNSW)
        6. Reciprocal Rank Fusion (RRF k=60)
        7. Cross-Encoder reranking
        8. Parent chunk resolution
        9. Evidence sufficiency gate (with 1 bounded retry)
        """
        t0 = time.time()
        storage = await get_storage()
        filters = filters or RetrievalFilters()
        target_top_k = top_k or settings.default_top_k
        max_retries = settings.max_retrieval_retries

        # 0. Check User Preferences from Memory Store if user_id is provided
        active_emb_model = embedding_model or settings.embedding_model
        active_emb_dim = embedding_dim or (
            768 if "gemini" in active_emb_model.lower() else settings.embedding_dim
        )
        active_reranker = reranker or settings.reranker_strategy

        if user_id:
            from deep_context.memory.stores import MemoryStoreManager

            mgr = MemoryStoreManager(storage)
            prefs = await mgr.get_embedding_preferences(user_id)
            if not embedding_model and "embedding_model" in prefs:
                active_emb_model = prefs["embedding_model"]
            if not embedding_dim and "embedding_dim" in prefs:
                active_emb_dim = prefs["embedding_dim"]
            if not reranker and "reranker" in prefs:
                active_reranker = prefs["reranker"]

        # 1. Classify Query and extract semantic filters (doc_types, section_prefix)
        shape, filters = await self.classifier.classify_and_filter(query, filters)
        current_query = query
        retry_count = 0

        # Anachronism gate
        if is_anachronism(query):
            latency_ms = int((time.time() - t0) * 1000)
            await storage.insert_event_trace(
                event_type="retrieval",
                payload={"query": query, "rejected_reason": "anachronism"},
                latency_ms=latency_ms,
            )
            return RetrievalResult(
                sufficient=False,
                parent_chunks=[],
                citations=[],
                query_shape=shape,
                retry_count=0,
                insufficiency_reason="anachronism",
            )

        while True:
            sub_queries = await self.rewriter.rewrite_or_decompose(current_query, shape)

            hybrid_retriever = HybridRetriever(storage)
            candidates = await hybrid_retriever.retrieve_candidates(
                sub_queries=sub_queries,
                filters=filters,
                limit=settings.first_stage_limit,
                embedding_model=active_emb_model,
                embedding_dim=active_emb_dim,
            )

            rerank_pool = min(24, max(target_top_k, len(candidates)))
            reranked_children = await Reranker.rerank(
                query=current_query,
                candidates=candidates,
                top_k=rerank_pool,
                strategy=active_reranker,
                embedding_model=active_emb_model,
                embedding_dim=active_emb_dim,
            )
            reranked_children = protect_consensus(candidates, reranked_children, top_k=rerank_pool)

            parents = await self._resolve_parent_chunks(storage, reranked_children)
            parents = parents[:target_top_k]

            missing_hops = hop_coverage(sub_queries, parents)
            if missing_hops and retry_count == 0:
                extra = await HybridRetriever(storage).retrieve_candidates(
                    sub_queries=missing_hops,
                    filters=filters,
                    limit=max(20, target_top_k * 2),
                    embedding_model=active_emb_model,
                    embedding_dim=active_emb_dim,
                )
                if extra:
                    merged = {c["id"]: c for c in candidates}
                    for c in extra:
                        merged[c["id"]] = c
                    reranked_children = await Reranker.rerank(
                        query=current_query,
                        candidates=list(merged.values()),
                        top_k=rerank_pool,
                        strategy=active_reranker,
                        embedding_model=active_emb_model,
                        embedding_dim=active_emb_dim,
                    )
                    parents = await self._resolve_parent_chunks(storage, reranked_children)
                    parents = parents[:target_top_k]

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
                        "sub_queries": sub_queries,
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

            current_query = f"{query} relevant details specifications context"
            retry_count += 1
            logger.info("Corrective retrieval retry %d for query: %s", retry_count, query)

    async def _resolve_parent_chunks(
        self, storage: Any, children: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fetch parent chunk contents with sibling window expansion so model receives complete, stitched context."""
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
            score_val = c.get("rerank_score", c.get("score", c.get("rrf_score", 0.0)))
            if pid and pid in parent_chunk_map:
                if pid in seen_parent_ids:
                    continue
                seen_parent_ids.add(pid)
                p = parent_chunk_map[pid]
                sec = p.section_path or c.get("section_path")
                page = p.page_number or c.get("page_number")
                p_meta = p.metadata or {}
                c_meta = c.get("metadata") or {}
                node_ids = p_meta.get("node_ids") or c_meta.get("node_ids", [])
                elem_types = p_meta.get("element_types") or c_meta.get("element_types", [])
                resolved.append(
                    {
                        "chunk_id": p.id,
                        "document_id": p.document_id,
                        "content": p.content,
                        "section_path": sec,
                        "page_number": page,
                        "node_ids": node_ids,
                        "element_types": elem_types,
                        "summary_text": c.get("summary_text") or p.summary_text,
                        "document_title": c.get("document_title", ""),
                        "source_uri": c.get("source_uri"),
                        "score": score_val,
                        "rrf_score": c.get("rrf_score", 0.0),
                    }
                )
            else:
                if c["id"] in seen_parent_ids:
                    continue
                seen_parent_ids.add(c["id"])
                c_meta = c.get("metadata") or {}
                resolved.append(
                    {
                        "chunk_id": c["id"],
                        "document_id": c["document_id"],
                        "content": c["content"],
                        "section_path": c.get("section_path"),
                        "page_number": c.get("page_number"),
                        "node_ids": c_meta.get("node_ids", []),
                        "element_types": c_meta.get("element_types", []),
                        "summary_text": c.get("summary_text"),
                        "document_title": c.get("document_title", ""),
                        "source_uri": c.get("source_uri"),
                        "score": score_val,
                        "rrf_score": c.get("rrf_score", 0.0),
                    }
                )

        # Sibling Window Merging: If two adjacent parents share the same document and consecutive pages,
        # note their continuous coverage for citations
        if len(resolved) >= 2:
            for i in range(len(resolved) - 1):
                cur = resolved[i]
                nxt = resolved[i + 1]
                if (
                    cur.get("document_id") == nxt.get("document_id")
                    and cur.get("page_number") is not None
                    and nxt.get("page_number") is not None
                ):
                    p1 = cur["page_number"]
                    p2 = nxt["page_number"]
                    if abs(p2 - p1) == 1 and not str(cur.get("section_path", "")).startswith(
                        "Pages"
                    ):
                        cur["section_path"] = f"Pages {min(p1, p2)}–{max(p1, p2)}"

        return resolved

    async def expand_chunk_context(
        self,
        chunk_id: str,
        *,
        include_ancestors: bool = True,
        include_table_structure: bool = True,
        include_figure_enrichment: bool = True,
    ) -> dict[str, Any]:
        """
        Retrieval context expansion:
        Resolves the parse tree nodes associated with a chunk, expanding context to:
        - The underlying parse tree elements (DocumentNode)
        - Surrounding sections and parent structural nodes (headings, chapters)
        - Complete table data (cell grid, row/column counts, markdown, captions)
        - Figure and chart metadata (captions, multimodal descriptions, chart fields)
        """
        storage = await get_storage()
        chunk = await storage.get_chunk(chunk_id)
        if not chunk:
            return {"error": f"Chunk '{chunk_id}' not found"}

        node_ids = chunk.node_ids or chunk.metadata.get("node_ids", [])
        nodes = []
        for nid in node_ids:
            n = await storage.get_tree_node(nid)
            if n:
                nodes.append(n)

        # If chunk didn't directly have node_ids, or it's a child chunk, check parent chunk
        if not nodes and chunk.parent_chunk_id:
            parent = await storage.get_chunk(chunk.parent_chunk_id)
            if parent:
                p_nids = parent.node_ids or parent.metadata.get("node_ids", [])
                for nid in p_nids:
                    n = await storage.get_tree_node(nid)
                    if n:
                        nodes.append(n)

        # If still no nodes by ID, try retrieving document tree nodes for this document and matching by page or section
        if not nodes and chunk.document_id:
            all_doc_nodes = await storage.get_tree_nodes(chunk.document_id)
            if all_doc_nodes:
                if chunk.page_number is not None:
                    nodes = [
                        n for n in all_doc_nodes if n.provenance.page_number == chunk.page_number
                    ]
                elif chunk.section_path:
                    nodes = [n for n in all_doc_nodes if n.section_path == chunk.section_path]

        tables = []
        figures = []
        ancestors = []
        seen_ancestor_ids = set()

        for n in nodes:
            if include_table_structure and n.table_data:
                tables.append(n.table_data.to_dict())
            if include_figure_enrichment and n.figure_data:
                figures.append(n.figure_data.to_dict())
            if include_ancestors and n.parent_id:
                p_node = await storage.get_tree_node(n.parent_id)
                while p_node and p_node.id not in seen_ancestor_ids:
                    seen_ancestor_ids.add(p_node.id)
                    ancestors.append(p_node.to_dict())
                    p_node = (
                        await storage.get_tree_node(p_node.parent_id) if p_node.parent_id else None
                    )

        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "section_path": chunk.section_path,
            "page_number": chunk.page_number,
            "node_ids": [n.id for n in nodes],
            "nodes": [n.to_dict() for n in nodes],
            "tables": tables,
            "figures": figures,
            "ancestors": ancestors,
            "element_types": list({n.node_type.value for n in nodes}),
        }

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
