"""End-to-end ingestion pipeline implementing workflows/01_ingestion_pipeline.md."""

from __future__ import annotations

import time
import uuid

from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import (
    Document,
    IngestRequest,
    IngestResponse,
    RetrievalMode,
)
from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.parser import DocumentParser
from deep_context.ingestion.tree_indexer import DocumentTreeIndexer
from deep_context.storage import get_storage


class IngestionPipeline:
    """Ingests raw documents, chunks hierarchically, generates bge-m3 embeddings, and stores them."""

    def __init__(self) -> None:
        self.parser = DocumentParser()
        self.chunker = ParentChildChunker()

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        t0 = time.time()
        storage = await get_storage()
        doc_id = str(uuid.uuid4())

        # 1. Parse document into structure-aware sections
        sections = self.parser.parse(request.content, doc_type=request.doc_type)

        # 2. Parent-child chunking
        parent_chunks, child_chunks = self.chunker.chunk_sections(doc_id, sections)

        # 3. Generate dense embeddings for child chunks using NVIDIA NIM bge-m3
        child_texts = [c.content for c in child_chunks]
        if child_texts:
            embeddings = await llm_client.get_embeddings(child_texts)
            for chunk, emb in zip(child_chunks, embeddings):
                chunk.embedding = emb

        # 4. Create document record
        doc = Document(
            id=doc_id,
            tenant_id=request.tenant_id,
            title=request.title,
            source_uri=request.source_uri,
            doc_type=request.doc_type,
            permission_scope=request.permission_scope,
            retrieval_mode=request.retrieval_mode,
            metadata=request.metadata,
        )

        # 5. Persist to storage (in order: document -> chunks -> optional tree nodes)
        await storage.insert_document(doc)
        all_chunks = parent_chunks + child_chunks
        await storage.insert_chunks(all_chunks)

        tree_nodes_count = 0
        if request.retrieval_mode == RetrievalMode.VECTORLESS:
            tree_nodes = DocumentTreeIndexer.build_tree_nodes(doc_id, sections, parent_chunks)
            await storage.insert_tree_nodes(tree_nodes)
            tree_nodes_count = len(tree_nodes)

        latency_ms = int((time.time() - t0) * 1000)

        # 6. Trace event logging
        await storage.insert_event_trace(
            event_type="ingestion",
            payload={
                "document_id": doc_id,
                "title": request.title,
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(child_chunks),
                "retrieval_mode": request.retrieval_mode.value,
            },
            latency_ms=latency_ms,
        )

        logger.info(
            "Successfully ingested document %s (%s) with %d parents and %d children in %d ms",
            doc_id,
            request.title,
            len(parent_chunks),
            len(child_chunks),
            latency_ms,
        )

        return IngestResponse(
            document_id=doc_id,
            title=request.title,
            parent_chunks_count=len(parent_chunks),
            child_chunks_count=len(child_chunks),
            retrieval_mode=request.retrieval_mode,
            tree_nodes_count=tree_nodes_count,
        )


ingestion_pipeline = IngestionPipeline()
