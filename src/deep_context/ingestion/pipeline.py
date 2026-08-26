"""End-to-end ingestion pipeline implementing workflows/01_ingestion_pipeline.md."""

from __future__ import annotations

import time
import uuid

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import (
    Document,
    IngestRequest,
    IngestResponse,
)
from deep_context.ingestion.chunker import ParentChildChunker
from deep_context.ingestion.parser import DocumentParser
from deep_context.ingestion.summarizer import ChunkSummarizer
from deep_context.storage import get_storage


class IngestionPipeline:
    """Ingests raw documents, chunks hierarchically, generates embeddings, summaries, and stores them."""

    def __init__(self) -> None:
        self.parser = DocumentParser()
        self.chunker = ParentChildChunker()
        self.summarizer = ChunkSummarizer()

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        t0 = time.time()
        storage = await get_storage()
        doc_id = str(uuid.uuid4())

        # 1. Parse document into structure-aware sections
        sections = self.parser.parse(request.content, doc_type=request.doc_type)

        # 2. Parent-child chunking
        parent_chunks, child_chunks = self.chunker.chunk_sections(doc_id, sections)

        # 3. Generate Qwen3 contextual summaries for child chunks if enabled
        should_summarize = (
            request.generate_summaries
            if request.generate_summaries is not None
            else settings.summary_enabled
        )

        summaries_count = 0
        parent_map = {p.id: p for p in parent_chunks}
        if should_summarize and child_chunks:
            logger.info(
                "Generating Qwen3 contextual summaries for %d child chunks...", len(child_chunks)
            )
            await self.summarizer.summarize_chunks(
                child_chunks,
                parent_chunk_map=parent_map,
                document_title=request.title,
            )
            summaries_count = len(child_chunks)
            self.summarizer.unload()

        # 4. Generate contextual dense embeddings for child chunks (Anthropic/Unstructured Standard)
        emb_model = request.embedding_model or settings.embedding_model
        emb_dim = request.embedding_dim or settings.embedding_dim

        child_contextual_texts = [
            f"{c.summary_text}\n\n{c.content}" if c.summary_text else c.content
            for c in child_chunks
        ]
        if child_contextual_texts:
            embeddings = await llm_client.get_embeddings(
                child_contextual_texts,
                model=emb_model,
                dim=emb_dim,
                title=request.title,
                is_query=False,
            )
            for chunk, emb in zip(child_chunks, embeddings):
                chunk.embedding = emb

        doc_metadata = request.metadata.copy()
        doc_metadata["embedding_model"] = emb_model
        doc_metadata["embedding_dim"] = emb_dim
        doc_metadata["summaries_generated"] = summaries_count > 0

        doc = Document(
            id=doc_id,
            tenant_id=request.tenant_id,
            title=request.title,
            source_uri=request.source_uri,
            doc_type=request.doc_type,
            permission_scope=request.permission_scope,
            retrieval_mode=request.retrieval_mode,
            metadata=doc_metadata,
        )

        # 5. Persist to storage (document -> chunks)
        await storage.insert_document(doc)
        all_chunks = parent_chunks + child_chunks
        await storage.insert_chunks(all_chunks)

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
                "embedding_model": emb_model,
                "embedding_dim": emb_dim,
            },
            latency_ms=latency_ms,
        )

        logger.info(
            "Successfully ingested document %s (%s) with %d parents and %d children using %s (%d-dim) in %d ms",
            doc_id,
            request.title,
            len(parent_chunks),
            len(child_chunks),
            emb_model,
            emb_dim,
            latency_ms,
        )

        return IngestResponse(
            document_id=doc_id,
            title=request.title,
            parent_chunks_count=len(parent_chunks),
            child_chunks_count=len(child_chunks),
            retrieval_mode=request.retrieval_mode,
            summaries_generated_count=summaries_count,
            embedding_model=emb_model,
            embedding_dim=emb_dim,
        )


ingestion_pipeline = IngestionPipeline()
