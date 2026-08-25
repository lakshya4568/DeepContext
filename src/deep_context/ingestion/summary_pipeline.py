"""End-to-end ingestion pipeline combining Parent-Child chunking with local LLM summarization."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

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
from deep_context.storage.base import StorageInterface


@dataclass
class IngestionResult:
    document_id: str
    title: str
    parent_chunks: int
    child_chunks: int
    summaries_generated: int
    errors: list[str] = field(default_factory=list)


class SummaryIngestionPipeline:
    """Production-grade ingestion combining structure-aware parent-child chunking and Qwen3 summaries."""

    def __init__(
        self,
        storage: StorageInterface | None = None,
        parser: DocumentParser | None = None,
        chunker: ParentChildChunker | None = None,
        summarizer: ChunkSummarizer | None = None,
    ):
        self.storage = storage
        self.parser = parser or DocumentParser()
        self.chunker = chunker or ParentChildChunker()
        self.summarizer = summarizer or ChunkSummarizer()

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Ingests a document, creates parent-child chunks, generates LLM summaries, and embeds."""
        t0 = time.time()
        storage = self.storage or await get_storage()
        doc_id = str(uuid.uuid4())

        # 1. Parse document into structure-aware sections
        sections = self.parser.parse(request.content, doc_type=request.doc_type)

        # 2. Parent-child hierarchical chunking
        parent_chunks, child_chunks = self.chunker.chunk_sections(doc_id, sections)

        # 3. Generate Qwen3 summaries for child chunks if enabled
        should_summarize = (
            request.generate_summaries
            if request.generate_summaries is not None
            else settings.summary_enabled
        )

        summaries_count = 0
        if should_summarize and child_chunks:
            logger.info(
                "Generating Qwen3 semantic summaries for %d child chunks in doc '%s'...",
                len(child_chunks),
                request.title,
            )
            await self.summarizer.summarize_chunks(child_chunks)
            summaries_count = sum(1 for c in child_chunks if c.summary_text)
            self.summarizer.unload()

        # 4. Generate dense embeddings for child chunks
        emb_model = request.embedding_model or settings.embedding_model
        emb_dim = request.embedding_dim or (
            768 if "gemini" in emb_model.lower() else settings.embedding_dim
        )

        child_texts = [c.content for c in child_chunks]
        if child_texts:
            embeddings = await llm_client.get_embeddings(
                child_texts,
                model=emb_model,
                dim=emb_dim,
                title=request.title,
                is_query=False,
            )
            for chunk, emb in zip(child_chunks, embeddings):
                chunk.embedding = emb

        # 5. Create document record
        doc_metadata = dict(request.metadata)
        doc_metadata["embedding_model"] = emb_model
        doc_metadata["embedding_dim"] = emb_dim
        doc_metadata["summary_model"] = settings.summary_model if should_summarize else None

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

        # 6. Persist to storage
        await storage.insert_document(doc)
        all_chunks = parent_chunks + child_chunks
        await storage.insert_chunks(all_chunks)

        latency_ms = int((time.time() - t0) * 1000)

        # 7. Trace event logging
        await storage.insert_event_trace(
            event_type="ingestion_summary",
            payload={
                "document_id": doc_id,
                "title": request.title,
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(child_chunks),
                "summaries_generated": summaries_count,
                "retrieval_mode": request.retrieval_mode.value,
                "embedding_model": emb_model,
                "embedding_dim": emb_dim,
            },
            latency_ms=latency_ms,
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

    async def ingest_batch(
        self,
        requests: list[IngestRequest],
        concurrency: int = 4,
    ) -> list[IngestResponse]:
        """Ingest multiple documents concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(req: IngestRequest) -> IngestResponse:
            async with semaphore:
                return await self.ingest(req)

        tasks = [_bounded(r) for r in requests]
        return await asyncio.gather(*tasks)

    async def ingest_document(
        self,
        file_path_or_content: str,
        title: str | None = None,
        doc_type: str = "auto",
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        generate_summaries: bool | None = None,
        tenant_id: str = "default",
        permission_scope: list[str] | None = None,
    ) -> IngestResponse:
        """Convenience method to ingest a file from path or raw text content."""
        from pathlib import Path

        p = Path(file_path_or_content)
        if p.exists() and p.is_file():
            doc_title = title or p.stem
            ext = p.suffix.lower()
            detected_type = (
                doc_type
                if doc_type != "auto"
                else (
                    "pdf"
                    if ext == ".pdf"
                    else ("markdown" if ext in (".md", ".markdown") else "text")
                )
            )
            if detected_type == "pdf":
                content = str(p.resolve())
            else:
                content = p.read_text(encoding="utf-8", errors="replace")
            source_uri = str(p.resolve())
        else:
            doc_title = title or "Document"
            detected_type = "markdown" if doc_type == "auto" else doc_type
            content = file_path_or_content
            source_uri = None

        req = IngestRequest(
            title=doc_title,
            content=content,
            doc_type=detected_type,
            source_uri=source_uri,
            embedding_model=embedding_model or settings.embedding_model,
            embedding_dim=embedding_dim or settings.embedding_dim,
            generate_summaries=generate_summaries,
            tenant_id=tenant_id,
            permission_scope=permission_scope or ["default"],
        )
        return await self.ingest(req)

    async def ingest_stream(self, request: IngestRequest) -> AsyncIterator[dict[str, Any]]:
        """Ingests a document while streaming real-time progress events for UI and CLI."""
        t0 = time.time()
        yield {
            "stage": "parsing",
            "percent": 5,
            "message": f"Extracting sections from '{request.title}'...",
        }

        storage = self.storage or await get_storage()
        doc_id = str(uuid.uuid4())

        # 1. Parse
        sections = self.parser.parse(request.content, doc_type=request.doc_type)
        yield {
            "stage": "chunking",
            "percent": 10,
            "message": f"Extracted {len(sections)} sections. Creating parent & child chunks...",
        }

        # 2. Chunk
        parent_chunks, child_chunks = self.chunker.chunk_sections(doc_id, sections)
        yield {
            "stage": "chunked",
            "percent": 15,
            "parents": len(parent_chunks),
            "children": len(child_chunks),
            "message": f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks.",
        }

        # 3. Summarize
        should_summarize = (
            request.generate_summaries
            if request.generate_summaries is not None
            else settings.summary_enabled
        )

        summaries_count = 0
        if should_summarize and child_chunks:
            total_children = len(child_chunks)
            yield {
                "stage": "summarizing_start",
                "percent": 18,
                "total": total_children,
                "message": f"Initializing Qwen3 model on GPU/MPS for {total_children} chunks...",
            }
            t_sum_start = time.time()

            await self.summarizer._ensure_model_loaded()
            loop = asyncio.get_running_loop()
            async with self.summarizer._load_lock:
                for idx, chunk in enumerate(child_chunks):
                    prompt = self.summarizer._build_prompt(chunk.content, chunk.section_path)
                    batch_res = await loop.run_in_executor(
                        None, self.summarizer._generate_batch_sync, [prompt]
                    )

                    summary, tokens = batch_res[0] if batch_res else ("", 0)
                    if not summary:
                        words = chunk.content.strip().split()
                        summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
                        tokens = len(summary.split())
                    chunk.summary_text = summary
                    chunk.summary_tokens = tokens
                    chunk.summary_model = settings.summary_model
                    chunk.generated_at = datetime.now(timezone.utc)
                    summaries_count += 1

                    done = summaries_count
                    elapsed = time.time() - t_sum_start
                    rate = elapsed / max(1, done)
                    remaining = total_children - done
                    eta_sec = int(rate * remaining)
                    pct = 18 + int(62 * (done / total_children))

                    sec_label = chunk.section_path or (
                        f"Page {chunk.page_number}" if chunk.page_number else "Section"
                    )
                    yield {
                        "stage": "summarizing",
                        "percent": pct,
                        "current": done,
                        "total": total_children,
                        "summary": summary,
                        "tokens": tokens,
                        "section": sec_label,
                        "eta_sec": eta_sec,
                        "rate_sec": round(rate, 2),
                        "message": f"Summarized child chunk {done}/{total_children} (ETA: {eta_sec // 60}m {eta_sec % 60}s)...",
                    }
            self.summarizer.unload()

        # 4. Embed
        emb_model = request.embedding_model or settings.embedding_model
        emb_dim = request.embedding_dim or (
            768 if "gemini" in emb_model.lower() else settings.embedding_dim
        )
        yield {
            "stage": "embedding",
            "percent": 82,
            "message": f"Generating {emb_dim}-dim embeddings via {emb_model}...",
        }

        child_texts = [c.content for c in child_chunks]
        if child_texts:
            embeddings = await llm_client.get_embeddings(
                child_texts,
                model=emb_model,
                dim=emb_dim,
                title=request.title,
                is_query=False,
            )
            for chunk, emb in zip(child_chunks, embeddings):
                chunk.embedding = emb

        # 5. Persist
        yield {
            "stage": "indexing",
            "percent": 94,
            "message": "Saving vectors and building HNSW indexes in PostgreSQL...",
        }

        doc_metadata = dict(request.metadata)
        doc_metadata["embedding_model"] = emb_model
        doc_metadata["embedding_dim"] = emb_dim
        doc_metadata["summary_model"] = settings.summary_model if should_summarize else None

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

        await storage.insert_document(doc)
        all_chunks = parent_chunks + child_chunks
        await storage.insert_chunks(all_chunks)

        elapsed_total = round(time.time() - t0, 1)
        yield {
            "stage": "complete",
            "percent": 100,
            "document_id": doc_id,
            "title": request.title,
            "parent_chunks_count": len(parent_chunks),
            "child_chunks_count": len(child_chunks),
            "summaries_generated_count": summaries_count,
            "embedding_model": emb_model,
            "embedding_dim": emb_dim,
            "elapsed_sec": elapsed_total,
            "message": f"Successfully ingested '{request.title}' in {elapsed_total}s!",
        }
