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

        # 3. Create and persist document & parent/child chunks immediately (Checkpoint 1)
        emb_model = request.embedding_model or settings.embedding_model
        emb_dim = request.embedding_dim or (
            768 if "gemini" in emb_model.lower() else settings.embedding_dim
        )

        should_summarize = (
            request.generate_summaries
            if request.generate_summaries is not None
            else settings.summary_enabled
        )

        # 3. Create and persist document & parent/child chunks immediately (Checkpoint 1: Zero Data Loss)
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

        await storage.insert_document_and_chunks(doc, parent_chunks + child_chunks)

        # 4. Generate Qwen3 contextual summaries for child chunks if enabled
        summaries_count = 0
        parent_map = {p.id: p for p in parent_chunks}
        if should_summarize and child_chunks:
            logger.info(
                "Generating Qwen3 contextual summaries for %d child chunks in doc '%s'...",
                len(child_chunks),
                request.title,
            )
            await self.summarizer.summarize_chunks(
                child_chunks,
                parent_chunk_map=parent_map,
                document_title=request.title,
            )
            summaries_count = sum(1 for c in child_chunks if c.summary_text)

            summary_updates = [
                (
                    c.id,
                    c.summary_text or "",
                    c.summary_tokens or 0,
                    c.summary_model or settings.summary_model,
                    c.generated_at,
                )
                for c in child_chunks
                if c.summary_text
            ]
            if summary_updates:
                await storage.update_chunk_summaries_batch(summary_updates)

            self.summarizer.unload()

        # 5. Generate contextual dense embeddings for child chunks in progressive batches
        embeddings_count = 0
        if child_chunks:
            emb_batch_size = 32
            for i in range(0, len(child_chunks), emb_batch_size):
                batch_chunks = child_chunks[i : i + emb_batch_size]
                batch_texts = [
                    f"{c.summary_text}\n\n{c.content}" if c.summary_text else c.content
                    for c in batch_chunks
                ]
                try:
                    embeddings = await llm_client.get_embeddings(
                        batch_texts,
                        model=emb_model,
                        dim=emb_dim,
                        title=request.title,
                        is_query=False,
                    )
                    emb_updates = []
                    for chunk, emb in zip(batch_chunks, embeddings):
                        chunk.embedding = emb
                        emb_updates.append((chunk.id, emb))
                    if emb_updates:
                        await storage.update_chunk_embeddings_batch(emb_updates)
                        embeddings_count += len(emb_updates)
                except Exception as emb_err:
                    logger.warning(
                        "Embedding batch %d-%d for doc '%s' paused due to rate limit/error (%s). All chunks and summaries remain safely preserved in DB.",
                        i,
                        i + len(batch_chunks),
                        request.title,
                        emb_err,
                    )
                    break

        latency_ms = int((time.time() - t0) * 1000)

        # 6. Trace event logging
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
            detected_type = p.suffix.lstrip(".").lower() if doc_type == "auto" else doc_type
            content = p.read_text(encoding="utf-8", errors="ignore")
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

        emb_model = request.embedding_model or settings.embedding_model
        emb_dim = request.embedding_dim or (
            768 if "gemini" in emb_model.lower() else settings.embedding_dim
        )

        should_summarize = (
            request.generate_summaries
            if request.generate_summaries is not None
            else settings.summary_enabled
        )

        # 3. Immediately persist document and chunks (Checkpoint 1: Zero Data Loss)
        yield {
            "stage": "indexing",
            "percent": 18,
            "message": "Persisting document & parent-child chunk hierarchy to database...",
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

        await storage.insert_document_and_chunks(doc, parent_chunks + child_chunks)

        # 4. Summarize child chunks in contextual vectorized batches
        summaries_count = 0
        parent_map = {p.id: p for p in parent_chunks}
        if should_summarize and child_chunks:
            total_children = len(child_chunks)
            yield {
                "stage": "summarizing_start",
                "percent": 22,
                "total": total_children,
                "message": f"Initializing Qwen3 model on GPU for {total_children} chunks...",
            }
            t_sum_start = time.time()

            await self.summarizer._ensure_model_loaded()
            loop = asyncio.get_running_loop()
            batch_size = max(1, self.summarizer.batch_size)
            pending_updates: list[tuple[str, str, int, str, Any]] = []

            async with self.summarizer._load_lock:
                for i in range(0, total_children, batch_size):
                    chunk_batch = child_chunks[i : i + batch_size]
                    prompts = []
                    for c in chunk_batch:
                        p_chunk = parent_map.get(c.parent_chunk_id) if c.parent_chunk_id else None
                        parent_text = p_chunk.content if p_chunk else None
                        prompts.append(
                            self.summarizer._build_prompt(
                                c.content,
                                section_path=c.section_path,
                                document_title=request.title,
                                parent_context=parent_text,
                            )
                        )

                    batch_res = await loop.run_in_executor(
                        None, self.summarizer._generate_batch_sync, prompts
                    )

                    for c_idx, chunk in enumerate(chunk_batch):
                        summary, tokens = batch_res[c_idx] if c_idx < len(batch_res) else ("", 0)
                        if not summary:
                            words = chunk.content.strip().split()
                            summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
                            tokens = len(summary.split())
                        chunk.summary_text = summary
                        chunk.summary_tokens = tokens
                        chunk.summary_model = settings.summary_model
                        chunk.generated_at = datetime.now(timezone.utc)
                        summaries_count += 1

                        pending_updates.append(
                            (chunk.id, summary, tokens, settings.summary_model, chunk.generated_at)
                        )

                    # Progressive incremental save to database
                    if pending_updates:
                        await storage.update_chunk_summaries_batch(pending_updates)
                        pending_updates.clear()

                    done = summaries_count
                    elapsed = time.time() - t_sum_start
                    rate = elapsed / max(1, done)
                    remaining = total_children - done
                    eta_sec = int(rate * remaining)
                    pct = 22 + int(58 * (done / total_children))

                    last_chunk = chunk_batch[-1]
                    sec_label = last_chunk.section_path or (
                        f"Page {last_chunk.page_number}" if last_chunk.page_number else "Section"
                    )
                    yield {
                        "stage": "summarizing",
                        "percent": pct,
                        "current": done,
                        "total": total_children,
                        "summary": last_chunk.summary_text,
                        "tokens": last_chunk.summary_tokens,
                        "section": sec_label,
                        "eta_sec": eta_sec,
                        "rate_sec": round(rate, 2),
                        "message": f"Summarized child chunk {done}/{total_children} (ETA: {eta_sec // 60}m {eta_sec % 60}s)...",
                    }

            self.summarizer.unload()

        # 5. Embed contextual child texts progressively
        yield {
            "stage": "embedding",
            "percent": 82,
            "message": f"Generating {emb_dim}-dim contextual embeddings via {emb_model}...",
        }

        embeddings_count = 0
        if child_chunks:
            total_emb_chunks = len(child_chunks)
            emb_batch_size = 32
            for i in range(0, total_emb_chunks, emb_batch_size):
                batch_chunks = child_chunks[i : i + emb_batch_size]
                batch_texts = [
                    f"{c.summary_text}\n\n{c.content}" if c.summary_text else c.content
                    for c in batch_chunks
                ]
                try:
                    embeddings = await llm_client.get_embeddings(
                        batch_texts,
                        model=emb_model,
                        dim=emb_dim,
                        title=request.title,
                        is_query=False,
                    )
                    emb_updates = []
                    for chunk, emb in zip(batch_chunks, embeddings):
                        chunk.embedding = emb
                        emb_updates.append((chunk.id, emb))
                    if emb_updates:
                        await storage.update_chunk_embeddings_batch(emb_updates)
                        embeddings_count += len(emb_updates)

                    emb_pct = 82 + int(16 * (embeddings_count / total_emb_chunks))
                    yield {
                        "stage": "embedding_progress",
                        "percent": emb_pct,
                        "current": embeddings_count,
                        "total": total_emb_chunks,
                        "message": f"Embedded child chunks {embeddings_count}/{total_emb_chunks}...",
                    }
                except Exception as emb_err:
                    logger.warning(
                        "Embedding batch %d-%d for doc '%s' paused due to rate limit (%s). All chunks and summaries remain safely preserved.",
                        i,
                        i + len(batch_chunks),
                        request.title,
                        emb_err,
                    )
                    yield {
                        "stage": "embedding_paused",
                        "percent": 95,
                        "embedded_count": embeddings_count,
                        "total_child_chunks": total_emb_chunks,
                        "message": f"Embedding quota paused at {embeddings_count}/{total_emb_chunks}. All chunks and summaries are 100% saved! Click 'Embed' anytime to resume.",
                    }
                    break

        elapsed_total = round(time.time() - t0, 1)
        complete_msg = (
            f"Successfully ingested and indexed '{request.title}' in {elapsed_total}s!"
            if embeddings_count == len(child_chunks)
            else f"Saved '{request.title}' structure & summaries! ({embeddings_count}/{len(child_chunks)} embeddings generated - can be resumed anytime)"
        )
        yield {
            "stage": "complete",
            "percent": 100,
            "document_id": doc_id,
            "title": request.title,
            "parent_chunks_count": len(parent_chunks),
            "child_chunks_count": len(child_chunks),
            "summaries_generated_count": summaries_count,
            "embeddings_generated_count": embeddings_count,
            "embedding_model": emb_model,
            "embedding_dim": emb_dim,
            "elapsed_sec": elapsed_total,
            "message": complete_msg,
        }

    async def resume_document_embeddings(
        self,
        document_id: str,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        batch_size: int = 32,
    ) -> AsyncIterator[dict[str, Any]]:
        """Resume generating embeddings for any chunks in a document that are missing embeddings."""
        storage = self.storage or await get_storage()
        doc = await storage.get_document(document_id)
        if not doc:
            yield {"stage": "error", "message": f"Document '{document_id}' not found."}
            return

        unembedded = await storage.get_unembedded_chunks(document_id)
        total_missing = len(unembedded)
        if total_missing == 0:
            yield {
                "stage": "complete",
                "percent": 100,
                "document_id": document_id,
                "title": doc.title,
                "embedded_count": 0,
                "message": f"All chunks for '{doc.title}' already have embeddings!",
            }
            return

        emb_model = (
            embedding_model or doc.metadata.get("embedding_model") or settings.embedding_model
        )
        emb_dim = (
            embedding_dim
            or doc.metadata.get("embedding_dim")
            or (768 if "gemini" in emb_model.lower() else settings.embedding_dim)
        )

        yield {
            "stage": "resuming_start",
            "percent": 5,
            "document_id": document_id,
            "title": doc.title,
            "total_missing": total_missing,
            "message": f"Generating missing embeddings for {total_missing} chunks in '{doc.title}'...",
        }

        completed = 0
        for i in range(0, total_missing, batch_size):
            batch_chunks = unembedded[i : i + batch_size]
            batch_texts = [
                f"{c.summary_text}\n\n{c.content}" if c.summary_text else c.content
                for c in batch_chunks
            ]
            try:
                embeddings = await llm_client.get_embeddings(
                    batch_texts,
                    model=emb_model,
                    dim=emb_dim,
                    title=doc.title,
                    is_query=False,
                )
                updates = [(c.id, emb) for c, emb in zip(batch_chunks, embeddings)]
                await storage.update_chunk_embeddings_batch(updates)
                completed += len(updates)
                pct = 5 + int(90 * (completed / total_missing))
                yield {
                    "stage": "resuming_progress",
                    "percent": pct,
                    "completed": completed,
                    "total": total_missing,
                    "message": f"Generated embeddings for {completed}/{total_missing} chunks...",
                }
            except Exception as e:
                logger.warning("Resume embeddings paused for doc '%s': %s", doc.title, e)
                yield {
                    "stage": "resuming_paused",
                    "percent": pct if "pct" in locals() else 50,
                    "completed": completed,
                    "total": total_missing,
                    "message": f"Embedding quota paused at {completed}/{total_missing} ({e}). You can resume again later.",
                }
                return

        yield {
            "stage": "complete",
            "percent": 100,
            "document_id": document_id,
            "title": doc.title,
            "embedded_count": completed,
            "total_missing": total_missing,
            "message": f"Successfully completed all embeddings for '{doc.title}'!",
        }
