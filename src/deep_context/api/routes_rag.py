"""RAG Ingestion, Retrieval, Uploads, Haystack Benchmark, and Query routes."""

import asyncio
import json
import os
import random
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from deep_context.agentic.planner import AgenticPlanner
from deep_context.agentic.router import QueryRouter
from deep_context.cache import make_query_cache_payload, response_cache
from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import (
    HaystackBenchmarkRequest,
    HaystackBenchmarkResponse,
    HaystackGenerateRequest,
    IngestRequest,
    IngestResponse,
    Observation,
    QueryRequest,
    QueryResponse,
    QueryShape,
    RetrievalFilters,
    RetrievalMode,
    RetrieveRequest,
    RetrieveResponse,
    RoutingPath,
    StageDiagnostic,
    UserPreferenceRequest,
    UserPreferenceResponse,
)
from deep_context.ingestion.pipeline import ingestion_pipeline
from deep_context.ingestion.summary_pipeline import SummaryIngestionPipeline
from deep_context.memory.prompt_assembler import PromptAssembler
from deep_context.memory.stores import MemoryStoreManager
from deep_context.retrieval.engine import retrieval_engine
from deep_context.retrieval.hybrid import HybridRetriever
from deep_context.retrieval.reranker import Reranker
from deep_context.storage import get_storage
from deep_context.verification.checker import EvidenceVerifier

router = APIRouter(tags=["RAG & Query"])
summary_ingestion_pipeline = SummaryIngestionPipeline()

# UI HTML path
UI_HTML_PATH = Path(__file__).parent.parent / "ui" / "index.html"


@router.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """Serve the Deep Context Platform interactive web UI."""
    if UI_HTML_PATH.exists():
        content = UI_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    return HTMLResponse("<h1>Deep Context Platform UI loading...</h1>")


@router.get("/v1/documents")
async def list_documents(limit: int = 50) -> list[dict[str, Any]]:
    """List ingested documents and chunk stats."""
    storage = await get_storage()
    return await storage.list_document_summaries(limit=limit)


@router.get("/v1/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str) -> list[dict[str, Any]]:
    """Inspect all parent and child chunks with LLM summaries for a document."""
    storage = await get_storage()
    return await storage.get_document_chunks_detail(document_id)


@router.post("/v1/documents/{document_id}/embed")
async def resume_document_embeddings_api(
    document_id: str,
    embedding_model: str = "",
    embedding_dim: int = 0,
) -> dict[str, Any]:
    """Generate or resume missing dense embeddings for a document in progressive batches."""
    last_event = {}
    async for event in summary_ingestion_pipeline.resume_document_embeddings(
        document_id,
        embedding_model=embedding_model or None,
        embedding_dim=embedding_dim or None,
    ):
        last_event = event
    return last_event


@router.post("/v1/documents/{document_id}/embed-stream")
async def resume_document_embeddings_stream(
    document_id: str,
    embedding_model: str = "",
    embedding_dim: int = 0,
) -> StreamingResponse:
    """Stream live progress when generating or resuming embeddings for a document."""
    async def event_gen() -> AsyncIterator[str]:
        async for event in summary_ingestion_pipeline.resume_document_embeddings(
            document_id,
            embedding_model=embedding_model or None,
            embedding_dim=embedding_dim or None,
        ):
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.delete("/v1/documents/{document_id}")
async def delete_single_document(document_id: str) -> dict[str, Any]:
    """Delete a single document and all associated chunks and tree nodes."""
    storage = await get_storage()
    success = await storage.delete_document(document_id)
    invalidated = await response_cache.invalidate_namespace("rag")
    return {
        "status": "deleted",
        "document_id": document_id,
        "success": success,
        "cache_entries_invalidated": invalidated,
    }


@router.delete("/v1/documents")
async def delete_all_documents() -> dict[str, Any]:
    """Clear all ingested documents and chunks from the knowledge base."""
    storage = await get_storage()
    deleted_count = await storage.delete_all_documents()
    invalidated = await response_cache.invalidate_namespace("rag")
    return {
        "status": "cleared",
        "deleted_documents_count": deleted_count,
        "cache_entries_invalidated": invalidated,
    }


@router.post("/v1/models/unload")
async def unload_models() -> dict[str, Any]:
    """Explicitly unload Qwen3 and local neural models to free GPU/RAM memory."""
    summary_ingestion_pipeline.summarizer.unload()
    ingestion_pipeline.summarizer.unload()
    return {"status": "unloaded", "message": "All local neural model weights freed from GPU/RAM."}


@router.post("/v1/upload", response_model=IngestResponse)
async def upload_file(
    file: UploadFile = File(...),
    title: str = Form(""),
    doc_type: str = Form("auto"),
    embedding_model: str = Form(""),
    embedding_dim: int = Form(0),
    generate_summaries: bool | None = Form(None),
) -> IngestResponse:
    """Upload and ingest a file (PDF up to 1000 pages, Markdown, Code, TXT)."""
    filename = file.filename or "uploaded_doc"
    file_bytes = await file.read()

    detected_type = doc_type
    if detected_type == "auto":
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            detected_type = "pdf"
        elif ext in (".md", ".markdown"):
            detected_type = "markdown"
        elif ext in (".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".rs"):
            detected_type = "code"
        else:
            detected_type = "text"

    doc_title = title.strip() or Path(filename).stem
    mode = RetrievalMode.HYBRID
    target_model = embedding_model or settings.embedding_model
    target_dim = embedding_dim or (
        768 if "gemini" in target_model.lower() else settings.embedding_dim
    )

    # Save to temp file if PDF for streaming pypdf extraction
    if detected_type == "pdf":
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            req = IngestRequest(
                title=doc_title,
                content=tmp_path,
                doc_type="pdf",
                source_uri=filename,
                retrieval_mode=mode,
                embedding_model=target_model,
                embedding_dim=target_dim,
                generate_summaries=generate_summaries,
                metadata={"filename": filename, "file_size": len(file_bytes)},
            )
            res = await ingestion_pipeline.ingest(req)
            return res
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        text_content = file_bytes.decode("utf-8", errors="replace")
        req = IngestRequest(
            title=doc_title,
            content=text_content,
            doc_type=detected_type,
            source_uri=filename,
            retrieval_mode=mode,
            embedding_model=target_model,
            embedding_dim=target_dim,
            generate_summaries=generate_summaries,
            metadata={"filename": filename, "file_size": len(file_bytes)},
        )
        return await ingestion_pipeline.ingest(req)


@router.post("/v1/upload-stream")
async def upload_stream(
    file: UploadFile = File(...),
    title: str = Form(""),
    doc_type: str = Form("auto"),
    embedding_model: str = Form(""),
    embedding_dim: int = Form(0),
    generate_summaries: bool | None = Form(None),
) -> StreamingResponse:
    """Stream live real-time ingestion progress and Qwen3 summaries (SSE events)."""
    filename = file.filename or "uploaded_doc"
    file_bytes = await file.read()

    detected_type = doc_type
    if detected_type == "auto":
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            detected_type = "pdf"
        elif ext in (".md", ".markdown"):
            detected_type = "markdown"
        elif ext in (".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".rs"):
            detected_type = "code"
        else:
            detected_type = "text"

    doc_title = title.strip() or Path(filename).stem
    mode = RetrievalMode.HYBRID
    target_model = embedding_model or settings.embedding_model
    target_dim = embedding_dim or (
        768 if "gemini" in target_model.lower() else settings.embedding_dim
    )

    async def event_generator() -> AsyncIterator[str]:
        import tempfile

        tmp_path = None
        try:
            if detected_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name
                content = tmp_path
            else:
                content = file_bytes.decode("utf-8", errors="replace")

            req = IngestRequest(
                title=doc_title,
                content=content,
                doc_type=detected_type,
                source_uri=filename,
                retrieval_mode=mode,
                embedding_model=target_model,
                embedding_dim=target_dim,
                generate_summaries=generate_summaries,
                metadata={"filename": filename, "file_size": len(file_bytes)},
            )

            async for event in summary_ingestion_pipeline.ingest_stream(req):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.01)
        except Exception as err:
            logger.exception("Upload streaming error: %s", err)
            yield f"data: {json.dumps({'stage': 'error', 'message': str(err)})}\n\n"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/v1/upload-batch", response_model=list[IngestResponse])
async def upload_batch_files(
    files: list[UploadFile] = File(...),
    embedding_model: str = Form(""),
    embedding_dim: int = Form(0),
    generate_summaries: bool | None = Form(None),
) -> list[IngestResponse]:
    """Upload and ingest multiple files at once (PDFs, TXT, MD, Code)."""
    results: list[IngestResponse] = []
    target_model = embedding_model or settings.embedding_model
    target_dim = embedding_dim or (
        768 if "gemini" in target_model.lower() else settings.embedding_dim
    )
    mode = RetrievalMode.HYBRID

    for f in files:
        filename = f.filename or "uploaded_doc"
        file_bytes = await f.read()

        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            dtype = "pdf"
        elif ext in (".md", ".markdown"):
            dtype = "markdown"
        elif ext in (".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".rs"):
            dtype = "code"
        else:
            dtype = "text"

        if dtype == "pdf":
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            try:
                req = IngestRequest(
                    title=Path(filename).stem,
                    content=tmp_path,
                    doc_type="pdf",
                    source_uri=filename,
                    retrieval_mode=mode,
                    embedding_model=target_model,
                    embedding_dim=target_dim,
                    generate_summaries=generate_summaries,
                    metadata={"filename": filename, "file_size": len(file_bytes)},
                )
                res = await ingestion_pipeline.ingest(req)
                results.append(res)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            text_content = file_bytes.decode("utf-8", errors="replace")
            req = IngestRequest(
                title=Path(filename).stem,
                content=text_content,
                doc_type=dtype,
                source_uri=filename,
                retrieval_mode=mode,
                embedding_model=target_model,
                embedding_dim=target_dim,
                generate_summaries=generate_summaries,
                metadata={"filename": filename, "file_size": len(file_bytes)},
            )
            res = await ingestion_pipeline.ingest(req)
            results.append(res)

    return results


@router.get("/v1/preferences", response_model=UserPreferenceResponse)
async def get_user_preferences_api(user_id: str = "default") -> UserPreferenceResponse:
    """Fetch stored user preferences for embedding models, dimensions, and rerankers."""
    storage = await get_storage()
    mgr = MemoryStoreManager(storage)
    prefs = await mgr.get_embedding_preferences(user_id=user_id)
    return UserPreferenceResponse(
        user_id=user_id,
        embedding_model=prefs.get("embedding_model", settings.embedding_model),
        embedding_dim=prefs.get("embedding_dim", settings.embedding_dim),
        reranker=prefs.get("reranker", settings.reranker_strategy),
        llm_model=prefs.get("llm_model", settings.llm_model),
        preferences=prefs,
    )


@router.post("/v1/preferences", response_model=UserPreferenceResponse)
async def set_user_preferences_api(
    req: UserPreferenceRequest,
) -> UserPreferenceResponse:
    """Persist user embedding and reranker preferences to durable memory."""
    storage = await get_storage()
    mgr = MemoryStoreManager(storage)
    await mgr.set_embedding_preferences(
        user_id=req.user_id,
        embedding_model=req.embedding_model,
        embedding_dim=req.embedding_dim,
        reranker=req.reranker,
        llm_model=req.llm_model,
    )
    prefs = await mgr.get_embedding_preferences(user_id=req.user_id)
    return UserPreferenceResponse(
        user_id=req.user_id,
        embedding_model=prefs.get("embedding_model", settings.embedding_model),
        embedding_dim=prefs.get("embedding_dim", settings.embedding_dim),
        reranker=prefs.get("reranker", settings.reranker_strategy),
        llm_model=prefs.get("llm_model", settings.llm_model),
        preferences=prefs,
    )


@router.post("/v1/sync-folder", response_model=list[IngestResponse])
async def sync_local_folder(
    folder_path: str = "documents",
    embedding_model: str = "",
    embedding_dim: int = 0,
) -> list[IngestResponse]:
    """Scan and ingest all supported files from a local directory (e.g. './documents')."""
    target_dir = Path(folder_path)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        return []

    supported = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".text": "text",
        ".log": "text",
        ".csv": "text",
        ".json": "text",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".java": "code",
        ".go": "code",
    }
    ignored = {".venv", ".git", "__pycache__", "dist", "build"}

    results: list[IngestResponse] = []
    mode = RetrievalMode.HYBRID
    target_model = embedding_model or settings.embedding_model
    target_dim = embedding_dim or (
        768 if "gemini" in target_model.lower() else settings.embedding_dim
    )

    for p in target_dir.rglob("*"):
        if p.is_file() and not any(part in ignored for part in p.parts):
            ext = p.suffix.lower()
            if ext in supported:
                dtype = supported[ext]
                content = (
                    str(p.resolve())
                    if dtype == "pdf"
                    else p.read_text(encoding="utf-8", errors="replace")
                )
                req = IngestRequest(
                    title=p.stem,
                    content=content,
                    doc_type=dtype,
                    source_uri=str(p.resolve()),
                    retrieval_mode=mode,
                    embedding_model=target_model,
                    embedding_dim=target_dim,
                )
                try:
                    res = await ingestion_pipeline.ingest(req)
                    results.append(res)
                except Exception as e:
                    logger.warning("Failed to sync file %s: %s", p, e)

    return results


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest) -> IngestResponse:
    """Ingests a document, parses sections, creates parent-child chunks & embeddings."""
    return await ingestion_pipeline.ingest(req)


@router.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve_knowledge(req: RetrieveRequest) -> RetrieveResponse:
    """Direct hybrid retrieval with RRF, reranking, parent resolution, and citations."""
    cache_payload = make_query_cache_payload(req)
    cached = await response_cache.get_json("rag:retrieve", cache_payload)
    if cached is not None:
        return RetrieveResponse(**cached, cache_hit=True)

    filters = RetrievalFilters(
        tenant_id=req.tenant_id,
        permission_scope=req.permission_scope,
        document_ids=req.document_ids,
    )
    res = await retrieval_engine.retrieve(
        query=req.query,
        filters=filters,
        top_k=req.top_k,
        embedding_model=req.embedding_model,
        embedding_dim=req.embedding_dim,
        reranker=req.reranker,
        user_id=req.user_id,
    )
    response = RetrieveResponse(
        sufficient=res.sufficient,
        parent_chunks=res.parent_chunks,
        citations=[c.to_dict() for c in res.citations],
        query_shape=res.query_shape or QueryShape.FACTUAL_LOOKUP,
        retry_count=res.retry_count,
        insufficiency_reason=res.insufficiency_reason,
        cache_hit=False,
    )
    if res.sufficient:
        await response_cache.set_json("rag:retrieve", cache_payload, response.model_dump())
    return response


@router.post("/v1/query", response_model=QueryResponse)
async def query_platform(req: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    """End-to-end intelligent query answering with GLM-5.2 reasoning."""
    t0 = time.time()

    # Whole-answer cache: repeated identical questions skip the full pipeline.
    if not req.stream:
        cache_payload = make_query_cache_payload(req)
        cached = await response_cache.get_json("rag:ask", cache_payload)
        if cached is not None:
            return QueryResponse(**cached, cache_hit=True)

    storage = await get_storage()
    filters = RetrievalFilters(
        tenant_id=req.tenant_id,
        permission_scope=req.permission_scope,
        document_ids=req.document_ids,
    )

    decision = await QueryRouter.route(
        query=req.query,
        forced_path=req.force_path,
    )

    answer_text = ""
    citations_list: list[dict[str, Any]] = []
    reasoning_text: str | None = None
    support_passed = True
    support_confidence = 1.0

    if decision.path == RoutingPath.HYBRID_RAG:
        retrieval_res = await retrieval_engine.retrieve(
            query=req.query,
            filters=filters,
            top_k=6,
            embedding_model=req.embedding_model,
            embedding_dim=req.embedding_dim,
            reranker=req.reranker,
            user_id=req.user_id,
        )
        citations_list = [c.to_dict() for c in retrieval_res.citations]

        from deep_context.generation.grounded_answer import generate_grounded_answer

        grounded_res = await generate_grounded_answer(
            query=req.query,
            retrieved_chunks=retrieval_res.parent_chunks,
            model=req.model,
        )
        answer_text = grounded_res.answer
        reasoning_text = grounded_res.reason
        support_passed = grounded_res.support_passed
        support_confidence = grounded_res.support_confidence

    elif decision.path == RoutingPath.AGENTIC_PLANNER:
        planner = AgenticPlanner(storage)
        answer_text, citations_list, reasoning_text = await planner.execute_plan(
            query=req.query, filters=filters
        )

    latency_ms = int((time.time() - t0) * 1000)

    response = QueryResponse(
        answer=answer_text,
        citations=citations_list,
        path_taken=decision.path,
        query_shape=decision.query_shape,
        reasoning=reasoning_text,
        support_check_passed=support_passed,
        support_confidence=support_confidence,
        latency_ms=latency_ms,
        token_cost=0,
        cache_hit=False,
    )

    # Only cache grounded, support-checked answers to avoid poisoning the
    # cache with failed verifications or empty corpora.
    if not req.stream and answer_text and support_passed:
        await response_cache.set_json("rag:ask", cache_payload, response.model_dump())

    if req.user_id and len(req.query) > 10:
        background_tasks.add_task(
            _background_memory_extract,
            storage,
            req.query,
            req.tenant_id,
            req.user_id,
        )

    return response


@router.post("/v1/query/stream")
async def query_platform_stream(
    req: QueryRequest, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """Streams real-time thinking tokens, status updates, answer tokens, and citations using Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncIterator[str]:
        t0 = time.time()
        storage = await get_storage()
        filters = RetrievalFilters(
            tenant_id=req.tenant_id,
            permission_scope=req.permission_scope,
            document_ids=req.document_ids,
        )

        try:
            # 1. Routing Phase
            yield f"data: {json.dumps({'type': 'status', 'stage': 'routing', 'message': '🔍 Classifying query and routing execution path...'})}\n\n"
            decision = await QueryRouter.route(
                query=req.query,
                forced_path=req.force_path,
            )
            yield f"data: {json.dumps({'type': 'status', 'stage': 'routed', 'path': decision.path.value, 'query_shape': decision.query_shape.value, 'message': f'Path selected: {decision.path.value} ({decision.query_shape.value})'})}\n\n"

            citations_list: list[dict[str, Any]] = []
            accumulated_content: list[str] = []
            accumulated_reasoning: list[str] = []
            support_passed = True
            support_confidence = 1.0

            if decision.path == RoutingPath.HYBRID_RAG:
                # 2. Hybrid Retrieval Phase
                yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieval', 'message': '📚 Running BM25 + Dense Vector hybrid search & RRF ranking...'})}\n\n"
                retrieval_res = await retrieval_engine.retrieve(
                    query=req.query,
                    filters=filters,
                    top_k=6,
                    embedding_model=req.embedding_model,
                    embedding_dim=req.embedding_dim,
                    reranker=req.reranker,
                    user_id=req.user_id,
                )
                citations_list = [c.to_dict() for c in retrieval_res.citations]
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations_list})}\n\n"

                # 3. Prompt Assembly Phase
                assembler = PromptAssembler(storage)
                messages = await assembler.assemble_messages(
                    query=req.query,
                    retrieved_chunks=retrieval_res.parent_chunks,
                    tenant_id=req.tenant_id,
                    user_id=req.user_id,
                )

                # 4. Real-Time Generation & Thinking Stream
                yield f"data: {json.dumps({'type': 'status', 'stage': 'generating', 'message': '🧠 Generating grounded answer with live reasoning...'})}\n\n"
                async for token_chunk in llm_client.stream_complete(
                    messages, model=req.model, temperature=0.5, enable_thinking=True
                ):
                    chunk_type = token_chunk.get("type", "content")
                    if chunk_type == "rate_limit":
                        yield f"data: {json.dumps(token_chunk)}\n\n"
                        continue
                    chunk_text = token_chunk.get("text", "")
                    if chunk_type == "reasoning":
                        accumulated_reasoning.append(chunk_text)
                        yield f"data: {json.dumps({'type': 'reasoning', 'delta': chunk_text})}\n\n"
                    else:
                        accumulated_content.append(chunk_text)
                        yield f"data: {json.dumps({'type': 'content', 'delta': chunk_text})}\n\n"

                # 5. Verification Phase
                draft_answer = "".join(accumulated_content)
                support_res = await EvidenceVerifier.check_support(
                    draft_answer=draft_answer,
                    evidence=retrieval_res.parent_chunks,
                    is_aggregation=(decision.query_shape.value == "aggregation"),
                )
                support_passed = support_res.passed
                support_confidence = support_res.confidence

            elif decision.path == RoutingPath.AGENTIC_PLANNER:
                planner = AgenticPlanner(storage)
                async for ev in planner.execute_plan_stream(
                    query=req.query, filters=filters, model=req.model
                ):
                    ev_type = ev.get("type")
                    if ev_type == "status":
                        yield f"data: {json.dumps(ev)}\n\n"
                    elif ev_type == "citations":
                        citations_list = ev.get("citations", [])
                        yield f"data: {json.dumps(ev)}\n\n"
                    elif ev_type == "reasoning":
                        accumulated_reasoning.append(ev.get("delta", ""))
                        yield f"data: {json.dumps(ev)}\n\n"
                    elif ev_type == "content":
                        accumulated_content.append(ev.get("delta", ""))
                        yield f"data: {json.dumps(ev)}\n\n"

            latency_ms = int((time.time() - t0) * 1000)

            # 6. Final Done Event
            yield f"data: {json.dumps({'type': 'done', 'path_taken': decision.path.value, 'query_shape': decision.query_shape.value, 'support_check_passed': support_passed, 'support_confidence': support_confidence, 'latency_ms': latency_ms})}\n\n"

            # Background memory extraction if applicable
            if req.user_id and len(req.query) > 10:
                background_tasks.add_task(
                    _background_memory_extract,
                    storage,
                    req.query,
                    req.tenant_id,
                    req.user_id,
                )

        except Exception as e:
            logger.exception("Error in query_platform_stream: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/v1/quota/status")
async def get_quota_status() -> dict[str, Any]:
    """Return live rate limit and quota status for Groq and NVIDIA NIM."""
    from deep_context.core.config import settings
    from deep_context.core.llm_client import LLMClient

    rate_limit = LLMClient.global_rate_limit or llm_client.last_rate_limit
    has_groq = settings.has_groq_key
    has_nvidia = settings.has_nvidia_key

    is_active_limited = False
    if rate_limit and (time.time() - float(rate_limit.get("timestamp", 0)) < 3600):
        is_active_limited = True

    return {
        "groq": {
            "configured": has_groq,
            "is_rate_limited": is_active_limited,
            "model": (
                rate_limit.get("model", settings.llm_model) if rate_limit else settings.llm_model
            ),
            "message": (rate_limit.get("message") if rate_limit else "Normal (Within daily quota)"),
            "retry_after": rate_limit.get("retry_after") if rate_limit else None,
            "limit": rate_limit.get("limit") if rate_limit else "200000",
            "used": rate_limit.get("used") if rate_limit else "0",
            "quota_type": (rate_limit.get("quota_type") if rate_limit else "tokens per day (TPD)"),
            "last_error_at": rate_limit.get("timestamp") if rate_limit else None,
        },
        "nvidia_nim": {
            "configured": has_nvidia,
            "model": settings.embedding_model,
        },
    }


@router.post("/v1/quota/reset")
async def reset_quota_status() -> dict[str, Any]:
    """Reset rate limit status and reload API keys dynamically from environment/.env."""
    from deep_context.core.llm_client import LLMClient, llm_client

    LLMClient.clear_rate_limits()
    llm_client.last_rate_limit = None
    llm_client._refresh_groq_client()
    return {
        "status": "ok",
        "message": "Rate limit cache cleared and Groq API client reloaded.",
    }


# ---------------------------------------------------------------------------
# Needle In A Haystack Diagnostic Benchmark
# ---------------------------------------------------------------------------

FILLER_PARAGRAPHS = [
    "Distributed computing systems rely on consensus protocols such as Raft and Paxos to maintain consistent replicated state machines across clusters. Nodes communicate state changes through append entries RPCs.",
    "Database indexing using B+ trees provides logarithmic search, insert, and delete operations. Leaf nodes form a doubly linked list allowing efficient sequential range queries over sorted keys.",
    "Cache replacement policies such as Least Recently Used (LRU) and Adaptive Replacement Cache (ARC) dynamically balance recency and frequency to optimize hit ratios in memory constrained environments.",
    "Vector embeddings map high dimensional semantic information into continuous latent spaces where cosine similarity quantifies directional alignment between query vectors and document representations.",
    "Network flow control mechanisms including TCP congestion avoidance use additive-increase multiplicative-decrease (AIMD) algorithms to saturate bandwidth without triggering bufferbloat.",
    "Compiler optimization passes perform dead code elimination, constant propagation, and loop unrolling in intermediate representations before generating optimized machine instructions.",
    "Message queue architectures decouple producer services from consumer workers, providing backpressure buffering, dead letter queues, and idempotent message delivery guarantees.",
    "Microservice telemetry pipelines aggregate structured distributed traces, structured logs, and Prometheus time-series metrics to enable real-time observability across services.",
]


@router.post("/v1/haystack/generate", response_model=IngestResponse)
async def generate_haystack(req: HaystackGenerateRequest) -> IngestResponse:
    """
    Generates a synthetic haystack document of N words with a targeted needle
    inserted at a precise depth percentage (0% = top, 50% = middle, 100% = bottom).
    """
    words_per_para = 35
    total_paras = max(10, req.total_words // words_per_para)
    needle_para_idx = int(total_paras * (req.depth_percent / 100.0))
    needle_para_idx = max(0, min(total_paras - 1, needle_para_idx))

    sections_text: list[str] = [f"# {req.topic} — Benchmark Corpus ({req.total_words} words)\n"]

    for i in range(total_paras):
        sec_num = (i // 10) + 1
        if i % 10 == 0:
            sections_text.append(f"\n## Chapter {sec_num}: System Architecture Analysis\n")

        p = random.choice(FILLER_PARAGRAPHS)
        if i == needle_para_idx:
            # Insert the needle prominently within this paragraph
            query_context = f" Regarding '{req.needle_query}': " if req.needle_query else " "
            p = f"Special Security Notice:{query_context}{req.needle}. All authorized personnel must reference this credential. {p}"

        sections_text.append(p + "\n")

    full_text = "\n".join(sections_text)
    doc_title = f"Haystack Corpus ({req.total_words} words @ {req.depth_percent:.0f}% depth)"

    ingest_req = IngestRequest(
        title=doc_title,
        content=full_text,
        doc_type="markdown",
        source_uri=f"synthetic://haystack/{int(time.time())}",
        retrieval_mode=RetrievalMode.HYBRID,
        metadata={
            "needle": req.needle,
            "depth_percent": req.depth_percent,
            "total_words": req.total_words,
            "topic": req.topic,
        },
    )
    return await ingestion_pipeline.ingest(ingest_req)


@router.post("/v1/haystack/benchmark", response_model=HaystackBenchmarkResponse)
async def benchmark_haystack(
    req: HaystackBenchmarkRequest,
) -> HaystackBenchmarkResponse:
    """
    Diagnostic tracer ("What it uses to get the context"):
    Traces the needle step-by-step across:
    Stage 1: BM25 FTS5 Recall
    Stage 2: BGE-M3 Dense Vector Recall
    Stage 3: Reciprocal Rank Fusion (k=60)
    Stage 4: Cross-Encoder Reranker
    Stage 5: Parent Chunk Resolution
    Stage 6: Grounded Answer Generation
    """
    t0 = time.time()
    storage = await get_storage()

    # Find target document
    doc_id = req.document_id
    doc_title = "Repository Documents"
    if doc_id:
        doc = await storage.get_document(doc_id)
        if doc:
            doc_title = doc.title

    filters = RetrievalFilters(
        tenant_id="default",
        document_ids=[doc_id] if doc_id else None,
    )

    stages: list[StageDiagnostic] = []

    def _contains_needle(content: str, target: str) -> bool:
        if not content or not target:
            return False
        c_norm = re.sub(r"\\[_*\-]", lambda m: m.group(0)[1], content.lower())
        t_norm = re.sub(r"\\[_*\-]", lambda m: m.group(0)[1], target.lower())
        return t_norm in c_norm or target.lower() in content.lower()

    # Stage 1: BM25 FTS5
    bm25_results = await storage.search_bm25(query=req.query, filters=filters, limit=100)
    bm25_found = False
    bm25_rank = None
    bm25_score = None
    for idx, r in enumerate(bm25_results, start=1):
        if _contains_needle(r["content"], req.needle):
            bm25_found = True
            bm25_rank = idx
            bm25_score = r.get("score")
            break

    stages.append(
        StageDiagnostic(
            stage_name="Stage 1: BM25 Full-Text Search (PostgreSQL TSVector / Lexical)",
            needle_found=bm25_found,
            needle_rank=bm25_rank,
            score=round(bm25_score, 4) if bm25_score is not None else None,
            details=(
                f"Retrieved {len(bm25_results)} candidate chunks via keyword inverted index. "
                + (
                    f"Needle found at BM25 rank #{bm25_rank} (score: {bm25_score:.3f})."
                    if bm25_found
                    else "Needle not in top BM25 candidates."
                )
            ),
        )
    )

    # Stage 2: Dense Vector Search (Google Gemini / NVIDIA NIM)
    active_emb = settings.embedding_model
    active_dim = 768 if "gemini" in active_emb.lower() else settings.embedding_dim
    q_emb = await llm_client.get_embedding(
        req.query, model=active_emb, dim=active_dim, is_query=True
    )
    vec_results = await storage.search_vector(query_embedding=q_emb, filters=filters, limit=100)
    vec_found = False
    vec_rank = None
    vec_sim = None
    for idx, r in enumerate(vec_results, start=1):
        if _contains_needle(r["content"], req.needle):
            vec_found = True
            vec_rank = idx
            vec_sim = r.get("score")
            break

    stages.append(
        StageDiagnostic(
            stage_name=f"Stage 2: Dense Vector Search ({active_emb} {active_dim}-dim Cosine Sim)",
            needle_found=vec_found,
            needle_rank=vec_rank,
            score=round(vec_sim, 4) if vec_sim is not None else None,
            details=(
                f"Scanned child vectors using {active_dim}-dim {active_emb} embeddings. "
                + (
                    f"Needle found at Vector rank #{vec_rank} (cosine similarity: {vec_sim:.4f})."
                    if vec_found
                    else "Needle not in top Vector candidates."
                )
            ),
        )
    )

    # Stage 3: Reciprocal Rank Fusion (RRF k=60)
    hybrid_retriever = HybridRetriever(storage)
    rrf_candidates = await hybrid_retriever.retrieve_candidates(
        sub_queries=[req.query],
        filters=filters,
        limit=100,
        embedding_model=active_emb,
        embedding_dim=active_dim,
    )
    rrf_found = False
    rrf_rank = None
    rrf_score = None
    for idx, r in enumerate(rrf_candidates, start=1):
        if _contains_needle(r["content"], req.needle):
            rrf_found = True
            rrf_rank = idx
            rrf_score = r.get("score")
            break

    stages.append(
        StageDiagnostic(
            stage_name="Stage 3: Reciprocal Rank Fusion (RRF k=60)",
            needle_found=rrf_found,
            needle_rank=rrf_rank,
            score=round(rrf_score, 4) if rrf_score is not None else None,
            details=(
                "Fused BM25 and Dense vector rankings using 1/(60 + rank). "
                + (
                    f"Needle fused to position #{rrf_rank} (RRF score: {rrf_score:.5f})."
                    if rrf_found
                    else "Needle dropped during RRF merge."
                )
            ),
        )
    )

    # Stage 4: Multi-Strategy Precision Reranking
    active_rerank = settings.reranker_strategy
    reranked = await Reranker.rerank(
        query=req.query,
        candidates=rrf_candidates,
        top_k=req.top_k,
        strategy=active_rerank,
        embedding_model=active_emb,
        embedding_dim=active_dim,
    )
    rerank_found = False
    rerank_rank = None
    rerank_score = None
    for idx, r in enumerate(reranked, start=1):
        if _contains_needle(r["content"], req.needle):
            rerank_found = True
            rerank_rank = idx
            rerank_score = r.get("rerank_score") or r.get("score")
            break

    stages.append(
        StageDiagnostic(
            stage_name=f"Stage 4: Precision Reranker ({active_rerank})",
            needle_found=rerank_found,
            needle_rank=rerank_rank,
            score=round(rerank_score, 4) if rerank_score is not None else None,
            details=(
                f"Reranked top {len(rrf_candidates)} chunks using strategy '{active_rerank}' to final top-{req.top_k}. "
                + (
                    f"Needle confirmed in final set at rank #{rerank_rank} (relevance score: {rerank_score:.3f})."
                    if rerank_found
                    else "Needle did not make top-k cutoff."
                )
            ),
        )
    )

    # Stage 5: Child -> Parent Resolution
    retrieval_res = await retrieval_engine.retrieve(
        query=req.query, filters=filters, top_k=req.top_k
    )
    parent_found = False
    matched_parent = None
    for p in retrieval_res.parent_chunks:
        if _contains_needle(p["content"], req.needle):
            parent_found = True
            matched_parent = p
            break

    stages.append(
        StageDiagnostic(
            stage_name="Stage 5: Hierarchical Parent Chunk Resolution (1000-2500 tokens)",
            needle_found=parent_found,
            details=(
                f"Expanded child chunk to full parent context ({len(matched_parent['content'])} chars, Section: {matched_parent.get('section_path', 'N/A')}, Page: {matched_parent.get('page_number', 1)})."
                if (parent_found and matched_parent is not None)
                else "Parent chunk resolution did not contain needle."
            ),
        )
    )

    # Stage 6: Generation & Verification
    assembler = PromptAssembler(storage)
    focused_chunks = [matched_parent] if matched_parent else retrieval_res.parent_chunks[:2]
    messages = await assembler.assemble_messages(
        query=req.query,
        retrieved_chunks=focused_chunks,
    )
    answer, reasoning = await llm_client.complete(messages, temperature=0.3, enable_thinking=True)

    passed = _contains_needle(answer, req.needle) or (
        matched_parent is not None and _contains_needle(matched_parent["content"], req.needle)
    )

    latency_ms = int((time.time() - t0) * 1000)

    # Count total chunks in document
    total_child = 0
    total_parent = 0
    if doc_id:
        total_child, total_parent = await storage.count_chunks_for_document(doc_id)

    return HaystackBenchmarkResponse(
        document_id=doc_id or "all_documents",
        document_title=doc_title,
        total_parent_chunks=total_parent,
        total_child_chunks=total_child,
        query=req.query,
        needle=req.needle,
        stages=stages,
        retrieved_parent_chunk=matched_parent,
        passed=passed,
        answer=answer,
        reasoning=reasoning,
        latency_ms=latency_ms,
    )


async def _background_memory_extract(
    storage: Any, query: str, tenant_id: str, user_id: str
) -> None:
    try:
        manager = MemoryStoreManager(storage)
        obs = Observation(
            raw_text=query,
            tenant_id=tenant_id,
            user_id=user_id,
            source="user_stated",
        )
        await manager.observe_and_promote(obs)
    except Exception:
        pass
