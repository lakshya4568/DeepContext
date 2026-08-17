"""RLM session execution routes."""

from __future__ import annotations

import json

from fastapi import APIRouter

from deep_context.core.types import RlmSessionRequest, RlmSessionResponse
from deep_context.rlm.orchestrator import RLMOrchestrator
from deep_context.storage import get_storage

router = APIRouter(prefix="/v1/rlm", tags=["Recursive Language Model (RLM)"])


@router.post("/session", response_model=RlmSessionResponse)
async def create_rlm_session(req: RlmSessionRequest) -> RlmSessionResponse:
    storage = await get_storage()
    orchestrator = RLMOrchestrator(storage)

    # Determine corpus
    corpus = req.corpus_text
    if not corpus and req.corpus_document_ids:
        doc_texts = []
        for did in req.corpus_document_ids:
            doc = await storage.get_document(did)
            if doc:
                doc_texts.append(f"=== Document: {doc.title} ===\n{json.dumps(doc.metadata)}")
        corpus = "\n\n".join(doc_texts)

    if not corpus:
        corpus = "No initial corpus provided. Use retrieve() or host tools to inspect knowledge."

    return await orchestrator.run_session(
        task_spec=req.task_spec,
        corpus=corpus,
        user_id=req.user_id,
        max_turns=req.max_turns,
        max_recursion_depth=req.max_recursion_depth,
    )
