"""Tests for evidence-sufficiency shared verification gate."""

import pytest

from deep_context.verification.checker import EvidenceVerifier


@pytest.mark.asyncio
async def test_evidence_support_check_passed() -> None:
    evidence = [
        {
            "id": "chunk_1",
            "content": "FastAPI is an async web framework for building APIs with Python 3.8+.",
        }
    ]
    draft_answer = (
        "FastAPI is an async web framework used for building high performance APIs with Python."
    )
    result = await EvidenceVerifier.check_support(draft_answer, evidence)

    assert result.passed is True
    assert result.confidence >= 0.75


@pytest.mark.asyncio
async def test_evidence_support_check_unsupported_claim() -> None:
    evidence = [
        {
            "id": "chunk_1",
            "content": "Hybrid retrieval combines BM25 and vector search fused with RRF.",
        }
    ]
    draft_answer = (
        "Hybrid retrieval combines BM25 and vector search with RRF. "
        "The system cures all known human diseases and grants immortality."
    )
    result = await EvidenceVerifier.check_support(draft_answer, evidence)

    assert result.passed is False
    assert len(result.failure_reasons) >= 1
