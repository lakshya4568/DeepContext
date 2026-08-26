"""Two-pass grounded generation: extract supported facts, then write only from them."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE, is_anachronism
from deep_context.verification.checker import EvidenceVerifier


@dataclass
class GroundedAnswer:
    answer: str
    supported_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str | None = None
    support_passed: bool = True
    support_confidence: float = 1.0


def _parse_extract_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    clean, _ = llm_client._parse_think_tags(text)
    clean = clean.strip()

    # 1. Check if enclosed in markdown json fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            if isinstance(data, dict) and "supported" in data:
                return data
        except Exception:
            pass

    # 2. Direct JSON load
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "supported" in data:
            return data
    except Exception:
        pass

    # 3. Targeted JSON search containing "supported"
    json_match = re.search(r"\{[\s\S]*?\"supported\"[\s\S]*?\}", clean)
    if json_match:
        try:
            data = json.loads(json_match.group(0).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


def _get_chunk_content(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("content", ""))
    return str(getattr(c, "content", ""))


def _strip_unsupported(answer: str, evidence: list[Any]) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return answer
    blob = " ".join(_get_chunk_content(c) for c in evidence).lower()
    kept: list[str] = []
    for sentence in sentences:
        words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
        if not words:
            kept.append(sentence)
            continue
        overlap = sum(1 for w in words if w in blob) / len(words)
        if overlap >= 0.45 or any(
            marker in sentence.lower()
            for marker in ("not in the", "insufficient", "not mentioned", "missing")
        ):
            kept.append(sentence)
    return " ".join(kept).strip() or REFUSAL_TEMPLATE


async def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[Any],
    *,
    model: str | None = None,
    timeout: float = 8.0,
) -> GroundedAnswer:
    """Extract supported facts, write only from them, then verify grounding."""
    if is_anachronism(query):
        return GroundedAnswer(
            answer=REFUSAL_TEMPLATE,
            refused=True,
            reason="anachronism",
        )
    if not retrieved_chunks:
        return GroundedAnswer(
            answer=REFUSAL_TEMPLATE,
            refused=True,
            reason="no_evidence",
        )

    evidence_text = "\n\n".join(
        f"[{idx}] {_get_chunk_content(chunk)}" for idx, chunk in enumerate(retrieved_chunks, start=1)
    )
    target_model = model or settings.llm_model

    extract_messages = [
        {
            "role": "system",
            "content": (
                "Read the evidence and list facts that help answer the question. "
                "Return ONLY JSON with keys supported (list of strings), missing (list of strings), "
                "and unanswerable (boolean). Set unanswerable=true only if supported is empty. "
                "Do not invent names, numbers, or events."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nEvidence:\n{evidence_text[:12000]}",
        },
    ]
    try:
        extract_raw, _ = await llm_client.complete(
            extract_messages,
            model=target_model,
            temperature=0.0,
            enable_thinking=False,
            timeout=timeout,
            max_retries=1,
            max_tokens=512,
        )
    except Exception as exc:
        logger.warning("Fact extraction failed (%s); falling back to single-pass write.", exc)
        extract_raw = "{}"

    payload = _parse_extract_payload(extract_raw)
    supported = [str(item).strip() for item in payload.get("supported", []) if str(item).strip()]
    missing = [str(item).strip() for item in payload.get("missing", []) if str(item).strip()]
    unanswerable = bool(payload.get("unanswerable")) or not supported

    if unanswerable:
        # Check if question has evidence overlap before hard refusal to prevent false abstention
        q_keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
        overlap_count = sum(1 for w in q_keywords if w in evidence_text.lower())
        if overlap_count >= 2 and not is_anachronism(query):
            fallback_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI answering questions strictly from the provided context.\n"
                        "Answer all parts of the question that the context supports. If some facts are missing, state that they are not mentioned.\n"
                        'If the question cannot be answered from the context at all, reply EXACTLY:\n"Based on the provided context, there is insufficient evidence to answer."'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nEvidence:\n{evidence_text[:12000]}",
                },
            ]
            try:
                ans, _ = await llm_client.complete(
                    fallback_messages,
                    model=target_model,
                    temperature=0.1,
                    enable_thinking=False,
                    timeout=timeout,
                )
                if ans and REFUSAL_TEMPLATE not in ans:
                    answer = ans.strip()
                    check = await EvidenceVerifier.check_support(answer, retrieved_chunks)
                    return GroundedAnswer(
                        answer=answer,
                        supported_facts=[answer],
                        refused=False,
                        support_passed=check.passed,
                        support_confidence=check.confidence,
                    )
            except Exception as exc:
                logger.warning("Fallback generation failed (%s).", exc)

        return GroundedAnswer(
            answer=REFUSAL_TEMPLATE,
            supported_facts=supported,
            missing_facts=missing,
            refused=True,
            reason="unanswerable",
        )

    write_messages = [
        {
            "role": "system",
            "content": (
                "/no_think\n"
                "Write the final answer directly using ONLY the supported facts, without preambles, meta-commentary, or thinking headers. "
                "Cover every supported fact clearly. If some requested details are missing, state that they are not mentioned. "
                "Do not add claims, names, or numbers that are not in the supported list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\nSupported facts:\n- "
                + "\n- ".join(supported)
                + ("\n\nMissing from evidence:\n- " + "\n- ".join(missing) if missing else "")
            ),
        },
    ]
    try:
        answer, _ = await llm_client.complete(
            write_messages,
            model=target_model,
            temperature=0.1,
            enable_thinking=False,
            timeout=timeout,
            max_retries=1,
            max_tokens=700,
        )
    except Exception as exc:
        logger.warning("Grounded write failed (%s).", exc)
        answer = " ".join(supported)

    answer = (answer or "").strip() or " ".join(supported)
    check = await EvidenceVerifier.check_support(answer, retrieved_chunks)
    if not check.passed:
        answer = _strip_unsupported(answer, retrieved_chunks)
        check = await EvidenceVerifier.check_support(answer, retrieved_chunks)
        if not check.passed and supported:
            # Fallback to direct supported facts extracted in Pass 1
            fallback_answer = " ".join(supported)
            fallback_check = await EvidenceVerifier.check_support(fallback_answer, retrieved_chunks)
            if fallback_check.passed:
                return GroundedAnswer(
                    answer=fallback_answer,
                    supported_facts=supported,
                    missing_facts=missing,
                    refused=False,
                    support_passed=True,
                    support_confidence=fallback_check.confidence,
                )

        if not check.passed:
            return GroundedAnswer(
                answer=REFUSAL_TEMPLATE,
                supported_facts=supported,
                missing_facts=missing,
                refused=True,
                reason="failed_verification",
                support_passed=False,
                support_confidence=check.confidence,
            )

    return GroundedAnswer(
        answer=answer,
        supported_facts=supported,
        missing_facts=missing,
        refused=False,
        support_passed=check.passed,
        support_confidence=check.confidence,
    )
