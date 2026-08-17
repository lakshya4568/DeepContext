"""Evidence-sufficiency shared verification gate implementing ARCHITECTURE.md §7 and FR20."""

from __future__ import annotations

import re
from typing import Any

from deep_context.core.config import settings
from deep_context.core.types import (
    Claim,
    ClaimSupport,
    SupportCheckResult,
)


class EvidenceVerifier:
    """Verifies that generated claims are grounded in retrieved evidence or marked as inference."""

    @classmethod
    async def check_support(
        cls,
        draft_answer: str,
        evidence: list[dict[str, Any]],
        *,
        is_aggregation: bool = False,
        total_candidate_count: int | None = None,
    ) -> SupportCheckResult:
        if not draft_answer.strip():
            return SupportCheckResult(passed=False, failure_reasons=["Empty draft answer."])

        if not evidence:
            if any(
                w in draft_answer.lower()
                for w in (
                    "not enough evidence",
                    "insufficient evidence",
                    "could not find",
                    "cannot answer",
                )
            ):
                return SupportCheckResult(passed=True, confidence=1.0, failure_reasons=[])
            return SupportCheckResult(
                passed=False,
                confidence=0.0,
                failure_reasons=["No retrieved evidence provided to support factual claims."],
            )

        claims_text = [
            s.strip()
            for s in re.split(r"(?<=[.?!])\s+", draft_answer)
            if len(s.strip()) > 10 and not s.strip().startswith("#")
        ]
        if not claims_text:
            claims_text = [draft_answer.strip()]

        claims: list[Claim] = []
        for c_text in claims_text:
            claim = cls._link_claim_to_evidence(c_text, evidence)
            claims.append(claim)

        unsupported = [c for c in claims if c.support == ClaimSupport.UNSUPPORTED]
        failure_reasons: list[str] = []

        if unsupported:
            failure_reasons.append(
                f"{len(unsupported)}/{len(claims)} claim(s) lack traceable evidence support: "
                + "; ".join(c.text[:80] for c in unsupported[:2])
            )

        coverage_ratio = None
        if is_aggregation and total_candidate_count and total_candidate_count > 0:
            covered_evidence = len({c.evidence_id for c in claims if c.evidence_id})
            coverage_ratio = covered_evidence / total_candidate_count
            if coverage_ratio < settings.min_aggregation_coverage:
                failure_reasons.append(
                    f"Aggregation coverage {coverage_ratio:.1%} is below required {settings.min_aggregation_coverage:.1%}."
                )

        supported_count = sum(1 for c in claims if c.support != ClaimSupport.UNSUPPORTED)
        supported_fraction = supported_count / len(claims) if claims else 1.0

        confidence = supported_fraction
        if is_aggregation and coverage_ratio is not None:
            confidence = min(supported_fraction, coverage_ratio)

        if confidence < settings.confidence_threshold:
            failure_reasons.append(
                f"Grounding confidence {confidence:.2f} is below threshold {settings.confidence_threshold:.2f}."
            )

        return SupportCheckResult(
            passed=len(failure_reasons) == 0,
            claims=claims,
            coverage_ratio=coverage_ratio,
            confidence=confidence,
            failure_reasons=failure_reasons,
        )

    @classmethod
    def _link_claim_to_evidence(cls, claim_text: str, evidence: list[dict[str, Any]]) -> Claim:
        claim_lower = claim_text.lower()

        if any(
            marker in claim_lower
            for marker in (
                "i infer",
                "this suggests",
                "likely",
                "hypothesize",
                "presumably",
                "my interpretation",
            )
        ):
            return Claim(text=claim_text, support=ClaimSupport.INFERENCE)

        if any(
            marker in claim_lower
            for marker in ("calculated", "sum is", "total of", "computed", "count is")
        ):
            return Claim(text=claim_text, support=ClaimSupport.COMPUTED)

        words = [w for w in re.findall(r"\w+", claim_lower) if len(w) > 2]
        if not words:
            return Claim(text=claim_text, support=ClaimSupport.UNSUPPORTED)

        for item in evidence:
            content = item.get("content", "").lower()
            overlap_count = sum(1 for w in words if w in content)
            overlap_ratio = overlap_count / len(words)
            if overlap_ratio >= 0.50 or overlap_count >= 4:
                return Claim(
                    text=claim_text,
                    support=ClaimSupport.RETRIEVED,
                    evidence_id=item.get("chunk_id") or item.get("id"),
                )

        return Claim(text=claim_text, support=ClaimSupport.UNSUPPORTED)
