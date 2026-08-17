"""
Reference implementation of the evidence-sufficiency gate described in SKILL.md and
docs/ARCHITECTURE.md §7. Shared by skills/rag-retrieval/ and skills/rlm-orchestrator/ --
this is the ONE gate both paths terminate at before an answer is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClaimSupport(str, Enum):
    RETRIEVED = "retrieved"  # traceable to a specific evidence chunk/citation
    COMPUTED = "computed"  # derived from a tool call / calculation, not retrieval
    INFERENCE = "inference"  # explicitly the model's own reasoning, flagged as such
    UNSUPPORTED = "unsupported"  # none of the above -- this is what fails the gate


@dataclass
class Claim:
    text: str
    support: ClaimSupport
    evidence_id: str | None = None  # chunk_id / citation, when support == RETRIEVED


@dataclass
class SupportCheckResult:
    passed: bool
    claims: list[Claim] = field(default_factory=list)
    coverage_ratio: float | None = None  # only meaningful for aggregation-shaped queries
    confidence: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)


CONFIDENCE_THRESHOLD = 0.75
MIN_AGGREGATION_COVERAGE = 0.95  # aggregation queries must cover ~all evidence, not a sample


def extract_claims(draft_answer: str) -> list[str]:
    """TODO: replace with a real LLM call that splits the draft answer into discrete,
    checkable claims (roughly one per sentence that asserts something, skipping filler).
    Naive sentence split as a runnable stub:"""
    return [s.strip() for s in draft_answer.split(".") if s.strip()]


def link_claim_to_evidence(claim_text: str, evidence: list[dict]) -> Claim:
    """TODO: replace with a real entailment/grounding check -- does any evidence chunk
    actually support this claim? (e.g. an NLI model, or an LLM-as-judge call scoped to
    the retrieved evidence only). This stub does a naive substring/keyword check, which is
    NOT adequate for production -- it exists only so the pipeline is runnable end-to-end."""
    claim_lower = claim_text.lower()
    for item in evidence:
        content = item.get("content", "").lower()
        overlap_words = set(claim_lower.split()) & set(content.split())
        if len(overlap_words) >= max(3, len(claim_lower.split()) // 3):
            return Claim(
                text=claim_text, support=ClaimSupport.RETRIEVED, evidence_id=item.get("id")
            )
    return Claim(text=claim_text, support=ClaimSupport.UNSUPPORTED)


def check_answer_support(
    draft_answer: str,
    evidence: list[dict],
    *,
    is_aggregation_query: bool = False,
    total_candidate_count: int | None = None,
) -> SupportCheckResult:
    """
    Implements the three checks from ARCHITECTURE.md §7:
      1. Every non-trivial claim traces to evidence, a computed value, or flagged inference.
      2. Aggregation queries: evidence must span the full retrieved/loaded set.
      3. Verifier confidence clears CONFIDENCE_THRESHOLD.
    """
    claim_texts = extract_claims(draft_answer)
    claims = [link_claim_to_evidence(text, evidence) for text in claim_texts]

    unsupported = [c for c in claims if c.support == ClaimSupport.UNSUPPORTED]
    failure_reasons: list[str] = []

    if unsupported:
        failure_reasons.append(
            f"{len(unsupported)}/{len(claims)} claims have no traceable support: "
            + "; ".join(c.text[:80] for c in unsupported[:3])
        )

    coverage_ratio = None
    if is_aggregation_query and total_candidate_count:
        covered = len({c.evidence_id for c in claims if c.evidence_id})
        coverage_ratio = covered / total_candidate_count
        if coverage_ratio < MIN_AGGREGATION_COVERAGE:
            failure_reasons.append(
                f"aggregation coverage {coverage_ratio:.0%} < required "
                f"{MIN_AGGREGATION_COVERAGE:.0%} -- likely a sampled, not exhaustive, answer"
            )

    supported_fraction = (
        1.0
        if not claims
        else sum(1 for c in claims if c.support != ClaimSupport.UNSUPPORTED) / len(claims)
    )
    confidence = (
        supported_fraction
        if not is_aggregation_query
        else min(supported_fraction, coverage_ratio if coverage_ratio is not None else 0.0)
    )

    if confidence < CONFIDENCE_THRESHOLD:
        failure_reasons.append(
            f"confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
        )

    return SupportCheckResult(
        passed=not failure_reasons,
        claims=claims,
        coverage_ratio=coverage_ratio,
        confidence=confidence,
        failure_reasons=failure_reasons,
    )


if __name__ == "__main__":
    fake_evidence = [
        {
            "id": "chunk_1",
            "content": "Hybrid retrieval combines BM25 and vector search fused with RRF.",
        },
    ]
    result = check_answer_support(
        "Hybrid retrieval combines BM25 and vector search fused with RRF. It also cures headaches.",
        fake_evidence,
    )
    print(result)
    # Expect: passed=False -- the second claim has no evidence support.
