"""
Reference implementation of the promotion gate described in SKILL.md and
docs/ARCHITECTURE.md §6.1.

Nothing in docs/DATA_MODEL.sql stops application code from writing to memory_fact or
memory_preference directly -- the schema allows it. This module IS the discipline that's
supposed to sit in front of every such write. Wire your real Postgres client in where marked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum


class MemoryType(str, Enum):
    POLICY = "policy"
    PREFERENCE = "preference"
    FACT = "fact"
    EPISODE = "episode"
    DISCARD = "discard"


class WriteDecision(str, Enum):
    WRITE = "write"
    REJECT = "reject"
    STAGE = "stage"  # inferred preference awaiting a 2nd corroborating observation


@dataclass
class Observation:
    raw_text: str
    tenant_id: str
    user_id: str | None
    source: str  # 'user_stated' | 'tool_output' | 'inferred'


@dataclass
class PromotionResult:
    decision: WriteDecision
    memory_type: MemoryType
    atomic_claim: str | None = None
    confidence: float | None = None
    expires_at: datetime | None = None
    superseded_id: str | None = None
    reject_reason: str | None = None


# Hedging language that should force a TTL rather than a permanent fact — see SKILL.md's
# TTL guidance ("might", "probably", "for now" => real TTL, not NULL).
HEDGE_MARKERS = ("might", "may", "probably", "possibly", "for now", "considering", "thinking about")

DEFAULT_SPECULATIVE_TTL = timedelta(days=30)
MIN_CORROBORATIONS_FOR_INFERRED_PREFERENCE = 2


# ---------------------------------------------------------------------------
# Step 1 — classify type
# ---------------------------------------------------------------------------


def classify_type(observation: Observation) -> MemoryType:
    """TODO: replace with a real LLM classification call in production. This heuristic exists
    so the pipeline is runnable/testable without a model call."""
    text = observation.raw_text.lower()
    if any(w in text for w in ("policy", "threshold", "must always", "require approval")):
        return MemoryType.POLICY
    if any(w in text for w in ("i prefer", "i like", "please always", "don't ever")):
        return MemoryType.PREFERENCE
    if any(w in text for w in ("thanks", "ok", "got it", "sounds good")):
        return MemoryType.DISCARD
    return MemoryType.FACT


# ---------------------------------------------------------------------------
# Step 2 — scope check
# ---------------------------------------------------------------------------


def check_scope(observation: Observation, memory_type: MemoryType) -> str | None:
    """Returns a reject_reason if scope is invalid, else None."""
    if memory_type == MemoryType.POLICY and observation.source != "operator":
        return "policy writes must come from an explicit operator/admin path, not inference"
    if memory_type in (MemoryType.PREFERENCE, MemoryType.EPISODE) and observation.user_id is None:
        return f"{memory_type.value} requires a user_id (these stores are always user-scoped)"
    return None


# ---------------------------------------------------------------------------
# Step 3 — extract the atomic claim (not the raw sentence)
# ---------------------------------------------------------------------------


def extract_atomic_claim(observation: Observation) -> str:
    """TODO: replace with a real LLM extraction call. Should strip conversational padding
    and hedges from the STORED claim while the hedge signal itself still drives TTL/confidence
    (see assign_confidence_and_ttl below) -- the hedge informs metadata, not the claim text."""
    return observation.raw_text.strip().rstrip(".")


# ---------------------------------------------------------------------------
# Step 4 — compare against existing memory; resolve contradictions
# ---------------------------------------------------------------------------


@dataclass
class ExistingMemory:
    id: str
    content: str
    confidence: float
    created_at: datetime


def find_contradiction(atomic_claim: str, existing: list[ExistingMemory]) -> ExistingMemory | None:
    """TODO: replace with a real semantic-similarity + NLI-style contradiction check (embed both
    claims, check similarity, then check for negation/contradiction). Stub: naive keyword overlap
    is NOT a real contradiction detector -- don't ship this heuristic as-is."""
    return None


def resolve_contradiction(
    new_claim: str,
    new_confidence: float,
    existing: ExistingMemory,
) -> tuple[WriteDecision, float, str | None]:
    """Returns (decision, adjusted_new_confidence, superseded_id)."""
    if new_confidence > existing.confidence + 0.15:
        return WriteDecision.WRITE, new_confidence, existing.id  # supersede old
    if existing.confidence > new_confidence + 0.15:
        return WriteDecision.REJECT, new_confidence, None  # keep old, reject new
    # Genuinely ambiguous: lower both, don't pick arbitrarily (see SKILL.md contradiction rule 2).
    # Caller is responsible for also lowering `existing.confidence` in storage when this fires.
    return WriteDecision.WRITE, max(new_confidence - 0.2, 0.1), None


# ---------------------------------------------------------------------------
# Step 5 — assign confidence + provenance + TTL
# ---------------------------------------------------------------------------


def assign_confidence_and_ttl(
    observation: Observation, atomic_claim: str
) -> tuple[float, datetime | None]:
    is_hedged = any(marker in observation.raw_text.lower() for marker in HEDGE_MARKERS)
    base_confidence = {
        "user_stated": 0.9,
        "tool_output": 0.85,
        "inferred": 0.55,
    }.get(observation.source, 0.5)

    if is_hedged:
        confidence = min(base_confidence, 0.6)
        expires_at = datetime.now(timezone.utc) + DEFAULT_SPECULATIVE_TTL
    else:
        confidence = base_confidence
        expires_at = (
            None
            if observation.source == "user_stated"
            else (datetime.now(timezone.utc) + DEFAULT_SPECULATIVE_TTL * 3)
        )
    return confidence, expires_at


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_promotion_gate(
    observation: Observation,
    existing_memory_for_scope: list[ExistingMemory],
    corroboration_count_for_key: int = 0,  # only relevant for inferred preferences
) -> PromotionResult:
    memory_type = classify_type(observation)

    if memory_type == MemoryType.DISCARD:
        return PromotionResult(
            decision=WriteDecision.REJECT,
            memory_type=memory_type,
            reject_reason="conversational filler, not a durable claim",
        )

    scope_error = check_scope(observation, memory_type)
    if scope_error:
        return PromotionResult(
            decision=WriteDecision.REJECT, memory_type=memory_type, reject_reason=scope_error
        )

    # Inferred preferences require corroboration before promotion (SKILL.md / FR7).
    if memory_type == MemoryType.PREFERENCE and observation.source == "inferred":
        if corroboration_count_for_key + 1 < MIN_CORROBORATIONS_FOR_INFERRED_PREFERENCE:
            return PromotionResult(
                decision=WriteDecision.STAGE,
                memory_type=memory_type,
                reject_reason=(
                    f"only {corroboration_count_for_key + 1} observation(s); "
                    f"need {MIN_CORROBORATIONS_FOR_INFERRED_PREFERENCE}"
                ),
            )

    atomic_claim = extract_atomic_claim(observation)
    confidence, expires_at = assign_confidence_and_ttl(observation, atomic_claim)

    contradiction = find_contradiction(atomic_claim, existing_memory_for_scope)
    superseded_id = None
    if contradiction is not None:
        decision, confidence, superseded_id = resolve_contradiction(
            atomic_claim, confidence, contradiction
        )
        if decision == WriteDecision.REJECT:
            return PromotionResult(
                decision=WriteDecision.REJECT,
                memory_type=memory_type,
                atomic_claim=atomic_claim,
                confidence=confidence,
                reject_reason=f"contradicted by higher-confidence existing memory {contradiction.id}",
            )

    return PromotionResult(
        decision=WriteDecision.WRITE,
        memory_type=memory_type,
        atomic_claim=atomic_claim,
        confidence=confidence,
        expires_at=expires_at,
        superseded_id=superseded_id,
    )


# ---------------------------------------------------------------------------
# Example (illustrates the worked example from ARCHITECTURE.md §6.1)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    obs = Observation(
        raw_text="I might use MongoDB for this prototype.",
        tenant_id="default",
        user_id="user_123",
        source="inferred",
    )
    result = run_promotion_gate(obs, existing_memory_for_scope=[])
    print(result)
    # Expect: decision=WRITE (fact, not preference), confidence <= 0.6, expires_at ~30 days out.
    # "User uses MongoDB" (no TTL, overconfident) is exactly the WRONG promotion this gate prevents.
