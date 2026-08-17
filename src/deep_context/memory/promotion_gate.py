"""Promotion gate implementing FR7–FR9, ARCHITECTURE.md §6.1, and typed-memory skill."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from deep_context.core.llm_client import llm_client
from deep_context.core.types import (
    ExistingMemory,
    MemoryType,
    Observation,
    PromotionResult,
    WriteDecision,
)
from deep_context.storage.base import StorageInterface

HEDGE_MARKERS = (
    "might",
    "may",
    "probably",
    "possibly",
    "for now",
    "considering",
    "thinking about",
    "evaluating",
)
DEFAULT_SPECULATIVE_TTL = timedelta(days=30)
MIN_CORROBORATIONS_FOR_INFERRED_PREFERENCE = 2


class PromotionGate:
    """Discipline layer gating all writes into durable long-term memory."""

    def __init__(self, storage: StorageInterface):
        self.storage = storage

    async def evaluate_and_promote(
        self,
        observation: Observation,
        corroboration_count: int = 0,
    ) -> PromotionResult:
        # Step 1: Classify Memory Type
        mem_type = self._classify_type(observation)
        if mem_type == MemoryType.DISCARD:
            return PromotionResult(
                decision=WriteDecision.REJECT,
                memory_type=mem_type,
                reject_reason="Conversational filler or non-durable observation.",
            )

        # Step 2: Scope Check
        scope_err = self._check_scope(observation, mem_type)
        if scope_err:
            return PromotionResult(
                decision=WriteDecision.REJECT,
                memory_type=mem_type,
                reject_reason=scope_err,
            )

        # Step 3: Preference Stage check
        if mem_type == MemoryType.PREFERENCE and observation.source == "inferred":
            if corroboration_count + 1 < MIN_CORROBORATIONS_FOR_INFERRED_PREFERENCE:
                return PromotionResult(
                    decision=WriteDecision.STAGE,
                    memory_type=mem_type,
                    reject_reason=f"Inferred preference requires >= {MIN_CORROBORATIONS_FOR_INFERRED_PREFERENCE} corroborating signals.",
                )

        # Step 4: Extract Atomic Claim & assign Confidence / TTL
        atomic_claim = self._extract_atomic_claim(observation)
        confidence, expires_at = self._assign_confidence_and_ttl(observation)

        # Step 5: Contradiction Check against existing memories
        existing_memories = await self.storage.get_facts_for_scope(
            tenant_id=observation.tenant_id, user_id=observation.user_id
        )
        contradiction = self._find_contradiction(atomic_claim, existing_memories)

        superseded_id = None
        if contradiction is not None:
            decision, confidence, superseded_id = self._resolve_contradiction(
                confidence, contradiction
            )
            if decision == WriteDecision.REJECT:
                return PromotionResult(
                    decision=WriteDecision.REJECT,
                    memory_type=mem_type,
                    atomic_claim=atomic_claim,
                    confidence=confidence,
                    reject_reason=f"Contradicted by higher-confidence existing memory {contradiction.id}",
                )

        # Step 6: Write to Storage
        if mem_type == MemoryType.FACT:
            fact_emb = await llm_client.get_embedding(atomic_claim)
            fact_id = await self.storage.insert_fact(
                tenant_id=observation.tenant_id,
                user_id=observation.user_id,
                content=atomic_claim,
                embedding=fact_emb,
                source=observation.source,
                confidence=confidence,
                expires_at=expires_at,
                superseded_by=None,
            )
            if superseded_id:
                await self.storage.supersede_fact(old_fact_id=superseded_id, new_fact_id=fact_id)

        elif mem_type == MemoryType.PREFERENCE and observation.user_id:
            pref_key = atomic_claim[:40].replace(" ", "_").lower()
            await self.storage.set_preference(
                user_id=observation.user_id,
                preference_key=pref_key,
                preference_value={"preference": atomic_claim},
                confidence=confidence,
                source=observation.source,
            )

        elif mem_type == MemoryType.POLICY and observation.source == "operator":
            pol_key = atomic_claim[:40].replace(" ", "_").lower()
            await self.storage.set_policy(
                tenant_id=observation.tenant_id,
                user_id=observation.user_id,
                policy_key=pol_key,
                policy_value={"policy": atomic_claim},
            )

        return PromotionResult(
            decision=WriteDecision.WRITE,
            memory_type=mem_type,
            atomic_claim=atomic_claim,
            confidence=confidence,
            expires_at=expires_at,
            superseded_id=superseded_id,
        )

    def _classify_type(self, observation: Observation) -> MemoryType:
        text = observation.raw_text.lower()
        if any(
            w in text
            for w in ("policy", "rule", "threshold", "must always", "require approval", "forbidden")
        ):
            return MemoryType.POLICY
        if any(
            w in text
            for w in ("i prefer", "i like", "please always", "don't ever", "my preference")
        ):
            return MemoryType.PREFERENCE
        words = set(re.findall(r"\b\w+\b", text))
        if (
            (words & {"thanks", "ok", "hello", "hi", "bye"})
            or "sounds good" in text
            or "got it" in text
        ):
            return MemoryType.DISCARD
        return MemoryType.FACT

    def _check_scope(self, observation: Observation, mem_type: MemoryType) -> str | None:
        if mem_type == MemoryType.POLICY and observation.source != "operator":
            return "Policy writes must originate from operator/admin path, not inference."
        if mem_type in (MemoryType.PREFERENCE, MemoryType.EPISODE) and not observation.user_id:
            return f"{mem_type.value} requires an explicit user_id."
        return None

    def _extract_atomic_claim(self, observation: Observation) -> str:
        return observation.raw_text.strip().rstrip(".")

    def _assign_confidence_and_ttl(self, observation: Observation) -> tuple[float, datetime | None]:
        text_lower = observation.raw_text.lower()
        is_hedged = any(marker in text_lower for marker in HEDGE_MARKERS)

        base_confidence = {
            "operator": 1.0,
            "user_stated": 0.9,
            "tool_output": 0.85,
            "inferred": 0.55,
        }.get(observation.source, 0.5)

        now = datetime.now(timezone.utc)
        expires_at: datetime | None = None
        if is_hedged:
            confidence = min(base_confidence, 0.6)
            expires_at = now + DEFAULT_SPECULATIVE_TTL
        else:
            confidence = base_confidence
            expires_at = (
                None
                if observation.source in ("user_stated", "operator")
                else (now + DEFAULT_SPECULATIVE_TTL * 3)
            )

        return confidence, expires_at

    def _find_contradiction(
        self, atomic_claim: str, existing_memories: list[ExistingMemory]
    ) -> ExistingMemory | None:
        claim_words = set(atomic_claim.lower().split())
        for em in existing_memories:
            em_words = set(em.content.lower().split())
            overlap = claim_words & em_words
            if len(overlap) >= 3:
                # Check for negation or contradiction keywords
                if ("not" in claim_words and "not" not in em_words) or (
                    "not" in em_words and "not" not in claim_words
                ):
                    return em
        return None

    def _resolve_contradiction(
        self, new_confidence: float, existing: ExistingMemory
    ) -> tuple[WriteDecision, float, str | None]:
        if new_confidence > existing.confidence + 0.15:
            return WriteDecision.WRITE, new_confidence, existing.id
        if existing.confidence > new_confidence + 0.15:
            return WriteDecision.REJECT, new_confidence, None
        return WriteDecision.WRITE, max(new_confidence - 0.2, 0.1), None
