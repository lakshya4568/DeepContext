"""Dynamic 8-layer prompt assembler implementing ARCHITECTURE.md §6.2 and FR9."""

from __future__ import annotations

from typing import Any

from deep_context.memory.stores import MemoryStoreManager
from deep_context.storage.base import StorageInterface


class PromptAssembler:
    """
    Constructs bounded, structured prompt context in 8 distinct layers:
    1. System instructions
    2. Active policies (exact lookup)
    3. User preferences (exact lookup)
    4. Short conversation summary
    5. Current task state
    6. Retrieved durable facts (hybrid search)
    7. Retrieved documents / evidence
    8. Current user query
    """

    def __init__(self, storage: StorageInterface):
        self.storage = storage
        self.memory_manager = MemoryStoreManager(storage)

    async def assemble_messages(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        tenant_id: str = "default",
        user_id: str | None = None,
        conversation_summary: str | None = None,
        task_state: str | None = None,
    ) -> list[dict[str, str]]:
        default_inst = (
            "You are an intelligent, strictly grounded AI assistant powered by the Deep Context Platform.\n\n"
            "RULES OF RESPONSE:\n"
            "1. Use ONLY the provided Retrieved Evidence. Do not use outside knowledge, assumptions, or external memory.\n"
            "2. Answer all parts of the question that are supported by the provided evidence. Be thorough, specific, and factual.\n"
            "3. If only some requested details are present in the context, provide the supported details fully and clearly state which specific parts are not mentioned.\n"
            "4. ONLY if the entire question cannot be answered at all from the provided context (e.g. asking about modern technology, real-world events, or characters completely absent from the text), reply EXACTLY:\n"
            '"Based on the provided context, there is insufficient evidence to answer."\n'
            "5. Do not guess or fabricate unsupported information."
        )
        base_instruction = system_instruction or default_inst
        sections: list[str] = [base_instruction]

        # Layer 2: Active Policies (exact lookup)
        policies = await self.memory_manager.get_active_policies(
            tenant_id=tenant_id, user_id=user_id
        )
        if policies:
            pol_text = "\n".join(f"- {p['policy_key']}: {p['policy_value']}" for p in policies)
            sections.append(f"### Mandatory Policies\n{pol_text}")

        # Layer 3: User Preferences (exact lookup)
        if user_id:
            preferences = await self.memory_manager.get_user_preferences(user_id=user_id)
            if preferences:
                pref_text = "\n".join(
                    f"- {p['preference_key']}: {p['preference_value']}" for p in preferences
                )
                sections.append(f"### User Preferences\n{pref_text}")

        # Layer 4: Conversation Summary
        if conversation_summary:
            sections.append(f"### Prior Conversation Context\n{conversation_summary}")

        # Layer 5: Task State
        if task_state:
            sections.append(f"### Current Task State\n{task_state}")

        # Layer 6: Retrieved Durable Facts
        facts = await self.memory_manager.search_relevant_facts(
            query=query, tenant_id=tenant_id, user_id=user_id, limit=4
        )
        if facts:
            fact_text = "\n".join(
                f"- {f['content']} (confidence: {f['confidence']:.2f})" for f in facts
            )
            sections.append(f"### Persistent Verified Facts\n{fact_text}")

        # Layer 7: Retrieved Evidence (bounded to ~3500 tokens to respect API limits)
        if retrieved_chunks:
            evidence_blocks = []
            char_budget = 12000
            total_chars = 0
            for idx, c in enumerate(retrieved_chunks, start=1):
                title = c.get("document_title", "Document")
                sec = c.get("section_path", "")
                content = c.get("content", "").strip()
                block = f"[{idx}] Source: {title} | Section: {sec}\n{content}"
                if total_chars + len(block) > char_budget:
                    remaining = char_budget - total_chars
                    if remaining > 200:
                        evidence_blocks.append(
                            block[:remaining] + "...\n[Additional context trimmed for brevity]"
                        )
                    break
                evidence_blocks.append(block)
                total_chars += len(block)

            evidence_text = "\n\n".join(evidence_blocks)
            sections.append(f"### Retrieved Evidence (Grounding Context)\n{evidence_text}")

        system_prompt = "\n\n".join(sections)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return messages
