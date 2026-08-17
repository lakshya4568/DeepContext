"""Tests for typed memory stores, promotion gate, and prompt assembly."""

import pytest

from deep_context.core.types import Observation, WriteDecision
from deep_context.memory.prompt_assembler import PromptAssembler
from deep_context.memory.stores import MemoryStoreManager
from deep_context.storage import get_storage


@pytest.mark.asyncio
async def test_memory_promotion_gate_hedged_fact() -> None:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)

    # Hedged observation -> should receive confidence <= 0.6 and a TTL
    obs = Observation(
        raw_text="I might use MongoDB for this prototype.",
        tenant_id="default",
        user_id="user_456",
        source="inferred",
    )
    result = await manager.observe_and_promote(obs)

    assert result.decision == WriteDecision.WRITE
    assert result.memory_type.value == "fact"
    assert result.confidence is not None and result.confidence <= 0.6
    assert result.expires_at is not None


@pytest.mark.asyncio
async def test_memory_promotion_gate_discard() -> None:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)

    obs = Observation(
        raw_text="Thanks, sounds good!",
        tenant_id="default",
        user_id="user_123",
        source="user_stated",
    )
    result = await manager.observe_and_promote(obs)
    assert result.decision == WriteDecision.REJECT


@pytest.mark.asyncio
async def test_policy_and_preference_exact_lookup() -> None:
    storage = await get_storage()
    manager = MemoryStoreManager(storage)

    # Set policy
    await storage.set_policy(
        tenant_id="default",
        policy_key="max_discount",
        policy_value={"max_percent": 15},
    )

    # Set user preference
    await storage.set_preference(
        user_id="user_789",
        preference_key="language",
        preference_value={"lang": "Python"},
    )

    policies = await manager.get_active_policies("default")
    assert any(p["policy_key"] == "max_discount" for p in policies)

    prefs = await manager.get_user_preferences("user_789")
    assert any(p["preference_key"] == "language" for p in prefs)


@pytest.mark.asyncio
async def test_prompt_assembler_8_layers() -> None:
    storage = await get_storage()
    assembler = PromptAssembler(storage)

    # Seed policy, preference, and fact
    await storage.set_policy("default", "security_rule", {"rule": "No plain passwords"})
    await storage.set_preference("user_1", "style", {"concise": True})
    await storage.insert_fact(
        tenant_id="default",
        user_id="user_1",
        content="Project uses Postgres with pgvector.",
        embedding=[0.1] * 1024,
        source="user_stated",
        confidence=0.9,
    )

    messages = await assembler.assemble_messages(
        query="What database is used?",
        retrieved_chunks=[
            {"document_title": "Arch Doc", "section_path": "DB", "content": "Database is Postgres."}
        ],
        user_id="user_1",
    )

    assert len(messages) == 2
    system_text = messages[0]["content"]

    # Verify all layers appear in prompt
    assert "Mandatory Policies" in system_text
    assert "User Preferences" in system_text
    assert "Persistent Verified Facts" in system_text
    assert "Retrieved Evidence" in system_text
    assert messages[1]["content"] == "What database is used?"
