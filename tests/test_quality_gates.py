"""Tests for retrieval quality gates."""

from deep_context.retrieval.quality_gates import (
    REFUSAL_TEMPLATE,
    hop_coverage,
    is_anachronism,
    protect_consensus,
)


def test_anachronism_detects_adversarial_queries() -> None:
    assert is_anachronism("What brand of smartphone did Sansa Stark use?")
    assert is_anachronism("Which electric airplane did Tyrion fly?")
    assert is_anachronism("What did Harry Potter tell Ned Stark?")
    assert not is_anachronism("What sword did Jon give Arya?")


def test_hop_coverage_flags_missing_entity() -> None:
    missing = hop_coverage(
        ["Why did Catelyn seize Tyrion and what did Littlefinger claim?", "Littlefinger dagger wager"],
        [{"content": "Catelyn seized Tyrion at the inn after Bran was attacked."}],
    )
    assert missing == ["Littlefinger dagger wager"]


def test_protect_consensus_keeps_dual_hits() -> None:
    original = [
        {"id": "keep", "bm25_rank": 2, "dense_rank": 3, "content": "needle"},
        {"id": "noise", "bm25_rank": 1, "dense_rank": 40, "content": "other"},
    ]
    reranked = [{"id": "noise", "content": "other"}]
    merged = protect_consensus(original, reranked, top_k=2)
    assert [c["id"] for c in merged] == ["noise", "keep"]
    assert REFUSAL_TEMPLATE.startswith("Based on the provided context")
