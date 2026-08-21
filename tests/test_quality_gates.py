"""Tests for retrieval quality gates."""

from deep_context.retrieval.quality_gates import (
    REFUSAL_TEMPLATE,
    hop_coverage,
    is_anachronism,
    is_top_consensus_candidate,
    protect_consensus,
)


def test_anachronism_detects_adversarial_queries() -> None:
    assert is_anachronism("What brand of smartphone did Sansa Stark use?")
    assert is_anachronism("Which electric airplane did Tyrion fly?")
    assert is_anachronism("What did Harry Potter tell Ned Stark?")
    assert not is_anachronism("What sword did Jon give Arya?")


def test_hop_coverage_flags_missing_entity() -> None:
    missing = hop_coverage(
        [
            "Why did Catelyn seize Tyrion and what did Littlefinger claim?",
            "Littlefinger dagger wager",
        ],
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


def test_consensus_requires_bm25_and_dense_membership() -> None:
    both = {
        "bm25_rank": 3,
        "dense_rank": 5,
    }

    bm25_only = {
        "bm25_rank": 3,
        "dense_rank": None,
    }

    dense_only = {
        "bm25_rank": None,
        "dense_rank": 5,
    }

    assert is_top_consensus_candidate(both) is True
    assert is_top_consensus_candidate(bm25_only) is False
    assert is_top_consensus_candidate(dense_only) is False


def test_protect_consensus_guarantees_retention_via_replacement() -> None:
    original = [
        {"id": "protected_chunk", "bm25_rank": 1, "dense_rank": 1, "content": "vital"},
        {"id": "item1", "bm25_rank": 15, "dense_rank": 15, "content": "c1"},
        {"id": "item2", "bm25_rank": 20, "dense_rank": 20, "content": "c2"},
    ]
    # Reranker filled all top_k slots with non-protected items
    reranked = [
        {"id": "item1", "content": "c1"},
        {"id": "item2", "content": "c2"},
    ]
    merged = protect_consensus(original, reranked, top_k=2)
    merged_ids = [c["id"] for c in merged]
    # The protected candidate must replace the weakest non-protected item
    assert "protected_chunk" in merged_ids
    assert len(merged) == 2
