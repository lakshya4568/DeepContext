"""Weighted Reciprocal Rank Fusion: each ranked list can carry its own vote (default: equal, as before)."""

from __future__ import annotations

from typing import Any

import pytest

from deep_context.core.config import settings
from deep_context.core.types import RetrievalFilters
from deep_context.retrieval import hybrid
from deep_context.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _items(*ids: str) -> list[dict[str, Any]]:
    return [{"id": i} for i in ids]


def test_default_weights_are_plain_rrf() -> None:
    fused = dict(reciprocal_rank_fusion([_items("a", "b"), _items("b", "c")], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_explicit_equal_weights_give_exactly_the_default_result() -> None:
    lists = [_items("a", "b", "c"), _items("c", "a", "d")]
    assert reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0]) == reciprocal_rank_fusion(
        lists, k=60
    )


def test_each_list_contributes_its_weight_times_the_reciprocal_rank() -> None:
    fused = dict(reciprocal_rank_fusion([_items("a"), _items("a", "b")], k=60, weights=[0.1, 1.0]))
    assert fused["a"] == pytest.approx(0.1 / 61 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_a_small_vote_lets_the_other_list_decide_the_order() -> None:
    keyword, dense = _items("a", "b", "c"), _items("b", "c", "a")
    equal = [cid for cid, _ in reciprocal_rank_fusion([keyword, dense], k=60)]
    dense_led = [
        cid for cid, _ in reciprocal_rank_fusion([keyword, dense], k=60, weights=[0.01, 1.0])
    ]
    assert equal == ["b", "a", "c"]
    assert dense_led == ["b", "c", "a"]


def test_a_zero_vote_adds_nothing_and_its_unique_items_rank_last() -> None:
    fused = reciprocal_rank_fusion([_items("x"), _items("a", "b")], k=60, weights=[0.0, 1.0])
    assert [cid for cid, _ in fused] == ["a", "b", "x"]
    assert dict(fused)["x"] == 0.0


def test_the_number_of_weights_must_match_the_number_of_lists() -> None:
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([_items("a"), _items("b")], weights=[1.0])


class _FakeStorage:
    """Keyword list a, b, c and dense list b, c, a; contents differ so nothing is treated as a duplicate."""

    @staticmethod
    def _row(cid: str) -> dict[str, Any]:
        return {"id": cid, "content": f"unique passage {cid} " * 5, "document_id": "d"}

    async def search_bm25(self, query: str, filters: Any, limit: int = 100) -> list[dict[str, Any]]:
        return [self._row(c) for c in ("a", "b", "c")]

    async def search_vector(self, emb: Any, filters: Any, limit: int = 100) -> list[dict[str, Any]]:
        return [self._row(c) for c in ("b", "c", "a")]


async def _order(monkeypatch: pytest.MonkeyPatch, lexical: float, dense: float) -> list[str]:
    async def fake_embeddings(texts: list[str], **_: Any) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 1.0] for _ in texts]

    monkeypatch.setattr(hybrid.llm_client, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(settings, "rrf_weight_lexical", lexical)
    monkeypatch.setattr(settings, "rrf_weight_dense", dense)
    candidates = await HybridRetriever(_FakeStorage()).retrieve_candidates(  # type: ignore[arg-type]
        ["a question"], RetrievalFilters(), limit=10
    )
    return [c["id"] for c in candidates]


async def test_the_retriever_defaults_to_equal_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _order(monkeypatch, 1.0, 1.0) == ["b", "a", "c"]


async def test_the_retriever_uses_the_configured_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _order(monkeypatch, 0.01, 1.0) == ["b", "c", "a"]
