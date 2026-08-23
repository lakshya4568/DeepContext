"""Tests for legacy reranker alias routing after local BGE-M3 removal.

The locally hosted BGE-reranker-v2-m3 ONNX path was removed; the
'local_cross_encoder' / 'bge' strategy names now resolve to the hosted
EcoHash BGE-M3 reranker exclusively.
"""

from unittest.mock import AsyncMock, patch

import pytest

from deep_context.retrieval.reranker import Reranker

CANDIDATES = [
    {
        "id": "1",
        "content": "Photosynthesis is how plants convert sunlight into energy.",
    },
    {"id": "2", "content": "The stock market closed higher on Tuesday."},
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    ["local_cross_encoder", "bge", "bge_reranker", "bge_m3", "ecohash"],
)
async def test_legacy_aliases_route_to_ecohash(strategy: str) -> None:
    """All BGE-flavored strategy names must dispatch to EcoHashReranker."""
    with patch(
        "deep_context.retrieval.ecohash_reranker.EcoHashReranker.rerank",
        new_callable=AsyncMock,
        return_value=[CANDIDATES[0], CANDIDATES[1]],
    ) as mock_eco:
        ranked = await Reranker.rerank(
            "What is photosynthesis?", CANDIDATES, strategy=strategy, top_k=2
        )
        mock_eco.assert_awaited_once()
        assert ranked[0]["id"] == "1"


@pytest.mark.asyncio
async def test_cross_encoder_strategy_still_available() -> None:
    """The heuristic cross_encoder strategy remains selectable."""
    ranked = await Reranker.rerank(
        "What is photosynthesis?", CANDIDATES, strategy="cross_encoder", top_k=1
    )
    assert len(ranked) == 1
