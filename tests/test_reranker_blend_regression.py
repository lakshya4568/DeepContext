"""Regression test locking in the reranker blend weight that produced 87.1% Hit@5.

Background: a 0.70 RRF / 0.30 raw blend with a 0.20/0.08 tiered consensus boost
was committed and regressed full-pipeline Hit@5 on the 36-query GoT benchmark
from 87.1% to 61.3%, with Direct Factual, Citation, and Multi-Hop Hit@5 each
dropping ~25 points. This test does not re-run the full benchmark (that needs
a live corpus), but it locks the default configuration values so a future edit
cannot silently reintroduce the regression without a deliberate settings change.
"""

import math

import pytest

from deep_context.core.config import settings
from deep_context.retrieval.reranker import (
    _blend_with_rrf,
    _normalize_reranker_scores,
    _sigmoid,
)


def test_default_blend_weight_matches_validated_configuration() -> None:
    assert settings.reranker_blend_rrf_weight == 0.60
    assert settings.reranker_consensus_top1_count == 3
    assert settings.reranker_consensus_boost_tier1 == 0.15


def test_blend_does_not_fully_override_raw_signal() -> None:
    """A candidate with a much stronger raw (lexical/semantic) score should be able
    to outrank a candidate with a merely-average RRF score, proving the raw signal
    still has meaningful influence at the default weighting.
    """
    candidates = [
        {"id": "top_rrf_weak_raw", "rrf_score": 0.04, "content": "unrelated filler text"},
        {"id": "weak_raw_ok_rrf", "rrf_score": 0.02, "content": "unrelated filler text"},
        {"id": "strong_raw_low_rrf", "rrf_score": 0.01, "content": "exact needle match"},
    ]
    raw_scores = [0.1, 0.1, 0.95]
    scored = _blend_with_rrf(candidates, raw_scores, score_type="probability")
    ranked_ids = [item["id"] for _, item in scored]
    assert ranked_ids.index("strong_raw_low_rrf") < ranked_ids.index("weak_raw_ok_rrf")


def test_sigmoid_is_numerically_stable() -> None:
    assert _sigmoid(-1000.0) == pytest.approx(0.0)
    assert _sigmoid(1000.0) == pytest.approx(1.0)
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_handles_non_finite_values() -> None:
    assert _sigmoid(float("nan")) == 0.0
    assert _sigmoid(float("inf")) == 1.0
    assert _sigmoid(float("-inf")) == 0.0


def test_logit_scores_are_sigmoid_normalized() -> None:
    scores = _normalize_reranker_scores(
        [-2.0, 0.0, 2.0],
        "logit",
    )

    assert scores[0] == pytest.approx(0.1192029, rel=1e-5)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.8807971, rel=1e-5)


def test_probability_scores_are_not_sigmoid_transformed() -> None:
    scores = _normalize_reranker_scores(
        [0.1, 0.4, 0.9],
        "probability",
    )

    assert scores == pytest.approx([0.1, 0.4, 0.9])


def test_probability_scores_are_clamped() -> None:
    scores = _normalize_reranker_scores(
        [-1.0, 0.5, 2.0],
        "probability",
    )

    assert scores == pytest.approx([0.0, 0.5, 1.0])


def test_non_finite_probability_scores_become_zero() -> None:
    scores = _normalize_reranker_scores(
        [float("nan"), float("inf"), float("-inf")],
        "probability",
    )

    assert scores == [0.0, 0.0, 0.0]


def test_unknown_score_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        _normalize_reranker_scores([0.5], "unknown")  # type: ignore[arg-type]


def test_equal_rrf_scores_do_not_divide_by_zero(monkeypatch) -> None:
    candidates = [
        {"id": "a", "rrf_score": 0.02},
        {"id": "b", "rrf_score": 0.02},
    ]

    monkeypatch.setattr(settings, "reranker_blend_rrf_weight", 0.5)
    monkeypatch.setattr(settings, "reranker_consensus_boost_tier1", 0.0)
    monkeypatch.setattr(settings, "reranker_consensus_boost_tier2", 0.0)

    result = _blend_with_rrf(
        candidates,
        [0.9, 0.1],
        score_type="probability",
    )

    assert result[0][1]["id"] == "a"
    assert all(math.isfinite(score) for score, _ in result)


def test_blend_rejects_mismatched_candidate_and_score_counts() -> None:
    with pytest.raises(ValueError, match="Candidate count"):
        _blend_with_rrf(
            [{"id": "a", "rrf_score": 0.02}],
            [],
            score_type="probability",
        )
