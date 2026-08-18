"""Regression test locking in the reranker blend weight that produced 87.1% Hit@5.

Background: a 0.70 RRF / 0.30 raw blend with a 0.20/0.08 tiered consensus boost
was committed and regressed full-pipeline Hit@5 on the 36-query GoT benchmark
from 87.1% to 61.3%, with Direct Factual, Citation, and Multi-Hop Hit@5 each
dropping ~25 points. This test does not re-run the full benchmark (that needs
a live corpus), but it locks the default configuration values so a future edit
cannot silently reintroduce the regression without a deliberate settings change.
"""

from deep_context.core.config import settings
from deep_context.retrieval.reranker import _blend_with_rrf


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
        {"id": "weak_raw_ok_rrf", "rrf_score": 0.02, "content": "unrelated filler text"},
        {"id": "strong_raw_low_rrf", "rrf_score": 0.01, "content": "exact needle match"},
    ]
    raw_scores = [0.1, 0.95]
    scored = _blend_with_rrf(candidates, raw_scores)
    ranked_ids = [item["id"] for _, item in scored]
    assert ranked_ids[0] == "strong_raw_low_rrf"
