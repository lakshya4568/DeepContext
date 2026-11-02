"""Corrected page key of the thesis set: every gold page that has a partner 20 pages later is the printed page number."""

from __future__ import annotations

import pytest
from eval import thesis_key as K


def test_a_gold_page_with_a_partner_twenty_pages_later_is_dropped_and_the_pdf_page_is_kept() -> (
    None
):
    assert K.corrected_relevant(["p0016", "p0036", "p0126", "p0146"]) == ["p0036", "p0146"]


def test_pages_without_a_partner_and_the_order_of_the_rest_are_kept() -> None:
    assert K.corrected_relevant(["p0050", "p0010", "p0031"]) == ["p0050", "p0010", "p0031"]


def test_in_a_chain_only_the_last_page_survives() -> None:
    assert K.corrected_relevant(["p0010", "p0030", "p0050"]) == ["p0050"]


def _row(relevant: list[str], ranked: list[str]) -> dict:
    return {
        "dataset": "pdf-quantum",
        "config": "dense_only",
        "qid": "q1",
        "relevant": relevant,
        "ranked": ranked,
        "metrics": {"recall@10": 0.5, "hit@5": 1.0, "ndcg@10": 0.6, "recall@100": 0.5},
    }


def test_rescoring_recomputes_the_metrics_with_the_corrected_pages_and_keeps_the_rest_of_the_row() -> (
    None
):
    row = _row(["p0016", "p0036"], ["p0036", "p0001", "p0002"])
    [fixed] = K.correct_group({"q1": row}).values()
    assert fixed["relevant"] == ["p0036"]
    assert fixed["metrics"]["recall@10"] == pytest.approx(1.0)
    assert fixed["metrics"]["ndcg@10"] == pytest.approx(1.0)
    assert fixed["metrics"]["hit@5"] == 1.0
    assert fixed["ranked"] == row["ranked"] and fixed["qid"] == "q1"
    assert row["relevant"] == ["p0016", "p0036"]  # the stored row is not changed
