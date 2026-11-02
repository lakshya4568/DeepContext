"""Far-apart evidence: questions whose gold pages sit in different chapters, scored at both ends."""

from __future__ import annotations

from typing import Any

import pytest
from eval import far_gold as FG


def _row(relevant: list[str], ranked: list[str]) -> dict[str, Any]:
    return {"relevant": relevant, "ranked": ranked, "metrics": {}}


def test_page_numbers_come_from_pdf_page_ids_only() -> None:
    assert FG.page_number("p0043") == 43
    assert FG.page_number("p0002") == 2
    assert FG.page_number("README.md#000") is None
    assert FG.page_number("doc12") is None


def test_gold_span_is_the_distance_between_the_earliest_and_latest_gold_page() -> None:
    assert FG.gold_span(["p0043", "p0666", "p0735"]) == 692
    assert FG.gold_span(["p0100"]) == 0
    assert FG.gold_span(["README.md#001", "README.md#009"]) == 0


def test_far_questions_need_two_gold_pages_at_least_min_span_apart() -> None:
    group = {
        "far": _row(["p0010", "p0200"], []),
        "exactly": _row(["p0010", "p0110"], []),
        "near": _row(["p0010", "p0050"], []),
        "single": _row(["p0300"], []),
    }
    assert set(FG.far_questions(group, min_span=100)) == {"far", "exactly"}
    assert set(FG.far_questions(group, min_span=101)) == {"far"}


def test_both_ends_needs_the_earliest_and_the_latest_gold_page_in_the_top_k() -> None:
    gold = ["p0010", "p0100", "p0300"]
    both = _row(gold, ["p0300", "p0001", "p0010", "p0002"])
    only_late = _row(gold, ["p0300", "p0100", "p0001", "p0010"])
    assert FG.both_ends_at_k(both, 3) == 1.0
    assert FG.both_ends_at_k(only_late, 3) == 0.0
    assert FG.both_ends_at_k(only_late, 4) == 1.0  # the middle page does not matter


def test_recall_at_k_counts_gold_pages_among_the_first_k() -> None:
    row = _row(["p0010", "p0100", "p0300"], ["p0300", "p0001", "p0010", "p0100"])
    assert FG.recall_at_k(row, 1) == pytest.approx(1 / 3)
    assert FG.recall_at_k(row, 3) == pytest.approx(2 / 3)
    assert FG.recall_at_k(row, 4) == 1.0


def test_compare_is_paired_over_the_questions_both_runs_answered() -> None:
    base = {
        "a": _row(["p0001", "p0500"], ["p0001", "p0002"]),
        "b": _row(["p0001", "p0500"], ["p0001", "p0002"]),
        "only_base": _row(["p0001", "p0500"], ["p0500", "p0001"]),
    }
    arm = {
        "a": _row(["p0001", "p0500"], ["p0001", "p0500"]),
        "b": _row(["p0001", "p0500"], ["p0001", "p0500"]),
        "only_arm": _row(["p0001", "p0500"], ["p0001"]),
    }
    delta = FG.compare(base, arm, lambda r: FG.both_ends_at_k(r, 2))
    assert delta is not None
    assert (delta.n, delta.base, delta.arm, delta.diff) == (2, 0.0, 1.0, 1.0)
    assert FG.compare({"x": base["a"]}, {"y": arm["a"]}, lambda r: 0.0) is None


def test_render_lists_each_setup_with_its_far_question_count() -> None:
    far = _row(["p0001", "p0500"], ["p0001", "p0500"])
    miss = _row(["p0001", "p0500"], ["p0001", "p0002"])
    runs = [
        ("dense only", {("pdf-got", "dense_only"): {"q1": far, "q2": far}}, "dense_only"),
        ("shipped", {("pdf-got", "rrf"): {"q1": miss, "q2": far}}, "rrf"),
    ]
    md = FG.render(runs, datasets=["pdf-got"], min_span=100, ks=(10,))
    assert "## pdf-got" in md
    assert "| dense only | 2 |" in md
    assert "| shipped | 2 |" in md
