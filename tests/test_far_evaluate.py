"""Scoring of the arms of the far-reference test (fast vs deep) from ranked page lists."""

from __future__ import annotations

import pytest
from eval.far_test import evaluate as E


def _items() -> list[E.Item]:
    return [
        E.Item("q1", (frozenset({"p1"}), frozenset({"p9"}))),
        E.Item("q2", (frozenset({"p2"}), frozenset({"p8"}))),
        E.Item("q3", (frozenset({"p3"}), frozenset({"p7"}))),
    ]


RANKINGS = {
    "fast": {"q1": ["p1", "p5"], "q2": ["p2", "p8"], "q3": ["p6", "p7"]},
    "deep": {"q1": ["p1", "p9"], "q2": ["p2", "p8"], "q3": ["p3", "p7"]},
}


def test_each_arm_counts_the_questions_found_part_by_part_and_complete() -> None:
    results = {r.arm: r for r in E.score_arms(_items(), RANKINGS, k=2, baseline="fast")}
    fast, deep = results["fast"], results["deep"]
    assert (fast.n, fast.a_found, fast.b_found, fast.complete) == (3, 2, 2, 1)
    assert (deep.a_found, deep.b_found, deep.complete) == (3, 3, 3)


def test_the_baseline_has_no_wins_and_the_other_arm_is_compared_question_by_question() -> None:
    results = {r.arm: r for r in E.score_arms(_items(), RANKINGS, k=2, baseline="fast")}
    assert (results["fast"].wins, results["fast"].losses, results["fast"].ties) == (0, 0, 3)
    assert (results["deep"].wins, results["deep"].losses, results["deep"].ties) == (2, 0, 1)
    assert results["deep"].p_value == pytest.approx(0.5)  # two wins, no loss: 2 * (1/2)^2


def test_a_depth_cuts_the_ranking() -> None:
    results = {r.arm: r for r in E.score_arms(_items(), RANKINGS, k=1, baseline="fast")}
    assert (
        results["deep"].a_found == 3 and results["deep"].b_found == 0
    )  # the second part is at rank 2
    assert results["deep"].complete == 0


def test_a_question_missing_from_an_arm_is_an_error() -> None:
    broken = {"fast": RANKINGS["fast"], "deep": {"q1": ["p1"]}}
    with pytest.raises(KeyError):
        E.score_arms(_items(), broken, k=2, baseline="fast")


def test_the_table_lists_every_arm_with_counts_and_an_interval() -> None:
    table = E.render_table(E.score_arms(_items(), RANKINGS, k=2, baseline="fast"), "pdf-got", k=2)
    assert "| fast | 1/3 |" in table
    assert "| deep | 3/3 |" in table
    assert "pdf-got" in table


def test_pages_of_view_documents_are_merged_to_pages_in_rank_order() -> None:
    assert E.pages_of(["p0003~0", "p0002~1", "p0003~2", "p0009~0"]) == ["p0003", "p0002", "p0009"]
