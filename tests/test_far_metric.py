"""Scoring of the far-reference test: a question is answered when every part of its evidence is found."""

from __future__ import annotations

import pytest
from eval.far_test import metric as M


def test_a_question_is_complete_when_every_part_has_a_page_in_the_top_k() -> None:
    parts = (frozenset({"p1", "p2"}), frozenset({"p9"}))
    assert M.complete_at_k(["p5", "p2", "p9"], parts, k=3)
    assert not M.complete_at_k(["p5", "p2", "p9"], parts, k=2)  # p9 is third
    assert not M.complete_at_k(["p1", "p2", "p3"], parts, k=3)  # the second part is never found


def test_any_page_of_a_part_counts_because_repeated_facts_are_alternatives() -> None:
    assert M.complete_at_k(["p2"], (frozenset({"p1", "p2"}),), k=1)


def test_parts_found_is_reported_part_by_part() -> None:
    parts = (frozenset({"p1"}), frozenset({"p9"}))
    assert M.parts_found(["p1", "p3"], parts, k=2) == (True, False)


def test_a_question_without_parts_cannot_be_scored() -> None:
    with pytest.raises(ValueError, match="no parts"):
        M.complete_at_k(["p1"], (), k=1)


def test_rrf_puts_pages_that_two_lists_agree_on_first_and_breaks_ties_by_first_appearance() -> None:
    assert M.rrf([["a", "b", "c"], ["c", "d", "a"]]) == ["a", "c", "b", "d"]


def test_rrf_weights_decide_which_list_leads() -> None:
    assert M.rrf([["a", "b"], ["b", "a"]], weights=[1.0, 0.1]) == ["a", "b"]
    assert M.rrf([["a", "b"], ["b", "a"]], weights=[0.1, 1.0]) == ["b", "a"]


def test_rrf_refuses_a_wrong_number_of_weights_and_accepts_no_lists() -> None:
    with pytest.raises(ValueError, match="weights"):
        M.rrf([["a"]], weights=[1.0, 2.0])
    assert M.rrf([]) == []


def test_paired_counts_are_wins_losses_and_ties_of_the_arm_against_the_baseline() -> None:
    assert M.paired_counts([True, False, False, True], [True, True, False, False]) == (1, 1, 2)


def test_sign_test_is_the_exact_two_sided_binomial_test_on_wins_and_losses() -> None:
    assert M.sign_test(0, 0) == 1.0
    assert M.sign_test(5, 0) == pytest.approx(0.0625)
    assert M.sign_test(3, 3) == 1.0
    assert M.sign_test(8, 2) == pytest.approx(0.109375)
    assert M.sign_test(2, 8) == M.sign_test(8, 2)


def test_exact_interval_matches_the_values_in_the_research_report() -> None:
    lo, hi = M.exact_ci(0, 5)
    assert (lo, hi) == (0.0, pytest.approx(0.5218, abs=1e-3))
    lo, hi = M.exact_ci(2, 5)
    assert lo == pytest.approx(0.0527, abs=1e-3)
    assert hi == pytest.approx(0.8534, abs=1e-3)
    assert M.exact_ci(5, 5)[1] == 1.0
