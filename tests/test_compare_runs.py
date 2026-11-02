"""Paired comparison of two retrieval runs on identical questions (used for the lexical-channel A/B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from eval.compare_runs import compare_metric, group_rows, load_groups


def _row(
    qid: str, ndcg: float, dataset: str = "d", config: str = "rrf", relevant: Any = ("p1",)
) -> dict:
    return {
        "run_tag": f"{dataset}|qall|dfull", "dataset": dataset, "config": config, "qid": qid,
        "relevant": list(relevant), "metrics": {"ndcg@10": ndcg, "hit@5": 1.0},
    }  # fmt: skip


def test_only_questions_present_in_both_runs_are_compared() -> None:
    base = group_rows([_row("q1", 0.1), _row("q2", 0.2), _row("q3", 0.3)])[("d", "rrf")]
    arm = group_rows([_row("q2", 0.3), _row("q3", 0.4), _row("q4", 0.9)])[("d", "rrf")]
    delta = compare_metric(base, arm, "ndcg@10")
    assert delta is not None and delta.n == 2


def test_difference_is_arm_minus_baseline_on_paired_questions() -> None:
    base = group_rows([_row("q1", 0.2), _row("q2", 0.4)])[("d", "rrf")]
    arm = group_rows([_row("q1", 0.4), _row("q2", 0.6)])[("d", "rrf")]
    delta = compare_metric(base, arm, "ndcg@10")
    assert delta is not None
    assert delta.base == pytest.approx(0.3)
    assert delta.arm == pytest.approx(0.5)
    assert delta.diff == pytest.approx(0.2)
    assert delta.lo == pytest.approx(0.2) and delta.hi == pytest.approx(0.2)  # same gain everywhere


def test_no_shared_questions_gives_no_comparison() -> None:
    base = group_rows([_row("q1", 0.2)])[("d", "rrf")]
    arm = group_rows([_row("q2", 0.4)])[("d", "rrf")]
    assert compare_metric(base, arm, "ndcg@10") is None


def test_unanswerable_questions_and_tolerant_pdf_rows_are_left_out() -> None:
    groups = group_rows(
        [
            _row("q1", 0.2),
            _row("q2", 0.2, relevant=()),
            _row("q3", 0.2, dataset="pdf-got ±1 page"),
        ]
    )
    assert set(groups) == {("d", "rrf")}
    assert set(groups[("d", "rrf")]) == {"q1"}


def test_a_failed_row_never_overrides_a_good_one_and_the_latest_good_row_wins(
    tmp_path: Path,
) -> None:
    rows = [
        _row("q1", 0.1),
        {**_row("q1", 0.0), "error": "boom"},
        _row("q1", 0.7),
    ]
    path = tmp_path / "raw_retrieval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_groups(path)[("d", "rrf")]["q1"]["metrics"]["ndcg@10"] == 0.7


def test_missing_file_gives_no_groups(tmp_path: Path) -> None:
    assert load_groups(tmp_path / "nope.jsonl") == {}
