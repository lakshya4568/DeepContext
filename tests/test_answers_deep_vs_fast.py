"""Answer-level comparison of the arms: rubric correctness, abstention, and the paired difference to the fast arm."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval import answers_deep_vs_fast as A


def _write(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    gen, rub = [], []
    for r in rows:
        gen.append(
            {
                "dataset": r["dataset"], "config": "rrf_bge", "qid": r["qid"],
                "unanswerable": r.get("unanswerable", False), "abstained": r.get("abstained", False),
            }
        )  # fmt: skip
        rub.append(
            {"dataset": r["dataset"], "config": "rrf_bge", "qid": r["qid"], "score": r["score"],
             "answerable": not r.get("unanswerable", False)}
        )  # fmt: skip
    (directory / "raw_generation.jsonl").write_text("\n".join(json.dumps(g) for g in gen))
    (directory / "rubric_scores.jsonl").write_text("\n".join(json.dumps(x) for x in rub))


def _rows(scores: list[float]) -> list[dict]:
    return [{"dataset": "docsqa", "qid": f"q{i}", "score": s} for i, s in enumerate(scores)]


def test_an_arm_is_read_as_one_record_per_question_with_score_and_abstention(
    tmp_path: Path,
) -> None:
    rows = _rows([1.0, 0.5]) + [
        {"dataset": "docsqa", "qid": "u1", "score": 1.0, "unanswerable": True, "abstained": True}
    ]
    _write(tmp_path / "fast", rows)
    arm = A.load_arm(tmp_path / "fast")
    assert arm[("docsqa", "q1")] == {"score": 0.5, "abstained": False, "unanswerable": False}
    assert (
        arm[("docsqa", "u1")]["unanswerable"] is True and arm[("docsqa", "u1")]["abstained"] is True
    )


def test_the_summary_gives_correctness_on_answerable_and_both_abstention_rates(
    tmp_path: Path,
) -> None:
    rows = _rows([1.0, 0.5, 0.0, 1.0])
    rows[2]["abstained"] = True  # a false abstention on an answerable question
    rows += [
        {"dataset": "docsqa", "qid": "u1", "score": 1.0, "unanswerable": True, "abstained": True},
        {"dataset": "docsqa", "qid": "u2", "score": 0.0, "unanswerable": True, "abstained": False},
    ]
    _write(tmp_path / "fast", rows)
    s = A.summarise(A.load_arm(tmp_path / "fast"), "docsqa")
    assert s["n_answerable"] == 4 and s["correctness"] == pytest.approx(0.625)
    assert s["false_abstention"] == pytest.approx(0.25)
    assert s["n_unanswerable"] == 2 and s["correct_abstention"] == pytest.approx(0.5)


def test_the_difference_is_paired_on_the_answerable_questions_both_arms_have(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "fast", _rows([0.0, 0.5, 0.5, 0.5, 0.5, 0.5]))
    _write(tmp_path / "deep", _rows([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]))
    fast, deep = A.load_arm(tmp_path / "fast"), A.load_arm(tmp_path / "deep")
    d = A.paired_difference(fast, deep, "docsqa")
    assert d.n == 6 and d.diff == pytest.approx(1.0 / 6)
    assert A.verdict(d) in {"no clear difference", "deep better"}


def test_the_table_has_one_row_per_arm_under_each_dataset(tmp_path: Path) -> None:
    _write(tmp_path / "fast", _rows([1.0, 0.5, 0.5, 1.0]))
    _write(tmp_path / "deep9b", _rows([1.0, 1.0, 0.5, 1.0]))
    table = A.render(
        {"fast": A.load_arm(tmp_path / "fast"), "deep9b": A.load_arm(tmp_path / "deep9b")},
        ["docsqa"],
    )
    assert "docsqa" in table and "| fast |" in table and "| deep9b |" in table
