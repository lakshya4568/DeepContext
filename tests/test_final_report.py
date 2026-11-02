"""Tables of the final round, built from the saved rows of the fast arm and the deep arm."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from eval import final_report as F

FOLDER_CONFIGS = {
    "equal": ["dense_only", "bm25_only", "rrf"],
    "vote0.02": ["rrf"],
    "rerank_vote0.02": ["rrf_bge"],
}


def _write_run(
    results_dir: Path, arm: str, suffix: str, dataset: str, ndcg: float, n: int = 6
) -> None:
    folder = results_dir / f"R_{arm}_{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_tag": f"{dataset}|qall|dfull",
            "dataset": dataset,
            "config": config,
            "qid": f"q{i}",
            "relevant": ["d1"],
            "ranked": ["d1", "d2"],
            "metrics": {"ndcg@10": ndcg, "hit@5": 1.0, "recall@100": 1.0},
        }  # fmt: skip
        for config in FOLDER_CONFIGS[suffix]
        for i in range(n)
    ]
    with (folder / "raw_retrieval.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_all_runs(results_dir: Path, dataset: str, fast: float, deep: float) -> None:
    for suffix in FOLDER_CONFIGS:
        _write_run(results_dir, "fast", suffix, dataset, fast)
        _write_run(results_dir, "deep", suffix, dataset, deep)


def _write_answers(results_dir: Path, arm: str, dataset: str, score: float, n: int = 8) -> None:
    folder = results_dir / f"A_{arm}"
    folder.mkdir(parents=True, exist_ok=True)
    raw, rubric = [], []
    for i in range(n):
        base: dict[str, Any] = {"dataset": dataset, "config": "rrf_bge_strict", "qid": f"q{i}"}
        raw.append({
            **base, "run_tag": "t", "answer": "A fact [1].", "abstained": False,
            "unanswerable": False, "contexts_seen": ["doc"], "context_ids_seen": ["d1"],
            "relevant": ["d1"],
        })  # fmt: skip
        rubric.append({**base, "score": score})
    (folder / "raw_generation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in raw), encoding="utf-8"
    )
    (folder / "rubric_scores.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rubric), encoding="utf-8"
    )


# --------------------------------------------------------------------------- retrieval tables
def test_each_dataset_has_one_row_for_each_of_the_five_search_settings(tmp_path: Path) -> None:
    _write_all_runs(tmp_path, "scifact", fast=0.5, deep=0.75)
    text = F.retrieval_markdown(tmp_path, ["scifact"])
    for label in (
        "dense only",
        "keyword only",
        "mixed, equal votes",
        "mixed, keyword vote 0.02",
        "mixed, keyword vote 0.02 + reranker",
    ):
        assert label in text
    assert "+0.250" in text


def test_a_search_setting_that_was_not_run_is_marked_not_run_and_not_shown_as_zero(
    tmp_path: Path,
) -> None:
    for suffix in ("equal", "vote0.02"):
        _write_run(tmp_path, "fast", suffix, "scifact", 0.5)
        _write_run(tmp_path, "deep", suffix, "scifact", 0.75)
    records = F.retrieval_records(tmp_path, ["scifact"])
    reranker = next(r for r in records if r["setting"].endswith("+ reranker"))
    assert reranker["verdict"] == "not run" and reranker["n"] == 0
    assert "not run" in F.retrieval_markdown(tmp_path, ["scifact"])


def test_a_setting_that_only_one_arm_has_is_not_run_either(tmp_path: Path) -> None:
    _write_run(tmp_path, "fast", "equal", "scifact", 0.5)
    records = F.retrieval_records(tmp_path, ["scifact"])
    assert {r["verdict"] for r in records} == {"not run"}


def test_the_thesis_is_shown_with_the_stored_key_and_with_the_corrected_key(
    tmp_path: Path,
) -> None:
    text = F.retrieval_markdown(tmp_path, ["pdf-quantum"])
    assert "(stored key)" in text and "(corrected key)" in text
    keys = {r["key"] for r in F.retrieval_records(tmp_path, ["pdf-quantum"])}
    assert keys == {"stored", "corrected"}


def test_a_dataset_with_only_the_fast_arm_gets_a_fast_only_table(tmp_path: Path) -> None:
    for suffix in FOLDER_CONFIGS:
        _write_run(tmp_path, "fast", suffix, "fiqa", 0.5)
    text = F.retrieval_markdown(tmp_path, ["fiqa"])
    assert "fast only" in text and "deep: not run" in text
    assert "0.500" in text and "dense only" in text


# --------------------------------------------------------------------------- numbers for the paper
def test_a_record_holds_both_means_the_paired_difference_and_the_verdict(tmp_path: Path) -> None:
    _write_all_runs(tmp_path, "scifact", fast=0.5, deep=0.75)
    dense = next(
        r for r in F.retrieval_records(tmp_path, ["scifact"]) if r["setting"] == "dense only"
    )
    assert dense["n"] == 6 and dense["fast_ndcg10"] == pytest.approx(0.5)
    assert dense["deep_ndcg10"] == pytest.approx(0.75)
    assert dense["diff_ndcg10"] == pytest.approx(0.25) and dense["verdict"] == "deep better"
    assert dense["ndcg10_lo"] == pytest.approx(0.25) and dense["ndcg10_hi"] == pytest.approx(0.25)


def test_the_glance_table_marks_clear_differences_and_counts_them(tmp_path: Path) -> None:
    _write_all_runs(tmp_path, "scifact", fast=0.5, deep=0.75)
    _write_all_runs(tmp_path, "fiqa", fast=0.75, deep=0.5)
    text = F.glance_markdown(F.retrieval_records(tmp_path, ["scifact", "fiqa"]))
    assert "+0.250 ▲" in text and "-0.250 ▼" in text
    assert "10 comparisons: 5 deep better, 5 deep worse, 0 no clear difference" in text


def _record(dataset: str, key: str, verdict: str) -> dict[str, Any]:
    return {
        "dataset": dataset, "key": key, "setting": "dense only", "n": 6, "diff_ndcg10": 0.1,
        "verdict": verdict,
    }  # fmt: skip


def test_the_counts_use_the_corrected_thesis_key_only_and_leave_out_runs_that_did_not_happen() -> (
    None
):
    records = [
        _record("pdf-quantum", "stored", "deep better"),
        _record("pdf-quantum", "corrected", "deep worse"),
        _record("scifact", "", "not run"),
    ]
    text = F.glance_markdown(records)
    assert "1 comparison: 0 deep better, 1 deep worse, 0 no clear difference" in text
    assert "pdf-quantum (stored key)" in text and "pdf-quantum (corrected key)" in text


def test_the_csv_has_one_line_per_record(tmp_path: Path) -> None:
    _write_all_runs(tmp_path, "scifact", fast=0.5, deep=0.75)
    records = F.retrieval_records(tmp_path, ["scifact"])
    path = tmp_path / "t.csv"
    F.write_csv(path, records)
    with path.open(encoding="utf-8") as fh:
        lines = list(csv.DictReader(fh))
    assert len(lines) == len(records) == 5
    assert lines[0]["dataset"] == "scifact" and lines[0]["verdict"] == "deep better"


# --------------------------------------------------------------------------- answers
def test_the_answer_records_give_correctness_and_the_paired_difference_per_arm(
    tmp_path: Path,
) -> None:
    _write_answers(tmp_path, "fast", "docsqa", 0.0)
    _write_answers(tmp_path, "deep", "docsqa", 1.0)
    records = F.answer_records(tmp_path, ["docsqa"])
    deep = next(r for r in records if r["arm"] == "deep")
    assert deep["correctness"] == pytest.approx(1.0) and deep["diff_correctness"] == pytest.approx(
        1.0
    )
    assert deep["verdict"] == "better"
    fast = next(r for r in records if r["arm"] == "fast")
    assert fast["diff_correctness"] is None


def test_an_answer_arm_that_was_not_run_is_said_so_in_the_report(tmp_path: Path) -> None:
    _write_answers(tmp_path, "fast", "docsqa", 0.0)
    assert "deep: not run" in F.answers_markdown(tmp_path, ["docsqa"])


def test_the_answers_glance_table_puts_both_arms_side_by_side_in_one_row_per_dataset(
    tmp_path: Path,
) -> None:
    _write_answers(tmp_path, "fast", "docsqa", 0.0)
    _write_answers(tmp_path, "deep", "docsqa", 1.0)
    text = F.answers_glance_markdown(F.answer_records(tmp_path, ["docsqa"]))
    row = next(line for line in text.splitlines() if line.startswith("| docsqa |"))
    assert "| 0.0 | 100.0 |" in row and "+100.0 [+100.0, +100.0]" in row and "better" in row
    assert "0% / 0% (+0.0 [+0.0, +0.0])" in row


def test_the_answer_records_carry_the_paired_difference_in_wrongly_saying_i_do_not_know(
    tmp_path: Path,
) -> None:
    _write_answers(tmp_path, "fast", "docsqa", 0.0)
    _write_answers(tmp_path, "deep", "docsqa", 1.0)
    deep = next(r for r in F.answer_records(tmp_path, ["docsqa"]) if r["arm"] == "deep")
    assert deep["diff_false_abstention"] == pytest.approx(0.0)
    assert deep["false_abstention_lo"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- the command
def test_the_command_writes_the_tables_and_the_csv_files(tmp_path: Path) -> None:
    _write_all_runs(tmp_path, "scifact", fast=0.5, deep=0.75)
    _write_answers(tmp_path, "fast", "docsqa", 0.0)
    _write_answers(tmp_path, "deep", "docsqa", 1.0)
    F.main(["--results-dir", str(tmp_path), "--datasets", "scifact", "--answer-datasets", "docsqa"])
    for name in ("RETRIEVAL.md", "ANSWERS.md", "table_retrieval.csv", "table_answers.csv"):
        assert (tmp_path / name).exists(), name
