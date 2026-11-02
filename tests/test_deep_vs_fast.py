"""Deep against fast ingestion on identical questions: one table row per search setting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval import deep_vs_fast as D


def _group(ndcg: list[float], hit: list[float] | None = None) -> dict[str, dict]:
    hit = hit or [1.0] * len(ndcg)
    return {
        f"q{i}": {"metrics": {"ndcg@10": n, "hit@5": h, "recall@100": 1.0}, "qid": f"q{i}"}
        for i, (n, h) in enumerate(zip(ndcg, hit, strict=True))
    }


def test_a_comparison_row_gives_both_means_the_paired_difference_and_the_number_of_questions() -> (
    None
):
    row = D.compare_row(_group([0.5, 0.5, 0.5, 0.5]), _group([0.75, 0.75, 0.75, 0.75]))
    assert row["n"] == 4
    assert row["fast"] == pytest.approx(0.5) and row["deep"] == pytest.approx(0.75)
    assert row["ndcg@10"].diff == pytest.approx(0.25)
    assert row["ndcg@10"].lo == pytest.approx(0.25) and row["ndcg@10"].hi == pytest.approx(0.25)


def test_only_questions_answered_by_both_runs_count() -> None:
    row = D.compare_row(_group([0.5, 0.5, 0.5]), _group([0.75, 0.75]))
    assert row["n"] == 2


def test_the_table_lists_each_setting_under_its_dataset_with_signed_differences() -> None:
    settings = [("dense only", _group([0.5, 0.5]), _group([0.75, 0.75]))]
    table = D.render_dataset("docsqa", settings)
    assert "docsqa" in table and "dense only" in table and "+0.250" in table


def test_the_verdict_says_better_worse_or_no_clear_difference_from_the_interval() -> None:
    assert (
        D.verdict(D.compare_row(_group([0.5] * 6), _group([0.75] * 6))["ndcg@10"]) == "deep better"
    )
    assert (
        D.verdict(D.compare_row(_group([0.75] * 6), _group([0.5] * 6))["ndcg@10"]) == "deep worse"
    )
    mixed = D.compare_row(_group([0.5, 0.9, 0.5, 0.9]), _group([0.9, 0.5, 0.9, 0.5]))["ndcg@10"]
    assert D.verdict(mixed) == "no clear difference"
    assert D.verdict(None) == "n/a"


def test_the_verdict_uses_the_name_of_the_second_run() -> None:
    better = D.compare_row(_group([0.5] * 6), _group([0.75] * 6))["ndcg@10"]
    worse = D.compare_row(_group([0.75] * 6), _group([0.5] * 6))["ndcg@10"]
    assert D.verdict(better, "deep 9B") == "deep 9B better"
    assert D.verdict(worse, "deep 9B") == "deep 9B worse"


def test_the_table_can_name_both_runs_so_a_deep_against_deep_table_is_not_headed_fast() -> None:
    settings = [("dense only", _group([0.5, 0.5]), _group([0.75, 0.75]))]
    table = D.render_dataset("docsqa", settings, names=("deep 0.6B", "deep 9B"))
    assert "| deep 0.6B nDCG@10 | deep 9B nDCG@10 | deep 9B - deep 0.6B, nDCG@10 |" in table
    assert "deep 9B better" in table
    assert "fast" not in table


def _write_rows(directory: Path, ndcg: float) -> None:
    directory.mkdir()
    rows = [
        {
            "run_tag": "t",
            "dataset": "docsqa",
            "config": "rrf",
            "qid": f"q{i}",
            "relevant": ["x"],
            "metrics": {"ndcg@10": ndcg, "hit@5": 1.0, "recall@100": 1.0},
        }
        for i in range(6)
    ]
    text = "\n".join(json.dumps(r) for r in rows)
    (directory / "raw_retrieval.jsonl").write_text(text, encoding="utf-8")


def test_the_command_takes_the_names_of_the_two_runs_and_a_title(tmp_path: Path) -> None:
    _write_rows(tmp_path / "a", 0.5)
    _write_rows(tmp_path / "b", 0.75)
    out = tmp_path / "table.md"
    D.main(
        [
            "--pair", f"mixed={tmp_path / 'a'}:rrf|{tmp_path / 'b'}:rrf",
            "--datasets", "docsqa",
            "--base-name", "deep 0.6B",
            "--other-name", "deep 9B",
            "--title", "Summariser size",
            "--out", str(out),
        ]
    )  # fmt: skip
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Summariser size")
    assert "deep 9B - deep 0.6B" in text and "deep 9B better" in text
