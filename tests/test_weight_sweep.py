"""Summary of a lexical-vote sweep on dev sets, with the pre-declared selection rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.weight_sweep import best_weight, summarise


def _row(dataset: str, qid: str, ndcg: float, recall: float, config: str = "rrf") -> dict:
    return {
        "run_tag": f"{dataset}|q300|dfull|dev", "dataset": dataset, "config": config, "qid": qid,
        "relevant": ["d1"], "metrics": {"ndcg@10": ndcg, "recall@100": recall},
    }  # fmt: skip


def _write(root: Path, weight: str, rows: list[dict]) -> None:
    folder = root / f"dev_pgs_w{weight}"
    folder.mkdir(parents=True)
    (folder / "raw_retrieval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_summarise_gives_mean_ndcg_recall_and_n_per_weight_and_dataset(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "0.1",
        [_row("a", "q1", 0.2, 1.0), _row("a", "q2", 0.4, 0.5), _row("b", "q1", 0.9, 1.0)],
    )
    table = summarise(tmp_path, ["0.1"])
    assert table["0.1"]["a"] == pytest.approx((0.3, 0.75, 2))
    assert table["0.1"]["b"] == pytest.approx((0.9, 1.0, 1))


def test_only_the_requested_config_is_summarised(tmp_path: Path) -> None:
    _write(
        tmp_path, "0.1", [_row("a", "q1", 0.2, 1.0), _row("a", "q1", 0.9, 1.0, config="dense_only")]
    )
    assert summarise(tmp_path, ["0.1"])["0.1"]["a"][0] == pytest.approx(0.2)


def test_a_weight_without_results_is_left_out(tmp_path: Path) -> None:
    _write(tmp_path, "0.1", [_row("a", "q1", 0.2, 1.0)])
    assert set(summarise(tmp_path, ["0.1", "0.5"])) == {"0.1"}


def test_best_weight_has_the_highest_mean_ndcg_over_the_datasets() -> None:
    table = {
        "0.05": {"a": (0.50, 0.9, 10), "b": (0.70, 0.9, 10)},  # mean 0.60
        "0.1": {"a": (0.60, 0.9, 10), "b": (0.65, 0.9, 10)},  # mean 0.625
        "0.5": {"a": (0.90, 0.9, 10), "b": (0.20, 0.9, 10)},  # mean 0.55 although a is higher
    }
    assert best_weight(table, ["a", "b"]) == "0.1"


def test_ties_go_to_the_smaller_weight() -> None:
    table = {
        "0.25": {"a": (0.5, 0.9, 10)},
        "0.05": {"a": (0.5, 0.9, 10)},
        "1.0": {"a": (0.5, 0.9, 10)},
    }
    assert best_weight(table, ["a"]) == "0.05"
