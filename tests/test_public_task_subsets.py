"""The deep arm covers exactly the corpora of the fast run, and RAGBench loads from its cache without `datasets`."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from eval import config as C
from eval import datasets as D
from eval import main_summaries as MS


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_build_task(name: str, **kwargs: Any) -> str:
        seen.update(name=name, **kwargs)
        return "task"

    monkeypatch.setattr(D, "build_task", fake_build_task)
    return seen


def test_hotpotqa_is_loaded_with_the_subset_the_fast_run_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    MS.task_for("hotpotqa")
    assert seen == {"name": "hotpotqa", "n_queries": 300, "max_docs": 30_000}


def test_a_dataset_the_fast_run_used_whole_is_loaded_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    MS.task_for("scifact")
    assert seen == {"name": "scifact"}


def _fake_ragbench(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "question": "What is X?",
            "response": "X is Y.",
            "documents": ["doc about X", "other doc"],
            "all_relevant_sentence_keys": ["0a"],
        }
    ]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *_a, **_k: rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_ragbench_is_cached_after_the_first_load_and_reloaded_without_the_datasets_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    _fake_ragbench(monkeypatch)
    first = D.load_ragbench("covidqa")
    assert list(tmp_path.glob("ragbench-covidqa_*.json"))

    monkeypatch.setitem(sys.modules, "datasets", None)  # importing it now would fail
    second = D.load_ragbench("covidqa")
    assert second.corpus == first.corpus
    assert second.queries == first.queries
    assert second.qrels == first.qrels
    assert second.answers == {"r0000": "X is Y."}


def test_the_cached_task_is_valid_json_with_the_reference_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    _fake_ragbench(monkeypatch)
    D.load_ragbench("covidqa")
    (cache,) = tmp_path.glob("ragbench-covidqa_*.json")
    assert json.loads(cache.read_text())["answers"] == {"r0000": "X is Y."}


# --------------------------------------------------------------------------- the FiQA subset (first documents + all relevant)
def _tiny_beir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ten documents d0..d9 in corpus order; q1 is answered by d9, q2 by d1."""
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)

    def source(name: str, split: str) -> Any:
        def corpus() -> Any:
            for i in range(10):
                yield f"d{i}", f"title {i}", f"text {i}"

        return (
            {"q1": "first question", "q2": "second question"},
            {"q1": {"d9": 1}, "q2": {"d1": 1}},
            corpus,
            {"source": "tiny"},
        )

    monkeypatch.setattr(D, "_beir_source", source)


def test_the_prefix_task_keeps_the_first_documents_and_every_relevant_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_beir(monkeypatch, tmp_path)
    task = D.load_beir("fiqa", first_docs=3)
    assert list(task.corpus) == ["d0", "d1", "d2", "d9"]
    assert set(task.queries) == {"q1", "q2"} and task.qrels == {"q1": {"d9": 1}, "q2": {"d1": 1}}
    assert "first 3 documents" in task.meta["sampling"]


def test_a_prefix_task_and_the_whole_task_do_not_share_a_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_beir(monkeypatch, tmp_path)
    D.load_beir("fiqa", first_docs=3)
    assert len(D.load_beir("fiqa").corpus) == 10
    assert len(D.load_beir("fiqa", first_docs=3).corpus) == 4


def test_first_documents_and_a_random_sample_cannot_be_asked_for_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_beir(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="first_docs"):
        D.load_beir("fiqa", first_docs=3, max_docs=5)


def test_fiqa_sub_is_the_prefix_task_under_its_own_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_beir(monkeypatch, tmp_path)
    monkeypatch.setattr(D, "FIQA_SUB_FIRST_DOCS", 3)
    task = D.build_task("fiqa-sub")
    assert task.name == "fiqa-sub" and list(task.corpus) == ["d0", "d1", "d2", "d9"]


def test_a_prefix_task_is_cut_from_the_cached_whole_task_when_there_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_beir(monkeypatch, tmp_path)
    D.load_beir("fiqa")  # writes the cache of the whole task

    def no_source(name: str, split: str) -> Any:
        raise AssertionError("the BEIR source must not be read")

    monkeypatch.setattr(D, "_beir_source", no_source)
    task = D.load_beir("fiqa", first_docs=3)
    assert list(task.corpus) == ["d0", "d1", "d2", "d9"]
    assert task.qrels == {"q1": {"d9": 1}, "q2": {"d1": 1}}
