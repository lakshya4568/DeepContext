"""Answer-level report of the final round: correctness, saying "I don't know", grounding and citations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from eval import answers_report as R
from eval import judge_grounding as G


def _write(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    raw, rubric, grounding = [], [], []
    for r in rows:
        base = {"dataset": r.get("dataset", "d"), "config": "rrf_bge_strict", "qid": r["qid"]}
        raw.append({
            **base, "run_tag": "t", "answer": r.get("answer", "An answer [1]."),
            "abstained": r.get("abstained", False), "unanswerable": r.get("unanswerable", False),
            "contexts_seen": r.get("seen", ["doc one", "doc two"]),
            "context_ids_seen": r.get("seen_ids", ["g1", "x2"]), "relevant": r.get("relevant", ["g1"]),
        })  # fmt: skip
        if "score" in r:
            rubric.append({**base, "score": r["score"]})
        if "claims" in r:
            claims = [G.Claim("c", c["supported"], c["support"], c["cited"]) for c in r["claims"]]
            grounding.append({**base, **G.summarise(claims)})
    for name, rows_ in (
        ("raw_generation", raw),
        ("rubric_scores", rubric),
        ("grounding_scores", grounding),
    ):
        (directory / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in rows_), encoding="utf-8"
        )


def _claim(supported: bool, support: list[int], cited: list[int]) -> dict[str, Any]:
    return {"supported": supported, "support": support, "cited": cited}


def test_a_record_holds_score_abstention_and_the_citation_facts(tmp_path: Path) -> None:
    _write(tmp_path / "a", [{"qid": "q1", "score": 1.0, "answer": "Yes [1][5]."}])
    rec = R.load_arm(tmp_path / "a")[("d", "q1")]
    assert rec["score"] == 1.0 and rec["abstained"] is False
    assert rec["valid_citations"] == [1] and rec["invalid_citations"] == [5]
    assert rec["cites_gold"] is True


def test_citations_are_compared_with_the_gold_documents_by_their_position(tmp_path: Path) -> None:
    rows = [
        {"qid": "q1", "answer": "A [2].", "seen_ids": ["g1", "x2"], "relevant": ["g1"]},
        {"qid": "q2", "answer": "A [1].", "seen_ids": ["g1", "x2"], "relevant": ["g1"]},
    ]
    _write(tmp_path / "a", rows)
    arm = R.load_arm(tmp_path / "a")
    assert arm[("d", "q1")]["cites_gold"] is False
    assert arm[("d", "q2")]["cites_gold"] is True


def test_a_set_without_relevance_labels_has_no_gold_citation_fact(tmp_path: Path) -> None:
    _write(tmp_path / "a", [{"qid": "q1", "relevant": []}])
    assert R.load_arm(tmp_path / "a")[("d", "q1")]["cites_gold"] is None


def test_the_summary_gives_both_abstention_rates_and_the_made_up_answers(tmp_path: Path) -> None:
    rows = [
        {"qid": "a1", "score": 1.0},
        {"qid": "a2", "score": 0.0, "abstained": True},  # false abstention
        {"qid": "u1", "score": 1.0, "unanswerable": True, "abstained": True},
        {"qid": "u2", "score": 0.0, "unanswerable": True},  # made-up answer
    ]
    _write(tmp_path / "a", rows)
    s = R.summarise(R.load_arm(tmp_path / "a"), "d")
    assert s["n_answerable"] == 2 and s["correctness"] == pytest.approx(0.5)
    assert s["false_abstention"] == pytest.approx(0.5)
    assert s["n_unanswerable"] == 2 and s["correct_abstention"] == pytest.approx(0.5)


def test_citation_shares_are_counted_among_answers_that_were_not_refusals(tmp_path: Path) -> None:
    rows = [
        {"qid": "a1", "answer": "Fact [1]."},
        {"qid": "a2", "answer": "Fact without a citation."},
        {"qid": "a3", "answer": "Fact [9]."},
        {"qid": "a4", "answer": "I do not know.", "abstained": True},
    ]
    _write(tmp_path / "a", rows)
    s = R.summarise(R.load_arm(tmp_path / "a"), "d")
    assert s["n_answers"] == 3
    assert s["with_valid_citation"] == pytest.approx(1 / 3)
    assert s["with_invented_citation"] == pytest.approx(1 / 3)
    assert s["cites_gold"] == pytest.approx(1 / 3)


def test_grounding_is_pooled_over_claims_and_citation_precision_over_cited_claims(
    tmp_path: Path,
) -> None:
    rows = [
        {"qid": "a1", "claims": [_claim(True, [1], [1]), _claim(True, [1], [2])]},
        {"qid": "a2", "claims": [_claim(False, [], []), _claim(True, [1], [])]},
    ]
    _write(tmp_path / "a", rows)
    s = R.summarise(R.load_arm(tmp_path / "a"), "d")
    assert s["n_graded"] == 2
    assert s["claims_backed"] == pytest.approx(3 / 4)
    assert s["answers_fully_backed"] == pytest.approx(0.5)
    assert s["citation_precision"] == pytest.approx(0.5)


def test_an_arm_without_grounding_rows_has_empty_grounding_numbers(tmp_path: Path) -> None:
    _write(tmp_path / "a", [{"qid": "a1"}])
    s = R.summarise(R.load_arm(tmp_path / "a"), "d")
    assert s["n_graded"] == 0 and s["claims_backed"] is None


def test_the_faithfulness_difference_is_paired_on_the_answers_both_arms_have(
    tmp_path: Path,
) -> None:
    unbacked = [_claim(False, [], [])]
    backed = [_claim(True, [1], [1])]
    _write(tmp_path / "fast", [{"qid": f"q{i}", "claims": unbacked} for i in range(8)])
    _write(tmp_path / "deep", [{"qid": f"q{i}", "claims": backed} for i in range(8)])
    fast, deep = R.load_arm(tmp_path / "fast"), R.load_arm(tmp_path / "deep")
    d = R.paired_field(fast, deep, "d", "faithfulness")
    assert d is not None and d.n == 8 and d.diff == pytest.approx(1.0)
    assert R.verdict_words(d) == "better"


def test_the_report_has_a_quality_table_and_a_grounding_table_per_dataset(tmp_path: Path) -> None:
    _write(tmp_path / "fast", [{"qid": "q1", "score": 1.0, "claims": [_claim(True, [1], [1])]}])
    _write(tmp_path / "deep", [{"qid": "q1", "score": 0.5, "claims": [_claim(True, [1], [1])]}])
    arms = {"fast": R.load_arm(tmp_path / "fast"), "deep": R.load_arm(tmp_path / "deep")}
    text = R.render(arms, ["d"])
    assert "### d: answers" in text and "### d: grounding and citations" in text
    assert text.count("| fast |") == 2 and text.count("| deep |") == 2
