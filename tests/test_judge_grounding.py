"""Grounding check: is every claim of an answer backed by the documents the model was shown, and are the citations right."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from eval import judge_grounding as G

from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE


def _claims(*claims: dict[str, Any]) -> str:
    return json.dumps({"claims": list(claims)})


def _claim(verdict: str, support: list[int], cited: list[int]) -> dict[str, Any]:
    return {"claim": "c", "verdict": verdict, "support": support, "cited": cited}


# --------------------------------------------------------------------------- prompt and parsing
def test_the_prompt_numbers_the_documents_and_holds_the_question_and_the_answer() -> None:
    prompt = G.build_prompt("Who ruled?", "Ned ruled [2].", ["First text.", "Second text."])
    assert "[1] First text." in prompt and "[2] Second text." in prompt
    assert "Who ruled?" in prompt and "Ned ruled [2]." in prompt


def test_the_judge_is_told_that_true_but_unstated_facts_are_unsupported() -> None:
    assert "true in the real world" in G.build_prompt("q", "a", ["d"])


def test_claims_are_read_with_their_verdict_support_and_citations() -> None:
    claims = G.parse_claims(_claims(_claim("supported", [1], [1]), _claim("unsupported", [], [])))
    assert claims is not None
    assert [c.supported for c in claims] == [True, False]
    assert claims[0].support == [1] and claims[0].cited == [1]


def test_json_wrapped_in_other_text_is_still_read() -> None:
    text = "Here you go:\n```json\n" + _claims(_claim("supported", [1], [])) + "\n```"
    claims = G.parse_claims(text)
    assert claims is not None and len(claims) == 1


def test_unusable_output_is_none_so_the_row_is_retried_not_scored() -> None:
    assert G.parse_claims("not json at all") is None
    assert (
        G.parse_claims(_claims({"claim": "c", "verdict": "maybe", "support": [], "cited": []}))
        is None
    )
    assert G.parse_claims(json.dumps({"other": 1})) is None


def test_an_answer_with_no_claims_is_an_empty_list_not_a_failure() -> None:
    assert G.parse_claims(_claims()) == []


# --------------------------------------------------------------------------- numbers
def test_faithfulness_is_the_share_of_supported_claims() -> None:
    claims = G.parse_claims(
        _claims(
            _claim("supported", [1], [1]),
            _claim("supported", [1], [1]),
            _claim("unsupported", [], []),
        )
    )
    assert claims is not None
    summary = G.summarise(claims)
    assert summary["n_claims"] == 3 and summary["n_supported"] == 2
    assert summary["faithfulness"] == pytest.approx(2 / 3)
    assert summary["fully_grounded"] is False


def test_an_answer_without_claims_has_no_faithfulness() -> None:
    summary = G.summarise([])
    assert summary["n_claims"] == 0
    assert summary["faithfulness"] is None and summary["fully_grounded"] is None


def test_a_cited_claim_is_right_only_when_a_cited_document_supports_it() -> None:
    claims = G.parse_claims(
        _claims(
            _claim("supported", [1], [1]),  # right
            _claim("supported", [1], [2]),  # cites the wrong document
            _claim("supported", [3], []),  # supported but cites nothing
        )
    )
    assert claims is not None
    summary = G.summarise(claims)
    assert summary["n_cited_claims"] == 2
    assert summary["n_citation_right"] == 1
    assert summary["citation_precision"] == 0.5
    assert summary["n_uncited_claims"] == 1


def test_a_claim_the_documents_do_not_support_cannot_be_cited_right() -> None:
    claims = G.parse_claims(_claims(_claim("unsupported", [], [1])))
    assert claims is not None
    assert G.summarise(claims)["n_citation_right"] == 0


# --------------------------------------------------------------------------- which rows are graded
def test_only_real_answers_are_graded_not_refusals_or_errors_or_empty_text() -> None:
    answer = {"answer": "Ned ruled [1].", "contexts_seen": ["t"]}
    assert G.needs_grading(answer)
    assert not G.needs_grading({**answer, "answer": REFUSAL_TEMPLATE})
    assert not G.needs_grading({**answer, "answer": "  "})
    assert not G.needs_grading({**answer, "error": "boom"})


def test_an_answer_that_names_a_missing_part_is_still_graded() -> None:
    row = {"answer": "Ned ruled [1]. The documents do not say when.", "contexts_seen": ["t"]}
    assert G.needs_grading(row)


# --------------------------------------------------------------------------- the command
def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(qid: str, answer: str) -> dict[str, Any]:
    return {
        "run_tag": "t", "dataset": "d", "config": "rrf_bge_strict", "qid": qid, "query": "q?",
        "answer": answer, "contexts_seen": ["doc text"], "unanswerable": False,
    }  # fmt: skip


def test_scoring_writes_one_row_per_graded_answer_and_a_second_run_adds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_rows(
        tmp_path / "raw_generation.jsonl",
        [_row("a", "Ned ruled [1]."), _row("b", REFUSAL_TEMPLATE), _row("c", "Jon left [1].")],
    )
    calls: list[str] = []

    async def fake_grade_all(
        rows: list[dict[str, Any]], model: str, concurrency: int
    ) -> list[dict[str, Any]]:
        calls.extend(r["qid"] for r in rows)
        return [
            {
                **G.summarise(G.parse_claims(_claims(_claim("supported", [1], [1]))) or []),
                "claims": [],
            }
            for _ in rows
        ]

    monkeypatch.setattr(G, "grade_all", fake_grade_all)
    G.score_results(tmp_path, "judge-x", concurrency=2, batch=10, flush=lambda: None)
    out = [json.loads(x) for x in (tmp_path / G.OUT_NAME).read_text().splitlines()]
    assert sorted(r["qid"] for r in out) == ["a", "c"]
    assert all(r["judge_model"] == "judge-x" and r["grounding_version"] == G.VERSION for r in out)

    calls.clear()
    G.score_results(tmp_path, "judge-x", concurrency=2, batch=10, flush=lambda: None)
    assert calls == []
    assert len((tmp_path / G.OUT_NAME).read_text().splitlines()) == 2


def test_all_batches_of_one_run_use_the_same_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A network client made during the first batch is used again in the later ones; a new event loop per batch
    # ended in "Event loop is closed" on the second batch of 200 answers.
    _write_rows(
        tmp_path / "raw_generation.jsonl", [_row(q, "Ned ruled [1].") for q in ("a", "b", "c")]
    )
    loops: list[asyncio.AbstractEventLoop] = []

    async def fake_grade_all(
        rows: list[dict[str, Any]], model: str, concurrency: int
    ) -> list[dict[str, Any]]:
        loops.append(asyncio.get_running_loop())
        return [
            {
                **G.summarise(G.parse_claims(_claims(_claim("supported", [1], [1]))) or []),
                "claims": [],
            }
            for _ in rows
        ]

    monkeypatch.setattr(G, "grade_all", fake_grade_all)
    G.score_results(tmp_path, "judge-x", concurrency=2, batch=1, flush=lambda: None)
    assert len(loops) == 3 and len({id(loop) for loop in loops}) == 1
