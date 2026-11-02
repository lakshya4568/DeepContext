"""The documents-only answer prompt of the final round, and the check of the citations it asks for."""

from __future__ import annotations

from typing import Any

from eval import strict_prompt as SP
from eval.adapter import EVIDENCE_CHAR_BUDGET, visible_contexts

from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE


def _parent(content: str, title: str = "Doc", section: str = "s", **extra: Any) -> dict[str, Any]:
    return {"content": content, "document_title": title, "section_path": section, **extra}


# --------------------------------------------------------------------------- the prompt
def test_the_system_prompt_allows_only_the_documents_and_gives_the_exact_refusal_sentence() -> None:
    system = SP.build_strict_prompt("Q?", [_parent("text")]).messages[0]
    assert system["role"] == "system"
    assert "ONLY" in system["content"]
    assert "outside knowledge" in system["content"]
    assert REFUSAL_TEMPLATE in system["content"]


def test_the_system_prompt_calls_the_documents_data_not_instructions() -> None:
    system = SP.build_strict_prompt("Q?", [_parent("text")]).messages[0]["content"]
    assert "not instructions" in system


def test_documents_are_numbered_from_one_and_the_question_comes_last() -> None:
    prompt = SP.build_strict_prompt(
        "Who ruled?", [_parent("First text.", "A", "s1"), _parent("Second text.", "B", "s2")]
    )
    user = prompt.messages[1]
    assert user["role"] == "user"
    assert "[1] Source: A | Section: s1\nFirst text." in user["content"]
    assert "[2] Source: B | Section: s2\nSecond text." in user["content"]
    assert user["content"].rstrip().endswith("Question: Who ruled?")
    assert user["content"].index("[2]") < user["content"].index("Question:")


def test_the_generated_summary_is_not_shown_because_it_is_not_a_document() -> None:
    parent = _parent("Real text.", summary_text="A model-written summary.")
    prompt = SP.build_strict_prompt("Q?", [parent])
    assert "model-written summary" not in prompt.messages[0]["content"]
    assert "model-written summary" not in prompt.messages[1]["content"]


def test_without_documents_the_prompt_still_tells_the_model_to_refuse() -> None:
    prompt = SP.build_strict_prompt("Q?", [])
    assert prompt.shown == []
    assert "no documents" in prompt.messages[1]["content"].lower()
    assert REFUSAL_TEMPLATE in prompt.messages[0]["content"]


# --------------------------------------------------------------------------- what the model saw
def test_the_budget_keeps_the_first_documents_cuts_one_and_drops_the_rest() -> None:
    parents = [_parent("a" * 500), _parent("b" * 500), _parent("c" * 500)]
    prompt = SP.build_strict_prompt("Q?", parents, budget=900)
    numbers = [n for n, _ in prompt.shown]
    assert numbers == [1, 2]
    assert prompt.shown[0][1] == "a" * 500
    assert 0 < len(prompt.shown[1][1]) < 500
    assert "ccc" not in prompt.messages[1]["content"]


def test_what_the_model_saw_equals_the_harnesss_own_record_of_visible_evidence() -> None:
    parents = [_parent("x" * 5000, "T1"), _parent("y" * 5000, "T2"), _parent("z" * 5000, "T3")]
    prompt = SP.build_strict_prompt("Q?", parents)
    assert prompt.shown == visible_contexts(parents, "one_pass")


def test_the_budget_is_the_one_the_other_generators_use() -> None:
    assert SP.EVIDENCE_CHAR_BUDGET == EVIDENCE_CHAR_BUDGET


# --------------------------------------------------------------------------- citations
def test_cited_numbers_reads_single_lists_and_ranges_once_each_in_order() -> None:
    answer = "A [2]. B [1, 3]; C [2][4]. D [5-7]."
    assert SP.cited_numbers(answer) == [2, 1, 3, 4, 5, 6, 7]


def test_brackets_that_are_not_citations_are_ignored() -> None:
    assert SP.cited_numbers("The array [a] and [1st] and [] are not citations.") == []


def test_a_citation_check_separates_real_invented_and_gold_documents() -> None:
    check = SP.check_citations("A [1]. B [2]. C [9].", n_shown=3, gold_numbers={2})
    assert check.cited == [1, 2, 9]
    assert check.valid == [1, 2]
    assert check.invalid == [9]
    assert check.cites_gold is True
    assert check.precision == 0.5


def test_without_gold_labels_the_gold_fields_stay_empty_not_zero() -> None:
    check = SP.check_citations("A [1].", n_shown=2, gold_numbers=None)
    assert check.cites_gold is None and check.precision is None
    assert check.valid == [1]


def test_an_answer_with_no_valid_citation_has_no_precision() -> None:
    check = SP.check_citations("Nothing cited.", n_shown=2, gold_numbers={1})
    assert check.valid == []
    assert check.cites_gold is False
    assert check.precision is None
