"""The strict configuration of the adapter answers with the documents-only prompt and records what the model saw."""

from __future__ import annotations

from typing import Any

import pytest
from eval import config as C
from eval.adapter import DeepContextAdapter
from eval.strict_prompt import PROMPT_VERSION

from deep_context.core.llm_client import llm_client
from deep_context.retrieval.engine import retrieval_engine
from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE

PARENT = {
    "id": "p1",
    "document_id": "uuid-1",
    "document_title": "Cats",
    "section_path": "Habits",
    "content": "The cat sleeps on the sofa.",
    "summary_text": "A model-written summary.",
}


def _adapter(
    monkeypatch: pytest.MonkeyPatch, answer: str, sent: dict[str, Any]
) -> DeepContextAdapter:
    adapter = DeepContextAdapter("test_strict_generation", config="rrf_bge_strict")
    adapter._ext_by_store = {"uuid-1": "doc-1"}

    async def fake_retrieve_children(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return [{"id": "c1"}]

    async def fake_resolve(_store: Any, _pool: Any) -> list[dict[str, Any]]:
        return [PARENT]

    async def fake_complete(messages: list[dict[str, str]], **kwargs: Any) -> tuple[str, Any]:
        sent["messages"], sent["kwargs"] = messages, kwargs
        return answer, None

    monkeypatch.setattr(adapter, "retrieve_children", fake_retrieve_children)
    monkeypatch.setattr(retrieval_engine, "_resolve_parent_chunks", fake_resolve)
    monkeypatch.setattr(llm_client, "complete", fake_complete)
    return adapter


def test_the_strict_configuration_reranks_like_the_default_and_uses_the_strict_generation_mode() -> (
    None
):
    _name, retrieval, generation = C.CONFIGS["rrf_bge_strict"]
    assert (retrieval, generation) == (C.CONFIGS["rrf_bge"][1], "strict")


async def test_it_answers_with_the_strict_prompt_at_temperature_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}
    adapter = _adapter(monkeypatch, "The cat sleeps on the sofa [1].", sent)
    out = await adapter.answer("Where does the cat sleep?")
    assert out["answer"] == "The cat sleeps on the sofa [1]."
    assert out["generation_mode"] == "strict"
    assert out["prompt_version"] == PROMPT_VERSION
    assert sent["kwargs"]["temperature"] == 0.0
    assert "ONLY" in sent["messages"][0]["content"]
    assert "A model-written summary." not in str(sent["messages"])


async def test_it_records_the_documents_the_model_saw_by_their_benchmark_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch, "Answer [1].", {})
    out = await adapter.answer("Q?")
    assert out["contexts_seen"] == ["The cat sleeps on the sofa."]
    assert out["context_ids_seen"] == ["doc-1"]
    assert out["evidence_chars_seen"] == len("The cat sleeps on the sofa.")


async def test_the_exact_refusal_sentence_counts_as_a_refusal_and_an_answer_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusing = await _adapter(monkeypatch, REFUSAL_TEMPLATE, {}).answer("Q?")
    assert refusing["refused"] is True
    answering = await _adapter(monkeypatch, "It sleeps [1].", {}).answer("Q?")
    assert answering["refused"] is False
