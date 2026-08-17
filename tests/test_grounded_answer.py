"""Tests for two-pass grounded generation."""

import pytest

from deep_context.generation.grounded_answer import generate_grounded_answer
from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE


@pytest.mark.asyncio
async def test_grounded_answer_refuses_anachronism() -> None:
    result = await generate_grounded_answer(
        "What brand of smartphone did Sansa Stark use to text Robb?",
        [{"content": "Sansa Stark lived in the Red Keep and sent ravens."}],
    )
    assert result.refused is True
    assert result.answer == REFUSAL_TEMPLATE
    assert result.reason == "anachronism"


@pytest.mark.asyncio
async def test_grounded_answer_refuses_empty_context() -> None:
    result = await generate_grounded_answer("What is Ice made of?", [])
    assert result.refused is True
    assert result.reason == "no_evidence"
