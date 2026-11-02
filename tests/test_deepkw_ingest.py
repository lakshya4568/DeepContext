"""``deepkw``: the summary stays in the keyword index but is not embedded (a remedy tested after the deep results)."""

from __future__ import annotations

from types import SimpleNamespace

from eval import adapter as A


def _chunk() -> SimpleNamespace:
    return SimpleNamespace(summary_text="In X (Y), a summary.", content="the chunk")


def test_deep_embeds_the_summary_a_blank_line_and_the_chunk() -> None:
    assert A.embedding_source("deep", _chunk()) == "In X (Y), a summary.\n\nthe chunk"


def test_deepkw_and_fast_embed_the_chunk_alone() -> None:
    assert A.embedding_source("deepkw", _chunk()) == "the chunk"
    assert A.embedding_source("fast", _chunk()) == "the chunk"
