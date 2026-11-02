"""The dataset metadata of a retrieval run must name the chunk sizes the chunker really used."""

from __future__ import annotations

from eval.run_retrieval import chunker_label

from deep_context.ingestion.chunker import ParentChildChunker


def test_the_chunker_label_names_the_sizes_the_chunker_really_uses() -> None:
    chunker = ParentChildChunker(
        child_min_tokens=10,
        child_max_tokens=20,
        parent_min_tokens=30,
        parent_max_tokens=40,
        overlap_pct=0.25,
    )
    assert (
        chunker_label(chunker) == "ParentChildChunker (child 10-20 tok, parent 30-40 tok, 25% overlap)"
    )
