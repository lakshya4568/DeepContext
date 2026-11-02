"""Page windows of the far-reference test: which pages are indexed, and the summary prompts for them."""

from __future__ import annotations

import pytest
from eval.far_test import window as W


def test_a_window_keeps_the_pages_between_start_and_end_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(W, "extract_pages", lambda name: [f"text {i}" for i in range(1, 11)])
    corpus = W.load_window("pdf-got", start=3, end=5)
    assert list(corpus) == ["p0003", "p0004", "p0005"]
    assert corpus["p0004"]["text"] == "text 4"


def test_a_window_page_carries_the_book_title_outside_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(W, "extract_pages", lambda name: ["a", "b", "c"])
    page = W.load_window("pdf-got", start=1, end=3)["p0002"]
    assert "Game of Thrones" in page["title"]
    assert page["title_in_content"] == "0"  # same convention as the full PDF sets


def test_empty_pages_are_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(W, "extract_pages", lambda name: ["a", "   ", "c"])
    assert list(W.load_window("pdf-got", start=1, end=3)) == ["p0001", "p0003"]


def test_a_window_beyond_the_document_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(W, "extract_pages", lambda name: ["a", "b"])
    with pytest.raises(ValueError, match="outside"):
        W.load_window("pdf-got", start=1, end=5)


def test_the_default_windows_cover_both_pdfs_and_are_far_larger_than_forty_pages() -> None:
    assert set(W.WINDOWS) == {"pdf-got", "pdf-quantum"}
    for lo, hi in W.WINDOWS.values():
        assert hi - lo + 1 >= 60


def test_one_summary_request_per_child_chunk_with_the_applications_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Tyrion read his book by the fire. " * 30
    monkeypatch.setattr(W, "extract_pages", lambda name: [text, text + "Then he left."])
    corpus = W.load_window("pdf-got", start=1, end=2)
    staged = W.stage_window(corpus)
    n_children = sum(len(children) for _doc, _parents, children in staged)
    requests = W.summary_requests(staged)
    assert n_children >= 2
    assert len(requests) == n_children
    assert all("Tyrion read his book" in r.prompt for r in requests)
    assert all("Game of Thrones" in r.prompt for r in requests)


def test_child_texts_are_the_children_the_index_stores_for_the_same_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Tyrion read his book by the fire. " * 40
    monkeypatch.setattr(W, "extract_pages", lambda name: [text])
    staged = W.stage_window(W.load_window("pdf-got", start=1, end=1))
    stored = [c.content for _doc, _parents, children in staged for c in children]
    assert W.child_texts(text) == stored
    assert len(stored) >= 2
