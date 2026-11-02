"""Tagging for the research-plan ingestion: which person or topic is a passage about, even when it says only "he"."""

from __future__ import annotations

import pytest
from eval.far_test import tagging as T


def _pages() -> list[str]:
    return [
        "Tyrion Lannister read his book. Then he left. The Imp laughed.",
        "Then the wolf came. He never saw it. Down fell Tyrion.",
        "Then Jon Snow set his mouth in a grim line. Tyrion grinned at him.",
        "He nodded, and then the night fell. Lord Eddard waited. The wolf howled.",
    ]


# --------------------------------------------------------------------------- the name registry
def test_proper_tokens_are_capitalised_words_that_never_occur_in_lower_case() -> None:
    proper = T.Registry(_pages(), kind="story").proper
    assert {"Tyrion", "Lannister", "Imp", "Jon", "Snow"} <= proper
    assert not ({"Then", "He", "The"} & proper)  # 'then', 'he', 'the' occur in lower case


def test_titles_and_navigation_words_are_not_names() -> None:
    assert "Lord" not in T.Registry(_pages(), kind="story").proper


def test_candidates_list_the_names_in_the_passage_first_then_the_frequent_earlier_ones() -> None:
    pages = _pages()
    reg = T.Registry(pages, kind="story")
    # earlier pages: Tyrion twice, Imp and Lannister once each (a tie is broken by name)
    assert reg.candidates(2, pages[2], top=5) == ["Jon", "Snow", "Tyrion", "Imp", "Lannister"]
    assert reg.candidates(2, pages[2], top=4) == ["Jon", "Snow", "Tyrion", "Imp"]


def test_candidates_look_only_at_earlier_pages_never_at_later_ones() -> None:
    pages = _pages()
    reg = T.Registry(pages, kind="story", lookback=12)
    assert "Jon" not in reg.candidates(1, pages[1], top=8)  # Jon first appears on page 2


def test_the_thesis_registry_also_keeps_acronyms() -> None:
    pages = [
        "EIT slows the light. The Raman scheme differs.",
        "Then the pulse is stored, and eit helps.",
    ]
    assert "Raman" in T.Registry(pages, kind="thesis").proper
    assert (
        "EIT" not in T.Registry(pages, kind="thesis").proper
    )  # 'eit' occurs in lower case on page 2


# --------------------------------------------------------------------------- prompt and answer
def test_the_prompt_asks_for_the_main_person_and_lists_the_candidates() -> None:
    prompt = T.build_prompt("He never saw the wolf.", ["Tyrion", "Jon"], kind="story")
    assert prompt.startswith("<|im_start|>system\n/no_think")
    assert "Names: Tyrion, Jon." in prompt and "He never saw the wolf." in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n</think>\nMain person:")
    assert T.build_prompt("x", ["EIT"], kind="thesis").endswith("Main topic:")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Tyrion\nWhat happens: something", "Tyrion"),
        (" tyrion", "Tyrion"),
        (" Jon Snow", "Jon"),  # the first candidate named in the answer, not the first in the list
        (" none", None),
        (" Bran", None),
        ("", None),
    ],
)
def test_an_answer_counts_only_when_it_names_a_candidate(raw: str, expected: str | None) -> None:
    assert T.parse_tag(raw, ["Snow", "Jon", "Tyrion"]) == expected


# --------------------------------------------------------------------------- the extra searchable text
def test_a_view_is_the_name_followed_by_the_start_of_the_passage() -> None:
    text = "He never saw the wolf,\nwhere it was or how it came at him. One moment he was walking."
    view = T.view_text("Tyrion", text, max_chars=40)
    assert view.startswith("Tyrion: He never saw the wolf, where it")
    assert len(view) <= len("Tyrion: ") + 40
    assert not view.endswith(" ")


# --------------------------------------------------------------------------- the entity index
def test_the_index_finds_pages_for_the_names_a_question_contains() -> None:
    index = T.EntityIndex()
    index.add("Tyrion", "p0120")
    index.add("Tyrion", "p0125")
    index.add("Jon", "p0121")
    assert index.mentioned_in("What did Tyrion tell Jon Snow?") == ["Tyrion", "Jon"]
    assert index.pages_for_query("What did Tyrion tell Jon Snow?") == {"p0120", "p0121", "p0125"}
    assert index.pages_for_query("What did tyrion say?") == set()  # names are matched as written
    assert index.pages_for_query("Who is Bran?") == set()


def test_restrict_keeps_the_order_of_the_ranking() -> None:
    assert T.restrict_to(["a", "b", "c", "d"], {"d", "b"}) == ["b", "d"]


# --------------------------------------------------------------------------- tagging a list of chunks
def test_tag_chunks_builds_one_prompt_per_chunk_in_batches_and_keeps_only_valid_names() -> None:
    pages = _pages()
    reg = T.Registry(pages, kind="story")
    chunks = [T.ChunkRef(page="p0002", index=0, page_index=1, text=pages[1])]
    chunks += [T.ChunkRef(page="p0003", index=0, page_index=2, text=pages[2])]
    chunks += [T.ChunkRef(page="p0004", index=0, page_index=3, text=pages[3])]
    seen_batches: list[int] = []

    def fake_generate(prompts: list[str]) -> list[str]:
        seen_batches.append(len(prompts))
        return [" Tyrion"] * len(prompts)

    tags = T.tag_chunks(reg, chunks, fake_generate, batch_size=2)
    assert seen_batches == [2, 1]
    assert [t.name for t in tags] == ["Tyrion", "Tyrion", "Tyrion"]
    assert [t.page for t in tags] == ["p0002", "p0003", "p0004"]
    assert all(t.raw == " Tyrion" for t in tags)


# --------------------------------------------------------------------------- possessives
def test_a_possessive_counts_as_the_name() -> None:
    pages = ["Tyrion's book was old. Then he left.", "He said then that Tyrion left."]
    reg = T.Registry(pages, kind="story")
    assert "Tyrion" in reg.proper and "Tyrion's" not in reg.proper
    assert reg.tokens_on(0) == {"Tyrion"}


def test_the_index_matches_a_possessive_in_the_question() -> None:
    index = T.EntityIndex()
    index.add("Tyrion", "p1")
    assert index.mentioned_in("What did Tyrion\u2019s father say?") == ["Tyrion"]


# --------------------------------------------------------------------------- stricter names and pronouns
def test_words_that_only_start_sentences_are_not_names() -> None:
    pages = [
        "Besides, the fire burned. Besides that, nothing. Said Tyrion to the king.",
        "He and Tyrion left, then.",
    ]
    proper = T.Registry(pages, kind="story").proper
    assert "Tyrion" in proper and "Besides" not in proper


def test_pronouns_are_counted_per_page() -> None:
    reg = T.Registry(["He took his book. She left.", "Nothing here."], kind="story")
    assert (reg.pronouns_on(0), reg.pronouns_on(1)) == (3, 0)
