"""How well the local model names the person a passage is about, scored against the chapter headings."""

from __future__ import annotations

from eval.far_test import tag_accuracy as A
from eval.far_test.tagging import ChunkRef


def _book() -> list[str]:
    he = "He took his axe and his book. "
    return [
        "previous | Table of Contents | next BRAN so Bran sat by the fire. "
        + he,  # heading page: not scored
        "so Bran talked and ran. " + he,  # POV Bran, name present
        he + "He said he ran down the hill. ",  # POV Bran, name absent
        "previous | Table of Contents | next JON so Jon sat. " + he,
        he + "He said he ran. ",  # POV Jon, name absent
        "so Jon and Jon and Bran talked. " + he,
    ]


def test_each_page_after_a_heading_belongs_to_the_headed_character_until_the_next_heading() -> None:
    assert A.pov_by_page(_book()) == {1: "Bran", 2: "Bran", 4: "Jon", 5: "Jon"}


def test_a_heading_that_is_not_a_name_is_ignored() -> None:
    pages = ["previous | Table of Contents | next PROLOGUE so it began.", "and so on and so on."]
    assert A.pov_by_page(pages) == {}


def test_the_formal_name_in_a_heading_also_accepts_the_short_name_of_the_text() -> None:
    assert A.accepted_names("Eddard") == {"Eddard", "Ned"}
    assert A.accepted_names("Bran") == {"Bran"}


def test_scores_split_passages_that_name_the_person_from_those_that_do_not() -> None:
    pages = _book()
    refs = [ChunkRef(f"p{i + 1:04d}", 0, i, pages[i]) for i in range(len(pages))]
    tags = {("p0002", 0): "Bran", ("p0003", 0): "Bran", ("p0005", 0): "Bran", ("p0006", 0): "Jon"}
    result = A.score_tags(pages, refs, tags)
    # scored chunks: pages 2, 3 (Bran) and 5, 6 (Jon); the name is absent from pages 3 and 5
    assert result["n"] == 4 and result["n_unnamed"] == 2
    assert result["qwen"] == (
        3,
        4,
        1,
        2,
    )  # right on 2, 3, 6; right on one of the two unnamed passages
    assert result["names"].keys() == {"qwen", "recent", "chunk name or recent"}
