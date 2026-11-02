"""Far-reference questions: programmatic checks, the checker's verdict, and the build loop (with a fake model)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from eval.far_test import questions as Q

WINDOW = (1, 20)


def _pages() -> list[str]:
    pages = [f"filler page {n} with nothing special." for n in range(1, 21)]
    pages[2] = "Tyrion Lannister read a book by the fire.\nHe said “I drink and I know things.”"
    pages[15] = (
        "He never saw the wolf. The little man fell flat on his back,\nthe book spinning away."
    )
    pages[17] = "Tyrion walked on. The little man laughed."
    return pages


def _candidate(**over: Any) -> Q.Candidate:
    base: dict[str, Any] = dict(
        id="c1",
        entity="Tyrion",
        names=("Tyrion", "Tyrion Lannister", "the Imp"),
        question="What did Tyrion read by the fire, and what happened to him when the wolf came?",
        answer="He read a book; he fell flat on his back.",
        a_page=3,
        a_quote="Tyrion Lannister read a book by the fire",
        b_page=16,
        b_quote="He never saw the wolf",
        link="'He' on page 16 is the man with the book.",
    )
    base.update(over)
    return Q.Candidate(**base)


# --------------------------------------------------------------------------- text matching
def test_normalize_unifies_quotes_dashes_case_and_line_breaks() -> None:
    assert Q.normalize("Ned’s  “word”\n— ok") == 'ned\'s "word" - ok'


def test_a_quote_is_found_across_a_line_break_and_an_altered_quote_is_not() -> None:
    page = "He said “I drink\nand I know things.”"
    assert Q.quote_on_page("I drink and I know things", page)
    assert not Q.quote_on_page("I drink and I know nothing", page)


def test_a_quote_matches_despite_hyphenation_and_spacing_of_the_pdf_text() -> None:
    page = "the photon- storage scheme (see Fig. 3) is used\nwith atomic en-\nsembles"
    assert Q.quote_on_page("photon-storage scheme (see Fig. 3) is used", page)
    assert Q.quote_on_page("used with atomic ensembles", page)
    assert not Q.quote_on_page("ab", "ab")  # too short to prove anything


def test_names_match_whole_words_and_possessives_only() -> None:
    assert Q.names_on_page(("Ned",), "Ned's sword")
    assert Q.names_on_page(("ned",), "NED spoke")
    assert not Q.names_on_page(("Ned",), "Nedra spoke")
    assert not Q.names_on_page(("Ned",), "He spoke")


def test_a_page_supports_a_quote_when_most_of_its_content_words_appear() -> None:
    quote = "the little man fell flat on his back"
    assert Q.supports_quote(quote, "The little man fell flat on his back, the book spinning away.")
    assert not Q.supports_quote(quote, "A different page about a wolf and a stream.")


# --------------------------------------------------------------------------- programmatic checks
def test_a_good_candidate_has_no_problems() -> None:
    assert Q.problems(_candidate(), _pages(), WINDOW, min_gap=10) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (dict(a_quote="Tyrion Lannister burned the book"), "quote A"),
        (dict(b_quote="He never saw the dragon"), "quote B"),
        (dict(a_page=25), "outside the window"),
        (dict(b_page=5), "pages apart"),
        (dict(b_page=18, b_quote="The little man laughed"), "named near fact B"),
        (dict(question="What did he read by the fire?"), "does not name"),
        (dict(question="What did Tyrion read on page 3?"), "page number"),
        (dict(answer=" "), "empty answer"),
    ],
)
def test_each_programmatic_check_names_what_is_wrong(change: dict[str, Any], expected: str) -> None:
    found = Q.problems(_candidate(**change), _pages(), WINDOW, min_gap=10)
    assert any(expected in p for p in found), found


# --------------------------------------------------------------------------- the checker's verdict
def _review(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        id="c1",
        same_entity=True,
        supported=True,
        a_alone=False,
        b_alone=False,
        pages_for_fact_a=[3],
        pages_for_fact_b=[16],
        comment="ok",
    )
    base.update(over)
    return base


def test_a_passed_review_gives_two_parts_of_page_ids() -> None:
    q, why = Q.finalize(_candidate(), _review(), _pages(), WINDOW, pdf="pdf-got")
    assert why is None and q is not None
    assert q.parts() == (frozenset({"p0003"}), frozenset({"p0016"}))


def test_extra_pages_listed_by_the_checker_join_a_part_only_when_they_support_the_quote() -> None:
    pages = _pages()
    pages[10] = "He never saw the wolf coming, the second time either."
    review = _review(pages_for_fact_b=[16, 11, 5])  # 11 supports the fact, 5 does not
    q, _ = Q.finalize(_candidate(), review, pages, WINDOW, pdf="pdf-got")
    assert q is not None
    assert q.parts()[1] == frozenset({"p0016", "p0011"})


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (dict(same_entity=False), "not the same entity"),
        (dict(supported=False), "not supported"),
        (dict(a_alone=True), "answered from page A alone"),
        (dict(b_alone=True), "answered from page B alone"),
    ],
)
def test_a_failed_review_drops_the_item_with_a_reason(change: dict[str, Any], why: str) -> None:
    q, reason = Q.finalize(_candidate(), _review(**change), _pages(), WINDOW, pdf="pdf-got")
    assert q is None and reason == why


def test_an_item_is_dropped_when_another_page_with_fact_b_names_the_entity() -> None:
    pages = _pages()
    pages[11] = "Tyrion never saw the wolf either."
    review = _review(pages_for_fact_b=[16, 12])
    q, reason = Q.finalize(_candidate(), review, pages, WINDOW, pdf="pdf-got")
    assert q is None and reason == "fact B is also stated near the entity's name"


# --------------------------------------------------------------------------- the build loop
def _writer_item(**over: Any) -> dict[str, Any]:
    c = _candidate(**over)
    return {
        "entity": c.entity, "names": list(c.names), "question": c.question, "answer": c.answer,
        "a_page": c.a_page, "a_quote": c.a_quote, "b_page": c.b_page, "b_quote": c.b_quote,
        "link": c.link,
    }  # fmt: skip


async def test_build_keeps_items_that_pass_every_check_and_counts_why_the_others_were_dropped() -> (
    None
):
    calls: list[str] = []
    writer_items = [_writer_item(), _writer_item(b_page=18, b_quote="The little man laughed")]

    async def fake_generate(model: str, messages: list[dict[str, str]], call: int) -> str:
        calls.append(model)
        if model == "writer":
            return json.dumps({"items": [writer_items[call]]})
        reviews = [_review(id=i) for i in _ids_in(messages)]
        return json.dumps({"reviews": reviews})

    tasks = [Q.Task("Tyrion", 16, (3,), (3,)), Q.Task("Tyrion", 18, (3,), (3,))]
    report = await Q.build_questions(
        pdf="pdf-got", pages=_pages(), window=WINDOW, kind="story", generate=fake_generate,
        writer="writer", checker="checker", min_gap=10, tasks=tasks,
    )  # fmt: skip
    assert [q.entity for q in report.questions] == ["Tyrion"]
    assert report.dropped == {"entity is named near fact B": 1}
    assert "checker" in calls and calls.count("writer") == 2


async def test_a_task_the_writer_skips_is_counted_not_hidden() -> None:
    async def fake_generate(model: str, messages: list[dict[str, str]], call: int) -> str:
        return json.dumps({"items": []})

    report = await Q.build_questions(
        pdf="pdf-got", pages=_pages(), window=WINDOW, kind="story", generate=fake_generate,
        writer="writer", checker="checker", min_gap=10, tasks=[Q.Task("Tyrion", 16, (3,), (3,))],
    )  # fmt: skip
    assert report.questions == [] and report.dropped == {"task skipped by the writer": 1}


def _ids_in(messages: list[dict[str, str]]) -> list[str]:
    user = next(m["content"] for m in messages if m["role"] == "user")
    start = user.index("ITEMS:") + len("ITEMS:")
    return [item["id"] for item in json.loads(user[start:].strip())]


def test_the_question_file_for_the_repository_holds_no_quotes_from_the_book() -> None:
    q, _ = Q.finalize(_candidate(), _review(), _pages(), WINDOW, pdf="pdf-got")
    assert q is not None
    public = q.to_public()
    assert "a_quote" not in public and "b_quote" not in public
    assert public["question"] == q.question
    assert public["part_a"] == [3] and public["part_b"] == [16]


# --------------------------------------------------------------------------- hints made by a script
def test_ranges_compacts_runs_of_consecutive_pages() -> None:
    assert Q.ranges([6, 7, 8, 20, 22, 23]) == "6-8, 20, 22-23"
    assert Q.ranges([]) == ""


# --------------------------------------------------------------------------- tasks chosen by the script
def test_the_task_prompt_shows_page_b_the_page_before_it_and_the_best_pages_a_only() -> None:
    task = Q.Task(name="Tyrion", b_page=16, a_pages=(3, 4), a_best=(3,))
    content = Q.task_messages("story", task, _pages(), WINDOW, min_gap=10)[1]["content"]
    assert '"Tyrion"' in content and "page 16" in content
    assert all(f"[[PAGE {n}]]" in content for n in (3, 15, 16))
    assert "[[PAGE 17]]" not in content and "[[PAGE 4]]" not in content


def test_a_wrong_page_number_is_corrected_from_the_quote_when_the_quote_is_on_exactly_one_page() -> (
    None
):
    fixed = Q.relocate(_candidate(a_page=5), _pages(), WINDOW)
    assert fixed.a_page == 3
    twice = _pages()
    twice[4] = twice[2]
    assert (
        Q.relocate(_candidate(a_page=7), twice, WINDOW).a_page == 7
    )  # two pages hold it: leave it


# --------------------------------------------------------------------------- window mode and page B
def test_page_b_is_corrected_from_its_quote_too() -> None:
    fixed = Q.relocate(_candidate(b_page=9), _pages(), WINDOW)
    assert fixed.b_page == 16


async def test_without_tasks_the_writer_reads_the_whole_window_and_everything_is_still_checked() -> (
    None
):
    seen: list[str] = []

    async def fake_generate(model: str, messages: list[dict[str, str]], call: int) -> str:
        seen.append(model)
        if model == "writer":
            assert "WINDOW" in messages[1]["content"] and "[[PAGE 3]]" in messages[1]["content"]
            return json.dumps(
                {
                    "items": [
                        _writer_item(),
                        _writer_item(b_page=17, b_quote="He never saw the dragon"),
                    ]
                }
            )
        return json.dumps({"reviews": [_review(id=i) for i in _ids_in(messages)]})

    report = await Q.build_questions(
        pdf="pdf-quantum", pages=_pages(), window=WINDOW, kind="thesis", generate=fake_generate,
        writer="writer", checker="checker", min_gap=10, calls=1, per_call=5,
    )  # fmt: skip
    assert [q.entity for q in report.questions] == ["Tyrion"]
    assert report.dropped == {"quote B is not on its page": 1}
    assert seen.count("writer") == 1


# --------------------------------------------------------------------------- tasks from chapter headings
def _book() -> list[str]:
    quiet = "the fire burned low and the night was long. "
    he = "He took his axe and his book. His own fire was his. He said he ran. "
    pages = [quiet] * 12
    pages[0] = "previous | Table of Contents | next BRAN so Bran sat by the fire. " + quiet
    pages[1] = "so Bran talked. " + he
    pages[2] = "previous | Table of Contents | next JON so Jon sat by the fire. " + quiet
    pages[3] = "so Jon talked. " + he
    pages[6] = "previous | Table of Contents | next BRAN so Bran ran. " + quiet
    pages[7] = he + quiet  # inside the second BRAN chapter, the name is missing, pronouns are there
    pages[8] = he + quiet
    pages[9] = "previous | Table of Contents | next JON so Jon sat. " + quiet
    return pages


def test_chapter_tasks_pick_pages_inside_a_chapter_that_lack_the_chapter_name() -> None:
    tasks = Q.pov_tasks(_book(), first_page=1, min_gap=5, min_pronouns=4, per_chapter=3)
    assert tasks == [
        Q.Task(name="Bran", b_page=8, a_pages=(1, 2), a_best=(1, 2)),
        Q.Task(name="Bran", b_page=9, a_pages=(1, 2), a_best=(1, 2)),
    ]


def test_a_chapter_page_needs_enough_pronouns_and_an_earlier_page_that_names_the_entity() -> None:
    assert Q.pov_tasks(_book(), first_page=1, min_gap=5, min_pronouns=99) == []
    # the first BRAN chapter has nothing before it, the JON chapters are too close to their first page
    assert [t.name for t in Q.pov_tasks(_book(), first_page=1, min_gap=5, min_pronouns=4)] == [
        "Bran",
        "Bran",
    ]


# --------------------------------------------------------------------------- chapter excerpts (thesis)
def _thesis() -> list[str]:
    return [
        "Chapter 1 Introduction and outline. The scheme X is summarised here.",
        "Chapter 1: Introduction 2 more of the introduction.",
        "Chapter 2 Storage in ensembles. Theory of the scheme.",
        "Chapter 2: Storage 4 more theory.",
        "Chapter 3 A phase gate. This scheme uses a gate.",
        "Chapter 3: A phase gate 6 more.",
    ]


def test_chapter_excerpts_pair_the_introduction_with_each_later_chapter() -> None:
    excerpts = Q.chapter_excerpts(_thesis(), first_page=21)
    assert [e.chapter for e in excerpts] == [2, 3]
    first = excerpts[0].text
    assert "[[PAGE 21]]" in first and "[[PAGE 22]]" in first and "[[PAGE 23]]" in first
    assert "[[PAGE 25]]" not in first  # chapter 3 is not in the chapter 2 excerpt


async def test_each_excerpt_gets_its_own_writer_call_and_the_items_are_still_checked() -> None:
    seen: list[str] = []

    async def fake_generate(model: str, messages: list[dict[str, str]], call: int) -> str:
        if model == "writer":
            seen.append(messages[1]["content"])
            return (
                json.dumps({"items": [_writer_item()]}) if call == 0 else json.dumps({"items": []})
            )
        return json.dumps({"reviews": [_review(id=i) for i in _ids_in(messages)]})

    excerpts = [
        Q.Excerpt(
            chapter=2,
            text="[[PAGE 3]]\nTyrion Lannister read a book by the fire.\n\n[[PAGE 16]]\nHe never saw the wolf.",
        ),
        Q.Excerpt(chapter=3, text="[[PAGE 5]]\nother"),
    ]
    report = await Q.build_questions(
        pdf="pdf-quantum", pages=_pages(), window=WINDOW, kind="thesis", generate=fake_generate,
        writer="writer", checker="checker", min_gap=10, excerpts=excerpts, calls=0,
    )  # fmt: skip
    assert len(seen) == 2 and "[[PAGE 16]]" in seen[0] and "[[PAGE 5]]" in seen[1]
    assert [q.entity for q in report.questions] == ["Tyrion"]


# --------------------------------------------------------------------------- the evidence is a chunk, not the page
def _long_page() -> str:
    return (
        "Tyrion read his book by the fire. " * 120
        + "The little man laughed at the wolf. " * 120
        + "At last the little man slept soundly under the stars."
    )


def test_the_evidence_text_is_the_chunk_that_holds_the_quote_not_the_whole_page() -> None:
    page = _long_page()
    chunk = Q.evidence_text(page, "At last the little man slept soundly under the stars")
    assert "Tyrion" not in chunk and len(chunk) < len(page) / 2


def test_without_the_quote_the_best_overlapping_chunk_is_used_and_else_the_whole_page() -> None:
    page = _long_page()
    assert "Tyrion" not in Q.evidence_text(
        page, "the little man slept soundly under the stars tonight"
    )
    assert Q.evidence_text(page, "nothing in common with this xyzzy plugh") == page


def test_a_name_elsewhere_on_page_b_does_not_drop_an_item_when_the_evidence_chunk_has_no_name() -> (
    None
):
    pages = _pages()
    pages[15] = _long_page()  # page 16: Tyrion is named in the first part only
    c = _candidate(b_quote="At last the little man slept soundly under the stars")
    assert Q.problems(c, pages, WINDOW, min_gap=10) == []
