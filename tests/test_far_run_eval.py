"""The runner of the far-reference test: how the arms are built from the ranked lists of each index."""

from __future__ import annotations

import json
from pathlib import Path

from eval.far_test import metric as M
from eval.far_test import run_eval as R


def test_arms_are_the_fast_list_the_views_the_name_route_and_their_fusions() -> None:
    fast, views, names = ["p1", "p2", "p3"], ["p3", "p9"], ["p9"]
    arms = R.build_arms(fast, None, views, names)
    assert list(arms) == ["fast", "views", "fast+views", "fast+names", "fast+views+names"]
    assert arms["fast"] == fast and arms["views"] == views
    assert arms["fast+views"] == M.rrf([fast, views])
    assert arms["fast+views+names"] == M.rrf([fast, views, names])


def test_the_notes_arm_is_added_only_when_the_notes_index_exists() -> None:
    arms = R.build_arms(["p1"], ["p2"], ["p3"], [])
    assert list(arms)[:2] == ["fast", "notes"] and arms["notes"] == ["p2"]
    assert arms["fast+names"] == [
        "p1"
    ]  # a question without a known name leaves the fast list as it is


def test_questions_are_read_from_the_public_file_as_parts_of_page_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Who?",
                    "part_a": [3],
                    "part_b": [16, 17],
                    "entity": "Tyrion",
                }
            ]
        )
    )
    [loaded] = R.load_questions(path)
    assert loaded.text == "Who?" and loaded.entity == "Tyrion"
    assert loaded.item.parts == (frozenset({"p0003"}), frozenset({"p0016", "p0017"}))


def test_the_name_route_reaches_part_b_when_a_question_name_is_tagged_on_a_part_b_page() -> None:
    from eval.far_test.tagging import EntityIndex

    index = EntityIndex()
    index.add("Tyrion", "p0016")
    item = (frozenset({"p0003"}), frozenset({"p0016"}))
    assert R.name_route_reaches_b(index, "What did Tyrion read?", item)
    assert not R.name_route_reaches_b(index, "What did Jon read?", item)
