"""Names in a contextual summary that its model was not shown (they came from its memory, not the document)."""

from __future__ import annotations

import json
from pathlib import Path

from eval import summary_names as S


def _prompt(title: str, context: str, passage: str) -> str:
    return (
        "<|im_start|>system\nNever write meta-phrases. State facts directly.<|im_end|>\n"
        f"<|im_start|>user\n<document>\nDocument: {title}\nSection: s\nContext: {context}\n"
        f"</document>\n\n<passage>\n{passage}\n</passage>\n\n"
        f"Complete the factual situating context:\nIn {title} (s), <|im_end|>\n"
        "<|im_start|>assistant\nIn something"
    )


def test_shown_text_is_the_document_block_and_the_passage_only() -> None:
    shown = S.shown_text(_prompt("The Book", "Ned rode north.", "Jon stayed."))
    assert "Ned rode north." in shown and "Jon stayed." in shown and "The Book" in shown
    assert "Never write" not in shown and "assistant" not in shown


def test_a_prompt_without_the_document_block_is_an_error() -> None:
    try:
        S.shown_text("plain text")
    except ValueError as err:
        assert "document" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_a_name_that_is_in_the_shown_text_is_supported_and_one_that_is_not_is_flagged() -> None:
    shown = "Ned rode north to Winterfell."
    assert S.unsupported_names("He met Ned at Winterfell and Bran there.", shown) == ["Bran"]


def test_the_match_ignores_case_so_a_common_noun_in_the_text_supports_the_name() -> None:
    assert S.unsupported_names("They crossed the Wall at dawn.", "the wall stood tall") == []


def test_words_that_open_a_sentence_and_short_words_are_not_names() -> None:
    summary = "Following the war, Ed met Jon. Later Tyrion left."
    assert S.unsupported_names(summary, "the war Jon") == ["Tyrion"]


def test_the_share_is_counted_per_group_by_the_title_prefix() -> None:
    novel = _prompt("A Game of Thrones", "Ned.", "Ned rode.")
    docs = _prompt("README.md", "Setup.", "Install it.")
    jobs = {"k1": novel, "k2": novel, "k3": docs}
    summaries = {"k1": "Ned rode with Tyrion.", "k2": "Ned rode.", "k3": "Install it."}
    share = S.unsupported_share(jobs, summaries, {"novel": "A Game of Thrones"})
    assert share == {"novel": (1, 2), "other": (0, 1)}


def test_the_command_reads_both_files_and_writes_a_table(tmp_path: Path) -> None:
    prompt = _prompt("A Game of Thrones", "Ned.", "Ned rode.")
    jobs, results = tmp_path / "jobs.jsonl", tmp_path / "results.jsonl"
    jobs.write_text(json.dumps({"key": "k", "prompt": prompt}) + "\n", encoding="utf-8")
    results.write_text(
        json.dumps({"key": "k", "summary": "Ned rode with Tyrion.", "model": "m"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "names.md"
    S.main(
        [
            "--jobs", str(jobs), "--results", str(results),
            "--group", "novel=A Game of Thrones", "--out", str(out),
        ]
    )  # fmt: skip
    text = out.read_text(encoding="utf-8")
    assert "| novel | 1 | 1 | 100% |" in text
