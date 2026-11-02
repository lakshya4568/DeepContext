"""Checks of the generated summaries, per dataset: missing, empty, cut at the token cap, length, unsupported names."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from eval import summary_checks as S


def _prompt(passage: str) -> str:
    return f"<document>\nDocument: Title\nSection: s\nContext: Alpha Beta\nPassage: {passage}</passage>"


def _result(summary: str, tokens: int) -> dict[str, Any]:
    return {"summary": summary, "tokens": tokens}


def test_prompts_without_a_summary_are_counted_as_missing_not_as_empty() -> None:
    jobs = {"a": _prompt("x"), "b": _prompt("y"), "c": _prompt("z")}
    stats = S.check_set(jobs, {"a": _result("In Title, it says x.", 10)}, cap=180)
    assert stats["prompts"] == 3 and stats["summarised"] == 1 and stats["missing"] == 2
    assert stats["empty"] == 0


def test_an_empty_summary_is_counted_empty() -> None:
    jobs = {"a": _prompt("x"), "b": _prompt("y")}
    stats = S.check_set(jobs, {"a": _result("", 0), "b": _result("In Title, y.", 8)}, cap=180)
    assert stats["empty"] == 1


def test_a_summary_that_used_the_whole_token_cap_is_counted_cut() -> None:
    jobs = {"a": _prompt("x"), "b": _prompt("y")}
    stats = S.check_set(
        jobs, {"a": _result("In Title, x.", 180), "b": _result("In Title, y.", 179)}, cap=180
    )
    assert stats["capped"] == 1


def test_the_mean_length_is_over_the_summaries_that_exist() -> None:
    jobs = {"a": _prompt("x"), "b": _prompt("y"), "c": _prompt("z")}
    results = {"a": _result("In Title, x.", 10), "b": _result("In Title, y.", 30)}
    assert S.check_set(jobs, results, cap=180)["mean_tokens"] == pytest.approx(20.0)


def test_results_for_prompts_that_are_not_in_this_set_are_ignored() -> None:
    stats = S.check_set(
        {"a": _prompt("x")}, {"a": _result("In Title, x.", 10), "zzz": _result("n", 5)}, cap=180
    )
    assert stats["summarised"] == 1


def test_a_name_that_is_nowhere_in_the_shown_text_is_an_unsupported_name() -> None:
    jobs = {"a": _prompt("a cat sat"), "b": _prompt("a dog ran")}
    results = {
        "a": _result(
            "In Title, a cat sat near Zorblax Prime.", 12
        ),  # names a thing the text never shows
        "b": _result("In Title, a dog ran near Alpha.", 12),  # Alpha is in the shown context
    }
    stats = S.check_set(jobs, results, cap=180)
    assert stats["unsupported"] == 1 and stats["unsupported_share"] == pytest.approx(0.5)


def test_a_set_without_summaries_has_no_means_or_shares() -> None:
    stats = S.check_set({"a": _prompt("x")}, {}, cap=180)
    assert stats["mean_tokens"] is None and stats["unsupported_share"] is None


def test_the_table_has_one_row_per_set_and_says_not_run_when_nothing_was_written() -> None:
    rows = {
        "done": S.check_set({"a": _prompt("x")}, {"a": _result("In Title, x.", 10)}, cap=180),
        "waiting": S.check_set({"a": _prompt("x")}, {}, cap=180),
    }
    text = S.render(rows)
    assert "| done | 1 | 1 | 0 | 0 | 0 | 10.0 |" in text
    assert "| waiting | 1 | 0 | 1 |" in text and "not run" in text


def test_the_command_reads_jobs_and_results_files_and_writes_the_table(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    results = tmp_path / "results.jsonl"
    jobs.write_text(json.dumps({"key": "a", "prompt": _prompt("x")}) + "\n", encoding="utf-8")
    results.write_text(
        json.dumps({"key": "a", **_result("In Title, x.", 10)}) + "\n", encoding="utf-8"
    )
    out = tmp_path / "out.md"
    S.main(["--set", f"demo={jobs}|{results}", "--cap", "180", "--out", str(out)])
    assert "| demo | 1 | 1 | 0 | 0 | 0 | 10.0 |" in out.read_text(encoding="utf-8")
