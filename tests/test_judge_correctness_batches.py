"""The rubric judge runs all its batches in one event loop (a new loop per batch broke the second batch)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from eval import judge_correctness as J


class _FakeRuntime:
    class cache:  # noqa: N801 - mirrors the attribute of the real runtime
        @staticmethod
        def flush() -> None:
            return None

    @staticmethod
    def usage_report() -> str:
        return ""


def test_all_batches_of_one_run_use_the_same_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "run_tag": "t",
            "dataset": "d",
            "config": "c",
            "qid": q,
            "query": "q?",
            "reference": "r",
            "answer": "a",
            "unanswerable": False,
        }  # fmt: skip
        for q in ("a", "b", "c")
    ]
    (tmp_path / "raw_generation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    loops: list[asyncio.AbstractEventLoop] = []

    async def fake_grade_all(
        batch_rows: list[dict[str, Any]], model: str, thinking_budget: int | None, concurrency: int
    ) -> list[dict[str, Any]]:
        loops.append(asyncio.get_running_loop())
        return [{"score": 1.0, "verdict": "correct", "reason": "ok"} for _ in batch_rows]

    monkeypatch.setattr(J, "grade_all", fake_grade_all)
    monkeypatch.setattr(J, "get_runtime", lambda: _FakeRuntime)
    J.score_results(tmp_path, "judge-x", None, 2, 1, None, None)
    assert len(loops) == 3 and len({id(loop) for loop in loops}) == 1
