"""Summaries from a self-hosted model: the application's own prompt, sent to an OpenAI-compatible completions server."""

from __future__ import annotations

import json
from typing import Any

import httpx
from eval.vm import summarize_vllm as V

OPEN = "<|im_start|>assistant\n<think>\n</think>\n"


def _prompt(prefix: str = "In Doc (Page 1), ") -> str:
    return f"<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nchunk<|im_end|>\n{OPEN}{prefix}"


def test_the_anchor_the_prompt_ends_with_is_put_back_in_front_of_the_continuation() -> None:
    assert (
        V.assemble(_prompt(), "the page lists rules.") == "In Doc (Page 1), the page lists rules."
    )


def test_an_output_that_already_starts_with_in_is_used_as_it_is() -> None:
    assert V.assemble(_prompt(), "In Doc, it lists rules.") == "In Doc, it lists rules."


def test_a_prompt_without_the_anchor_keeps_the_raw_output() -> None:
    assert V.assemble("plain prompt", "text") == "text"


async def test_each_job_gets_one_record_and_finished_keys_are_skipped() -> None:
    jobs = [{"key": f"k{i}", "prompt": _prompt()} for i in range(3)]
    records: list[dict[str, Any]] = []

    async def complete(prompt: str) -> tuple[str, int]:
        return "it lists rules.", 7

    n = await V.summarize_jobs(
        jobs,
        done={"k1"},
        complete=complete,
        clean=str.strip,
        emit=records.append,
        model_name="Qwen/Qwen3.5-9B",
    )
    assert n == 2
    assert sorted(r["key"] for r in records) == ["k0", "k2"]
    assert records[0] == {
        "key": records[0]["key"],
        "summary": "In Doc (Page 1), it lists rules.",
        "tokens": 7,
        "model": "Qwen/Qwen3.5-9B",
    }


async def test_an_empty_generation_is_recorded_as_empty_not_dropped() -> None:
    records: list[dict[str, Any]] = []

    async def complete(prompt: str) -> tuple[str, int]:
        return "", 0

    await V.summarize_jobs(
        [{"key": "k", "prompt": "plain"}],
        done=set(),
        complete=complete,
        clean=str.strip,
        emit=records.append,
        model_name="m",
    )
    assert records[0]["summary"] == "" and records[0]["tokens"] == 0


async def test_the_completions_request_is_greedy_and_carries_the_raw_prompt_and_the_token_cap() -> (
    None
):
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/completions"
        return httpx.Response(
            200, json={"choices": [{"text": " it lists rules."}], "usage": {"completion_tokens": 5}}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    ) as http:
        complete = V.make_complete(http, served_model="qwen3.5-9b", max_tokens=180)
        text, tokens = await complete(_prompt())
    assert (text, tokens) == (" it lists rules.", 5)
    assert seen[0]["model"] == "qwen3.5-9b" and seen[0]["temperature"] == 0
    assert seen[0]["max_tokens"] == 180 and seen[0]["prompt"] == _prompt()
