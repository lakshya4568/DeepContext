"""GPU-side step with a bigger summariser: the application's prompt, answered by a self-hosted model.

``eval/vm/summarize_prompts.py`` runs the 0.6B model through transformers. This script sends the *same prompts*
(written by ``python -m eval.main_summaries export``, with ``SUMMARY_MODEL`` set to the bigger model so the cache keys
name it) to an OpenAI-compatible completions server such as vLLM, so that only the model changes. The output is
post-processed exactly as ``ChunkSummarizer._generate_batch_sync`` does (the anchor the prompt ends with is put back in
front of the continuation, then ``_clean_and_complete_summary``). Needs ``src/deep_context``, ``httpx`` and a running
server; no database and no credentials.

    SUMMARY_MODEL=Qwen/Qwen3.5-9B PYTHONPATH=src python summarize_vllm.py jobs_9b.jsonl results_9b.jsonl \\
        --url http://127.0.0.1:8000 --served-model qwen3.5-9b --model-name Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

ASSISTANT_OPEN = "<|im_start|>assistant\n<think>\n</think>\n"


def assemble(prompt: str, raw: str) -> str:
    """Anchor + continuation, as the application does before cleaning."""
    prefix = prompt.split(ASSISTANT_OPEN)[-1] if ASSISTANT_OPEN in prompt else ""
    return prefix + raw if not raw.startswith("In ") else raw


def make_complete(
    http: httpx.AsyncClient, served_model: str, max_tokens: int
) -> Callable[[str], Awaitable[tuple[str, int]]]:
    async def complete(prompt: str) -> tuple[str, int]:
        response = await http.post(
            "/v1/completions",
            json={
                "model": served_model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"], int(data.get("usage", {}).get("completion_tokens", 0))

    return complete


async def summarize_jobs(
    jobs: Sequence[dict[str, Any]],
    done: set[str],
    complete: Callable[[str], Awaitable[tuple[str, int]]],
    clean: Callable[[str], str],
    emit: Callable[[dict[str, Any]], None],
    model_name: str,
    concurrency: int = 32,
) -> int:
    """Summarise every job whose key is not in ``done``; returns how many were run."""
    todo = [job for job in jobs if job["key"] not in done]
    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    finished = 0

    async def one(job: dict[str, Any]) -> None:
        nonlocal finished
        async with sem:
            raw, tokens = await complete(job["prompt"])
        summary = clean(assemble(job["prompt"], raw)) if raw.strip() else ""
        emit({"key": job["key"], "summary": summary, "tokens": tokens, "model": model_name})
        finished += 1
        if finished % 100 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"{finished}/{len(todo)} done, {elapsed:.0f}s, {finished / elapsed:.2f} prompts/s",
                flush=True,
            )

    await asyncio.gather(*(one(job) for job in todo))
    return len(todo)


def load_done(results: Path) -> set[str]:
    if not results.exists():
        return set()
    with results.open(encoding="utf-8") as fh:
        return {json.loads(line)["key"] for line in fh if line.strip()}


async def _run(args: argparse.Namespace) -> int:
    from deep_context.ingestion.summarizer import ChunkSummarizer

    with args.jobs.open(encoding="utf-8") as fh:
        jobs = [json.loads(line) for line in fh if line.strip()]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    done = load_done(args.results)
    async with httpx.AsyncClient(base_url=args.url.rstrip("/"), timeout=600.0) as http:
        complete = make_complete(http, args.served_model, args.max_tokens)
        with args.results.open("a", encoding="utf-8") as out:

            def emit(record: dict[str, Any]) -> None:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

            n = await summarize_jobs(
                jobs,
                done,
                complete,
                ChunkSummarizer._clean_and_complete_summary,
                emit,
                args.model_name,
                args.concurrency,
            )
    print(f"finished {n} new summaries ({len(done)} were already in {args.results})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("jobs", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", required=True, help="name the server answers to")
    parser.add_argument("--model-name", required=True, help="model id written into every record")
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke test)")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
