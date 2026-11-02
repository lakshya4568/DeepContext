"""GPU-side step of the deep-ingestion arm: run the application's ChunkSummarizer over a file of prompts.

Needs ``src/deep_context`` on the path plus torch, transformers and pydantic-settings. No database and no
credentials. Input: JSONL ``{"key", "prompt"}`` (written by ``python -m eval.main_summaries export``).
Output: JSONL ``{"key", "summary", "tokens", "model"}``, appended after every batch, so a stopped run
resumes where it ended. An empty generation is written as an empty summary; the harness then applies the
application's own fallback (first 25 words) when it attaches the summary.

    PYTHONPATH=src python summarize_prompts.py jobs.jsonl results.jsonl --batch-size 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def summarize_jobs(
    jobs: Sequence[dict[str, Any]],
    done: set[str],
    summarizer: Any,
    batch_size: int,
    emit: Callable[[dict[str, Any]], None],
) -> int:
    """Generate a summary for every job whose key is not in ``done``; returns how many were run.

    The longest prompts go first: padding is smallest within a batch of similar lengths, and a
    memory problem shows up in the first batch instead of the last.
    """
    todo = sorted(
        (job for job in jobs if job["key"] not in done),
        key=lambda j: len(j["prompt"]),
        reverse=True,
    )
    started = time.perf_counter()
    finished = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        outputs = summarizer._generate_batch_sync([job["prompt"] for job in batch])
        for job, (summary, tokens) in zip(batch, outputs, strict=True):
            emit(
                {
                    "key": job["key"],
                    "summary": summary,
                    "tokens": tokens,
                    "model": summarizer.model_name,
                }
            )
        finished += len(batch)
        elapsed = time.perf_counter() - started
        print(
            f"{finished}/{len(todo)} done, {elapsed:.0f}s, {finished / elapsed:.2f} prompts/s",
            flush=True,
        )
    return finished


def load_done(results: Path) -> set[str]:
    if not results.exists():
        return set()
    with results.open(encoding="utf-8") as fh:
        return {json.loads(line)["key"] for line in fh if line.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("jobs", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke test)")
    args = parser.parse_args(argv)

    with args.jobs.open(encoding="utf-8") as fh:
        jobs = [json.loads(line) for line in fh if line.strip()]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    from deep_context.ingestion.summarizer import ChunkSummarizer

    summarizer = ChunkSummarizer(device=args.device)
    if not asyncio.run(summarizer._ensure_model_loaded()):
        print("The summarisation model did not load; nothing was generated.", file=sys.stderr)
        return 2
    print(
        f"model {summarizer.model_name} on {summarizer._device}, batch {args.batch_size}",
        flush=True,
    )

    done = load_done(args.results)
    with args.results.open("a", encoding="utf-8") as out:

        def emit(record: dict[str, Any]) -> None:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

        n = summarize_jobs(jobs, done, summarizer, args.batch_size, emit)
    print(f"finished {n} new summaries ({len(done)} were already in {args.results})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
