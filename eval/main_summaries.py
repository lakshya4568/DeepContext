"""Contextual summaries of the deep-ingestion arm.

The application's deep ingestion (``SummaryIngestionPipeline``) asks a small local LLM (``ChunkSummarizer``,
Qwen3-0.6B) for a one-to-two sentence context per child chunk and embeds ``summary + blank line + content``.
That model needs a GPU for thousands of chunks, so the harness splits the work in three steps:

1. ``export``: chunk the benchmark corpora exactly as the fast index does and write the prompts that are not
   cached yet. The prompt is built by the application's own ``ChunkSummarizer._build_prompt``.
2. A GPU machine runs ``eval/vm/summarize_prompts.py`` (the application's own generation code) over that file.
3. ``import``: the raw outputs go into the Postgres cache (``eval_cache.kv``, namespace ``main_summaries``).

``EVAL_INGEST=deep`` then builds the index from the cache: each summary is attached to its child chunk, stored
in ``chunks.summary_text`` and embedded together with the content. A missing summary stops the build. Nothing is
generated on the fly and nothing is replaced by a stand-in; only an empty model output gets the application's
own fallback (the first 25 words), as in ``ChunkSummarizer.summarize_batch``.

    uv run --no-sync python -m eval.main_summaries status --datasets docsqa pdf-quantum pdf-got
    uv run --no-sync python -m eval.main_summaries export --datasets docsqa pdf-quantum pdf-got --out jobs.jsonl
    uv run --no-sync python -m eval.main_summaries import results.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_context.core.types import Chunk, Document
from eval.pg_cache import KV

NAMESPACE = "main_summaries"
FALLBACK_WORDS = (
    25  # ChunkSummarizer.summarize_batch: first 25 words when the model returns nothing
)

# Subsets of the public benchmark runs (REPORT.md "Datasets"): the deep arm must cover exactly the corpora the fast
# run indexed. HotpotQA is a 30,000-document sample (all relevant documents plus seeded distractors); the others are whole.
RUN_SUBSETS: dict[str, dict[str, int]] = {"hotpotqa": {"n_queries": 300, "max_docs": 30_000}}

Staged = tuple[Document, list[Chunk], list[Chunk]]  # (document, parent chunks, child chunks)


@dataclass(frozen=True)
class SummaryRequest:
    key: str
    prompt: str


@dataclass(frozen=True)
class SummaryRecord:
    """Raw output of the application's summariser for one prompt (empty when the model returned nothing)."""

    summary: str
    tokens: int
    model: str


@dataclass(frozen=True)
class AttachReport:
    n_children: int
    n_fallback: int


@dataclass(frozen=True)
class ExportReport:
    total: int
    unique: int
    cached: int
    written: int


class MissingSummariesError(RuntimeError):
    def __init__(self, n_missing: int, n_total: int) -> None:
        self.n_missing = n_missing
        self.n_total = n_total
        super().__init__(
            f"{n_missing} of {n_total} child chunks have no cached summary. Export their prompts "
            "(python -m eval.main_summaries export ...), run eval/vm/summarize_prompts.py on a GPU "
            "machine, then import the results (python -m eval.main_summaries import ...)."
        )


def summary_key(model: str, max_new_tokens: int, prompt: str) -> str:
    """Cache key: the same model, token budget and prompt always give the same key."""
    return hashlib.sha256(f"summary|{model}|{max_new_tokens}|{prompt}".encode()).hexdigest()


def build_requests(
    summarizer: Any, children: Sequence[Chunk], parents_by_id: Mapping[str, Chunk], title: str
) -> list[SummaryRequest]:
    """One request per child, with the prompt ``ChunkSummarizer.summarize_batch`` would build for it."""
    requests = []
    for child in children:
        parent = parents_by_id.get(child.parent_chunk_id) if child.parent_chunk_id else None
        prompt = summarizer._build_prompt(
            child.content,
            section_path=child.section_path,
            document_title=title,
            parent_context=parent.content if parent else None,
        )
        key = summary_key(summarizer.model_name, summarizer.max_summary_tokens, prompt)
        requests.append(SummaryRequest(key, prompt))
    return requests


class SummaryCache:
    """Summaries by request key, on top of a :class:`eval.pg_cache.KV` namespace."""

    def __init__(self, kv: KV) -> None:
        self._kv = kv

    def get_many(self, keys: Iterable[str]) -> dict[str, SummaryRecord]:
        found: dict[str, SummaryRecord] = {}
        for key in keys:
            raw = self._kv.get(key)
            if raw is not None:
                d = json.loads(raw)
                found[key] = SummaryRecord(d["summary"], int(d["tokens"]), d["model"])
        return found

    def put_many(self, records: Mapping[str, SummaryRecord]) -> int:
        """Store the missing keys only (first write wins); returns how many were new."""
        pairs = (
            (key, json.dumps({"summary": r.summary, "tokens": r.tokens, "model": r.model}).encode())
            for key, r in records.items()
        )
        return self._kv.put_many(pairs)


def fallback_summary(content: str) -> tuple[str, int]:
    """The application's stand-in for an empty generation: the first 25 words of the chunk."""
    words = content.strip().split()
    summary = " ".join(words[:FALLBACK_WORDS]) + ("..." if len(words) > FALLBACK_WORDS else "")
    return summary, len(summary.split())


def contextual_text(chunk: Chunk) -> str:
    """Text the application embeds for a child chunk: the summary, a blank line, then the content."""
    return f"{chunk.summary_text}\n\n{chunk.content}" if chunk.summary_text else chunk.content


def attach_summaries(
    staged: Iterable[Staged], cache: SummaryCache, summarizer: Any
) -> AttachReport:
    """Set ``summary_*`` on every child chunk from the cache; all or nothing."""
    plan: list[tuple[Chunk, SummaryRequest]] = []
    for doc, parents, children in staged:
        requests = build_requests(summarizer, children, {p.id: p for p in parents}, doc.title)
        plan.extend(zip(children, requests, strict=True))
    records = cache.get_many({request.key for _, request in plan})
    n_missing = sum(1 for _, request in plan if request.key not in records)
    if n_missing:
        raise MissingSummariesError(n_missing, len(plan))
    n_fallback = 0
    for child, request in plan:
        record = records[request.key]
        summary, tokens = record.summary, record.tokens
        if not summary:
            summary, tokens = fallback_summary(child.content)
            n_fallback += 1
        child.summary_text, child.summary_tokens, child.summary_model = (
            summary,
            tokens,
            record.model,
        )
    return AttachReport(n_children=len(plan), n_fallback=n_fallback)


def export_jobs(
    requests: Sequence[SummaryRequest], cache: SummaryCache, path: Path
) -> ExportReport:
    """Write the prompts that have no cached summary, each once, as JSONL ``{"key", "prompt"}``."""
    unique: dict[str, SummaryRequest] = {}
    for request in requests:
        unique.setdefault(request.key, request)
    cached = cache.get_many(unique)
    todo = [r for key, r in unique.items() if key not in cached]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for request in todo:
            fh.write(json.dumps({"key": request.key, "prompt": request.prompt}, ensure_ascii=False))
            fh.write("\n")
    return ExportReport(len(requests), len(unique), len(cached), len(todo))


def import_results(path: Path, cache: SummaryCache) -> int:
    """Load the GPU side's JSONL into the cache; returns how many summaries were new."""
    records: dict[str, SummaryRecord] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            records[d["key"]] = SummaryRecord(d["summary"], int(d["tokens"]), d["model"])
    return cache.put_many(records)


# --------------------------------------------------------------------------- command line
def task_for(name: str) -> Any:
    """The benchmark task ``name`` exactly as the fast run loaded it (same subset, so the same corpus)."""
    from eval import datasets

    return datasets.build_task(name, **RUN_SUBSETS.get(name, {}))


def requests_for_dataset(name: str) -> list[SummaryRequest]:
    """Requests for every child chunk of a benchmark corpus, chunked exactly as the fast index does."""
    from deep_context.ingestion.chunker import ParentChildChunker
    from deep_context.ingestion.summarizer import ChunkSummarizer
    from eval.adapter import DeepContextAdapter

    summarizer = ChunkSummarizer()
    chunker = ParentChildChunker()
    adapter = DeepContextAdapter("summaries_export", config="rrf")  # only used to stage documents
    requests: list[SummaryRequest] = []
    for ext_id, d in task_for(name).corpus.items():
        staged = adapter.stage_document(chunker, ext_id, d)
        if staged is not None:
            doc, parents, children = staged
            requests += build_requests(summarizer, children, {p.id: p for p in parents}, doc.title)
    return requests


def _open_cache() -> SummaryCache:
    from eval.pg_cache import PgKV

    return SummaryCache(PgKV(NAMESPACE))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("status", "export"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="+", required=True)
        if name == "export":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument(
                "--limit", type=int, default=None, help="first N prompts only (smoke test)"
            )
    sub.add_parser("import").add_argument("results", type=Path)
    args = parser.parse_args(argv)

    cache = _open_cache()
    if args.cmd == "import":
        print(f"{import_results(args.results, cache)} new summaries stored")
        return 0
    per_dataset = {name: requests_for_dataset(name) for name in args.datasets}
    for name, requests in per_dataset.items():
        have = cache.get_many({r.key for r in requests})
        missing = sum(1 for r in requests if r.key not in have)
        print(
            f"{name}: {len(requests)} child chunks, {len(requests) - missing} cached, {missing} missing"
        )
    if args.cmd == "export":
        requests = [r for rs in per_dataset.values() for r in rs]
        if args.limit is not None:
            requests = requests[: args.limit]
        report = export_jobs(requests, cache, args.out)
        print(
            f"wrote {report.written} prompts to {args.out} ({report.unique} unique, {report.cached} cached)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
