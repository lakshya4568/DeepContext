"""Page windows of the far-reference test, kept small so the local 0.6B model can ingest them.

* ``pdf-got``     : PDF pages 6-125 (the prologue and the first chapters; the story starts on page 6).
* ``pdf-quantum`` : PDF pages 21-97 (chapters 1-8; the bibliography and the appendices follow).

Page ids are the PDF page numbers of the full PDF sets (``p0042`` = PDF page 42), so a page keeps its name
in every index. Chunking is the application's (``ParentChildChunker``, one page = one parent).

    uv run --no-sync python -m eval.far_test.window export --out eval/cache/far_test/summary_jobs.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from eval import main_summaries as MS
from eval.pdf_sets import SETS, extract_pages, page_id

WINDOWS: dict[str, tuple[int, int]] = {"pdf-got": (6, 125), "pdf-quantum": (21, 97)}

Corpus = dict[str, dict[str, str]]


def load_window(name: str, start: int | None = None, end: int | None = None) -> Corpus:
    """``{page id: {title, text, title_in_content}}`` for the PDF pages ``start``..``end`` (inclusive)."""
    default_start, default_end = WINDOWS[name]
    lo, hi = start or default_start, end or default_end
    pages = extract_pages(name)
    if lo < 1 or hi > len(pages) or lo > hi:
        raise ValueError(f"window {lo}-{hi} is outside {name} ({len(pages)} pages)")
    title = SETS[name]["title"]
    return {
        page_id(n): {"title": title, "text": pages[n - 1], "title_in_content": "0"}
        for n in range(lo, hi + 1)
        if pages[n - 1].strip()
    }


def child_texts(text: str) -> list[str]:
    """The child chunks of one page, in order, without the adapter (the local Qwen process stays small).

    Same steps as ``DeepContextAdapter.stage_document`` for a page whose title is not part of the text.
    """
    from deep_context.ingestion.chunker import ParentChildChunker
    from deep_context.ingestion.parser import DocumentParser

    sections = DocumentParser.parse(text.strip(), doc_type="text")
    _parents, children = ParentChildChunker().chunk_sections("window-page", sections)
    return [child.content for child in children]


def stage_window(corpus: Corpus) -> list[MS.Staged]:
    """Chunk every page exactly as the fast index does (parents, children, deterministic ids)."""
    from deep_context.ingestion.chunker import ParentChildChunker
    from eval.adapter import DeepContextAdapter

    chunker = ParentChildChunker()
    adapter = DeepContextAdapter("far_window_staging", config="rrf")  # only used to stage documents
    staged = (adapter.stage_document(chunker, ext_id, d) for ext_id, d in corpus.items())
    return [s for s in staged if s is not None]


def summary_requests(staged: Sequence[MS.Staged]) -> list[MS.SummaryRequest]:
    """The application's own summariser prompt for every child chunk (see ``eval.main_summaries``)."""
    from deep_context.ingestion.summarizer import ChunkSummarizer

    summarizer = ChunkSummarizer()
    requests: list[MS.SummaryRequest] = []
    for doc, parents, children in staged:
        requests += MS.build_requests(summarizer, children, {p.id: p for p in parents}, doc.title)
    return requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("cmd", choices=["export"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from eval.pg_cache import PgKV

    requests: list[MS.SummaryRequest] = []
    for name in WINDOWS:
        staged = stage_window(load_window(name))
        per_window = summary_requests(staged)
        print(f"{name}: {len(staged)} pages, {len(per_window)} child chunks")
        requests += per_window
    report = MS.export_jobs(requests, MS.SummaryCache(PgKV(MS.NAMESPACE)), args.out)
    print(
        f"wrote {report.written} prompts to {args.out} ({report.unique} unique, {report.cached} cached)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
