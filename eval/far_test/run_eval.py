"""Run the far-reference test: fast ingestion against the deep variants, on one window per PDF.

Arms (all dense search over the Gemini embeddings, no reranker, one ranked list of pages per question):

* ``fast``              : the chunks as the fast ingestion stores them.
* ``notes``             : the application's own summariser (local Qwen3-0.6B) in front of every chunk.
* ``views``             : a second index of short texts "<name>: <start of the chunk>", one per chunk the local
                          Qwen tagged with a person or topic; a hit counts for the chunk's page.
* ``fast+views``        : Reciprocal Rank Fusion (k=60, equal votes) of ``fast`` and ``views``.
* ``fast+names``        : fusion of ``fast`` and the *name route*: the pages tagged with a name that the question
                          contains, in the order of the fast ranking.
* ``fast+views+names``  : fusion of all three lists.

The arms, the depth (top 10 pages) and the equal votes were fixed before any deep result existed.

    EVAL_STORE=stock PYTHONPATH=.:src uv run --no-sync python -m eval.far_test.run_eval \\
        --pdfs pdf-got pdf-quantum --out eval/far_test/RESULTS.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval import config as C
from eval.far_test import evaluate as E
from eval.far_test import metric as M
from eval.far_test.run_tagger import chunk_refs, load_tags, tags_path
from eval.far_test.tagging import EntityIndex, restrict_to, view_text
from eval.far_test.window import WINDOWS, load_window
from eval.pdf_sets import SETS, page_id

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
CACHE = Path(C.CACHE_DIR) / "far_test"
DEPTH = 400  # children per query: the whole window, because the name route needs every page ranked
SHORT = {"pdf-got": "got", "pdf-quantum": "qtm"}
Parts = tuple[frozenset[str], ...]


@dataclass(frozen=True)
class LoadedQuestion:
    item: E.Item
    text: str
    entity: str


def load_questions(path: Path) -> list[LoadedQuestion]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        LoadedQuestion(
            E.Item(
                r["id"],
                (
                    frozenset(page_id(p) for p in r["part_a"]),
                    frozenset(page_id(p) for p in r["part_b"]),
                ),
            ),
            r["question"],
            r.get("entity", ""),
        )
        for r in rows
    ]


def build_arms(
    fast: list[str], notes: list[str] | None, views: list[str], names: list[str]
) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {"fast": fast}
    if notes is not None:
        arms["notes"] = notes
    arms["views"] = views
    arms["fast+views"] = M.rrf([fast, views])
    arms["fast+names"] = M.rrf([fast, names])
    arms["fast+views+names"] = M.rrf([fast, views, names])
    return arms


def name_route_reaches_b(index: EntityIndex, query: str, parts: Parts) -> bool:
    return bool(index.pages_for_query(query) & parts[1])


async def ranked_pages(adapter: Any, query: str) -> list[str]:
    children = await adapter.retrieve_children(query, depth=DEPTH, config="dense_only")
    return E.pages_of([doc for doc, _score in adapter.rank_documents(children)])


async def run_pdf(pdf: str, use_notes: bool) -> dict[str, Any]:
    from eval import main_summaries as MS
    from eval.adapter import DeepContextAdapter
    from eval.pg_cache import PgKV

    corpus = load_window(pdf)
    questions = load_questions(DATA / f"questions_{pdf}.json")
    if not questions:
        raise SystemExit(f"{pdf}: no questions yet; run eval.far_test.questions build first")
    _pages, refs = chunk_refs(pdf)
    tags = load_tags(tags_path(pdf))
    title = SETS[pdf]["title"]
    index, views = EntityIndex(), {}
    for ref in refs:
        name = tags.get((ref.page, ref.index))
        if name:
            index.add(name, ref.page)
            views[f"{ref.page}~{ref.index}"] = {
                "title": title,
                "text": view_text(name, ref.text),
                "title_in_content": "0",
            }

    short = SHORT[pdf]
    os.environ["EVAL_INGEST"] = "fast"
    fast = DeepContextAdapter(f"fw_{short}_fast", "dense_only")
    view_index = DeepContextAdapter(f"fw_{short}_views", "dense_only")
    await fast.index(corpus)
    if views:
        await view_index.index(views)

    notes_index = None
    n_fallback = None
    if use_notes:
        cache = MS.SummaryCache(PgKV(MS.NAMESPACE))
        results = CACHE / "summary_results.jsonl"
        if results.exists():
            MS.import_results(results, cache)
        os.environ["EVAL_INGEST"] = "deep"
        try:
            notes_index = DeepContextAdapter(f"fw_{short}_notes", "dense_only", summaries=cache)
            stats = await notes_index.index(corpus)
            n_fallback = (await notes_index._read_meta() or {}).get("n_summary_fallbacks")
            print(f"{pdf}: notes index {stats.n_children} chunks", flush=True)
        except MS.MissingSummariesError as err:
            print(f"{pdf}: notes arm skipped ({err})", flush=True)
            notes_index = None
        finally:
            os.environ["EVAL_INGEST"] = "fast"

    sem = asyncio.Semaphore(4)

    async def one(q: LoadedQuestion) -> tuple[LoadedQuestion, dict[str, list[str]], bool]:
        async with sem:
            fast_pages = await ranked_pages(fast, q.text)
            view_pages = await ranked_pages(view_index, q.text) if views else []
            notes_pages = await ranked_pages(notes_index, q.text) if notes_index else None
        route = restrict_to(fast_pages, index.pages_for_query(q.text))
        arms = build_arms(fast_pages, notes_pages, view_pages, route)
        return q, arms, name_route_reaches_b(index, q.text, q.item.parts)

    done = await asyncio.gather(*(one(q) for q in questions))
    for adapter in (fast, view_index, notes_index):
        if adapter is not None:
            await adapter.close()

    rankings = {arm: {q.item.id: arms[arm] for q, arms, _ in done} for arm in done[0][1]}
    return {
        "pdf": pdf,
        "window": WINDOWS[pdf],
        "questions": questions,
        "rankings": rankings,
        "n_chunks": len(refs),
        "n_tagged": len(views),
        "route_reaches_b": sum(reaches for _q, _a, reaches in done),
        "n_fallback": n_fallback,
        "fast_pages_listed": min(len(r) for r in rankings["fast"].values()),
    }


def write_rows(run: dict[str, Any], k: int) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{run['pdf']}_rows.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for q in run["questions"]:
            row: dict[str, Any] = {
                "id": q.item.id,
                "question": q.text,
                "part_a": sorted(q.item.parts[0]),
                "part_b": sorted(q.item.parts[1]),
            }
            row["arms"] = {
                arm: {
                    "top": ranked[q.item.id][:k],
                    "parts_found": list(M.parts_found(ranked[q.item.id], q.item.parts, k)),
                }
                for arm, ranked in run["rankings"].items()
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


READ_FIRST = """**Read this first.** This is a pilot. The questions are the few that survived strict checks (written by
gemini-3.7-flash, reviewed by gemini-3.8-flash), so the counts are tiny and no difference below is evidence of
anything. The application's own note-writer (arm `notes`) was not run: it freezes an 8 GB Mac. Search is dense
only, top pages, no reranker, one run."""


def render_report(
    runs: Sequence[dict[str, Any]], k: int = 10, deeper: int = 20, extra: str = ""
) -> str:
    out = ["# Far-reference test: fast ingestion against deep ingestion (local Qwen)\n", READ_FIRST]
    for run in runs:
        items = [q.item for q in run["questions"]]
        lo, hi = run["window"]
        out.append(f"\n## {run['pdf']}: pages {lo}-{hi}, {len(items)} questions\n")
        notes = (
            ""
            if run["n_fallback"] is None
            else f", {run['n_fallback']} empty note(s) replaced by the fallback"
        )
        out.append(
            f"{run['n_chunks']} chunks; the local model tagged {run['n_tagged']} of them with a name or topic{notes}. "
            f"The name route reaches a part-B page for {run['route_reaches_b']} of {len(items)} questions "
            "(it is the only way a question that names the entity can reach a page that never spells it).\n"
        )
        for depth in (k, deeper):
            results = E.score_arms(items, run["rankings"], depth, baseline="fast")
            out.append(E.render_table(results, run["pdf"], depth))
    out.append(extra)
    return "\n".join(out)


def tagging_section() -> str:
    """Accuracy of the local model's tags on the novel, scored against the chapter headings."""
    from eval.far_test import tag_accuracy

    return (
        "\n## Does the local model name the right person? (novel)\n\n"
        + tag_accuracy.report()
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pdfs", nargs="+", default=["pdf-got", "pdf-quantum"], choices=sorted(WINDOWS)
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "RESULTS.md")
    parser.add_argument("--no-notes", action="store_true", help="skip the app's summariser arm")
    args = parser.parse_args(argv)

    async def run_all() -> list[dict[str, Any]]:
        # one event loop for all PDFs: the Vertex client of the first PDF is reused by the second
        return [await run_pdf(pdf, use_notes=not args.no_notes) for pdf in args.pdfs]

    runs = asyncio.run(run_all())
    for run in runs:
        print(f"{run['pdf']}: rows in {write_rows(run, k=20)}")
    extra = tagging_section() if "pdf-got" in args.pdfs else ""
    args.out.write_text(render_report(runs, extra=extra), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
