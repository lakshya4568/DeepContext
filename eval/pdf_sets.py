"""Custom long-document case study: the project's two PDFs and their existing 36-query sets.

* ``pdf-got``     - ``Eval 1.pdf`` (755 pages, a novel) with ``tests/eval_dataset.json``
* ``pdf-quantum`` - ``documents/quantum_computing_dissertation.pdf`` (296 pages, a PhD thesis) with
                    ``tests/eval_quantum_dataset.json``

Both query sets were written for this project before the benchmark (they are not public data, and
their labels were not audited by a human). Each PDF page is one corpus document (``p0017`` = PDF
page 17); a query's relevant documents are its ``expected_pages``. The shipped parser also bounds
sections by page, so no chunk crosses a page. Text comes from ``pypdf`` (the project's fast path);
no layout model is used. Extracted text is cached under ``eval/cache`` and is never committed.

Page labels are imprecise (a few are off by a page or point to a neighbouring chapter), so
``label_audit`` reports a transparent heuristic and ``make_tables`` also scores a +/-1-page tolerant
relevance. The heuristic is a diagnostic, not a filter: labels are never changed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eval import config as C
from eval.datasets import Task

SETS: dict[str, dict[str, str]] = {
    "pdf-got": {
        "pdf": "Eval 1.pdf",
        "queries": "tests/eval_dataset.json",
        "title": "A Game of Thrones (G. R. R. Martin), Book One of A Song of Ice and Fire",
    },
    "pdf-quantum": {
        "pdf": "documents/quantum_computing_dissertation.pdf",
        "queries": "tests/eval_quantum_dataset.json",
        "title": "Novel Systems and Methods for Quantum Communication, Quantum Computation, "
        "and Quantum Simulation (A. V. Gorshkov, Harvard PhD thesis, 2009)",
    },
}


def page_id(page: int) -> str:
    return f"p{page:04d}"


def page_number(doc_id: str) -> int:
    return int(doc_id[1:])


def extract_pages(name: str) -> list[str]:
    """Per-page text via pypdf (cached as JSON)."""
    cache = C.DATA_DIR / f"{name}_pages.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import pypdf

    reader = pypdf.PdfReader(C.REPO_ROOT / SETS[name]["pdf"])
    pages = [(p.extract_text() or "") for p in reader.pages]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages))
    return pages


def _words(text: str) -> set[str]:
    return {
        w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("\n", " ")).split() if len(w) > 3
    }


def label_audit(name: str, items: list[dict[str, Any]], pages: list[str]) -> dict[str, Any]:
    """Heuristic label-quality indicator (not a filter; labels are never changed).

    For every expected fact of an answerable item: the share of its content words (length > 3)
    that occur anywhere on the union of the item's expected pages. A fact counts as *supported*
    when that share is >= 0.8. Paraphrased facts under-count and generic facts over-count, so this
    is only a rough signal of how noisy the page labels are.
    """
    covs: list[float] = []
    n_items = n_items_all = 0
    for it in items:
        if it["is_unanswerable"]:
            continue
        have: set[str] = set()
        for p in it["expected_pages"]:
            if 1 <= p <= len(pages):
                have |= _words(pages[p - 1])
        item_covs = []
        for f in it["expected_facts"]:
            w = _words(f)
            if w:
                item_covs.append(len(w & have) / len(w))
        covs += item_covs
        n_items += 1
        n_items_all += bool(item_covs) and all(c >= 0.8 for c in item_covs)
    return {
        "dataset": name,
        "answerable_items": n_items,
        "facts_total": len(covs),
        "mean_fact_word_coverage_on_expected_pages": round(sum(covs) / len(covs), 3)
        if covs
        else None,
        "share_facts_supported_ge_0.8": round(sum(c >= 0.8 for c in covs) / len(covs), 3)
        if covs
        else None,
        "items_with_all_facts_supported": n_items_all,
        "heuristic": "share of an expected fact's content words (len>3) found on the union of the expected pages; supported = share >= 0.8",
    }


def load_pdf_set(name: str) -> Task:
    spec = SETS[name]
    pages = extract_pages(name)
    items = json.loads((C.REPO_ROOT / spec["queries"]).read_text())
    corpus = {
        page_id(i): {"title": spec["title"], "text": text}
        for i, text in enumerate(pages, start=1)
        if text.strip()
    }
    queries, qrels, answers, qtypes = {}, {}, {}, {}
    unanswerable: set[str] = set()
    for it in items:
        qid = it["id"]
        queries[qid] = it["question"]
        answers[qid] = it["ground_truth"]
        qtypes[qid] = it["category"]
        qrels[qid] = {} if it["is_unanswerable"] else {
            page_id(p): 1 for p in it["expected_pages"] if page_id(p) in corpus
        }  # fmt: skip
        if it["is_unanswerable"]:
            unanswerable.add(qid)
    audit = label_audit(name, items, pages)
    return Task(
        name=name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        answers=answers,
        qtypes=qtypes,
        unanswerable=unanswerable,
        meta={
            "source": f"{spec['pdf']} + {spec['queries']} (project files)",
            "license": "copyrighted work supplied by the project owner; text is not redistributed",
            "queries_available": len(items),
            "queries_used": len(items),
            "corpus_used_docs": len(corpus),
            "corpus_total_docs": len(pages),
            "sampling": f"all {len(items)} items ({len(unanswerable)} unanswerable); one document per PDF page",
            "label_provenance": (
                "queries, reference answers and expected_pages were written for this project before "
                "the benchmark; page labels were not human-audited and are imprecise"
            ),
            "label_audit": audit,
        },
    )


def write_label_audits(results_dir: Path) -> None:
    out = {n: load_pdf_set(n).meta["label_audit"] for n in SETS}
    (results_dir / "pdf_label_audit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for n in SETS:
        t = load_pdf_set(n)
        print(
            n,
            len(t.corpus),
            "pages;",
            len(t.queries),
            "queries;",
            json.dumps(t.meta["label_audit"]),
        )
