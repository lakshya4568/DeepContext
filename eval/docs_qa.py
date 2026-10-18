"""Documentation-QA task: closed-book questions over the repo's own markdown documentation.

Documents are markdown *sections* (H1-H3), so retrieval has real granularity; relevance labels are
the sections that contain each gold evidence quote (verified, never hand-tagged).
"""

from __future__ import annotations

import re
from pathlib import Path

from eval import config as C
from eval.datasets import Task
from eval.docs_qa_items import ITEMS

DOC_FILES = [
    "README.md", "AGENTS.md", "RERANKER_UPGRADE.md", "Rag eval.md",
    "docs/ARCHITECTURE.md", "docs/CACHE_SCHEDULER_AGENTIC.md",
    "docs/CRITICAL_ASSESSMENT_AND_SCOPE.md", "docs/PRD.md", "docs/RAG_QUALITY_NEXT.md",
    "docs/SYSTEM_HANDOFF.md", "docs/TECH_STACK.md", "docs/VERIFICATION_AND_SOURCES.md",
]  # fmt: skip

_HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip()


def build_sections(root: Path = C.REPO_ROOT, min_chars: int = 80) -> dict[str, dict[str, str]]:
    """Split each doc into H1-H3 sections (headings inside code fences are ignored)."""
    sections: dict[str, dict[str, str]] = {}
    for rel in DOC_FILES:
        lines = (root / rel).read_text(encoding="utf-8").splitlines()
        trail: list[str] = []
        buf: list[str] = []
        in_fence = False
        n = 0

        def flush() -> None:
            nonlocal n
            body = "\n".join(buf).strip()
            if len(body) >= min_chars:
                title = " > ".join(trail) if trail else rel
                sections[f"{rel}#{n:03d}"] = {"title": f"{rel} › {title}", "text": body}
                n += 1

        for line in lines:
            if _FENCE.match(line):
                in_fence = not in_fence
            m = None if in_fence else _HEADING.match(line)
            if m:
                flush()
                buf = [line]
                level = len(m.group(1))
                trail = trail[: level - 1] + [re.sub(r"[`*_]", "", m.group(2)).strip()]
            else:
                buf.append(line)
        flush()
    return sections


def locate(quote: str, sections: dict[str, dict[str, str]]) -> list[str]:
    q = normalize(quote)
    return [sid for sid, s in sections.items() if q in normalize(s["title"] + " " + s["text"])]


def verify_gold() -> list[str]:
    """Return a list of problems (empty = every evidence quote exists in the corpus)."""
    sections = build_sections()
    problems = []
    for it in ITEMS:
        for quote in it["ev"]:
            if not locate(quote, sections):
                problems.append(f"{it['id']}: quote not found: {quote[:90]!r}")
    return problems


def load_docs_qa() -> Task:
    sections = build_sections()
    queries, qrels, answers, qtypes = {}, {}, {}, {}
    unanswerable: set[str] = set()
    for it in ITEMS:
        qid = it["id"]
        queries[qid] = it["q"]
        answers[qid] = it["a"]
        qtypes[qid] = it["cat"]
        rel: dict[str, int] = {}
        for quote in it["ev"]:
            hits = locate(quote, sections)
            if not hits:
                raise ValueError(f"{qid}: evidence quote not found in docs: {quote[:80]!r}")
            for sid in hits:
                rel[sid] = 1
        qrels[qid] = rel
        if it["cat"] == "unanswerable" or (it["cat"] == "false_prem" and not it["ev"]):
            unanswerable.add(qid)
    return Task(
        name="docsqa",
        corpus=sections,
        queries=queries,
        qrels=qrels,
        answers=answers,
        qtypes=qtypes,
        unanswerable=unanswerable,
        meta={
            "source": "repo documentation on branch eval/public-benchmarks ("
            + ", ".join(DOC_FILES)
            + ")",
            "license": "repository documentation (Apache-2.0 repo)",
            "queries_available": len(queries),
            "queries_used": len(queries),
            "corpus_used_docs": len(sections),
            "sampling": "all items; sections = markdown H1-H3 blocks >= 80 chars",
            "gold_authoring": (
                "questions and answers authored by the harness author (an AI assistant) from the docs; "
                "every evidence quote is machine-verified verbatim; answers were NOT human-reviewed"
            ),
        },
    )


if __name__ == "__main__":
    probs = verify_gold()
    secs = build_sections()
    print(f"sections: {len(secs)} | items: {len(ITEMS)}")
    print("\n".join(probs) if probs else "all evidence quotes verified")
