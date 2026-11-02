"""Far-reference test questions: a fact about an entity on one page, and a fact about the same entity many
pages later, on a page that never names it (it says "he", "the old man", "this scheme").

Two Gemini models do the reading, because they can hold a whole window of pages at once while the small
local model cannot: the *writer* proposes items, the *checker* (a different model) reviews them. Before the
checker, plain code rejects every item whose quotes are not on the stated pages, whose later page names the
entity, or whose question does not name it. After the checker, an item survives only if both facts are needed
and every page that states the later fact is free of the entity's names, so that finding that page needs the
link "he = the entity" and not just a word match. Each part keeps all pages that state its fact: a repeated
fact is found on any of them (see ``eval.far_test.metric``).

    uv run --no-sync python -m eval.far_test.questions build --pdf pdf-got
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from eval.far_test.tagging import PRONOUN, words
from eval.far_test.window import child_texts
from eval.pdf_sets import page_id

# --------------------------------------------------------------------------- text matching
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "­": ""})


def normalize(text: str) -> str:
    """Lower case, one space between words, straight quotes and dashes, ligatures expanded (NFKC)."""
    flat = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    return re.sub(r"\s+", " ", flat).strip()


def _squash(text: str) -> str:
    """Letters and digits only: hyphenation, line breaks and spacing of the PDF text no longer matter."""
    return re.sub(r"[^a-z0-9]+", "", normalize(text))


def quote_on_page(quote: str, page: str) -> bool:
    wanted = _squash(quote)
    return len(wanted) >= 12 and wanted in _squash(page)


def names_on_page(names: Sequence[str], text: str) -> bool:
    """True when any name occurs as a whole word (or phrase) in ``text``; possessives count."""
    body = normalize(text)
    return any(
        re.search(rf"(?<!\w){re.escape(normalize(name))}(?!\w)", body)
        for name in names
        if name.strip()
    )


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text)) if len(w) > 3}


def supports_quote(quote: str, page: str, min_share: float = 0.6) -> bool:
    """A page 'states the same fact' when most content words of the quote occur on it."""
    words = _content_words(quote)
    return bool(words) and len(words & _content_words(page)) / len(words) >= min_share


def evidence_text(page: str, quote: str) -> str:
    """The chunk of ``page`` that holds ``quote`` (or overlaps it most); the whole page when none does.

    The search unit is the chunk, so what matters is whether the chunk that states the fact names the entity.
    """
    chunks = child_texts(page)
    for chunk in chunks:
        if quote_on_page(quote, chunk):
            return chunk
    words_wanted = _content_words(quote)
    if words_wanted:
        share, best = max(
            (len(words_wanted & _content_words(c)) / len(words_wanted), c) for c in chunks
        )
        if share >= 0.5:
            return best
    return page


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Candidate:
    """An item proposed by the writer (not yet reviewed)."""

    id: str
    entity: str
    names: tuple[str, ...]
    question: str
    answer: str
    a_page: int
    a_quote: str
    b_page: int
    b_quote: str
    link: str


@dataclass(frozen=True)
class Question:
    """A reviewed item. ``a_pages``/``b_pages`` hold every page that states the fact of that part."""

    id: str
    pdf: str
    entity: str
    names: tuple[str, ...]
    question: str
    answer: str
    a_page: int
    b_page: int
    a_pages: tuple[int, ...]
    b_pages: tuple[int, ...]
    a_quote: str
    b_quote: str

    def parts(self) -> tuple[frozenset[str], frozenset[str]]:
        return (
            frozenset(page_id(p) for p in self.a_pages),
            frozenset(page_id(p) for p in self.b_pages),
        )

    def to_public(self) -> dict[str, Any]:
        """The record kept in the repository: no quotes from the book."""
        return {
            "id": self.id,
            "pdf": self.pdf,
            "entity": self.entity,
            "question": self.question,
            "answer": self.answer,
            "a_page": self.a_page,
            "b_page": self.b_page,
            "part_a": sorted(self.a_pages),
            "part_b": sorted(self.b_pages),
        }


@dataclass
class BuildReport:
    questions: list[Question]
    dropped: dict[str, int]
    n_candidates: int


# --------------------------------------------------------------------------- programmatic checks
def problems(
    c: Candidate, pages: Sequence[str], window: tuple[int, int], min_gap: int
) -> list[str]:
    """What is wrong with a candidate, found by plain code (an empty list = nothing wrong)."""
    lo, hi = window
    if not (lo <= c.a_page <= hi and lo <= c.b_page <= hi):
        return ["page outside the window"]
    found: list[str] = []
    if c.b_page - c.a_page < min_gap:
        found.append("pages apart: too close")
    a_text, b_text = pages[c.a_page - 1], pages[c.b_page - 1]
    if not quote_on_page(c.a_quote, a_text):
        found.append("quote A is not on its page")
    if not quote_on_page(c.b_quote, b_text):
        found.append("quote B is not on its page")
    if not names_on_page(c.names, a_text):
        found.append("entity is not named on page A")
    if names_on_page(c.names, evidence_text(b_text, c.b_quote)):
        found.append("entity is named near fact B")
    if not names_on_page(c.names, c.question):
        found.append("question does not name the entity")
    if re.search(r"\bpages?\s*\d", c.question, re.IGNORECASE):
        found.append("question contains a page number")
    if not c.answer.strip():
        found.append("empty answer")
    return found


def finalize(
    c: Candidate,
    review: dict[str, Any],
    pages: Sequence[str],
    window: tuple[int, int],
    pdf: str,
) -> tuple[Question | None, str | None]:
    """Combine a candidate with the checker's review: a Question, or the reason it was dropped."""
    if not review.get("same_entity"):
        return None, "not the same entity"
    if not review.get("supported"):
        return None, "not supported"
    if review.get("a_alone"):
        return None, "answered from page A alone"
    if review.get("b_alone"):
        return None, "answered from page B alone"
    lo, hi = window

    def with_alternatives(page: int, quote: str, listed: Sequence[Any]) -> tuple[int, ...]:
        extra = {
            int(p)
            for p in listed
            if isinstance(p, int) and lo <= p <= hi and supports_quote(quote, pages[p - 1])
        }
        return tuple(sorted({page} | extra))

    a_pages = with_alternatives(c.a_page, c.a_quote, review.get("pages_for_fact_a", []))
    b_pages = with_alternatives(c.b_page, c.b_quote, review.get("pages_for_fact_b", []))
    if any(names_on_page(c.names, evidence_text(pages[p - 1], c.b_quote)) for p in b_pages):
        return None, "fact B is also stated near the entity's name"
    question = Question(
        id=c.id, pdf=pdf, entity=c.entity, names=c.names, question=c.question, answer=c.answer,
        a_page=c.a_page, b_page=c.b_page, a_pages=a_pages, b_pages=b_pages,
        a_quote=c.a_quote, b_quote=c.b_quote,
    )  # fmt: skip
    return question, None


# --------------------------------------------------------------------------- prompts
_ENTITY = {
    "story": (
        "a character or a named group or object of the story",
        "a pronoun (he, she, they, it), a description (the old man, the boy, the lord) or a role",
    ),
    "thesis": (
        "a named method, device, protocol, system or physical quantity",
        "a pronoun (it, they), a description (this scheme, the protocol, the system, the quantity) "
        "or a symbol without its name",
    ),
}


def window_text(pages: Sequence[str], window: tuple[int, int]) -> str:
    lo, hi = window
    return "\n\n".join(f"[[PAGE {n}]]\n{pages[n - 1].strip()}" for n in range(lo, hi + 1))


# --------------------------------------------------------------------------- tasks chosen by the script
@dataclass(frozen=True)
class Task:
    """One thing for the writer to do: a page B where ``name`` never occurs although it occurs just before."""

    name: str
    b_page: int
    a_pages: tuple[int, ...]  # pages that name the entity, at least ``min_gap`` pages before page B
    a_best: tuple[int, ...]  # the (up to three) of those pages where the word occurs most often


def ranges(pages: Sequence[int]) -> str:
    """'6-8, 20, 22-23' for the sorted pages 6, 7, 8, 20, 22, 23."""
    out: list[str] = []
    run: list[int] = []
    for p in [*sorted(pages), None]:
        if p is not None and run and p == run[-1] + 1:
            run.append(p)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [] if p is None else [p]
    return ", ".join(out)


def _spread(items: list[int], k: int) -> list[int]:
    """``k`` items spread evenly over a sorted list (all of them when there are fewer)."""
    if len(items) <= k:
        return items
    if k == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


_CHAPTER_HEAD = re.compile(r"next\s+([A-Z]{3,})\b")


def chapter_heads(pages: Sequence[str]) -> list[tuple[int, str]]:
    """``(page index, name)`` of every page that starts a chapter headed by a capitalised word (BRAN -> Bran)."""
    heads = []
    for i, text in enumerate(pages):
        found = _CHAPTER_HEAD.search(text[:120])
        if found:
            heads.append((i, found.group(1).capitalize()))
    return heads


def pov_tasks(
    pages: Sequence[str],
    first_page: int,
    min_gap: int,
    min_pronouns: int = 4,
    per_chapter: int = 3,
    top: int = 60,
) -> list[Task]:
    """Tasks from chapters headed by a character's name (the novel's point-of-view chapters).

    Inside a chapter headed ``BRAN`` the pages are about Bran even where the page never spells the name. A page
    qualifies when the chapter name does not occur on it, it has ``min_pronouns`` pronouns, and the name occurs
    on at least one page that is ``min_gap`` or more pages earlier. The heading page itself is left out.
    """
    heads = chapter_heads(pages)
    per_chapter_tasks: list[list[Task]] = []
    for k, (start, name) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(pages)
        counts = [words(text).count(name) for text in pages]
        if sum(counts) < 3:  # a heading such as PROLOGUE is not a name
            continue
        later = [
            j
            for j in range(start + 1, end)
            if counts[j] == 0
            and len(PRONOUN.findall(pages[j])) >= min_pronouns
            and any(counts[i] for i in range(0, j - min_gap + 1))
        ]
        tasks = []
        for j in _spread(later, per_chapter):
            early = [i for i in range(0, j - min_gap + 1) if counts[i]]
            best = sorted(early, key=lambda i: (-counts[i], i))[:3]
            tasks.append(
                Task(
                    name,
                    first_page + j,
                    tuple(first_page + i for i in early),
                    tuple(sorted(first_page + i for i in best)),
                )
            )
        if tasks:
            per_chapter_tasks.append(tasks)
    out: list[Task] = []
    for depth in range(per_chapter):
        for tasks in per_chapter_tasks:
            if depth < len(tasks) and len(out) < top:
                out.append(tasks[depth])
    return out


@dataclass(frozen=True)
class Excerpt:
    """A part of the window shown to the writer: the introduction and one later chapter."""

    chapter: int
    text: str


_CHAPTER_START = re.compile(r"^\s*Chapter\s+(\d+)\b")


def chapter_excerpts(pages: Sequence[str], first_page: int) -> list[Excerpt]:
    """Chapter 1 (the introduction, which summarises the later chapters) paired with each later chapter."""
    chapter_of: list[int | None] = []
    current: int | None = None
    for text in pages:
        found = _CHAPTER_START.match(text)
        if found:
            current = int(found.group(1))
        chapter_of.append(current)

    def block(chapter: int) -> str:
        return "\n\n".join(
            f"[[PAGE {first_page + i}]]\n{text.strip()}"
            for i, text in enumerate(pages)
            if chapter_of[i] == chapter
        )

    later = sorted({c for c in chapter_of if c is not None and c > 1})
    return [Excerpt(c, f"{block(1)}\n\n{block(c)}") for c in later]


def relocate(c: Candidate, pages: Sequence[str], window: tuple[int, int]) -> Candidate:
    """Correct a page number from its quote when the quote is on exactly one page of the window."""
    lo, hi = window

    def fixed(page: int, quote: str) -> int:
        if lo <= page <= hi and quote_on_page(quote, pages[page - 1]):
            return page
        where = [n for n in range(lo, hi + 1) if quote_on_page(quote, pages[n - 1])]
        return where[0] if len(where) == 1 else page

    return replace(c, a_page=fixed(c.a_page, c.a_quote), b_page=fixed(c.b_page, c.b_quote))


def window_messages(
    kind: str, text: str, window: tuple[int, int], n: int, min_gap: int
) -> list[dict[str, str]]:
    """The writer reads the whole window and finds the pairs itself (used when no task can be prepared)."""
    entity_kind, referring = _ENTITY[kind]
    rules = f"""You build a TEST for a document search system. Below is a window of a document; every page
starts with a marker like [[PAGE 57]]. Write {n} test items. Each item links two facts that are far apart.

- The ENTITY is {entity_kind}.
- FACT A is stated on page A, where the entity is NAMED (by one of its names).
- FACT B is stated on a LATER page B, at least {min_gap} pages after page A. The passage that states fact B
  (about 150 words around "b_quote") never names the entity by any of its names: the text points back to it
  only with {referring}. The entity may be named elsewhere on page B.
- The QUESTION names the entity (use one of its names) and can only be answered by combining fact A and
  fact B. Neither fact alone may answer it. It never mentions a page number.
- "names" lists EVERY way the text names the entity (full name, acronym, short form). The passage around
  "b_quote" must contain none of them. Use specific names, not common words.
- "a_quote" and "b_quote" are exact copies (at most 200 characters) of text from page A and page B.
  "b_quote" contains the pronoun or description that points back to the entity.
- Use only what the text says; no outside knowledge. Use different entities and different pages.

Return JSON only: {{"items": [{{"entity": str, "names": [str], "question": str, "answer": str,
"a_page": int, "a_quote": str, "b_page": int, "b_quote": str, "link": str}}]}}
where "link" says in one sentence why the wording on page B refers to the entity."""
    return [
        {"role": "system", "content": "You write careful, verifiable test items from a text."},
        {
            "role": "user",
            "content": f"{rules}\n\nWINDOW (pages {window[0]}-{window[1]}):\n\n{text}",
        },
    ]


def task_messages(
    kind: str, task: Task, pages: Sequence[str], window: tuple[int, int], min_gap: int
) -> list[dict[str, str]]:
    """The writer sees only the few pages of one task: page B, the pages around it, and the best pages A."""
    entity_kind, referring = _ENTITY[kind]
    lo, hi = window
    shown = sorted({*task.a_best, task.b_page - 1, task.b_page})
    excerpt = "\n\n".join(f"[[PAGE {n}]]\n{pages[n - 1].strip()}" for n in shown if lo <= n <= hi)
    rules = f"""You build ONE test item for a document search system from the pages below. Every page starts
with a marker like [[PAGE 57]]. The ENTITY is {entity_kind}; the text names it "{task.name}" (it may have
other names too).

- PAGE B is page {task.b_page}. The word "{task.name}" does not occur on it, but it occurs on the page before.
  If page B is clearly about this entity (the text points back to it only with {referring}), continue.
  Otherwise, or if "{task.name}" is not something that can be referred to like that, return {{"items": []}}.
  Any other name of the entity (a nickname, a short form) must not occur in the passage around "b_quote".
- PAGE A is one of the pages {ranges(task.a_best)}: a page where the entity is named and a fact about it is
  stated. It is at least {min_gap} pages before page B.
- FACT B is stated on page B and is about the entity. FACT A is stated on page A.
- The QUESTION names the entity (use one of its names) and can only be answered by combining fact A and
  fact B. Neither fact alone may answer it. It never mentions a page number.
- "names" lists EVERY way the text names the entity (full name, first name, nickname, title and name).
  The passage around "b_quote" (about 150 words) must contain none of them.
- "a_quote" and "b_quote" are exact copies (at most 200 characters) of text from page A and page B.
  "b_quote" contains the pronoun or description that points back to the entity.
- Use only what the text says; no outside knowledge.

Return JSON only: {{"items": [{{"entity": str, "names": [str], "question": str, "answer": str,
"a_page": int, "a_quote": str, "b_page": int, "b_quote": str, "link": str}}]}} with at most one item, where
"link" says in one sentence why the wording on page B refers to the entity."""
    return [
        {"role": "system", "content": "You write careful, verifiable test items from a text."},
        {"role": "user", "content": f"{rules}\n\nPAGES:\n\n{excerpt}"},
    ]


def checker_messages(
    text: str, window: tuple[int, int], batch: Sequence[Candidate]
) -> list[dict[str, str]]:
    rules = """You review test items for a document search system. The document window is below; every page
starts with a marker like [[PAGE 57]]. For each item decide, using only the text:
- same_entity: does the wording of b_quote (on page b_page) refer to the same entity that is named on a_page?
- supported: are both facts really stated on those pages, and is the answer correct according to the text?
- a_alone: could the question be answered correctly from page a_page alone?
- b_alone: could the question be answered correctly from page b_page alone?
- pages_for_fact_a: ALL pages of the window (up to 6) that state the fact the answer takes from a_quote.
- pages_for_fact_b: ALL pages of the window (up to 6) that state the fact the answer takes from b_quote.
Be strict. Return JSON only: {"reviews": [{"id": str, "same_entity": bool, "supported": bool,
"a_alone": bool, "b_alone": bool, "pages_for_fact_a": [int], "pages_for_fact_b": [int], "comment": str}]}
with one review per item, in any order. The items are listed after the word ITEMS and the window."""
    items = [
        {
            "id": c.id,
            "entity": c.entity,
            "names": list(c.names),
            "question": c.question,
            "answer": c.answer,
            "a_page": c.a_page,
            "a_quote": c.a_quote,
            "b_page": c.b_page,
            "b_quote": c.b_quote,
        }  # fmt: skip
        for c in batch
    ]
    return [
        {"role": "system", "content": "You are a strict, literal reviewer."},
        {
            "role": "user",
            "content": f"{rules}\n\nWINDOW (pages {window[0]}-{window[1]}):\n\n{text}\n\n"
            f"ITEMS:\n{json.dumps(items, ensure_ascii=False)}",
        },
    ]


# --------------------------------------------------------------------------- build
Generate = Callable[[str, list[dict[str, str]], int], Awaitable[str]]


def parse_items(raw: str, prefix: str) -> list[Candidate]:
    """Candidates from the writer's JSON; malformed entries are skipped."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    out: list[Candidate] = []
    for n, it in enumerate(items):
        try:
            out.append(
                Candidate(
                    id=f"{prefix}-{n}",
                    entity=str(it["entity"]).strip(),
                    names=tuple(str(x).strip() for x in it["names"] if str(x).strip()),
                    question=str(it["question"]).strip(),
                    answer=str(it["answer"]).strip(),
                    a_page=int(it["a_page"]),
                    a_quote=str(it["a_quote"]).strip(),
                    b_page=int(it["b_page"]),
                    b_quote=str(it["b_quote"]).strip(),
                    link=str(it.get("link", "")).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


# Fixed before any deep result existed. The novel: the script prepares tasks from the chapters headed by a
# character (page B = a page of that chapter without the name). The thesis: the writer reads the window itself.
MIN_GAP = {
    "pdf-got": 20,
    "pdf-quantum": 10,
}  # the thesis: a different chapter, at least 10 pages on


async def build_questions(
    *,
    pdf: str,
    pages: Sequence[str],
    window: tuple[int, int],
    kind: str,
    generate: Generate,
    writer: str,
    checker: str,
    min_gap: int = 40,
    batch_size: int = 8,
    tasks: Sequence[Task] = (),
    excerpts: Sequence[Excerpt] = (),
    calls: int = 0,
    per_call: int = 8,
) -> BuildReport:
    text = window_text(pages, window)
    dropped: Counter[str] = Counter()
    seen: set[tuple[str, int, int]] = set()
    valid: list[Candidate] = []
    n_candidates = 0
    prompts = [task_messages(kind, task, pages, window, min_gap) for task in tasks]
    prompts += [window_messages(kind, e.text, window, per_call, min_gap) for e in excerpts]
    prompts += [window_messages(kind, text, window, per_call, min_gap)] * calls  # the whole window
    written = await asyncio.gather(*(generate(writer, p, n) for n, p in enumerate(prompts)))
    for n, raw in enumerate(written):
        for c in (relocate(c, pages, window) for c in parse_items(raw, f"{pdf}-t{n}")):
            n_candidates += 1
            key = (c.entity.lower(), c.a_page, c.b_page)
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            found = problems(c, pages, window, min_gap)
            if found:
                dropped[found[0]] += 1
            else:
                valid.append(c)
    skipped = sum(1 for raw in written if not parse_items(raw, "x"))
    if skipped:
        dropped["task skipped by the writer"] = skipped

    questions: list[Question] = []
    for start in range(0, len(valid), batch_size):
        batch = valid[start : start + batch_size]
        raw = await generate(checker, checker_messages(text, window, batch), start // batch_size)
        try:
            reviews = {r["id"]: r for r in json.loads(raw).get("reviews", [])}
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            reviews = {}
        for c in batch:
            review = reviews.get(c.id)
            if review is None:
                dropped["no review"] += 1
                continue
            question, why = finalize(c, review, pages, window, pdf)
            if question is None:
                dropped[str(why)] += 1
            else:
                questions.append(question)
    return BuildReport(questions, dict(dropped), n_candidates)


# --------------------------------------------------------------------------- command line
def vertex_generate(
    runtime: Any, temperature: float = 0.5, thinking_budget: int | None = 2048
) -> Generate:
    async def generate(model: str, messages: list[dict[str, str]], call: int) -> str:
        res = await runtime.generate(
            model=model, messages=messages, temperature=temperature, max_tokens=8192,
            thinking_budget=thinking_budget, json_mode=True, candidate_idx=call, stage="far_questions",
        )  # fmt: skip
        return res.text

    return generate


async def _build_cli(args: argparse.Namespace) -> None:
    from eval import config as C
    from eval.far_test.window import WINDOWS
    from eval.pdf_sets import extract_pages
    from eval.strict_clients import install_strict_patches

    runtime = install_strict_patches()
    pages = extract_pages(args.pdf)
    kind = "story" if args.pdf == "pdf-got" else "thesis"
    lo, hi = WINDOWS[args.pdf]
    min_gap = args.min_gap or MIN_GAP[args.pdf]
    tasks = (
        pov_tasks(pages[lo - 1 : hi], first_page=lo, min_gap=min_gap, per_chapter=6, top=args.tasks)
        if args.pdf == "pdf-got"
        else []
    )
    excerpts = (
        chapter_excerpts(pages[lo - 1 : hi], first_page=lo) if args.pdf == "pdf-quantum" else []
    )
    print(
        f"{args.pdf}: {len(tasks)} chapter tasks, {len(excerpts)} chapter excerpts, "
        f"{args.calls} whole-window calls (pages at least {min_gap} apart)"
    )
    report = await build_questions(
        pdf=args.pdf, pages=pages, window=(lo, hi), kind=kind,
        generate=vertex_generate(runtime), writer=args.writer, checker=args.checker,
        min_gap=min_gap, tasks=tasks, excerpts=excerpts, calls=args.calls,
    )  # fmt: skip
    runtime.cache.flush()
    full = Path(C.CACHE_DIR) / "far_test" / f"questions_{args.pdf}.full.json"
    public = Path(__file__).parent / "data" / f"questions_{args.pdf}.json"
    full.parent.mkdir(parents=True, exist_ok=True)
    public.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(
        json.dumps([q.__dict__ for q in report.questions], indent=1, ensure_ascii=False)
    )
    public.write_text(
        json.dumps([q.to_public() for q in report.questions], indent=1, ensure_ascii=False)
    )
    print(f"{args.pdf}: {report.n_candidates} proposed, {len(report.questions)} kept")
    for reason, n in sorted(report.dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:3d}  {reason}")
    print(runtime.usage_report())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--pdf", required=True, choices=["pdf-got", "pdf-quantum"])
    build.add_argument("--writer", default="gemini-3.7-flash")
    build.add_argument("--checker", default="gemini-3.8-flash")
    build.add_argument(
        "--tasks", type=int, default=60, help="how many chapter tasks the writer gets"
    )
    build.add_argument("--calls", type=int, default=2, help="whole-window writer calls")
    build.add_argument(
        "--min-gap",
        type=int,
        default=None,
        help="default: 20 pages for the novel, 10 for the thesis",
    )
    asyncio.run(_build_cli(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
