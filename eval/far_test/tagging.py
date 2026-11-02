"""Name tagging for the research-plan ingestion, sized for a 0.6B model with a short context.

The research report's first choice is an index from a person (or topic) to the pages about it, including pages
that only say "he". A small model cannot resolve pronouns over a book, so each step here stays inside what it
can see in one prompt:

1. ``Registry`` finds candidate names from the document text alone: capitalised words whose lower-case form
   never occurs in the window (proper nouns, acronyms), without titles and navigation words.
2. For every chunk the model gets the candidates (names in the chunk first, then the frequent names of the
   previous pages) and must answer with one of them. An answer that is not a candidate counts as "no tag":
   the index never holds a name the model invented.
3. ``view_text`` turns a tag into a second, separately searchable text ("Tyrion: He never saw the wolf ...").
   The original chunk vector stays untouched. ``EntityIndex`` maps a name to its pages for query time.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_STOP = frozenset(
    """Lord Lady Ser King Queen Prince Princess Maester Septa Grand House Wall Watch North South East West
    Seven Kingdoms God Gods Night Hand Chapter Table Contents Fig Figure Eq Eqs Sec Section Appendix Ref
    Refs Phys Rev Lett Theorem Lemma Proof""".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_POSSESSIVE = re.compile(r"(?:['’]s|['’])$")
_MID_SENTENCE = re.compile(
    r"(?<=[a-z,] )([A-Z][A-Za-z]{2,})\b"
)  # a capital after a lower-case word or comma
PRONOUN = re.compile(r"\b(?:he|him|his|himself|she|her|hers|herself)\b", re.IGNORECASE)


def words(text: str) -> list[str]:
    """Words of ``text`` with a possessive ending removed ("Tyrion's" -> "Tyrion")."""
    return [_POSSESSIVE.sub("", w) for w in _WORD.findall(text)]


_PROPER_SHAPE = {
    "story": re.compile(r"^[A-Z][a-z]{2,}$"),
    "thesis": re.compile(r"^(?:[A-Z][a-z]{2,}|[A-Z]{2,6})$"),
}

_PROMPTS = {
    "story": ("person", "book", "Arya", "Arya, Sansa, Septa",
              "She dropped her needlework and ran to the window. The old woman scolded her."),
    "thesis": ("topic", "thesis", "EIT", "EIT, Raman, Rydberg",
               "The control pulse slows the light to zero group velocity and maps it onto a spin wave."),
}  # fmt: skip


class Registry:
    """Candidate names of one window, from the text alone."""

    def __init__(self, pages: Sequence[str], kind: str = "story", lookback: int = 12) -> None:
        shape = _PROPER_SHAPE[kind]
        self._lookback = lookback
        self._pages = list(pages)
        self._tokens = [[w for w in words(p) if shape.match(w)] for p in pages]
        lower = {w for p in pages for w in words(p) if w.islower()}
        mid = Counter(w for p in pages for w in _MID_SENTENCE.findall(p))
        # a proper noun is capitalised wherever it stands, so it also occurs in the middle of a sentence
        self.proper = frozenset(
            w
            for tokens in self._tokens
            for w in tokens
            if w.lower() not in lower and w not in _STOP and mid[w] >= 1
        )
        self._counts = [Counter(w for w in tokens if w in self.proper) for tokens in self._tokens]

    def __len__(self) -> int:
        return len(self._counts)

    def tokens_on(self, page_index: int) -> set[str]:
        return set(self._counts[page_index])

    def count_on(self, page_index: int, name: str) -> int:
        return self._counts[page_index][name]

    def pronouns_on(self, page_index: int) -> int:
        return len(PRONOUN.findall(self._pages[page_index]))

    def candidates(self, page_index: int, current: str, top: int = 8) -> list[str]:
        """Names in ``current`` first (in order), then the most frequent names of the previous pages."""
        names: list[str] = []
        for w in words(current):
            if w in self.proper and w not in names:
                names.append(w)
        earlier: Counter[str] = Counter()
        for counts in self._counts[max(0, page_index - self._lookback) : page_index]:
            earlier.update(counts)
        for w, _n in sorted(earlier.items(), key=lambda kv: (-kv[1], kv[0])):
            if w not in names:
                names.append(w)
        return names[:top]


def build_prompt(chunk: str, names: Sequence[str], kind: str = "story") -> str:
    noun, doc, example, example_names, example_text = _PROMPTS[kind]
    system = (
        f"/no_think\nYou name the main {noun} of a passage of a {doc}. Answer with ONE name taken from the "
        "list or from the passage, even if the passage only says he, she, it or a description. "
        "If you cannot tell, answer none."
    )
    user = (
        f"Example\nNames: {example_names}.\n<passage>{example_text}</passage>\nMain {noun}: {example}\n\n"
        f"Now the real one\nNames: {', '.join(names)}.\n<passage>{chunk.strip()}</passage>"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\nMain {noun}:"
    )


def parse_tag(raw: str, names: Sequence[str]) -> str | None:
    """The first candidate named in the first line of the answer; None when there is none."""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    by_lower = {n.lower(): n for n in names}
    for word in words(line):
        if word.lower() in by_lower:
            return by_lower[word.lower()]
    return None


def view_text(name: str, chunk: str, max_chars: int = 300) -> str:
    """The name, then the start of the passage cut at a word boundary: a second text to embed."""
    flat = " ".join(chunk.split())
    if len(flat) > max_chars:
        flat = flat[:max_chars].rsplit(" ", 1)[0]
    return f"{name}: {flat}".rstrip()


class EntityIndex:
    """Name -> pages tagged with it; used at query time to reach pages that never spell the name."""

    def __init__(self) -> None:
        self._pages: dict[str, set[str]] = {}

    def add(self, name: str, page: str) -> None:
        self._pages.setdefault(name, set()).add(page)

    def pages(self, name: str) -> set[str]:
        return set(self._pages.get(name, ()))

    def mentioned_in(self, query: str) -> list[str]:
        """Indexed names that occur in the query as written, in the order they occur."""
        seen: list[str] = []
        for word in words(query):
            if word in self._pages and word not in seen:
                seen.append(word)
        return seen

    def pages_for_query(self, query: str) -> set[str]:
        return set().union(*(self._pages[n] for n in self.mentioned_in(query)))


def restrict_to(ranked: Sequence[str], allowed: set[str]) -> list[str]:
    return [page for page in ranked if page in allowed]


@dataclass(frozen=True)
class ChunkRef:
    page: str
    index: int
    page_index: int  # position of the page in the window (for the registry)
    text: str


@dataclass(frozen=True)
class Tag:
    page: str
    index: int
    name: str | None
    raw: str


def tag_chunks(
    registry: Registry,
    chunks: Sequence[ChunkRef],
    generate: Callable[[list[str]], list[str]],
    batch_size: int = 8,
    kind: str = "story",
) -> list[Tag]:
    """Ask the model for the main person or topic of every chunk, ``batch_size`` prompts at a time."""
    candidates = [registry.candidates(c.page_index, c.text) for c in chunks]
    tags: list[Tag] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        prompts = [build_prompt(c.text, candidates[start + n], kind) for n, c in enumerate(batch)]
        for n, (c, raw) in enumerate(zip(batch, generate(prompts), strict=True)):
            tags.append(Tag(c.page, c.index, parse_tag(raw, candidates[start + n]), raw))
    return tags
