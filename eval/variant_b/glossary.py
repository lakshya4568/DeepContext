"""B1 (extract entries with evidence) and B2 (merge) of variant B, and the per-chunk glossary slice.

Every entry must be backed by a quote that really occurs in the passage and must name something that occurs there;
entries that fail either test are dropped (and counted), so the glossary cannot contain what the text never said.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from eval.variant_b import prompts
from eval.variant_b.llm import SlmClient
from eval.variant_b.profiles import Profile

log = logging.getLogger("eval.variant_b.glossary")

STOP_NAMES = frozenset(
    "it he she they we i you this that these those the a an his her its their who which".split()
)
MAX_NAMES = 4
MAX_DEF_CHARS = 220
SLICE_DEF_CHARS = 160


@dataclass
class Entry:
    names: list[str]
    kind: str
    definition: str
    quote: str
    first_seen: int  # span position (0-based) in the source document where it first appears
    timeline: list[tuple[int, str]] = field(
        default_factory=list
    )  # (span position, stated state change)
    mentions: int = 1
    support: str = "exact"  # how the quote was found in the passage: exact | fuzzy

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Entry:
        return cls(**{**d, "timeline": [(int(p), str(s)) for p, s in d.get("timeline", [])]})


# ---------------------------------------------------------------------------------------- text helpers
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*`#_>|]", " ", s)).strip().lower()


def name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def usable_name(name: str) -> bool:
    n = name.strip()
    if not n or n.lower() in STOP_NAMES:
        return False
    if len(n) >= 3:
        return True
    return len(n) == 2 and (n.isupper() or any(c.isdigit() for c in n))


def specific_name(name: str) -> bool:
    """False for a plain lowercase word such as "context" or "agent": it names different things in different places."""
    n = name.strip()
    return " " in n or any(c.isupper() or c.isdigit() or c in "_./-:" for c in n)


def match_names(e: Entry) -> list[str]:
    """Names to look for in a chunk. An entry with many aliases may have lumped unrelated things together: then only
    its specific names count."""
    names = [n for n in e.names if usable_name(n)]
    return names if len(names) <= 3 else [n for n in names if specific_name(n)]


@lru_cache(maxsize=8192)
def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(name.strip()) + r"(?![A-Za-z0-9])", re.IGNORECASE
    )


def count_name(name: str, text: str) -> int:
    return len(_name_pattern(name).findall(text))


def quote_support(quote: str, span: str) -> str:
    """``exact`` if the quote occurs in the passage, ``fuzzy`` if >= 80% of it does as one block, else ``failed``."""
    nq, ns = norm_text(quote), norm_text(span)
    if len(nq) < 8:
        return "failed"
    if nq in ns:
        return "exact"
    block = SequenceMatcher(None, nq, ns, autojunk=False).find_longest_match(0, len(nq), 0, len(ns))
    return "fuzzy" if block.size >= max(12, 0.8 * len(nq)) else "failed"


def split_spans(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split at paragraph (else line, else sentence) boundaries into pieces of at most ``max_chars``.

    Returns ``(start_offset, piece)`` so a chunk can be mapped back to the piece that contains it.
    """
    spans: list[tuple[int, str]] = []
    pos, n = 0, len(text)
    while pos < n:
        end = min(n, pos + max_chars)
        if end < n:
            floor = pos + max_chars // 2
            cut = text.rfind("\n\n", floor, end)
            if cut == -1:
                cut = text.rfind("\n", floor, end)
            if cut == -1:
                cut = text.rfind(". ", floor, end)
                if cut != -1:
                    cut += 1
            if cut > pos:
                end = cut
        if text[pos:end].strip():
            spans.append((pos, text[pos:end]))
        pos = end
        while pos < n and text[pos].isspace():
            pos += 1
    return spans


# ---------------------------------------------------------------------------------------- B1 extract
def entry_schema(profile: Profile) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": MAX_NAMES,
                        },  # fmt: skip
                        "kind": {"type": "string", "enum": [*profile.kinds, "other"]},
                        "definition": {"type": "string"},
                        "status": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["names", "kind", "definition", "status", "quote"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entries"],
        "additionalProperties": False,
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip()[:80])
    return out


def _parse_entries(text: str) -> list[dict[str, Any]] | None:
    try:
        entries = json.loads(text)["entries"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return entries if isinstance(entries, list) else None


async def extract_span(
    llm: SlmClient, profile: Profile, span: str, position: int
) -> tuple[list[Entry], dict[str, int]]:
    """B1 on one passage. Returns the verified entries and counters (raw / kept / dropped_*)."""
    stats: Counter[str] = Counter()
    user = prompts.EXTRACT_USER.format(
        profile=profile.name,
        kinds=", ".join(profile.kinds),
        capture=profile.capture or "nothing in particular",
        span=span,
    )
    schema = entry_schema(profile)
    reply = await llm.chat(
        prompts.EXTRACT_SYSTEM, user, max_tokens=2400, temperature=0.0, top_p=1.0,
        json_schema=schema,
    )  # fmt: skip
    raw = _parse_entries(reply.text)
    if raw is None:
        # Greedy decoding under a JSON schema sometimes loops (endless whitespace or escapes) on a 2B model;
        # one retry with a little randomness and a presence penalty breaks the loop.
        stats["retry"] += 1
        reply = await llm.chat(
            prompts.EXTRACT_SYSTEM, user, max_tokens=1800, temperature=0.3, top_p=0.95,
            presence_penalty=0.5, json_schema=schema,
        )  # fmt: skip
        raw = _parse_entries(reply.text)
    if raw is None:
        stats["parse_error"] += 1
        return [], dict(stats)
    entries: list[Entry] = []
    for item in raw:
        stats["raw"] += 1
        names = _dedupe([n for n in item.get("names", []) if usable_name(str(n))])[:MAX_NAMES]
        definition = str(item.get("definition") or "").strip()[:MAX_DEF_CHARS]
        if not names or not definition:
            stats["dropped_empty"] += 1
            continue
        if not any(count_name(n, span) for n in names):
            stats["dropped_name"] += 1
            continue
        support = quote_support(str(item.get("quote") or ""), span)
        if support == "failed":
            stats["dropped_quote"] += 1
            continue
        status = str(item.get("status") or "").strip()[:60]
        entries.append(
            Entry(
                names=names,
                kind=str(item.get("kind") or "other"),
                definition=definition,
                quote=str(item["quote"]).strip()[:200],
                first_seen=position,
                timeline=[(position, status)] if status else [],
                support=support,
            )
        )
        stats["kept"] += 1
        stats[f"support_{support}"] += 1
    return entries, dict(stats)


# ---------------------------------------------------------------------------------------- B2 merge
def merge_entries(entries: list[Entry]) -> list[Entry]:
    """Merge entries that share a specific surface form (case and punctuation ignored); keep the earliest definition."""
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Only specific names (see ``specific_name``) can join entries; a generic spelling of the same name may then
    # join too ("rlm" next to "RLM"), but two entries that share nothing except a generic word stay apart.
    owner: dict[str, int] = {}
    for i, e in enumerate(entries):
        for n in e.names:
            k = name_key(n)
            if not k or not specific_name(n):
                continue
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    for i, e in enumerate(entries):
        for n in e.names:
            k = name_key(n)
            if k in owner:
                parent[find(i)] = find(owner[k])
    clusters: dict[int, list[Entry]] = defaultdict(list)
    for i, e in enumerate(entries):
        clusters[find(i)].append(e)

    merged: list[Entry] = []
    for members in clusters.values():
        members.sort(key=lambda e: e.first_seen)
        freq = Counter(n for e in members for n in e.names)
        names = [n for n, _ in sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))]
        kinds = Counter(e.kind for e in members if e.kind != "other") or Counter(
            e.kind for e in members
        )
        first = members[0]
        merged.append(
            Entry(
                names=names[:MAX_NAMES],
                kind=kinds.most_common(1)[0][0],
                definition=first.definition,
                quote=first.quote,
                first_seen=first.first_seen,
                timeline=sorted({(p, s) for e in members for p, s in e.timeline}),
                mentions=len(members),
                support=first.support,
            )
        )
    merged.sort(key=lambda e: (e.first_seen, e.names[0].lower()))
    return merged


# ---------------------------------------------------------------------------------------- per-chunk slice
def select_slice(
    entries: list[Entry], text: str, position: int, *, causal: bool, max_entries: int = 8
) -> list[Entry]:
    """Entries whose names occur in ``text``, most mentioned first. Causal: only what is known at ``position``."""
    scored: list[tuple[int, Entry]] = []
    for e in entries:
        if causal and e.first_seen > position:
            continue
        hits = sum(count_name(n, text) for n in match_names(e))
        if hits:
            scored.append((hits, e))
    scored.sort(key=lambda t: (-t[0], t[1].first_seen, t[1].names[0].lower()))
    return [e for _, e in scored[:max_entries]]


def render_slice(
    entries: list[Entry], position: int, *, causal: bool, max_chars: int = 1400
) -> str:
    """One line per entry (about 350 tokens at most). Causal: the latest state stated up to ``position``."""
    lines: list[str] = []
    used = 0
    for e in entries:
        states = [s for p, s in e.timeline if not causal or p <= position]
        status = f" [{states[-1]}]" if states else ""
        line = f"- {e.names[0]} ({e.kind}): {e.definition[:SLICE_DEF_CHARS].rstrip()}{status}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "none"
