"""How often a contextual summary names something its model was not shown.

A capitalised word of a summary (not the first word of a sentence, at least three letters) is *unsupported*
when it appears nowhere in the part of the prompt that carries the document: the ``<document>`` block (title,
section path, first characters of the parent) and the ``<passage>``. Such a word came from the model's memory
(or was made up), not from the document. The match ignores case, so the rate is a lower bound.

    uv run --no-sync python -m eval.summary_names \\
        --jobs eval/cache/summaries/jobs_9b.jsonl --results eval/cache/summaries/results_9b.jsonl \\
        --group "novel=A Game of Thrones" --group "thesis=Novel Systems and Methods"

Summaries are grouped by the prefix of the title on the prompt's ``Document:`` line; the rest are ``other``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

CAPITALISED_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")
WORD = re.compile(r"[A-Za-z]+")
DOCUMENT_LINE = re.compile(r"^Document:\s*(.*)$", re.MULTILINE)
SENTENCE_END = ".!?:"
OTHER = "other"


def shown_text(prompt: str) -> str:
    """The part of a summary prompt that carries the document: header, parent context and passage."""
    start, end = prompt.find("<document>"), prompt.find("</passage>")
    if start < 0 or end < 0:
        raise ValueError("prompt has no <document> ... </passage> block")
    return prompt[start:end]


def unsupported_names(summary: str, shown: str) -> list[str]:
    """Capitalised words of ``summary`` that occur nowhere in ``shown``."""
    known = {w.lower() for w in WORD.findall(shown)}
    names = []
    for match in CAPITALISED_WORD.finditer(summary):
        before = summary[: match.start()].rstrip()
        if not before or before[-1] in SENTENCE_END:
            continue
        if match.group().lower() not in known:
            names.append(match.group())
    return names


def _group_of(prompt: str, groups: Mapping[str, str]) -> str:
    found = DOCUMENT_LINE.search(prompt)
    title = found.group(1) if found else ""
    return next((label for label, prefix in groups.items() if title.startswith(prefix)), OTHER)


def unsupported_share(
    jobs: Mapping[str, str], summaries: Mapping[str, str], groups: Mapping[str, str]
) -> dict[str, tuple[int, int]]:
    """Per group: (summaries with at least one unsupported name, summaries). ``jobs`` maps key to prompt."""
    counts: dict[str, list[int]] = {label: [0, 0] for label in groups}
    for key, summary in summaries.items():
        if key not in jobs:
            continue
        counted = counts.setdefault(_group_of(jobs[key], groups), [0, 0])
        counted[1] += 1
        counted[0] += bool(unsupported_names(summary, shown_text(jobs[key])))
    return {label: (hits, total) for label, (hits, total) in counts.items() if total}


def render(share: Mapping[str, tuple[int, int]]) -> str:
    lines = [
        "| group | summaries | with an unsupported name | share |",
        "|---|---|---|---|",
    ]
    for label, (hits, total) in share.items():
        lines.append(f"| {label} | {total} | {hits} | {100 * hits / total:.0f}% |")
    return "\n".join(lines) + "\n"


def _read_field(path: Path, field: str) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["key"]: row[field] for row in rows}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--jobs", type=Path, required=True, help="JSONL of {key, prompt}")
    parser.add_argument("--results", type=Path, required=True, help="JSONL of {key, summary, ...}")
    parser.add_argument(
        "--group", action="append", default=[], help="LABEL=TITLE_PREFIX (repeatable)"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    groups = dict(spec.split("=", 1) for spec in args.group)
    share = unsupported_share(
        _read_field(args.jobs, "prompt"), _read_field(args.results, "summary"), groups
    )
    table = render(share)
    print(table)
    if args.out is not None:
        args.out.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
