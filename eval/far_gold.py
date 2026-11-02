"""Far-apart evidence: questions whose gold pages sit in different chapters, scored at both ends.

A question whose gold pages span at least ``min_span`` pages has labelled evidence in different parts
of the book. The questions are the existing ones of the PDF sets, selected by the gold span alone; the
threshold and the depths are fixed before any run is looked at (span >= 100 pages, K = 10 and 20).

Caution (the 10 selected questions were read on 2026-11-01): the labels were never audited for necessity.
None of the 10 was written as a "name here, pronoun there" question, and in the novel the dying-words
question has the same sentence on all three of its gold pages, so one page is enough. In the thesis 184 of
188 gold pages come with a partner exactly 20 pages away (the printed page number and the PDF page number
of the same page); the later page holds more of the answer words in 84 of 92 pairs, so both-ends@K is out
of reach there. Read both-ends@K as a strict score on these labels, not as a test of the pronoun problem.
Two measures per question at depth K:

* both-ends@K: 1 if the earliest and the latest gold page are both among the first K ranked pages;
* recall@K: share of all gold pages among the first K ranked pages.

Only the PDF sets have page ids (``p0043``).

    uv run --no-sync python -m eval.far_gold --datasets pdf-got pdf-quantum --out far.md \\
        --run "dense only=eval/results_main/M0_stock:dense_only" \\
        --run "shipped=eval/results_main/M0_stock:rrf"
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from eval.compare_runs import Delta, Group, load_groups
from eval.make_tables import fmt_ci
from eval.stats import bootstrap_ci, paired_test

Row = dict[str, Any]
Measure = Callable[[Row], float]
Run = tuple[str, dict[tuple[str, str], Group], str]  # (label, groups by (dataset, config), config)

MIN_SPAN = 100
KS = (10, 20)
_PAGE_ID = re.compile(r"^p(\d+)$")


def page_number(doc_id: str) -> int | None:
    match = _PAGE_ID.match(doc_id)
    return int(match.group(1)) if match else None


def _gold_pages(relevant: Sequence[str]) -> list[int]:
    return sorted(p for p in (page_number(d) for d in relevant) if p is not None)


def gold_span(relevant: Sequence[str]) -> int:
    """Pages between the earliest and the latest gold page (0 with fewer than two page ids)."""
    pages = _gold_pages(relevant)
    return pages[-1] - pages[0] if len(pages) >= 2 else 0


def far_questions(group: Group, min_span: int = MIN_SPAN) -> Group:
    return {q: r for q, r in group.items() if gold_span(r["relevant"]) >= min_span}


def both_ends_at_k(row: Row, k: int) -> float:
    pages = _gold_pages(row["relevant"])
    if len(pages) < 2:
        raise ValueError("both-ends needs at least two page ids among the gold pages")
    top = {p for p in (page_number(d) for d in row["ranked"][:k]) if p is not None}
    return float(pages[0] in top and pages[-1] in top)


def recall_at_k(row: Row, k: int) -> float:
    gold = set(row["relevant"])
    return len(gold & set(row["ranked"][:k])) / len(gold)


def compare(base: Group, arm: Group, measure: Measure) -> Delta | None:
    """Arm minus baseline on the questions both runs answered (paired bootstrap)."""
    keys = sorted(set(base) & set(arm))
    if not keys:
        return None
    b = [measure(base[k]) for k in keys]
    a = [measure(arm[k]) for k in keys]
    res = paired_test(b, a)
    return Delta(
        len(keys), sum(b) / len(b), sum(a) / len(a), res.mean_diff, res.ci_lo, res.ci_hi,
        res.p_bootstrap,
    )  # fmt: skip


def _mean_ci(group: Group, measure: Measure) -> str:
    mean, lo, hi = bootstrap_ci([measure(r) for r in group.values()])
    return fmt_ci(mean, lo, hi)


def _fmt_delta(d: Delta | None) -> str:
    return "n/a" if d is None else f"{d.diff:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}] (p={d.p:.3f})"


def render(
    runs: Sequence[Run], datasets: Sequence[str], min_span: int = MIN_SPAN, ks: Sequence[int] = KS
) -> str:
    """One table per dataset; every row is a setup, the first run is the baseline of the deltas."""
    k0 = ks[0]
    md = [
        "# Far-apart evidence\n",
        f"Questions whose gold pages are at least {min_span} pages apart. Retrieval only, first stage. "
        "Differences are setup minus the first row on identical questions (paired bootstrap, 95% CI).\n",
    ]
    for ds in datasets:
        md.append(f"\n## {ds}\n")
        header = ["setup", "far questions"]
        for k in ks:
            header += [f"both-ends@{k}", f"recall@{k}"]
        header += [f"Δ both-ends@{k0}", f"Δ recall@{k0}"]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        base: Group | None = None
        for label, groups, config in runs:
            group = groups.get((ds, config))
            if not group:
                continue
            far = far_questions(group, min_span)
            if not far:
                continue
            if base is None:
                base = far
            cells = [label, str(len(far))]
            for k in ks:
                cells += [
                    _mean_ci(far, lambda r, k=k: both_ends_at_k(r, k)),
                    _mean_ci(far, lambda r, k=k: recall_at_k(r, k)),
                ]
            cells += [
                ""
                if far is base
                else _fmt_delta(compare(base, far, lambda r: both_ends_at_k(r, k0))),
                "" if far is base else _fmt_delta(compare(base, far, lambda r: recall_at_k(r, k0))),
            ]
            md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="LABEL=DIR:CONFIG (repeatable)")
    ap.add_argument("--datasets", nargs="+", default=["pdf-got", "pdf-quantum"])
    ap.add_argument("--min-span", type=int, default=MIN_SPAN)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    loaded: dict[str, dict[tuple[str, str], Group]] = {}
    runs: list[Run] = []
    for spec in a.run:
        label, rest = spec.split("=", 1)
        directory, config = rest.rsplit(":", 1)
        if directory not in loaded:
            loaded[directory] = load_groups(Path(directory) / "raw_retrieval.jsonl")
        runs.append((label, loaded[directory], config))
    a.out.write_text(render(runs, a.datasets, a.min_span, a.ks), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
