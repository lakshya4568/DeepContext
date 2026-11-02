"""Scoring of the arms of the far-reference test: fast ingestion against the deep variants.

Every arm gives one ranked list of pages per question. A question is *answered* when each part of its
evidence has an acceptable page among the first K pages (``eval.far_test.metric``). Arms are compared with
the baseline question by question (wins, losses) and an exact sign test; with a few dozen questions only
large differences can show, and the table says so with an exact interval for each arm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from eval.far_test import metric as M


@dataclass(frozen=True)
class Item:
    id: str
    parts: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class ArmResult:
    arm: str
    n: int
    a_found: int
    b_found: int
    complete: int
    wins: int
    losses: int
    ties: int
    p_value: float
    ci: tuple[float, float]


def pages_of(doc_ids: Sequence[str]) -> list[str]:
    """Pages of ranked documents; a view document ``p0003~0`` belongs to page ``p0003``. First hit wins."""
    seen: list[str] = []
    for doc_id in doc_ids:
        page = doc_id.split("~", 1)[0]
        if page not in seen:
            seen.append(page)
    return seen


def score_arms(
    items: Sequence[Item],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    k: int,
    baseline: str,
) -> list[ArmResult]:
    complete = {
        arm: [M.complete_at_k(ranked[item.id], item.parts, k) for item in items]
        for arm, ranked in rankings.items()
    }
    results = []
    for arm, ranked in rankings.items():
        found = [M.parts_found(ranked[item.id], item.parts, k) for item in items]
        wins, losses, ties = M.paired_counts(complete[baseline], complete[arm])
        results.append(
            ArmResult(
                arm=arm,
                n=len(items),
                a_found=sum(f[0] for f in found),
                b_found=sum(f[1] for f in found),
                complete=sum(complete[arm]),
                wins=wins,
                losses=losses,
                ties=ties,
                p_value=M.sign_test(wins, losses),
                ci=M.exact_ci(sum(complete[arm]), len(items)),
            )
        )
    return results


def render_table(results: Sequence[ArmResult], title: str, k: int) -> str:
    lines = [
        f"### {title} (top {k} pages)\n",
        "| arm | answered (both parts) | 95% interval | part A found | part B found | wins / losses vs fast | sign test p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        versus = (
            "-"
            if (r.wins, r.losses) == (0, 0) and r.arm == results[0].arm
            else f"{r.wins} / {r.losses}"
        )
        p = "-" if versus == "-" else f"{r.p_value:.3f}"
        lines.append(
            f"| {r.arm} | {r.complete}/{r.n} | {r.ci[0]:.0%}-{r.ci[1]:.0%} | {r.a_found}/{r.n} | "
            f"{r.b_found}/{r.n} | {versus} | {p} |"
        )
    return "\n".join(lines) + "\n"
