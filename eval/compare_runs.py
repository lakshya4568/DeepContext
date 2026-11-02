"""Paired comparison of retrieval runs on identical questions (used for the lexical-channel A/B).

    uv run --no-sync python -m eval.compare_runs --baseline eval/results_final \\
        --arm "pg_search, vote 0.1=eval/results_paradedb/B_w0.1" \\
        --arm "pg_search, vote 1.0=eval/results_paradedb/C_w1.0" \\
        --configs bm25_only rrf --out eval/results_paradedb/LEXICAL_AB.md

Means carry a 95% bootstrap CI (10,000 resamples, seed 0); the difference to the baseline is a paired
bootstrap over the questions both runs answered, as in every other table of this harness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.make_tables import fmt_ci, latest_per_key, read_jsonl
from eval.stats import bootstrap_ci, paired_test

METRICS = (("ndcg@10", "nDCG@10"), ("hit@5", "Hit@5"), ("recall@100", "Recall@100"))
Group = dict[str, dict[str, Any]]  # qid -> retrieval row


@dataclass(frozen=True)
class Delta:
    n: int
    base: float
    arm: float
    diff: float
    lo: float
    hi: float
    p: float


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], Group]:
    """Rows by (dataset, config) then question id; unanswerable questions and +/-1-page rows are left out."""
    out: dict[tuple[str, str], Group] = {}
    for r in rows:
        if r["relevant"] and "±" not in r["dataset"]:
            out.setdefault((r["dataset"], r["config"]), {})[r["qid"]] = r
    return out


def load_groups(path: Path) -> dict[tuple[str, str], Group]:
    return group_rows(latest_per_key(read_jsonl(path)))


def compare_metric(base: Group, arm: Group, metric: str) -> Delta | None:
    """Arm minus baseline on the questions present in both runs; None if there are none."""
    keys = sorted(set(base) & set(arm))
    if not keys:
        return None
    b = [base[k]["metrics"][metric] for k in keys]
    a = [arm[k]["metrics"][metric] for k in keys]
    res = paired_test(b, a)
    return Delta(
        len(keys), sum(b) / len(b), sum(a) / len(a), res.mean_diff, res.ci_lo, res.ci_hi,
        res.p_bootstrap,
    )  # fmt: skip


def _fmt_delta(d: Delta | None) -> str:
    return "n/a" if d is None else f"{d.diff:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}] (p={d.p:.3f})"


def _mean_ci(group: Group, metric: str) -> str:
    mean, lo, hi = bootstrap_ci([r["metrics"][metric] for r in group.values()])
    return fmt_ci(mean, lo, hi)


def render(
    base: dict[tuple[str, str], Group],
    arms: list[tuple[str, dict[tuple[str, str], Group]]],
    configs: list[str],
) -> str:
    datasets = sorted({d for d, _ in base})
    md = [
        "# Lexical channel A/B: ParadeDB pg_search BM25 vs the baseline\n",
        "Retrieval only (no reranker, no LLM). Same questions, same indexes, same settings except the arm; "
        "differences are arm minus baseline on identical questions (paired bootstrap, 95% CI).\n",
        "## Summary: change in nDCG@10 vs the baseline\n",
        "| dataset | config | n | baseline nDCG@10 | "
        + " | ".join(label for label, _ in arms)
        + " |",
        "|---|---|---|---|" + "---|" * len(arms),
    ]
    for ds in datasets:
        for cfg in configs:
            b = base.get((ds, cfg))
            if not b:
                continue
            cells = [
                _fmt_delta(compare_metric(b, g[(ds, cfg)], "ndcg@10")) if (ds, cfg) in g else "n/a"
                for _, g in arms
            ]
            md.append(
                f"| {ds} | {cfg} | {len(b)} | {_mean_ci(b, 'ndcg@10')} | "
                + " | ".join(cells)
                + " |"
            )
    for ds in datasets:
        md.append(f"\n## {ds}\n")
        md.append(
            "| config | setup | "
            + " | ".join(lbl for _, lbl in METRICS)
            + " | Δ nDCG@10 [95% CI] | n |"
        )
        md.append("|---|---|" + "---|" * (len(METRICS) + 2))
        for cfg in configs:
            b = base.get((ds, cfg))
            if not b:
                continue
            md.append(
                f"| {cfg} | baseline | "
                + " | ".join(_mean_ci(b, m) for m, _ in METRICS)
                + f" | | {len(b)} |"
            )
            for label, groups in arms:
                g = groups.get((ds, cfg))
                if g:
                    md.append(
                        f"| {cfg} | {label} | "
                        + " | ".join(_mean_ci(g, m) for m, _ in METRICS)
                        + f" | {_fmt_delta(compare_metric(b, g, 'ndcg@10'))} | {len(g)} |"
                    )
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="eval/results_final")
    ap.add_argument("--arm", action="append", required=True, help="LABEL=DIR (repeatable)")
    ap.add_argument("--configs", nargs="+", default=["bm25_only", "rrf"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = load_groups(Path(a.baseline) / "raw_retrieval.jsonl")
    arms = []
    for spec in a.arm:
        label, _, d = spec.partition("=")
        arms.append((label, load_groups(Path(d) / "raw_retrieval.jsonl")))
    Path(a.out).write_text(render(base, arms, a.configs))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
