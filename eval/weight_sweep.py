"""Summary of a lexical-vote sweep on dev sets, with the pre-declared selection rule.

    uv run --no-sync python -m eval.weight_sweep --root eval/results_paradedb \\
        --weights 0.02 0.05 0.1 0.25 0.5 1.0 --datasets scifact fiqa

Reads ``<root>/dev_pgs_w<weight>/raw_retrieval.jsonl`` (ParadeDB BM25) and, for comparison, the Okapi runs of the
original study under ``eval/results_fix/dev_w<weight>`` when they exist. Rule, written down before any result was
seen: the chosen weight has the best mean dev nDCG@10 over the datasets; ties go to the smaller weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.compare_runs import load_groups

Table = dict[
    str, dict[str, tuple[float, float, int]]
]  # weight -> dataset -> (nDCG@10, Recall@100, n)


def summarise(
    root: Path, weights: list[str], config: str = "rrf", prefix: str = "dev_pgs_w"
) -> Table:
    out: Table = {}
    for w in weights:
        groups = load_groups(root / f"{prefix}{w}" / "raw_retrieval.jsonl")
        for (dataset, cfg), rows in groups.items():
            if cfg != config:
                continue
            n = len(rows)
            ndcg = sum(r["metrics"]["ndcg@10"] for r in rows.values()) / n
            recall = sum(r["metrics"]["recall@100"] for r in rows.values()) / n
            out.setdefault(w, {})[dataset] = (ndcg, recall, n)
    return out


def best_weight(table: Table, datasets: list[str]) -> str:
    """Highest mean nDCG@10 over ``datasets``; equal means (to 6 decimals) go to the smaller weight."""

    def mean(w: str) -> float:
        return sum(table[w][d][0] for d in datasets) / len(datasets)

    return min(table, key=lambda w: (-round(mean(w), 6), float(w)))


def render(pgs: Table, okapi: Table, weights: list[str], datasets: list[str]) -> str:
    lines = [
        "| lexical vote | " + " | ".join(f"{d} nDCG@10 (ParadeDB / Okapi)" for d in datasets)
        + " | mean ParadeDB | " + " | ".join(f"{d} Recall@100 (ParadeDB)" for d in datasets) + " |",
        "|---|" + "---|" * (2 * len(datasets) + 1),
    ]  # fmt: skip
    for w in sorted(set(pgs) | set(okapi), key=float):
        cells = []
        for d in datasets:
            p = pgs.get(w, {}).get(d)
            o = okapi.get(w, {}).get(d)
            cells.append(f"{p[0]:.4f}" if p else "n/a" if not o else "n/a")
            cells[-1] += f" / {o[0]:.4f}" if o else " / –"
        mean = (
            f"{sum(pgs[w][d][0] for d in datasets) / len(datasets):.4f}"
            if w in pgs and all(d in pgs[w] for d in datasets)
            else "n/a"
        )
        recalls = [f"{pgs[w][d][1]:.3f}" if w in pgs and d in pgs[w] else "n/a" for d in datasets]
        lines.append(f"| {w} | " + " | ".join(cells) + f" | {mean} | " + " | ".join(recalls) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval/results_paradedb")
    ap.add_argument("--okapi-root", default="eval/results_fix")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    a = ap.parse_args(argv)
    pgs = summarise(Path(a.root), a.weights)
    okapi = summarise(Path(a.okapi_root), a.weights, prefix="dev_w")
    print(render(pgs, okapi, a.weights, a.datasets))
    complete = {w: v for w, v in pgs.items() if all(d in v for d in a.datasets)}
    if complete:
        print(
            f"\nChosen by the pre-declared rule (ParadeDB BM25): lexical vote {best_weight(complete, a.datasets)}"
        )


if __name__ == "__main__":
    main()
