"""Fast ingestion vs variant B retrieval on the same questions: mean with 95% bootstrap CI, and the paired difference.

    uv run --no-sync python -m eval.variant_b.compare --dataset docsqa \
        --fast eval/results_final/raw_retrieval.jsonl \
        --vb eval/results_variant_b/retrieval_docsqa/raw_retrieval.jsonl --configs dense_only rrf rrf_bge
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.make_tables import fmt_ci, latest_per_key, read_jsonl, tolerant_page_rows
from eval.stats import bootstrap_ci, paired_test

METRICS = (("hit@5", "Hit@5"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10"), ("recall@10", "Recall@10"))


def by_query(rows: list[dict[str, Any]], dataset: str, config: str) -> dict[str, dict[str, Any]]:
    return {
        r["qid"]: r
        for r in latest_per_key(rows)
        if r["dataset"] == dataset and r["config"] == config and r["relevant"]
    }


def compare(
    fast_rows: list[dict[str, Any]], vb_rows: list[dict[str, Any]], dataset: str, configs: list[str]
) -> str:
    lines = [
        "| config | metric | fast (n) | variant B | variant B minus fast, paired 95% CI | p (bootstrap) |",
        "|---|---|---|---|---|---|",
    ]
    for config in configs:
        fast, vb = by_query(fast_rows, dataset, config), by_query(vb_rows, dataset, config)
        qids = sorted(set(fast) & set(vb))
        if not qids:
            lines.append(f"| {config} | – | no common questions | | | |")
            continue
        for key, label in METRICS:
            a = [fast[q]["metrics"][key] for q in qids]
            b = [vb[q]["metrics"][key] for q in qids]
            fm, flo, fhi = bootstrap_ci(a)
            vm, vlo, vhi = bootstrap_ci(b)
            d = paired_test(a, b)
            lines.append(
                f"| {config} | {label} | {fmt_ci(fm, flo, fhi)} (n={len(qids)}) | {fmt_ci(vm, vlo, vhi)} "
                f"| {d.mean_diff:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}] | {d.p_bootstrap:.3f} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fast", required=True, type=Path)
    ap.add_argument("--vb", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=["dense_only", "bm25_only", "rrf", "rrf_bge"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--tolerant",
        action="store_true",
        help="PDF sets: relevance widened to +/-1 page (labels are imprecise)",
    )
    args = ap.parse_args()
    fast_rows, vb_rows, dataset = read_jsonl(args.fast), read_jsonl(args.vb), args.dataset
    if args.tolerant:
        fast_rows, vb_rows = (
            tolerant_page_rows(latest_per_key(fast_rows)),
            tolerant_page_rows(latest_per_key(vb_rows)),
        )
        dataset = f"{dataset} (±1 page)"
    table = compare(fast_rows, vb_rows, dataset, args.configs)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
