"""Before/after comparison of the search fixes (P1 BM25, P2 weighted merge, P3 duplicate removal).

Baseline  = rows in ``eval/results/raw_retrieval.jsonl`` (ts_rank lexical channel, prefix dedup, equal RRF votes)
Variants  = ``eval/results_fix/<variant>/raw_retrieval.jsonl``
              v1_bm25         : BM25 lexical channel (bm25_only, rrf)
              v2_bm25_dedup   : + exact-duplicate removal (rrf)
              v3_weighted     : + lexical vote 0.1 (chosen on dev questions only) (rrf)
Paired tests are over identical test questions. Writes ``BEFORE_AFTER.md`` / ``before_after.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from eval.make_tables import fmt_ci, latest_per_key, read_jsonl
from eval.stats import bootstrap_ci, paired_test

METRICS = [("ndcg@10", "nDCG@10"), ("hit@5", "Hit@5"), ("recall@100", "Recall@100")]
VARIANTS = [
    ("v1_bm25", "bm25_only", "BM25 lexical channel"),
    ("v1_bm25", "rrf", "merged, BM25 only (P1)"),
    ("v2_bm25_dedup", "rrf", "merged, BM25 + exact dedup (P1+P3)"),
    ("v3_weighted", "rrf", "merged, BM25 + dedup + lexical vote 0.1 (P1+P2+P3)"),
    ("v3_weighted", "rrf_bge", "merged + reranker, all fixes"),
    ("v3_weighted", "engine_as_shipped", "full engine (≤6 documents), all fixes"),
]


def by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r["relevant"] and "±" not in r["dataset"]:
            out.setdefault((r["dataset"], r["config"]), {})[r["qid"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default="eval/results")
    ap.add_argument("--fix-dir", default="eval/results_fix")
    a = ap.parse_args()
    base = by_key(latest_per_key(read_jsonl(Path(a.baseline_dir) / "raw_retrieval.jsonl")))
    variants = {
        v: by_key(latest_per_key(read_jsonl(Path(a.fix_dir) / v / "raw_retrieval.jsonl")))
        for v in {v for v, _, _ in VARIANTS}
    }
    datasets = sorted({d for d, _ in base})
    recs: list[dict[str, Any]] = []
    md = [
        "# Search fixes: before vs after (test questions only)\n",
        "Mean [95% bootstrap CI, 10,000 resamples, seed 0]. Δ = after − before on identical questions with a paired "
        "bootstrap CI and p-value. `dense_only` is unchanged by the fixes and shown for reference.\n",
        "- **before** = original code (ts_rank word search, drop-by-first-120-characters, equal votes)",
        "- **P1** real BM25 word search · **P3** exact-duplicate removal · **P2** lexical vote 0.1 (picked on SciFact-train/FiQA-dev questions, not on these)\n",
    ]
    for ds in datasets:
        n_q = len(base.get((ds, "dense_only"), {}))
        md.append(f"\n## {ds}  (n = {n_q})\n")
        header = ["setup"] + [lbl for _, lbl in METRICS] + ["Δ nDCG@10 vs before", "p"]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")

        def row(
            label: str, rows: dict[str, dict[str, Any]], ref: dict[str, dict[str, Any]] | None
        ) -> None:
            if not rows:
                return
            cells = [label]
            rec: dict[str, Any] = {"dataset": ds, "setup": label, "n": len(rows)}
            for m, _ in METRICS:
                mean, lo, hi = bootstrap_ci([r["metrics"][m] for r in rows.values()])
                cells.append(fmt_ci(mean, lo, hi))
                rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
            if ref:
                keys = sorted(set(rows) & set(ref))
                res = paired_test(
                    [ref[k]["metrics"]["ndcg@10"] for k in keys],
                    [rows[k]["metrics"]["ndcg@10"] for k in keys],
                )
                cells += [
                    f"{res.mean_diff:+.3f} [{res.ci_lo:+.3f}, {res.ci_hi:+.3f}]",
                    f"{res.p_bootstrap:.4f}",
                ]
                rec.update(
                    delta_ndcg10=res.mean_diff,
                    delta_lo=res.ci_lo,
                    delta_hi=res.ci_hi,
                    p=res.p_bootstrap,
                )
            else:
                cells += ["", ""]
            md.append("| " + " | ".join(cells) + " |")
            recs.append(rec)

        row("dense only (unchanged)", base.get((ds, "dense_only"), {}), None)
        row("word search — before (ts_rank)", base.get((ds, "bm25_only"), {}), None)
        row("merged — before", base.get((ds, "rrf"), {}), None)
        row("merged + reranker — before", base.get((ds, "rrf_bge"), {}), None)
        row("full engine (≤6 documents) — before", base.get((ds, "engine_as_shipped"), {}), None)
        for v, cfg, label in VARIANTS:
            ref_cfg = cfg
            row(
                label + (" — after" if cfg == "bm25_only" else ""),
                variants[v].get((ds, cfg), {}),
                base.get((ds, ref_cfg)),
            )
    (Path(a.fix_dir) / "BEFORE_AFTER.md").write_text("\n".join(md) + "\n")
    pd.DataFrame(recs).to_csv(Path(a.fix_dir) / "before_after.csv", index=False)
    print(f"wrote {Path(a.fix_dir) / 'BEFORE_AFTER.md'}")


if __name__ == "__main__":
    main()
