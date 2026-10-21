"""Rerank-depth study: how many fused candidates should the reranker score?

The shipped engine scores every fused candidate (up to FIRST_STAGE_LIMIT = 100) and keeps the best 24 in
reranked order. This study scores only the first N fused candidates (N = 12, 24, 48) and compares with the
shipped depth (all, up to 100), on identical questions, for the ``rrf_bge`` setup with the fixed search.

Reads  eval/results_final/raw_retrieval.jsonl (depth "all", also ``rrf`` = no reranker) and
       eval/results_fix/rerank_depth/d{12,24,48}/raw_retrieval.jsonl
Writes eval/results_fix/RERANK_DEPTH.md and rerank_depth.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from eval.make_tables import fmt_ci, latest_per_key, read_jsonl
from eval.stats import bootstrap_ci, paired_test, percentile

DEPTHS = (12, 24, 48)
METRICS = (("ndcg@10", "nDCG@10"), ("hit@5", "Hit@5"), ("mrr", "MRR"))


def by_query(
    rows: list[dict[str, Any]], config: str
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r["config"] == config and r["relevant"] and "±" not in r["dataset"]:
            out.setdefault((r["dataset"], "all-tags"), {})[r["qid"]] = r
    return out


def rerank_latency(rows: dict[str, dict[str, Any]]) -> float:
    vals = [
        r["trace_summary"]["rerank"]["latency_s"]
        for r in rows.values()
        if "rerank" in r.get("trace_summary", {})
    ]
    return percentile(vals, 50) if vals else float("nan")


def main() -> None:
    base_dir, fix_dir = Path("eval/results_final"), Path("eval/results_fix")
    final = latest_per_key(read_jsonl(base_dir / "raw_retrieval.jsonl"))
    full = by_query(final, "rrf_bge")
    plain = by_query(final, "rrf")
    depth_rows = {
        d: by_query(
            latest_per_key(read_jsonl(fix_dir / "rerank_depth" / f"d{d}" / "raw_retrieval.jsonl")),
            "rrf_bge",
        )
        for d in DEPTHS
    }
    md = [
        "# Rerank-depth study (rrf_bge, fixed search)\n",
        "The shipped engine scores **all** fused candidates (up to 100) and keeps the best 24 in reranked order. "
        "Here the reranker scores only the first N fused candidates. Same questions for every depth; Δ = depth N minus all, "
        "paired bootstrap 95% CI. `no rerank` = the merged list without the reranker. Rerank p50 = median reranker call time "
        "(a call recorded when the answer was first computed; it includes the network path to the self-hosted GPU, so use it only to compare depths).\n",
    ]
    recs: list[dict[str, Any]] = []
    for (ds, tag), ref in sorted(full.items()):
        keys_by_depth = {d: set(depth_rows[d].get((ds, tag), {})) for d in DEPTHS}
        if not any(keys_by_depth.values()):
            continue
        common = set(ref)
        for d in DEPTHS:
            if keys_by_depth[d]:
                common &= keys_by_depth[d]
        if len(common) < 10:
            continue
        qs = sorted(common)
        md.append(f"\n## {ds} (n = {len(qs)})\n")
        header = [
            "rerank depth",
            *[lbl for _, lbl in METRICS],
            "Δ nDCG@10 vs all",
            "rerank p50 (s)",
        ]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")

        def row(
            label: str, rows: dict[str, dict[str, Any]], ref_rows: dict[str, dict[str, Any]] | None
        ) -> None:
            cells = [label]
            rec: dict[str, Any] = {"dataset": ds, "depth": label, "n": len(qs)}
            for m, _ in METRICS:
                mean, lo, hi = bootstrap_ci([rows[q]["metrics"][m] for q in qs])
                cells.append(fmt_ci(mean, lo, hi))
                rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
            if ref_rows is not None:
                res = paired_test(
                    [ref_rows[q]["metrics"]["ndcg@10"] for q in qs],
                    [rows[q]["metrics"]["ndcg@10"] for q in qs],
                )
                cells.append(f"{res.mean_diff:+.3f} [{res.ci_lo:+.3f}, {res.ci_hi:+.3f}]")
                rec.update(
                    delta=res.mean_diff, delta_lo=res.ci_lo, delta_hi=res.ci_hi, p=res.p_bootstrap
                )
            else:
                cells.append("")
            p50 = rerank_latency({q: rows[q] for q in qs})
            cells.append("–" if p50 != p50 else f"{p50:.2f}")
            rec["rerank_p50_s"] = p50
            md.append("| " + " | ".join(cells) + " |")
            recs.append(rec)

        pr = plain.get((ds, tag), {})
        if all(q in pr for q in qs):
            row("no rerank (merged list)", pr, ref)
        for d in DEPTHS:
            row(f"first {d} candidates", depth_rows[d][(ds, tag)], ref)
        row("all (shipped, up to 100)", ref, None)
    (fix_dir / "RERANK_DEPTH.md").write_text("\n".join(md) + "\n")
    pd.DataFrame(recs).to_csv(fix_dir / "rerank_depth.csv", index=False)
    print(f"wrote {fix_dir / 'RERANK_DEPTH.md'}")


if __name__ == "__main__":
    main()
