"""Compare judge variants scored on the same rows: mean scores, agreement with a reference judge, cost."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MET = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness_f1",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--reference", default="gemini-2.5-pro")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    by: dict[str, dict[tuple[str, str, str], dict]] = defaultdict(dict)
    for line in (rd / "ragas_scores.jsonl").open():
        r = json.loads(line)
        by[r["judge_model"]][(r["run_tag"], r["config"], r["qid"])] = r
    usage = {}
    if (rd / "judge_usage.jsonl").exists():
        for line in (rd / "judge_usage.jsonl").open():
            u = json.loads(line)
            usage[u["judge_model"]] = u
    common = set.intersection(*(set(d) for d in by.values())) if by else set()
    out = [
        f"# Judge comparison (same {len(common)} rows scored by every judge)\n",
        "| judge | est. list-price cost (rows scored) | " + " | ".join(MET) + " |",
        "|" + "|".join(["---"] * (len(MET) + 2)) + "|",
    ]
    for j, d in by.items():
        means = []
        for m in MET:
            v = [d[k][m] for k in common if d[k][m] == d[k][m]]
            means.append(f"{np.mean(v):.3f} (n={len(v)})" if v else "n/a")
        u = usage.get(j)
        cost = (
            f"${u['est_list_price_usd']:.2f} ({u['rows_scored']} rows, {u['live_out_tokens']:,} out-tokens)"
            if u and u.get("est_list_price_usd") is not None
            else "n/a"
        )
        out.append(f"| {j} | {cost} | " + " | ".join(means) + " |")
    if a.reference in by:
        for j, d in by.items():
            if j == a.reference:
                continue
            out.append(f"\n**{j} vs {a.reference}** — mean |diff| / Pearson r over rows")
            cells = []
            for m in MET:
                x = np.array([by[a.reference][k][m] for k in common])
                y = np.array([d[k][m] for k in common])
                ok = ~(np.isnan(x) | np.isnan(y))
                r = (
                    np.corrcoef(x[ok], y[ok])[0, 1]
                    if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0
                    else float("nan")
                )
                cells.append(
                    f"{m}: {np.mean(np.abs(x[ok] - y[ok])):.3f} / {r:.2f} (n={int(ok.sum())})"
                )
            out.append("- " + "; ".join(cells))
    out.append(
        "\nSmall-n caution: these rows come from one dataset's smoke batch; correlations on ~12 rows are indicative only."
    )
    (rd / "judge_comparison.md").write_text("\n".join(out) + "\n")
    print(f"wrote {rd / 'judge_comparison.md'}")


if __name__ == "__main__":
    main()
