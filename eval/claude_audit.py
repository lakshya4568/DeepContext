"""Cross-family validation of the Gemini judge by Claude (an Anthropic model).

Workflow
  1. ``make_tables`` writes ``claude_audit_items.jsonl`` (sampled answers + the evidence the
     generator saw + the Gemini judge's scores).
  2. Claude reads the items and writes ``claude_labels.jsonl`` - one line per item::
         {"dataset":..., "config":..., "qid":..., "faithfulness": 0|0.5|1, "correctness": 0|0.5|1, "note": "..."}
     faithfulness: share of the answer's factual claims supported by the evidence shown (1 = all,
     0.5 = some, 0 = none/contradicted). correctness: does the answer convey the reference answer.
  3. ``python -m eval.claude_audit --results-dir ...`` compares them with the Gemini judge.

These are model labels, not human labels; ``human_label`` stays blank for manual review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args()
    rd = Path(a.results_dir)
    items = {
        (d["dataset"], d["config"], d["qid"]): d
        for d in map(json.loads, (rd / "claude_audit_items.jsonl").open())
    }
    labels = {
        (d["dataset"], d["config"], d["qid"]): d
        for d in map(json.loads, (rd / "claude_labels.jsonl").open())
    }
    keys = sorted(set(items) & set(labels))
    out: dict = {
        "labeller": "Claude (Anthropic model) - NOT human labels",
        "n_items": len(items),
        "n_labelled": len(keys),
    }
    for metric, jkey in (
        ("faithfulness", "faithfulness"),
        ("correctness", "factual_correctness_f1"),
        ("correctness_rubric", "rubric_correctness"),
    ):
        lab_metric = "correctness" if metric.startswith("correctness") else metric
        sub = [k for k in keys if items[k]["judge_scores"].get(jkey) is not None]
        if not sub:
            continue
        c = np.array([labels[k][lab_metric] for k in sub], dtype=float)
        g = np.array([items[k]["judge_scores"][jkey] for k in sub], dtype=float)
        ok = ~np.isnan(g)
        res = {"n": int(ok.sum()), "claude_mean": float(c[ok].mean()), "gemini_judge_mean": float(g[ok].mean()),
               "mean_abs_diff": float(np.mean(np.abs(c[ok] - g[ok])))}  # fmt: skip
        if ok.sum() > 3 and c[ok].std() > 0 and g[ok].std() > 0:
            res["pearson_r"] = float(np.corrcoef(c[ok], g[ok])[0, 1])
            res["spearman_rho"] = float(sp_stats.spearmanr(c[ok], g[ok])[0])
        # binary agreement: "mostly supported/correct" (claude >= 0.5 vs judge >= 0.5)
        res["binary_agreement_at_0.5"] = float(np.mean((c[ok] >= 0.5) == (g[ok] >= 0.5)))
        res["judge_more_lenient_share"] = float(np.mean(g[ok] > c[ok] + 0.25))
        res["judge_stricter_share"] = float(np.mean(g[ok] < c[ok] - 0.25))
        out[metric] = res
    by_ds: dict[str, dict] = {}
    for k in keys:
        d = by_ds.setdefault(k[0], {"n": 0, "claude_faith": [], "judge_faith": []})
        d["n"] += 1
        d["claude_faith"].append(labels[k]["faithfulness"])
        d["judge_faith"].append(items[k]["judge_scores"]["faithfulness"])
    out["faithfulness_by_dataset"] = {
        ds: {
            "n": d["n"],
            "claude_mean": float(np.mean(d["claude_faith"])),
            "gemini_judge_mean": float(np.nanmean(d["judge_faith"])),
        }
        for ds, d in by_ds.items()
    }
    (rd / "claude_audit_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
