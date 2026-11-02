"""Aggregate raw per-query results into the paper tables (CSV + markdown).

Inputs  (in --results-dir): raw_retrieval.jsonl, raw_generation.jsonl, ragas_scores.jsonl,
                            rubric_scores.jsonl, raw_latency.jsonl (optional live latency pass)
Outputs: table_retrieval.csv, table_ragas.csv, table_correctness.csv, table_robustness.csv,
         table_system.csv, significance.csv, ablation.md

Every number in REPORT.md is read from these files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval import config as C
from eval.metrics import RANK_METRIC_NAMES, contains_answer, exact_match, rank_metrics, token_f1
from eval.stats import bootstrap_ci, paired_test, percentile

CONFIG_ORDER = ["dense_only", "bm25_only", "rrf", "rrf_bge", "rrf_bge_two_pass", C.ENGINE_CONFIG]
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness_f1",
]
RUBRIC_METRICS = ["rubric_correctness", "rubric_strict"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open()] if path.exists() else []


def latest_per_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last non-error row per (run_tag, config, qid)."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("error"):
            continue
        out[(r["run_tag"], r["config"], r["qid"])] = r
    return list(out.values())


def fmt_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if mean != mean:
        return "n/a"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ------------------------------------------------------------------------------ retrieval
def tolerant_page_rows(ret: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PDF case-study rows re-scored with relevance widened to +/-1 page (labels are imprecise)."""
    out = []
    for r in ret:
        if not r["dataset"].startswith("pdf-") or not r["relevant"]:
            continue
        rel = {
            f"p{m:04d}": 1
            for d in r["relevant"]
            for m in (int(d[1:]) - 1, int(d[1:]), int(d[1:]) + 1)
            if m >= 1
        }
        out.append({**r, "dataset": f"{r['dataset']} (±1 page)", "run_tag": r["run_tag"] + "|tol1",
                    "relevant": sorted(rel), "metrics": rank_metrics(r["ranked"], rel)})  # fmt: skip
    return out


def retrieval_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    recs = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["run_tag"], r["config"])].append(r)
    for (ds, tag, cfg), rs in groups.items():
        rec: dict[str, Any] = {"dataset": ds, "run_tag": tag, "config": cfg}
        valid = [r for r in rs if r["relevant"]]
        rec["n_queries"] = len(valid)
        for m in RANK_METRIC_NAMES:
            vals = [r["metrics"][m] for r in valid]
            mean, lo, hi = bootstrap_ci(vals)
            rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "run_tag", "_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------ generation
def short_answer_metrics(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """EM / token-F1 / contains for datasets whose gold answers are short (MultiHop-RAG)."""
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        if r["dataset"] != "multihoprag" or r["unanswerable"] or not r["reference"]:
            continue
        out[(r["run_tag"], r["config"], r["qid"])] = {
            "exact_match": exact_match(r["answer"], [r["reference"]]),
            "token_f1": token_f1(r["answer"], [r["reference"]]),
            "contains_answer": contains_answer(r["answer"], [r["reference"]]),
        }
    return out


def ragas_table(
    gen: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    judge: str | None,
    rubric: list[dict[str, Any]] | None = None,
    rubric_judge: str | None = None,
) -> pd.DataFrame:
    sc = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in scores
        if judge is None or s["judge_model"] == judge
    }
    rb = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in (rubric or [])
        if rubric_judge is None or s["judge_model"] == rubric_judge
    }
    short = short_answer_metrics(gen)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in gen:
        if not r["unanswerable"] and r["reference"].strip():
            groups[(r["dataset"], r["run_tag"], r["config"])].append(r)
    recs = []
    for (ds, tag, cfg), rs in groups.items():
        rec: dict[str, Any] = {"dataset": ds, "run_tag": tag, "config": cfg, "n_answerable": len(rs),
                               "judge_model": judge or "any", "generator_model": C.GENERATOR_MODEL}  # fmt: skip
        scored = [sc[(tag, cfg, r["qid"])] for r in rs if (tag, cfg, r["qid"]) in sc]
        rec["n_scored"] = len(scored)
        for m in RAGAS_METRICS:
            vals = [s[m] for s in scored if s.get(m) is not None]
            mean, lo, hi = bootstrap_ci(vals)
            rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
            rec[f"{m}_n_valid"] = int(sum(1 for v in vals if v == v))
        rscored = [rb[(tag, cfg, r["qid"])] for r in rs if (tag, cfg, r["qid"]) in rb]
        rec["n_rubric_scored"] = len(rscored)
        for m, vals in (
            ("rubric_correctness", [x["score"] for x in rscored]),
            ("rubric_strict", [float(x["verdict"] == "correct") for x in rscored]),
        ):
            mean, lo, hi = bootstrap_ci(vals)
            rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
        for m in ("exact_match", "token_f1", "contains_answer"):
            vals = [short[(tag, cfg, r["qid"])][m] for r in rs if (tag, cfg, r["qid"]) in short]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "run_tag", "_o"]).drop(columns="_o").reset_index(drop=True)


def rubric_table(
    gen: list[dict[str, Any]], rubric: list[dict[str, Any]], rubric_judge: str | None
) -> pd.DataFrame:
    """LLM-rubric correctness per (dataset, config): answerable questions and unanswerable ones."""
    rb = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in rubric
        if rubric_judge is None or s["judge_model"] == rubric_judge
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in gen:
        groups[(r["dataset"], r["run_tag"], r["config"])].append(r)
    # Compare configs on the same questions: keep only questions graded under every config of the run.
    graded_by: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    for (ds, tag, cfg), rs in groups.items():
        graded_by[(ds, tag)].append({r["qid"] for r in rs if (tag, cfg, r["qid"]) in rb})
    common = {k: set.intersection(*v) for k, v in graded_by.items() if v}
    recs = []
    for (ds, tag, cfg), rs in groups.items():
        rs = [r for r in rs if r["qid"] in common[(ds, tag)]]
        rec: dict[str, Any] = {"dataset": ds, "run_tag": tag, "config": cfg,
                               "judge_model": rubric_judge or "any", "rubric_version": "v1"}  # fmt: skip
        ans = [
            rb[(tag, cfg, r["qid"])]
            for r in rs
            if not r["unanswerable"] and (tag, cfg, r["qid"]) in rb
        ]
        una = [
            rb[(tag, cfg, r["qid"])] for r in rs if r["unanswerable"] and (tag, cfg, r["qid"]) in rb
        ]
        rec["n_answerable"] = sum(1 for r in rs if not r["unanswerable"])
        rec["n_scored_answerable"] = len(ans)  # same questions for every config of the dataset
        for m, vals in (
            ("rubric_correctness", [x["score"] for x in ans]),
            ("share_correct", [float(x["verdict"] == "correct") for x in ans]),
            ("share_partial", [float(x["verdict"] == "partial") for x in ans]),
            ("share_incorrect", [float(x["verdict"] == "incorrect") for x in ans]),
            ("share_abstained", [float(x["verdict"] == "abstained") for x in ans]),
        ):
            mean, lo, hi = bootstrap_ci(vals)
            rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
        rec["n_unanswerable"] = sum(1 for r in rs if r["unanswerable"])
        rec["n_scored_unanswerable"] = len(una)
        mean, lo, hi = bootstrap_ci([x["score"] for x in una])
        rec["unanswerable_handled_correctly"], rec["unanswerable_handled_correctly_lo"] = mean, lo
        rec["unanswerable_handled_correctly_hi"] = hi
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "run_tag", "_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------ robustness
def robustness_table(gen: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in gen:
        groups[(r["dataset"], r["run_tag"], r["config"])].append(r)
    recs = []
    for (ds, tag, cfg), rs in groups.items():
        unans = [r for r in rs if r["unanswerable"]]
        ans = [r for r in rs if not r["unanswerable"]]
        rec: dict[str, Any] = {"dataset": ds, "run_tag": tag, "config": cfg,
                               "n_unanswerable": len(unans), "n_answerable": len(ans)}  # fmt: skip
        if unans:
            (
                rec["abstention_rate_on_unanswerable"],
                rec["abstention_rate_on_unanswerable_lo"],
                rec["abstention_rate_on_unanswerable_hi"],
            ) = bootstrap_ci([float(r["abstained"]) for r in unans])
        if ans:
            (
                rec["false_abstention_rate"],
                rec["false_abstention_rate_lo"],
                rec["false_abstention_rate_hi"],
            ) = bootstrap_ci([float(r["abstained"]) for r in ans])
            # false abstention while the gold evidence WAS in the supplied context
            both = [
                r
                for r in ans
                if r["relevant"] and set(r["relevant"]) & set(r.get("context_doc_ids", []))
            ]
            if both:
                rec["false_abstention_given_gold_in_context"] = float(
                    np.mean([float(r["abstained"]) for r in both])
                )
                rec["n_gold_in_context"] = len(both)
        if cfg == C.ENGINE_CONFIG:
            rec["anachronism_gate_fire_rate"] = float(
                np.mean([float(r.get("anachronism_gate_fired", False)) for r in rs])
            )
            rec["engine_insufficient_rate"] = float(
                np.mean([float(not r.get("engine_sufficient", True)) for r in rs])
            )
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "run_tag", "_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------ RGB
def rgb_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for r in rows:
        if not r.get("error"):
            seen[(r["kind"], r["mode"], r["id"], r["noise"])] = r
    for (kind, mode, _id, noise), r in seen.items():
        groups[(kind, mode, noise)].append(r)
    recs = []
    for (kind, mode, noise), rs in sorted(groups.items()):
        rec: dict[str, Any] = {"kind": kind, "mode": mode, "noise_ratio": noise, "n": len(rs)}
        if kind == "en_fact":
            for m in ("mentions_true", "mentions_fake", "abstained"):
                rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = bootstrap_ci([float(r[m]) for r in rs])
        else:
            if noise < 1.0:
                rec["accuracy"], rec["accuracy_lo"], rec["accuracy_hi"] = bootstrap_ci(
                    [float(r["correct"]) for r in rs]
                )
            rec["abstention_rate"], rec["abstention_lo"], rec["abstention_hi"] = bootstrap_ci(
                [float(r["abstained"]) for r in rs]
            )
            if noise == 1.0:
                rec["rejection_rate"] = rec[
                    "abstention_rate"
                ]  # RGB negative rejection (all passages are noise)
        recs.append(rec)
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------ evidence visibility
def visibility_table(gen: list[dict[str, Any]]) -> pd.DataFrame:
    """How much of the retrieved evidence reaches the generator (12k-char prompt budget)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in gen:
        groups[(r["dataset"], r["run_tag"], r["config"])].append(r)
    recs = []
    for (ds, tag, cfg), rs in groups.items():
        gold = [r for r in rs if r.get("gold_in_context") is not None]
        rec: dict[str, Any] = {"dataset": ds, "run_tag": tag, "config": cfg, "n": len(rs)}
        rec["evidence_chars_total_mean"] = float(np.mean([r["evidence_chars_total"] for r in rs]))
        rec["evidence_chars_seen_mean"] = float(np.mean([r["evidence_chars_seen"] for r in rs]))
        rec["fraction_of_evidence_seen"] = rec["evidence_chars_seen_mean"] / max(
            1.0, rec["evidence_chars_total_mean"]
        )
        rec["contexts_seen_over_retrieved"] = float(
            np.mean(
                [
                    len(r["contexts_seen"])
                    / max(1, r.get("n_contexts", len(r.get("contexts", []))))
                    for r in rs
                ]
            )
        )
        if gold:
            rec["gold_in_context_rate"] = float(
                np.mean([float(r["gold_in_context"]) for r in gold])
            )
            rec["gold_in_seen_context_rate"] = float(
                np.mean([float(r["gold_in_seen_context"]) for r in gold])
            )
            rec["gold_docs_seen_fraction"] = float(np.mean([r["gold_frac_seen"] for r in gold]))
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "run_tag", "_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------ judge audit
def write_judge_audit(
    gen: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    judge: str,
    path: Path,
    n: int = 50,
    rubric: list[dict[str, Any]] | None = None,
    rubric_judge: str | None = None,
) -> int:
    """50 sampled judgments for validation.

    ``human_label`` is intentionally left blank. ``claude_*`` columns are filled from
    ``claude_labels.jsonl`` (labels by Claude, an Anthropic model - NOT human labels) when present.
    Also writes ``claude_audit_items.jsonl`` with the full evidence the generator saw.
    """
    import csv
    import random

    sc = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in scores
        if s["judge_model"] == judge and s["faithfulness"] == s["faithfulness"]
    }
    rb = {
        (x["run_tag"], x["config"], x["qid"]): x
        for x in (rubric or [])
        if rubric_judge is None or x["judge_model"] == rubric_judge
    }
    cands = [r for r in gen if (r["run_tag"], r["config"], r["qid"]) in sc and not r["abstained"]]
    cands.sort(key=lambda r: (r["dataset"], r["config"], r["qid"]))
    picked = random.Random(C.BOOTSTRAP_SEED).sample(cands, min(n, len(cands)))
    labels: dict[tuple[str, str, str], dict[str, Any]] = {}
    lp = path.parent / "claude_labels.jsonl"
    if lp.exists():
        for line in lp.open():
            d = json.loads(line)
            labels[(d["dataset"], d["config"], d["qid"])] = d
    with (
        path.open("w", newline="") as f,
        (path.parent / "claude_audit_items.jsonl").open("w") as items,
    ):
        w = csv.writer(f)
        w.writerow(["dataset", "config", "qid", "query", "answer", "context_snippet", "metric", "judge_model",
                    "judge_score", "rubric_verdict", "rubric_score", "claude_faithfulness", "claude_correctness",
                    "human_label"])  # fmt: skip
        for r in picked:
            s_ = sc[(r["run_tag"], r["config"], r["qid"])]
            ctx_full = r.get("contexts_seen") or r.get("contexts", [])
            lab = labels.get((r["dataset"], r["config"], r["qid"]), {})
            rbr = rb.get((r["run_tag"], r["config"], r["qid"]), {})
            w.writerow([r["dataset"], r["config"], r["qid"], r["query"], r["answer"],
                        " ||| ".join(ctx_full)[:900], "faithfulness", judge, round(s_["faithfulness"], 4),
                        rbr.get("verdict", ""), rbr.get("score", ""),
                        lab.get("faithfulness", ""), lab.get("correctness", ""), ""])  # fmt: skip
            items.write(json.dumps({
                "dataset": r["dataset"], "config": r["config"], "qid": r["qid"], "query": r["query"],
                "answer": r["answer"], "reference": r["reference"], "contexts_seen": ctx_full,
                "judge_scores": {
                    **{k: s_[k] for k in RAGAS_METRICS},
                    "rubric_correctness": rbr.get("score"),
                },
                "rubric_verdict": rbr.get("verdict"),
                "rubric_reason": rbr.get("reason"),
            }) + "\n")  # fmt: skip
    return len(picked)


# ------------------------------------------------------------------------------ system
def _price(model: str) -> dict[str, float]:
    return C.PRICES_USD_PER_M.get(model, {"in": float("nan"), "out": float("nan")})


def system_table(
    ret: list[dict[str, Any]], gen: list[dict[str, Any]], lat: list[dict[str, Any]]
) -> pd.DataFrame:
    """Tokens/cost from the main runs; latency percentiles from the live latency pass if present."""
    recs = []
    lat_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in lat:
        if not r.get("error"):
            lat_groups[(r["dataset"], r["config"])].append(r)

    def stage(tr: dict[str, Any], s: str, key: str) -> float:
        return float(tr.get(s, {}).get(key, 0.0))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in gen:
        groups[(r["dataset"], r["config"])].append(r)
    for (ds, cfg), rs in groups.items():
        rec: dict[str, Any] = {"dataset": ds, "config": cfg, "n_generated": len(rs)}
        gin = [stage(r["trace_summary"], "generation", "in_tokens") for r in rs]
        gout = [stage(r["trace_summary"], "generation", "out_tokens") for r in rs]
        rin = [stage(r["trace_summary"], "retrieval", "in_tokens") for r in rs]
        rout = [stage(r["trace_summary"], "retrieval", "out_tokens") for r in rs]
        rr = [stage(r["trace_summary"], "rerank", "in_tokens") for r in rs]
        rec.update(gen_in_tokens_mean=np.mean(gin), gen_out_tokens_mean=np.mean(gout),
                   retrieval_tokens_mean=np.mean(rin) + np.mean(rout), rerank_tokens_mean=np.mean(rr))  # fmt: skip
        p = _price(C.GENERATOR_MODEL)
        # engine path also bills rewriter/classifier calls (stage 'retrieval' LLM calls)
        llm_in = np.mean(gin) + (np.mean(rin) if cfg == C.ENGINE_CONFIG else 0)
        llm_out = np.mean(gout) + (np.mean(rout) if cfg == C.ENGINE_CONFIG else 0)
        rec["generator_cost_per_1k_queries_usd"] = (
            (llm_in * p["in"] + llm_out * p["out"]) / 1e6 * 1000
        )
        rec["query_embedding_cost_per_1k_usd"] = (
            25 * _price(C.EMBEDDING_MODEL)["in"] / 1e6 * 1000
        )  # ~25 tok/query
        rec["rerank_cost_note"] = (
            "EcoHash BGE pricing is not published in the repo; tokens reported, $ excluded"
        )
        recs.append(rec)
        live = lat_groups.get((ds, cfg), [])
        if live:
            tot = [r["latency_s"] for r in live]
            rec["live_n"] = len(live)
            rec["total_p50_s"], rec["total_p95_s"] = percentile(tot, 50), percentile(tot, 95)
            for s in ("retrieval", "rerank", "generation"):
                vals = [stage(r["trace_summary"], s, "latency_s") for r in live]
                rec[f"{s}_p50_s"], rec[f"{s}_p95_s"] = percentile(vals, 50), percentile(vals, 95)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------ significance
def significance_table(
    ret: list[dict[str, Any]],
    gen: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    judge: str | None,
    rubric: list[dict[str, Any]] | None = None,
    rubric_judge: str | None = None,
) -> pd.DataFrame:
    recs = []

    def add(
        ds: str, tag: str, comparison: str, metric: str, a: dict[str, float], b: dict[str, float]
    ) -> None:
        keys = sorted(set(a) & set(b))
        if len(keys) < 3:
            return
        res = paired_test([a[k] for k in keys], [b[k] for k in keys])
        recs.append({"dataset": ds, "run_tag": tag, "comparison": comparison, "metric": metric, "n": res.n,
                     "mean_baseline": res.mean_a, "mean_variant": res.mean_b, "mean_diff": res.mean_diff,
                     "ci_lo": res.ci_lo, "ci_hi": res.ci_hi, "p_bootstrap": res.p_bootstrap,
                     "p_wilcoxon": res.p_wilcoxon, "cohens_dz": res.cohens_dz})  # fmt: skip

    by: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in ret:
        if r["relevant"]:
            by[(r["dataset"], r["run_tag"], r["config"])][r["qid"]] = r["metrics"]
    # Pre-specified retrieval contrasts (baseline -> variant); the engine row is depth-6 limited,
    # so it is compared only on metrics defined at k <= 6.
    retrieval_pairs = (
        ("dense_only", "rrf", "dense vs rrf", ("hit@5", "mrr", "ndcg@10", "recall@100", "map")),
        ("dense_only", "rrf_bge", "dense vs rrf+bge", ("hit@5", "mrr", "ndcg@10", "recall@100", "map")),
        ("rrf", "rrf_bge", "rrf vs rrf+bge", ("hit@5", "mrr", "ndcg@10", "recall@100", "map")),
        ("rrf_bge", C.ENGINE_CONFIG, "rrf+bge vs engine_as_shipped", ("hit@5", "recall@5")),
    )  # fmt: skip
    for ds, tag in sorted({(d, t) for d, t, _ in by}):
        for base, var, name, metrics in retrieval_pairs:
            a_rows, b_rows = by.get((ds, tag, base)), by.get((ds, tag, var))
            if a_rows is None or b_rows is None:
                continue
            for m in metrics:
                add(ds, tag, name, m, {q: v[m] for q, v in a_rows.items()},
                    {q: v[m] for q, v in b_rows.items()})  # fmt: skip

    sc = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in scores
        if judge is None or s["judge_model"] == judge
    }
    gby: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in gen:
        gby[(r["dataset"], r["run_tag"], r["config"])][r["qid"]] = r
    for (ds, tag, cfg), rows in list(gby.items()):
        if cfg != "rrf_bge":
            continue
        two = gby.get((ds, tag, "rrf_bge_two_pass"))
        if two is None:
            continue
        for m in RAGAS_METRICS:
            a = {
                q: sc[(tag, "rrf_bge", q)][m]
                for q in rows
                if (tag, "rrf_bge", q) in sc
                and sc[(tag, "rrf_bge", q)].get(m) == sc[(tag, "rrf_bge", q)].get(m)
            }
            b = {
                q: sc[(tag, "rrf_bge_two_pass", q)][m]
                for q in two
                if (tag, "rrf_bge_two_pass", q) in sc
                and sc[(tag, "rrf_bge_two_pass", q)].get(m)
                == sc[(tag, "rrf_bge_two_pass", q)].get(m)
            }
            add(ds, tag, "one-pass vs two-pass", m, a, b)
        for name, fn in (("abstained", lambda r: float(r["abstained"])),):
            a = {q: fn(r) for q, r in rows.items() if not r["unanswerable"]}
            b = {q: fn(r) for q, r in two.items() if not r["unanswerable"]}
            add(ds, tag, "one-pass vs two-pass", "false_abstention", a, b)
            a = {q: fn(r) for q, r in rows.items() if r["unanswerable"]}
            b = {q: fn(r) for q, r in two.items() if r["unanswerable"]}
            add(ds, tag, "one-pass vs two-pass", "abstention_on_unanswerable", a, b)

    rb = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in (rubric or [])
        if rubric_judge is None or s["judge_model"] == rubric_judge
    }
    answerable = {
        (r["dataset"], r["run_tag"], r["config"], r["qid"]) for r in gen if not r["unanswerable"]
    }

    def rubric_scores(ds: str, tag: str, cfg: str) -> dict[str, float]:
        return {
            q: rb[(tag, c, q)]["score"]
            for (d, t, c, q) in answerable
            if d == ds and t == tag and c == cfg and (tag, c, q) in rb
        }

    rubric_pairs = (
        ("dense_only", "rrf", "dense vs rrf"),
        ("rrf", "rrf_bge", "rrf vs rrf+bge"),
        ("rrf_bge", "rrf_bge_two_pass", "one-pass vs two-pass"),
        ("rrf_bge_two_pass", C.ENGINE_CONFIG, "two-pass vs engine_as_shipped"),
    )
    for ds, tag in sorted({(d, t) for d, t, _ in gby}):
        for base, var, name in rubric_pairs:
            add(
                ds,
                tag,
                name,
                "rubric_correctness",
                rubric_scores(ds, tag, base),
                rubric_scores(ds, tag, var),
            )
    df = pd.DataFrame(recs)
    if not df.empty:  # Holm step-down adjustment over every contrast in this table
        order = np.argsort(df["p_bootstrap"].to_numpy())
        m = len(df)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * float(df["p_bootstrap"].iloc[i]))
            adj[i] = min(1.0, running)
        df["p_bootstrap_holm"] = adj
    return df


# ------------------------------------------------------------------------------ retrieval latency
def retrieval_latency_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Live (uncached) latency of embedding + search (+ fusion + hosted reranker) per config."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r.get("error") and "latency_s" in r:
            groups[(r["dataset"], r["config"])].append(r)
    recs = []
    for (ds, cfg), rs in groups.items():
        lat = [r["latency_s"] for r in rs]
        rec: dict[str, Any] = {"dataset": ds, "config": cfg, "n": len(rs),
                               "p50_s": percentile(lat, 50), "p95_s": percentile(lat, 95)}  # fmt: skip
        for stage in ("retrieval", "rerank"):
            vals = [
                r["trace_summary"][stage]["latency_s"] for r in rs if stage in r["trace_summary"]
            ]
            rec[f"{stage}_p50_s"] = percentile(vals, 50) if vals else float("nan")
        rec["calls_with_retries"] = sum(
            1 for r in rs if any(v.get("retries", 0) for v in r["trace_summary"].values())
        )
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_o"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    return df.sort_values(["dataset", "_o"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------ markdown
ABLATION_METRICS = ["hit@1", "hit@5", "recall@10", "recall@100", "mrr", "map", "ndcg@10"]


def ablation_markdown(ret_df: pd.DataFrame, ragas_df: pd.DataFrame, robust_df: pd.DataFrame) -> str:
    out = [
        "# Ablation tables\n",
        "Mean [95% bootstrap CI, 10,000 resamples, seed 0]. **Bold** = best value in the column "
        "(ties within 0.0005 all bold). For `false abstention`, lower is better.\n",
    ]
    for (ds, tag), g in ret_df.groupby(["dataset", "run_tag"], sort=False):
        out.append(
            f"\n## {ds}  \n`{tag}` — {int(g['n_queries'].iloc[0])} queries with relevance labels\n"
        )
        merged = g.copy()
        rg = (
            ragas_df[(ragas_df["dataset"] == ds) & (ragas_df["run_tag"] == tag)]
            if not ragas_df.empty
            else pd.DataFrame()
        )
        cols = list(ABLATION_METRICS)
        extra = []
        if not rg.empty:
            extra = [m for m in RAGAS_METRICS + RUBRIC_METRICS if m in rg and rg[m].notna().any()]
        header = ["config"] + cols + extra
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        cfgs = list(
            dict.fromkeys(list(merged["config"]) + (list(rg["config"]) if not rg.empty else []))
        )
        cfgs = [c for c in CONFIG_ORDER if c in cfgs]
        best: dict[str, float] = {}
        for m in cols:
            v = merged.set_index("config")[m]
            best[m] = v.max() if v.notna().any() else float("nan")
        for m in extra:
            v = rg.set_index("config")[m]
            best[m] = v.max() if v.notna().any() else float("nan")
        for cfg in cfgs:
            cells = [cfg]
            for m in cols:
                row = merged[merged["config"] == cfg]
                if row.empty and cfg == "rrf_bge_two_pass":  # same retrieval as rrf_bge
                    row = merged[merged["config"] == "rrf_bge"]
                if row.empty:
                    cells.append("–")
                    continue
                r = row.iloc[0]
                s = fmt_ci(r[m], r[f"{m}_lo"], r[f"{m}_hi"])
                cells.append(f"**{s}**" if r[m] >= best[m] - 5e-4 else s)
            for m in extra:
                row = rg[rg["config"] == cfg]
                if row.empty:
                    cells.append("–")
                    continue
                r = row.iloc[0]
                s = fmt_ci(r[m], r[f"{m}_lo"], r[f"{m}_hi"])
                cells.append(f"**{s}**" if r[m] >= best[m] - 5e-4 else s)
            out.append("| " + " | ".join(cells) + " |")
        rb = (
            robust_df[(robust_df["dataset"] == ds) & (robust_df["run_tag"] == tag)]
            if not robust_df.empty
            else pd.DataFrame()
        )
        if not rb.empty:
            out.append(
                "\n| config | abstention on unanswerable (↑) | false abstention (↓) |\n|---|---|---|"
            )
            for _, r in rb.iterrows():
                a = (
                    fmt_ci(
                        r.get("abstention_rate_on_unanswerable", float("nan")),
                        r.get("abstention_rate_on_unanswerable_lo", float("nan")),
                        r.get("abstention_rate_on_unanswerable_hi", float("nan")),
                    )
                    if r.get("n_unanswerable", 0)
                    else "n/a"
                )
                f = (
                    fmt_ci(
                        r.get("false_abstention_rate", float("nan")),
                        r.get("false_abstention_rate_lo", float("nan")),
                        r.get("false_abstention_rate_hi", float("nan")),
                    )
                    if r.get("n_answerable", 0)
                    else "n/a"
                )
                out.append(
                    f"| {r['config']} | {a} (n={int(r['n_unanswerable'])}) | {f} (n={int(r['n_answerable'])}) |"
                )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--judge", default=C.JUDGE_MODEL)
    ap.add_argument("--rubric-judge", default=None, help="default: <judge>@think0")
    args = ap.parse_args()
    rubric_judge = args.rubric_judge or f"{args.judge.split('@')[0]}@think0"
    rdir = Path(args.results_dir)
    ret = latest_per_key(read_jsonl(rdir / "raw_retrieval.jsonl"))
    gen = latest_per_key(read_jsonl(rdir / "raw_generation.jsonl"))
    scores = read_jsonl(rdir / "ragas_scores.jsonl")
    rubric = read_jsonl(rdir / "rubric_scores.jsonl")
    lat = read_jsonl(rdir / "raw_latency.jsonl")
    rgb = read_jsonl(rdir / "raw_rgb.jsonl")

    rt = retrieval_table(ret + tolerant_page_rows(ret))
    rg = ragas_table(gen, scores, args.judge, rubric, rubric_judge)
    rc = rubric_table(gen, rubric, rubric_judge)
    rb = robustness_table(gen)
    sy = system_table(ret, gen, lat)
    sg = significance_table(ret, gen, scores, args.judge, rubric, rubric_judge)
    vis = visibility_table(gen)
    if not vis.empty:
        vis.to_csv(rdir / "table_visibility.csv", index=False)
        print(f"table_visibility.csv: {len(vis)} rows")
    n_audit = write_judge_audit(
        gen, scores, args.judge, rdir / "judge_audit.csv", rubric=rubric, rubric_judge=rubric_judge
    )
    print(f"judge_audit.csv: {n_audit} rows (human_label blank)")
    if any(r["dataset"].startswith("pdf-") for r in ret):
        from eval.pdf_sets import write_label_audits

        write_label_audits(rdir)
    rl = retrieval_latency_table(read_jsonl(rdir / "raw_latency_retrieval.jsonl"))
    if not rl.empty:
        rl.to_csv(rdir / "table_latency_retrieval.csv", index=False)
        print(f"table_latency_retrieval.csv: {len(rl)} rows")
    rgbt = rgb_table(rgb)
    if not rgbt.empty:
        rgbt.to_csv(rdir / "table_rgb.csv", index=False)
        print(f"table_rgb.csv: {len(rgbt)} rows")
    for name, df in (
        ("table_retrieval", rt),
        ("table_ragas", rg),
        ("table_correctness", rc),
        ("table_robustness", rb),
        ("table_system", sy),
        ("significance", sg),
    ):
        df.to_csv(rdir / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows")
    (rdir / "ablation.md").write_text(ablation_markdown(rt, rg, rb))


if __name__ == "__main__":
    main()
