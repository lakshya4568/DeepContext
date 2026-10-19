"""Generate failure_analysis.md: worst queries per dataset/config with a mechanical diagnosis.

Diagnoses are rule-based on stored evidence (ranks, context visibility, abstention flags, RAGAS
scores), so every label is reproducible from the raw files. Nothing here is hand-written prose
about specific queries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval import config as C
from eval.make_tables import CONFIG_ORDER, latest_per_key, read_jsonl


def first_rel_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return i
    return None


def diagnose_retrieval(r: dict[str, Any]) -> str:
    rel = set(r["relevant"])
    rank = first_rel_rank(r["ranked"], rel)
    if r.get("config") == C.ENGINE_CONFIG:
        if r.get("engine_sufficient") is False:
            return (
                f"engine declined: insufficient evidence ({r.get('engine_insufficiency_reason')})"
            )
        if rank is None:
            return "retrieval miss (no relevant doc among the engine's ≤6 returned documents)"
    if rank is None:
        return "retrieval miss (no relevant doc in top-100)"
    if rank > 5:
        return f"ranking error (first relevant at rank {rank})"
    if len(rel) > 1 and r["metrics"]["evidence_all@10"] < 1:
        return "partial evidence (multi-doc query; not all required docs in top-10)"
    return "ok at rank ≤5 (nDCG penalty only)"


def diagnose_generation(g: dict[str, Any], score: dict[str, Any] | None) -> str:
    if g["unanswerable"]:
        return (
            "hallucinated answer to unanswerable query"
            if not g["abstained"]
            else "ok (correct abstention)"
        )
    gold_ctx = g.get("gold_in_context")
    gold_seen = g.get("gold_in_seen_context")
    if g["abstained"]:
        if gold_ctx is False:
            return "false abstention — gold evidence was NOT retrieved (retrieval miss)"
        if gold_ctx and not gold_seen:
            return "false abstention — gold retrieved but cut by the 12k-char evidence budget"
        return "false abstention — gold evidence was visible to the generator"
    if gold_ctx is False:
        return "wrong generation — gold not retrieved (answer built on other docs)"
    if gold_ctx and not gold_seen:
        return "wrong generation — gold retrieved but cut by the 12k-char evidence budget"
    if score is not None:
        f, fc = score.get("faithfulness"), score.get("factual_correctness_f1")
        if f is not None and f == f and f < 0.5:
            return "hallucination (faithfulness < 0.5 with gold visible)"
        if fc is not None and fc == fc and fc < 0.3:
            return "wrong generation (gold visible, factual correctness < 0.3)"
    return "ok / minor"


def snippet(text: str, n: int = 160) -> str:
    return text.replace("\n", " ").replace("|", "/")[:n]


def reranker_evidence(ret: list[dict[str, Any]]) -> list[str]:
    by: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in ret:
        if r["relevant"]:
            by[(r["dataset"], r["run_tag"])][r["config"]][r["qid"]] = r
    out: list[str] = [
        "\n## Where did the BGE reranker help or hurt? (rrf → rrf_bge)\n",
        "Rank = position of the first relevant document in the deduplicated top-100. Lower is better.\n",
    ]
    for (ds, tag), cfgs in by.items():
        if "rrf" not in cfgs or "rrf_bge" not in cfgs:
            continue
        rows = []
        for qid, a in cfgs["rrf"].items():
            b = cfgs["rrf_bge"].get(qid)
            if b is None:
                continue
            rel = set(a["relevant"])
            ra, rb = first_rel_rank(a["ranked"], rel), first_rel_rank(b["ranked"], rel)
            rows.append((qid, a, b, ra or 101, rb or 101))
        if not rows:
            continue
        better = [x for x in rows if x[4] < x[3]]
        worse = [x for x in rows if x[4] > x[3]]
        same = len(rows) - len(better) - len(worse)
        d_ndcg = np.mean(
            [b["metrics"]["ndcg@10"] - a["metrics"]["ndcg@10"] for _, a, b, _, _ in rows]
        )
        out.append(f"\n### {ds} (`{tag}`, n={len(rows)})\n")
        out.append(
            f"- first-relevant rank improved for **{len(better)}** queries, worsened for **{len(worse)}**, unchanged for **{same}**; "
            f"mean Δ nDCG@10 = {d_ndcg:+.4f}."
        )
        beyond = [x for x in rows if x[3] > 24 or x[4] > 24]
        out.append(
            f"- the reranker only reorders the top 24 candidates; {len(beyond)} queries have their first relevant doc outside that window in at least one config (no reranker effect possible)."
        )
        multi = [x for x in rows if len(x[1]["relevant"]) > 1]
        if multi:
            dm = np.mean(
                [b["metrics"]["ndcg@10"] - a["metrics"]["ndcg@10"] for _, a, b, _, _ in multi]
            )
            out.append(
                f"- multi-relevant-doc queries (n={len(multi)}): mean Δ nDCG@10 = {dm:+.4f}."
            )
        for title, group, key in (
            ("largest rank gains", better, lambda x: x[3] - x[4]),
            ("largest rank losses", worse, lambda x: x[4] - x[3]),
        ):
            if not group:
                continue
            out.append(
                f"\n*{title}*\n\n| qid | query | rank rrf → bge | top-3 rrf | top-3 bge |\n|---|---|---|---|---|"
            )
            for qid, a, b, ra, rb in sorted(group, key=key, reverse=True)[:5]:
                rel = set(a["relevant"])
                mark = lambda ids: ", ".join(("✔" if d in rel else "") + d[:18] for d in ids[:3])  # noqa: E731
                out.append(
                    f"| {qid} | {snippet(a['query'], 90)} | {ra if ra <= 100 else '>100'} → {rb if rb <= 100 else '>100'} | {mark(a['ranked'])} | {mark(b['ranked'])} |"
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--judge", default=C.JUDGE_MODEL)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    ret = latest_per_key(read_jsonl(rdir / "raw_retrieval.jsonl"))
    gen = latest_per_key(read_jsonl(rdir / "raw_generation.jsonl"))
    scores = {
        (s["run_tag"], s["config"], s["qid"]): s
        for s in read_jsonl(rdir / "ragas_scores.jsonl")
        if s["judge_model"] == args.judge
    }

    out = [
        "# Failure analysis\n",
        f"Worst {args.top} queries per dataset and config. Labels are rule-based (see `eval/failure_analysis.py`). "
        "Retrieval rows are ordered by nDCG@10 ascending; generation rows by RAGAS factual-correctness F1 ascending "
        "(unscored rows first by diagnosis severity).\n",
    ]

    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in ret:
        if r["relevant"]:
            groups[(r["dataset"], r["run_tag"])][r["config"]].append(r)
    for (ds, tag), cfgs in groups.items():
        out.append(f"\n## Retrieval — {ds} (`{tag}`)\n")
        for cfg in [c for c in CONFIG_ORDER if c in cfgs]:
            rs = sorted(
                cfgs[cfg],
                key=lambda r: (r["metrics"]["ndcg@10"], r["metrics"]["hit@5"], r["metrics"]["mrr"]),
            )[: args.top]
            out.append(
                f"\n### {cfg}\n\n| qid | query | nDCG@10 | diagnosis | gold | top-3 retrieved |\n|---|---|---|---|---|---|"
            )
            for r in rs:
                rel = set(r["relevant"])
                top3 = ", ".join(("✔" if d in rel else "") + d[:20] for d in r["ranked"][:3])
                out.append(
                    f"| {r['qid']} | {snippet(r['query'], 100)} | {r['metrics']['ndcg@10']:.2f} | {diagnose_retrieval(r)} | {', '.join(sorted(rel))[:60]} | {top3} |"
                )
    out += reranker_evidence(ret)

    ggroups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for g in gen:
        ggroups[(g["dataset"], g["run_tag"])][g["config"]].append(g)
    for (ds, tag), cfgs in ggroups.items():
        out.append(f"\n## Generation — {ds} (`{tag}`)\n")
        for cfg in [c for c in CONFIG_ORDER if c in cfgs]:

            def sort_key(g: dict[str, Any]) -> tuple[float, int]:
                sc = scores.get((g["run_tag"], g["config"], g["qid"]))
                fc = (
                    sc["factual_correctness_f1"]
                    if sc and sc["factual_correctness_f1"] == sc["factual_correctness_f1"]
                    else 9.0
                )
                bad = 0 if (g["abstained"] != g["unanswerable"]) else 1
                return (bad, fc)  # type: ignore[return-value]

            worst = sorted(cfgs[cfg], key=sort_key)[: args.top]
            counts: dict[str, int] = defaultdict(int)
            for g in cfgs[cfg]:
                counts[
                    diagnose_generation(g, scores.get((g["run_tag"], g["config"], g["qid"])))
                ] += 1
            out.append(
                f"\n### {cfg}\n\nDiagnosis counts over all {len(cfgs[cfg])} queries: "
                + "; ".join(f"{k} = {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
                + "\n"
            )
            out.append(
                "| qid | query | diagnosis | answer (start) | reference (start) | FC-F1 |\n|---|---|---|---|---|---|"
            )
            for g in worst:
                sc = scores.get((g["run_tag"], g["config"], g["qid"]))
                fc = (
                    f"{sc['factual_correctness_f1']:.2f}"
                    if sc and sc["factual_correctness_f1"] == sc["factual_correctness_f1"]
                    else "–"
                )
                out.append(
                    f"| {g['qid']} | {snippet(g['query'], 80)} | {diagnose_generation(g, sc)} | {snippet(g['answer'], 110)} | {snippet(g['reference'], 70)} | {fc} |"
                )
    (rdir / "failure_analysis.md").write_text("\n".join(out) + "\n")
    print(f"wrote {rdir / 'failure_analysis.md'}")


if __name__ == "__main__":
    main()
