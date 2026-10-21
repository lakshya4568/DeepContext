"""Mechanical gap analysis: where and why does the pipeline lose points?

Every number is computed from ``raw_retrieval.jsonl`` / ``raw_generation.jsonl`` (no model calls).
Writes ``gap_analysis.md`` and ``gap_analysis.json`` in the results directory.

Sections
  A. merge effect       - per query, does RRF place the first relevant page above or below dense-only?
  B. reranker limits    - how often the first relevant page is outside the 24-candidate rerank window
  C. misses by cause    - first relevant rank buckets per config
  D. engine behaviour   - declines, query shapes, ≤6-document limit
  E. PDF label noise    - pages per query, strict vs ±1 page
  F. generation losses  - (needs raw_generation) gold retrieved?, visible under the 12k cap?, abstentions,
                          hallucinated answers on unanswerable questions, one-pass vs two-pass
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval import config as C
from eval.make_tables import latest_per_key, read_jsonl


def first_rank(ranked: list[str], rel: set[str]) -> int | None:
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return i
    return None


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "–"


def table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def retrieval_sections(ret: list[dict[str, Any]], out: list[str], data: dict[str, Any]) -> None:
    by: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in ret:
        if r["relevant"] and not r["dataset"].startswith("pdf-got (") and "±" not in r["dataset"]:
            by[(r["dataset"], r["run_tag"])][r["config"]][r["qid"]] = r

    # A. merge effect
    out.append("## A. Does merging with the word search help or hurt? (dense_only vs rrf)\n")
    out.append(
        "Per question: position of the first correct page. 'hurt' = rrf puts it lower than dense-only, "
        "'helped' = higher. 'lexical found nothing useful' = the word-search list had no correct page in its top 100.\n"
    )
    rows = []
    for (ds, tag), cfgs in by.items():
        if "dense_only" not in cfgs or "rrf" not in cfgs or "bm25_only" not in cfgs:
            continue
        hurt = helped = same = lex_none = n = 0
        for qid, d in cfgs["dense_only"].items():
            r = cfgs["rrf"].get(qid)
            b = cfgs["bm25_only"].get(qid)
            if r is None or b is None:
                continue
            rel = set(d["relevant"])
            rd, rr = first_rank(d["ranked"], rel) or 999, first_rank(r["ranked"], rel) or 999
            rb = first_rank(b["ranked"], rel)
            n += 1
            lex_none += rb is None
            hurt += rr > rd
            helped += rr < rd
            same += rr == rd
        rows.append([ds, n, pct(hurt, n), pct(helped, n), pct(same, n), pct(lex_none, n)])
        data.setdefault("merge_effect", {})[ds] = {
            "n": n, "hurt": hurt, "helped": helped, "same": same, "lexical_found_nothing": lex_none
        }  # fmt: skip
    out.append(
        table(["dataset", "n", "hurt", "helped", "same", "lexical found nothing useful"], rows)
    )

    # B. reranker window
    out.append("## B. What the reranker receives (it scores every fused candidate, up to 100, and keeps the best 24)\n")
    rows = []
    for (ds, tag), cfgs in by.items():
        if "rrf" not in cfgs:
            continue
        n = outside = missing = fixable = 0
        for qid, r in cfgs["rrf"].items():
            rel = set(r["relevant"])
            fr = first_rank(r["ranked"], rel)
            n += 1
            if fr is None:
                missing += 1
            elif fr > 24:
                outside += 1
            else:
                fixable += 1
        rows.append([ds, n, pct(fixable, n), pct(outside, n), pct(missing, n)])
    out.append(table(["dataset", "n", "correct page already in fused top 24",
                      "correct page at fused rank 25–100 (reranker still scores it and can promote it)",
                      "not in the fused top 100 (reranker never sees it)"], rows))  # fmt: skip

    # C. where the first correct page lands, per config
    out.append("## C. Where does the first correct page land? (share of questions)\n")
    rows = []
    for (ds, tag), cfgs in sorted(by.items()):
        for cfg in ("dense_only", "bm25_only", "rrf", "rrf_bge", C.ENGINE_CONFIG):
            if cfg not in cfgs:
                continue
            ranks = [first_rank(r["ranked"], set(r["relevant"])) for r in cfgs[cfg].values()]
            n = len(ranks)
            rows.append([ds, cfg, n,
                         pct(sum(1 for x in ranks if x == 1), n),
                         pct(sum(1 for x in ranks if x and 2 <= x <= 5), n),
                         pct(sum(1 for x in ranks if x and 6 <= x <= 24), n),
                         pct(sum(1 for x in ranks if x and 25 <= x <= 100), n),
                         pct(sum(1 for x in ranks if x is None), n)])  # fmt: skip
    out.append(
        table(["dataset", "config", "n", "rank 1", "2–5", "6–24", "25–100", "not found"], rows)
    )

    # D. engine behaviour
    out.append("## D. The full system ('engine_as_shipped'): declines and shapes\n")
    rows = []
    eng: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ret:
        if r["config"] == C.ENGINE_CONFIG and r["relevant"] and "±" not in r["dataset"]:
            eng[r["dataset"]].append(r)
    for ds, rs in sorted(eng.items()):
        n = len(rs)
        declined = sum(1 for r in rs if r.get("engine_sufficient") is False)
        empty = sum(1 for r in rs if not r["ranked"])
        retries = sum(1 for r in rs if (r.get("engine_retry_count") or 0) > 0)
        shapes = Counter(r.get("engine_query_shape") for r in rs)
        rows.append([ds, n, pct(declined, n), pct(empty, n), pct(retries, n),
                     ", ".join(f"{k}:{v}" for k, v in shapes.most_common(4))])  # fmt: skip
        data.setdefault("engine", {})[ds] = {
            "n": n,
            "declined": declined,
            "empty": empty,
            "retried": retries,
        }
    out.append(
        table(
            [
                "dataset",
                "n",
                "declined as insufficient",
                "returned no document",
                "needed a retry",
                "query shapes",
            ],
            rows,
        )
    )

    # E. PDF labels
    out.append("## E. PDF case study: how many pages does a question point to?\n")
    rows = []
    for ds in ("pdf-got", "pdf-quantum"):
        rs = [
            r
            for (d, _t), cfgs in by.items()
            if d == ds
            for r in cfgs.get("dense_only", {}).values()
        ]
        if not rs:
            continue
        sizes = sorted(len(r["relevant"]) for r in rs)
        rows.append(
            [
                ds,
                len(rs),
                sizes[len(sizes) // 2],
                max(sizes),
                pct(sum(1 for s in sizes if s >= 4), len(sizes)),
            ]
        )
    out.append(
        table(
            [
                "set",
                "answerable questions",
                "median pages per question",
                "max",
                "questions with ≥4 pages",
            ],
            rows,
        )
    )
    out.append(
        "A question with many 'correct' pages is hard to score: Recall@k can never reach 1 until all of them are in the top k, "
        "and the labels are rough (see the label audit in REPORT.md §2).\n"
    )


def generation_sections(gen: list[dict[str, Any]], out: list[str], data: dict[str, Any]) -> None:
    out.append("## F. Where answers lose points\n")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for g in gen:
        groups[(g["dataset"], g["config"])].append(g)
    rows = []
    for (ds, cfg), rs in sorted(groups.items()):
        ans = [g for g in rs if not g["unanswerable"]]
        una = [g for g in rs if g["unanswerable"]]
        n = len(ans)
        if not n:
            continue
        no_gold = sum(1 for g in ans if g.get("gold_in_context") is False)
        cut = sum(1 for g in ans if g.get("gold_in_context") and not g.get("gold_in_seen_context"))
        seen = n - no_gold - cut
        refused = sum(1 for g in ans if g["abstained"])
        false_abst_seen = sum(1 for g in ans if g["abstained"] and g.get("gold_in_seen_context"))
        all_seen = [g for g in ans if (g.get("gold_frac_seen") or 0) >= 0.999]
        false_abst_all = sum(1 for g in all_seen if g["abstained"])
        hallu = sum(1 for g in una if not g["abstained"])
        rows.append([ds, cfg, n, pct(no_gold, n), pct(cut, n), pct(seen, n), pct(refused, n),
                     pct(false_abst_seen, n), pct(len(all_seen), n), pct(false_abst_all, len(all_seen)),
                     f"{hallu}/{len(una)}" if una else "–"])  # fmt: skip
        data.setdefault("generation", {})[f"{ds}|{cfg}"] = {
            "answerable": n, "gold_not_retrieved": no_gold, "gold_cut_by_budget": cut,
            "gold_visible": seen, "abstained": refused, "all_gold_visible": len(all_seen), "refused_with_all_gold_visible": false_abst_all, "false_abstain_with_gold_visible": false_abst_seen,
            "answered_unanswerable": hallu, "unanswerable": len(una),
        }  # fmt: skip
    out.append(
        table(
            [
                "dataset",
                "config",
                "answerable",
                "correct page not retrieved",
                "retrieved but cut by the 12k-character limit",
                "correct page visible to the writer",
                "refused to answer",
                "refused although page visible",
                "answered an unanswerable question",
            ],
            rows,
        )
    )
    out.append(
        "'Gold' = the correct page(s). 'Visible' means at least one correct page's text reached the writer after the "
        "12,000-character evidence limit both generators apply.\n"
    )
    # one-pass vs two-pass refusals
    out.append("## G. One-pass vs two-pass writing: refusal behaviour\n")
    rows = []
    for ds in sorted({g["dataset"] for g in gen}):
        for cfg in ("rrf_bge", "rrf_bge_two_pass"):
            rs = [g for g in groups.get((ds, cfg), []) if not g["unanswerable"]]
            if rs:
                rows.append([ds, cfg, len(rs), pct(sum(1 for g in rs if g["abstained"]), len(rs))])
    out.append(table(["dataset", "config", "answerable", "refused"], rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    rd = Path(a.results_dir)
    ret = latest_per_key(read_jsonl(rd / "raw_retrieval.jsonl"))
    gen = latest_per_key(read_jsonl(rd / "raw_generation.jsonl"))
    out: list[str] = [
        "# Gap analysis (mechanical, from raw result files)\n",
        "Everything below is computed from `raw_retrieval.jsonl` and `raw_generation.jsonl`; no model was called.\n",
    ]
    data: dict[str, Any] = {}
    retrieval_sections(ret, out, data)
    if gen:
        generation_sections(gen, out, data)
    (rd / "gap_analysis.md").write_text("\n".join(out))
    (rd / "gap_analysis.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {rd / 'gap_analysis.md'}")


if __name__ == "__main__":
    main()
