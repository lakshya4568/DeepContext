"""Why did variant B score differently from fast ingestion? Measured diagnostics (no summariser needed).

    PYTHONPATH=. uv run --no-sync python -m eval.variant_b.diagnose --dataset docsqa \
        --fast eval/results_final/raw_retrieval.jsonl \
        --vb eval/results_variant_b/retrieval_docsqa/raw_retrieval.jsonl

Reads the fast and variant B Postgres schemas, the cached query embeddings and the saved summaries; writes
``eval/results_variant_b/<dataset>/diagnostics.json``. Query embeddings come from the on-disk call cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from eval import pg
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.datasets import build_task
from eval.make_tables import read_jsonl
from eval.metrics import rank_metrics
from eval.variant_b import glossary as G
from eval.variant_b import pipeline as P
from eval.variant_b import summarize as Z
from eval.variant_b.compare import by_query
from eval.variant_b.profiles import profile_for_dataset

STOP = frozenset(
    "a an and are as at be been by for from has have how in is it its of on or that the their this to was were what "
    "when where which who whom why with does do did can could should would will not no into than then there these "
    "those about after before between during each other such via".split()
)


def words(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[a-z0-9][a-z0-9_\-./]*", text.lower())
        if len(w) >= 3 and w not in STOP
    ]


def _vec(x: Any) -> np.ndarray:
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


async def child_vectors(schema: str) -> tuple[list[str], list[str], np.ndarray]:
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    await register_vector(conn)
    try:
        rows = await conn.fetch(
            f'SELECT c.id::text AS id, d.metadata AS meta, c.embedding FROM "{P._check_ident(schema)}".chunks c '
            f'JOIN "{schema}".documents d ON d.id = c.document_id WHERE c.level = $1 ORDER BY c.id',
            "child",
        )
    finally:
        await conn.close()
    ids = [r["id"] for r in rows]
    meta = [json.loads(r["meta"]) if isinstance(r["meta"], str) else dict(r["meta"]) for r in rows]
    docs = [m["external_id"] for m in meta]
    return ids, docs, _norm(np.stack([_vec(r["embedding"]) for r in rows]).astype(np.float64))


def doc_scores(q: np.ndarray, docs: list[str], m: np.ndarray) -> dict[str, float]:
    best: dict[str, float] = {}
    for d, s in zip(docs, m @ q, strict=True):
        if s > best.get(d, -9.0):
            best[d] = float(s)
    return best


def ndcg10(order: list[str], rel: dict[str, int]) -> float:
    return float(rank_metrics(order, rel)["ndcg@10"])


def first_rank(order: list[str], rel: set[str]) -> int | None:
    return next((i + 1 for i, d in enumerate(order) if d in rel), None)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    task = build_task(args.dataset)
    fp = corpus_fingerprint(task.corpus)[:10]
    src = pg.schema_name(f"{task.name}_{fp}")
    vb = f"{src}_vb"
    out_dir = Path(args.out) if args.out else Path("eval/results_variant_b") / task.name
    answerable = [
        q for q in sorted(task.queries) if task.qrels.get(q) and q not in task.unanswerable
    ]

    f_ids, f_docs, fm = await child_vectors(src)
    v_ids, v_docs, vm = await child_vectors(vb)
    assert f_ids == v_ids, "fast and variant B indexes must hold the same chunks"
    adapter = DeepContextAdapter(f"{task.name}_{fp}", "dense_only")
    qv = {}
    for qid in answerable:
        v = np.asarray(await adapter._query_vector(task.queries[qid]), dtype=np.float64)
        qv[qid] = v / np.linalg.norm(v)
    await adapter.close()

    summaries = [
        json.loads(ln)
        for ln in (out_dir / "summaries.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    summ = {r["id"]: r for r in summaries}
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    text = {
        r["id"]: r["content"]
        for r in await conn.fetch(f'SELECT id::text, content FROM "{src}".chunks')
    }
    await conn.close()
    docs_of = dict(zip(f_ids, f_docs, strict=True))

    # -------------------------------------------------------------- 1. does exact cosine reproduce the stored runs?
    fast_rows = by_query(read_jsonl(Path(args.fast)), task.name, "dense_only")
    vb_rows = by_query(read_jsonl(Path(args.vb)), task.name, "dense_only")
    per_q: list[dict[str, Any]] = []
    for qid in answerable:
        rel = task.qrels[qid]
        fs, vs = doc_scores(qv[qid], f_docs, fm), doc_scores(qv[qid], v_docs, vm)
        fo, vo = sorted(fs, key=lambda d: -fs[d]), sorted(vs, key=lambda d: -vs[d])
        gold = set(rel)
        g_f, g_v = max(fs[d] for d in gold if d in fs), max(vs[d] for d in gold if d in vs)
        c_f = max(s for d, s in fs.items() if d not in gold)
        c_v = max(s for d, s in vs.items() if d not in gold)
        per_q.append(
            {
                "qid": qid,
                "category": task.qtypes.get(qid, ""),
                "query": task.queries[qid],
                "gold": sorted(gold),
                "cos_ndcg_fast": ndcg10(fo, rel),
                "cos_ndcg_vb": ndcg10(vo, rel),
                "rank_fast": first_rank(fo, gold),
                "rank_vb": first_rank(vo, gold),
                "run_ndcg_fast": fast_rows[qid]["metrics"]["ndcg@10"] if qid in fast_rows else None,
                "run_ndcg_vb": vb_rows[qid]["metrics"]["ndcg@10"] if qid in vb_rows else None,
                "gold_sim_fast": g_f,
                "gold_sim_vb": g_v,
                "best_other_fast": c_f,
                "best_other_vb": c_v,
                "top1_vb": vo[0],
                "top1_fast": fo[0],
            }
        )
    reproduces = sum(
        abs(p["cos_ndcg_fast"] - (p["run_ndcg_fast"] or 0)) < 1e-9
        and abs(p["cos_ndcg_vb"] - (p["run_ndcg_vb"] or 0)) < 1e-9
        for p in per_q
    )
    reproduction = {
        "questions": len(per_q),
        "exact_cosine_matches_stored_dense_run": reproduces,
        "mean_ndcg10_fast_cosine": statistics.mean(p["cos_ndcg_fast"] for p in per_q),
        "mean_ndcg10_vb_cosine": statistics.mean(p["cos_ndcg_vb"] for p in per_q),
    }

    # -------------------------------------------------------------- 2. geometry of the two embedding spaces
    def crowd(m: np.ndarray) -> dict[str, float]:
        s = m @ m.T
        np.fill_diagonal(s, np.nan)
        return {
            "mean_pairwise_cosine": float(np.nanmean(s)),
            "mean_nearest_neighbour_cosine": float(np.nanmean(np.nanmax(s, axis=1))),
        }

    geometry = {
        "fast": crowd(fm),
        "variant_b": crowd(vm),
        "gold_similarity_mean": {
            "fast": statistics.mean(p["gold_sim_fast"] for p in per_q),
            "vb": statistics.mean(p["gold_sim_vb"] for p in per_q),
        },
        "best_other_similarity_mean": {
            "fast": statistics.mean(p["best_other_fast"] for p in per_q),
            "vb": statistics.mean(p["best_other_vb"] for p in per_q),
        },
        "margin_mean": {
            "fast": statistics.mean(p["gold_sim_fast"] - p["best_other_fast"] for p in per_q),
            "vb": statistics.mean(p["gold_sim_vb"] - p["best_other_vb"] for p in per_q),
        },
        "questions_margin_smaller_in_vb": sum(
            (p["gold_sim_vb"] - p["best_other_vb"]) < (p["gold_sim_fast"] - p["best_other_fast"])
            for p in per_q
        ),
        "query_gold_similarity_change_mean": statistics.mean(
            p["gold_sim_vb"] - p["gold_sim_fast"] for p in per_q
        ),
        "query_best_other_similarity_change_mean": statistics.mean(
            p["best_other_vb"] - p["best_other_fast"] for p in per_q
        ),
    }

    # -------------------------------------------------------------- 3. what the summaries look like
    kept = [r for r in summaries if r["status"] in P.SUMMARISED_OK]
    novel, ratio, unclear = [], [], 0
    starts: Counter[str] = Counter()
    for r in kept:
        own = set(words(text[r["id"]]))
        sw = words(r["summary"])
        novel.append(sum(w not in own for w in sw) / max(1, len(sw)))
        ratio.append(len(r["summary"].split()) / max(1, len(text[r["id"]].split())))
        unclear += any(b.strip().lower().startswith("- unclear") for b in r["summary"].splitlines())
        for b in r["summary"].splitlines():
            starts[" ".join(b.strip().lstrip("- ").split()[:2]).lower()] += 1
    summary_stats = {
        "summaries_embedded": len(kept),
        "with_an_unclear_bullet": unclear,
        "mean_share_of_summary_words_not_in_own_chunk": statistics.mean(novel),
        "p90_share_not_in_own_chunk": sorted(novel)[int(0.9 * (len(novel) - 1))],
        "mean_summary_to_chunk_length_ratio": statistics.mean(ratio),
        "most_common_bullet_openings": starts.most_common(8),
    }
    chunk_words = {r["id"]: len(text[r["id"]].split()) for r in summaries}
    sum_words = {r["id"]: len(r["summary"].split()) for r in kept}
    summary_stats.update(
        children=len(chunk_words),
        median_chunk_words=statistics.median(chunk_words.values()),
        chunks_under_100_words=sum(w < 100 for w in chunk_words.values()),
        summaries_longer_than_half_their_chunk=sum(
            s > 0.5 * chunk_words[i] for i, s in sum_words.items()
        ),
        summaries_longer_than_their_chunk=sum(s > chunk_words[i] for i, s in sum_words.items()),
    )

    # -------------------------------------------------------------- 4. where do words that are not in the chunk come from?
    profile = profile_for_dataset(task.name)
    groups = P.build_groups(task.name, task.corpus, profile)
    children = await P.load_children(src, groups, task.name)
    gl = json.loads((out_dir / "glossary.json").read_text())
    sy = json.loads((out_dir / "synopsis.json").read_text())
    by_group: dict[str, list[P.Child]] = defaultdict(list)
    for c in children:
        by_group[c.group].append(c)
    borrowed: Counter[str] = Counter()
    flagged: set[str] = set()
    n_entries: dict[str, int] = {}
    for key, kids in by_group.items():
        g = P.GroupContext(
            [G.Entry.from_json(e) for e in gl[key]], sy[key]["synopsis"], sy[key]["notes"], {}
        )
        previous: list[str] = []
        for child in sorted(kids, key=lambda c: (c.pos, c.ordinal)):
            ctx, n = P.child_context(profile, g, child, previous)
            n_entries[child.id] = n
            r = summ[child.id]
            if r["status"] in P.SUMMARISED_OK:
                slots = {
                    "glossary": Z._fold(ctx.glossary), "previous": Z._fold(" ".join(ctx.previous)),
                    "parent_context": Z._fold(ctx.parent_context), "synopsis": Z._fold(ctx.synopsis),
                    "title": Z._fold(ctx.title),
                }  # fmt: skip
                for tok in Z.unsupported_tokens(r["summary"], child.text):
                    where = next(
                        (s for s, t in slots.items() if Z._supported(tok, t)),
                        "no slot (model's own words)",
                    )
                    borrowed[where] += 1
                    flagged.add(child.id)
                previous.append(r["summary"])
    borrowed_stats = {
        "summaries_with_a_name_or_number_not_in_their_own_chunk": len(flagged),
        "of_summaries": len(kept),
        "tokens_by_where_they_came_from": dict(borrowed),
    }

    # -------------------------------------------------------------- 5. do the questions copy the chunk's words?
    recall_chunk, recall_sum, recall_title = [], [], []
    gold_children: dict[str, list[str]] = defaultdict(list)
    for cid, d in docs_of.items():
        gold_children[d].append(cid)
    for p in per_q:
        qw = set(words(p["query"]))
        if not qw:
            continue
        cids = [cid for d in p["gold"] for cid in gold_children.get(d, [])]
        recall_chunk.append(max(len(qw & set(words(text[c]))) / len(qw) for c in cids))
        recall_sum.append(max(len(qw & set(words(summ[c]["summary"]))) / len(qw) for c in cids))
        recall_title.append(
            max(len(qw & set(words(task.corpus[d]["title"]))) / len(qw) for d in p["gold"])
        )
        p["query_word_recall_in_chunk"], p["query_word_recall_in_summary"] = (
            recall_chunk[-1],
            recall_sum[-1],
        )
    term_recall = {
        "mean_share_of_question_words_found_in_gold_chunk": statistics.mean(recall_chunk),
        "mean_share_found_in_gold_summary": statistics.mean(recall_sum),
        "mean_share_found_in_gold_title": statistics.mean(recall_title),
    }

    # -------------------------------------------------------------- 6. which questions got worse, and why
    for p in per_q:
        p["delta_ndcg_cosine"] = p["cos_ndcg_vb"] - p["cos_ndcg_fast"]
        top = p["top1_vb"]
        p["vb_top1_is_gold"] = top in p["gold"]
        same_file = (
            P.group_and_order(task.name, top)[0] == P.group_and_order(task.name, p["gold"][0])[0]
        )
        p["vb_top1_same_file_as_gold"] = same_file
        gc = [c for d in p["gold"] for c in gold_children.get(d, [])]
        p["gold_summary_has_unclear"] = any(
            any(b.strip().lower().startswith("- unclear") for b in summ[c]["summary"].splitlines())
            for c in gc
        )
        p["gold_summary_flagged_borrowed"] = any(c in flagged for c in gc)
        p["gold_glossary_entries"] = max(n_entries.get(c, 0) for c in gc)
    worse = [p for p in per_q if p["delta_ndcg_cosine"] < -1e-9]
    categories: dict[str, dict[str, float]] = {}
    for cat in sorted({p["category"] for p in per_q}):
        sub = [p for p in per_q if p["category"] == cat]
        categories[cat] = {"n": len(sub), "mean_delta_ndcg10_dense": statistics.mean(p["delta_ndcg_cosine"] for p in sub),
                           "worse": sum(p["delta_ndcg_cosine"] < -1e-9 for p in sub), "better": sum(p["delta_ndcg_cosine"] > 1e-9 for p in sub)}  # fmt: skip

    def subgroup(flag: str) -> dict[str, Any]:
        out = {}
        for val in (True, False):
            sub = [p for p in per_q if bool(p[flag]) is val]
            out[str(val)] = {
                "n": len(sub),
                "mean_delta_ndcg10_dense": statistics.mean(p["delta_ndcg_cosine"] for p in sub)
                if sub
                else None,
            }
        return out

    breakdown = {
        "worse_questions": len(worse),
        "worse_where_vb_top1_is_in_the_same_file_as_gold": sum(
            p["vb_top1_same_file_as_gold"] for p in worse
        ),
        "by_category": categories,
        "by_gold_summary_has_unclear": subgroup("gold_summary_has_unclear"),
        "by_gold_summary_borrowed_a_name_or_number": subgroup("gold_summary_flagged_borrowed"),
    }

    # -------------------------------------------------------------- 7. reranker and fusion interplay
    interplay: dict[str, Any] = {}
    allrows = {name: (by_query(read_jsonl(Path(args.fast)), task.name, name), by_query(read_jsonl(Path(args.vb)), task.name, name)) for name in ("dense_only", "bm25_only", "rrf", "rrf_bge")}  # fmt: skip
    for name, (f, v) in allrows.items():
        qs = sorted(set(f) & set(v) & set(answerable))
        interplay[name] = {"n": len(qs), "mean_ndcg10_fast": statistics.mean(f[q]["metrics"]["ndcg@10"] for q in qs),
                           "mean_ndcg10_vb": statistics.mean(v[q]["metrics"]["ndcg@10"] for q in qs)}  # fmt: skip
        interplay[name]["delta"] = (
            interplay[name]["mean_ndcg10_vb"] - interplay[name]["mean_ndcg10_fast"]
        )
    f, v = allrows["rrf_bge"]
    qs = sorted(set(f) & set(v) & set(answerable))
    jac = [
        len(set(f[q]["ranked"][:24]) & set(v[q]["ranked"][:24]))
        / len(set(f[q]["ranked"][:24]) | set(v[q]["ranked"][:24]))
        for q in qs
    ]
    fd, vd = allrows["dense_only"]
    fr, vr = allrows["rrf"]
    interplay["reranked_pool_top24"] = {
        "mean_overlap_fast_vs_vb": statistics.mean(jac),
        "questions_with_identical_top24_set": sum(j == 1.0 for j in jac),
        "questions_with_identical_top24_order": sum(
            f[q]["ranked"][:24] == v[q]["ranked"][:24] for q in qs
        ),
    }
    if interplay["rrf"]["delta"]:
        interplay["share_of_hybrid_loss_recovered_by_reranker"] = (
            1 - interplay["rrf_bge"]["delta"] / interplay["rrf"]["delta"]
        )
    interplay["rrf_tracks_dense"] = {
        "dense_delta": interplay["dense_only"]["delta"], "rrf_delta": interplay["rrf"]["delta"],
        "questions_where_rrf_top1_equals_dense_top1_fast": sum(fr[q]["ranked"][0] == fd[q]["ranked"][0] for q in qs if q in fr and q in fd),
        "questions_where_rrf_top1_equals_dense_top1_vb": sum(vr[q]["ranked"][0] == vd[q]["ranked"][0] for q in qs if q in vr and q in vd),
    }  # fmt: skip

    result = {
        "dataset": task.name, "fast_schema": src, "variant_b_schema": vb,
        "reproduction": reproduction, "geometry": geometry, "summary_stats": summary_stats,
        "borrowed": borrowed_stats, "question_word_recall": term_recall, "breakdown": breakdown,
        "reranker_and_fusion": interplay,
        "per_child": [
            {"id": r["id"], "status": r["status"], "borrowed": r["id"] in flagged,
             "chunk_words": chunk_words[r["id"]], "summary_words": sum_words.get(r["id"], 0)}
            for r in summaries
        ],
        "per_question": per_q,
    }  # fmt: skip
    (out_dir / "diagnostics.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=float)
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--vb", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = asyncio.run(run(args))
    brief = {k: v for k, v in res.items() if k not in ("per_question", "per_child")}
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
