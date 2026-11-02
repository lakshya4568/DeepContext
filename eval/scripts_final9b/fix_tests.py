"""Remedy tests for the loss of deep ingestion (post-hoc; the questions are the benchmark's own, so read them as exploration, not as a new benchmark).

``ensemble``  full corpus, exact dense search (pgvector scan, no approximate index): fast vector only, deep vector only, the mean of the two
              cosines, and the larger of the two. Uses the vectors already stored in both indexes: no new embedding.
``rrf``       from the saved rankings: dense list of one arm fused with the keyword list of the same or the other arm (document-level
              reciprocal rank fusion, k = 60, as the application does at chunk level); checked against the saved fused rankings
``ablation``  FiQA subset, on the documents each question's candidates come from (its relevant documents plus the top 10 of the fast
              and of the deep dense ranking): the chunk embedded with (A) only the summary's opening words, (B) only the summary body
              without them, next to the fast text, the deep text and the two ensembles. Needs about 10,000 new embeddings.

    PYTHONPATH=.:src SUMMARY_MODEL=Qwen/Qwen3.5-9B EVAL_STORE=stock uv run --no-sync python eval/scripts_final9b/fix_tests.py ensemble --dataset fiqa-sub
    PYTHONPATH=.:src SUMMARY_MODEL=Qwen/Qwen3.5-9B EVAL_STORE=stock uv run --no-sync python eval/scripts_final9b/fix_tests.py ablation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
from eval import config as C
from eval import pg
from eval.main_summaries import task_for
from eval.metrics import rank_metrics
from eval.stats import paired_test
from eval.strict_clients import _sha, get_runtime

RESULTS = Path("eval/results_final9b")
ANCHOR = re.compile(r"^In .*?\([^)]*\),\s*")


def _query_vector(rt: Any, query: str) -> np.ndarray | None:
    formatted = rt.format_embedding_text(query, is_query=True, title=None, task=None)
    hit = rt.cache.get(_sha("emb", C.EMBEDDING_MODEL, C.EMBEDDING_DIM, formatted))
    return None if hit is None else np.frombuffer(hit, dtype=np.float32)


def _literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


async def _schemas(conn: asyncpg.Connection, dataset: str) -> tuple[str, str]:
    names = [
        r["nspname"]
        for r in await conn.fetch(
            "select nspname from pg_namespace where nspname like $1",
            dataset.replace("-", "_") + "_%",
        )
    ]
    return next(n for n in names if n.endswith("_main")), next(
        n for n in names if n.endswith("_maindeep9b")
    )


def _line(label: str, values: dict[str, list[float]], base: str, key: str) -> str:
    mine, ref = values[label], values[base]
    res = paired_test(ref, mine)
    tag = (
        ""
        if label == base
        else f" ({res.mean_diff:+.3f} [{res.ci_lo:+.3f}, {res.ci_hi:+.3f}] against {base})"
    )
    return f"| {label} | {sum(mine) / len(mine):.3f}{tag} |"


async def ensemble(dataset: str) -> None:
    rt = get_runtime()
    task = task_for(dataset)
    conn = await asyncpg.connect(pg.dsn_for("deepcontext_eval"))
    try:
        fast_s, deep_s = await _schemas(conn, dataset)
        sql = (
            f"with s as (select d.metadata->>'external_id' as ext, 1 - (cf.embedding <=> $1::vector) as sf, "
            f'1 - (cd.embedding <=> $1::vector) as sd from "{deep_s}".chunks cd '
            f'join "{fast_s}".chunks cf on cf.id = cd.id join "{deep_s}".documents d on d.id = cd.document_id where cd.level = \'child\') '
            f"select ext, max(sf) as sf, max(sd) as sd, max(0.5 * (sf + sd)) as smean, max(greatest(sf, sd)) as smax from s group by ext"
        )
        variants = ("fast vector", "deep vector", "mean of the two", "larger of the two")
        columns = ("sf", "sd", "smean", "smax")
        metrics = {m: {v: [] for v in variants} for m in ("ndcg@10", "hit@5", "recall@100")}
        n = 0
        for qid, query in task.queries.items():
            rel = {d: 1 for d in task.qrels.get(qid, {})}
            qv = _query_vector(rt, query)
            if not rel or qv is None:
                continue
            rows = await conn.fetch(sql, _literal(qv / np.linalg.norm(qv)))
            n += 1
            for variant, column in zip(variants, columns, strict=True):
                ranked = [r["ext"] for r in sorted(rows, key=lambda r: -r[column])[:100]]
                m = rank_metrics(ranked, rel)
                for name in metrics:
                    metrics[name][variant].append(m[name])
        print(f"### {dataset}: exact dense search over the whole corpus, {n} questions\n")
        print(
            "| dense score used | nDCG@10 (difference to the fast vector, 95% interval) | Hit@5 | Recall@100 |"
        )
        print("|---|---|---|---|")
        for variant in variants:
            cells = []
            for name in ("ndcg@10", "hit@5", "recall@100"):
                mine, ref = metrics[name][variant], metrics[name]["fast vector"]
                res = paired_test(ref, mine)
                tag = (
                    ""
                    if variant == "fast vector"
                    else f" ({res.mean_diff:+.3f} [{res.ci_lo:+.3f}, {res.ci_hi:+.3f}])"
                )
                cells.append(f"{sum(mine) / len(mine):.3f}{tag}")
            print(f"| {variant} | " + " | ".join(cells) + " |")
        print()
    finally:
        await conn.close()


async def ablation(
    n_questions: int, dataset: str, max_cost_usd: float, count_only: bool = False
) -> None:
    rt = get_runtime()
    task = task_for(dataset)
    fast_rows, deep_rows = {}, {}
    for folder, store in (("R_fast_equal", fast_rows), ("R_deep_equal", deep_rows)):
        for line in (RESULTS / folder / "raw_retrieval.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if (
                r["dataset"] == dataset
                and r["config"] == "dense_only"
                and "error" not in r
                and r["relevant"]
            ):
                store[r["qid"]] = r
    qids = sorted(set(fast_rows) & set(deep_rows))
    random.Random(21).shuffle(qids)
    qids = [q for q in qids if _query_vector(rt, task.queries[q]) is not None][:n_questions]
    pools = {
        q: set(fast_rows[q]["relevant"])
        | set(fast_rows[q]["ranked"][:10])
        | set(deep_rows[q]["ranked"][:10])
        for q in qids
    }
    all_docs = sorted(set().union(*pools.values()))
    conn = await asyncpg.connect(pg.dsn_for("deepcontext_eval"))
    try:
        fast_s, deep_s = await _schemas(conn, dataset)
        chunk_rows = await conn.fetch(
            f"select cd.id, cd.content, cd.summary_text, d.title, d.metadata->>'external_id' as ext, cf.embedding::real[] as fe, cd.embedding::real[] as de "
            f'from "{deep_s}".chunks cd join "{fast_s}".chunks cf on cf.id = cd.id join "{deep_s}".documents d on d.id = cd.document_id '
            f"where cd.level = 'child' and d.metadata->>'external_id' = any($1::text[])",
            all_docs,
        )
    finally:
        await conn.close()
    print(
        f"{len(qids)} questions, {len(all_docs)} candidate documents, {len(chunk_rows)} chunks",
        flush=True,
    )

    # the two new texts for every chunk
    texts: dict[str, list[tuple[Any, str]]] = {"opening only": [], "body only": []}
    for r in chunk_rows:
        summary = r["summary_text"] or ""
        match = ANCHOR.match(summary)
        opening, body = (match.group(0).strip(), summary[match.end() :]) if match else ("", summary)
        title = r["title"] if r["title"] != r["ext"] else None
        texts["opening only"].append(
            (
                r,
                rt.format_embedding_text(
                    f"{opening}\n\n{r['content']}" if opening else r["content"],
                    is_query=False,
                    title=title,
                    task=None,
                ),
            )
        )
        texts["body only"].append(
            (
                r,
                rt.format_embedding_text(
                    f"{body}\n\n{r['content']}" if body else r["content"],
                    is_query=False,
                    title=title,
                    task=None,
                ),
            )
        )
    all_tokens = sum(
        max(len(t.split()), len(t) // 4) for variant in texts.values() for _r, t in variant
    )
    print(
        f"{dataset}: all variant texts: {sum(len(v) for v in texts.values())}, about {all_tokens:,} tokens = ${all_tokens * 0.20 / 1e6:.2f} at the list price if none were cached",
        flush=True,
    )
    if count_only:
        return
    todo = {
        t
        for variant in texts.values()
        for _r, t in variant
        if rt.cache.get(_sha("emb", C.EMBEDDING_MODEL, C.EMBEDDING_DIM, t)) is None
    }
    approx_tokens = sum(max(len(t.split()), len(t) // 4) for t in todo)
    print(
        f"new embeddings needed: {len(todo)} (about {approx_tokens:,} tokens, about ${approx_tokens * 0.20 / 1e6:.2f} at the list price of ${0.20}/M)",
        flush=True,
    )
    if approx_tokens * 0.20 / 1e6 > max_cost_usd:
        raise SystemExit(
            f"stopped: the estimate is above the limit of ${max_cost_usd:.2f} (set --max-cost to allow it)"
        )
    vectors: dict[str, dict[Any, np.ndarray]] = {}
    for variant, items in texts.items():
        embedded = await rt.embed_many(
            [t for _r, t in items], model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM
        )
        vectors[variant] = {r["id"]: v for (r, _t), v in zip(items, embedded, strict=True)}
        rt.cache.flush()
    vectors["fast"] = {r["id"]: np.asarray(r["fe"], dtype=np.float32) for r in chunk_rows}
    vectors["deep"] = {r["id"]: np.asarray(r["de"], dtype=np.float32) for r in chunk_rows}

    labels = (
        "fast",
        "deep",
        "opening only",
        "body only",
        "mean of fast and deep",
        "larger of fast and deep",
    )
    scores: dict[str, list[float]] = {label: [] for label in labels}
    for qid in qids:
        q = _unit(_query_vector(rt, task.queries[qid]))
        rel = {d: 1 for d in task.qrels[qid]}
        best: dict[str, dict[str, float]] = {label: {} for label in labels}
        for r in chunk_rows:
            if r["ext"] not in pools[qid]:
                continue
            cos = {
                name: float(_unit(vectors[name][r["id"]]) @ q)
                for name in ("fast", "deep", "opening only", "body only")
            }
            cos["mean of fast and deep"] = 0.5 * (cos["fast"] + cos["deep"])
            cos["larger of fast and deep"] = max(cos["fast"], cos["deep"])
            for label in labels:
                best[label][r["ext"]] = max(best[label].get(r["ext"], -1.0), cos[label])
        for label in labels:
            ranked = [d for d, _s in sorted(best[label].items(), key=lambda kv: -kv[1])]
            scores[label].append(rank_metrics(ranked, rel)["ndcg@10"])
    print(
        f"\n### {dataset}: re-ranking the candidates of {len(qids)} questions with different embedded texts (nDCG@10 inside the candidate set)\n"
    )
    print("| text that is embedded | nDCG@10 (difference to fast, 95% interval) |")
    print("|---|---|")
    for label in labels:
        print(_line(label, scores, "fast", "ndcg@10"))


def rrf_mix(dataset: str) -> None:
    def load(folder: str, config: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for line in (RESULTS / folder / "raw_retrieval.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if (
                r["dataset"] == dataset
                and r["config"] == config
                and "error" not in r
                and r["relevant"]
            ):
                out[r["qid"]] = r
        return out

    def fuse(lists: list[list[str]], weights: tuple[float, float]) -> list[str]:
        score: dict[str, float] = {}
        for weight, ranked in zip(weights, lists, strict=True):
            for rank, doc in enumerate(ranked, start=1):
                score[doc] = score.get(doc, 0.0) + weight / (60 + rank)
        return [d for d, _s in sorted(score.items(), key=lambda kv: -kv[1])][:100]

    fd, fk, dd, dk = (
        load("R_fast_equal", "dense_only"),
        load("R_fast_equal", "bm25_only"),
        load("R_deep_equal", "dense_only"),
        load("R_deep_equal", "bm25_only"),
    )
    saved = {
        (1.0, "fast"): load("R_fast_equal", "rrf"),
        (1.0, "deep"): load("R_deep_equal", "rrf"),
        (0.02, "fast"): load("R_fast_vote0.02", "rrf"),
        (0.02, "deep"): load("R_deep_vote0.02", "rrf"),
    }
    qids = sorted(set(fd) & set(fk) & set(dd) & set(dk))
    print(
        f"### {dataset}: fusing the saved dense and keyword rankings (document level), {len(qids)} questions\n"
    )
    print(
        "| keyword vote | dense list from | keyword list from | nDCG@10 (difference to fast and fast) | the application's own saved number |"
    )
    print("|---|---|---|---|---|")
    for vote in (1.0, 0.02):
        results: dict[str, list[float]] = {}
        for dense_arm, key_arm in (
            ("fast", "fast"),
            ("deep", "deep"),
            ("fast", "deep"),
            ("deep", "fast"),
        ):
            d, k = (fd if dense_arm == "fast" else dd), (fk if key_arm == "fast" else dk)
            results[f"{dense_arm}/{key_arm}"] = [
                rank_metrics(
                    fuse([d[q]["ranked"], k[q]["ranked"]], (1.0, vote)),
                    {x: 1 for x in fd[q]["relevant"]},
                )["ndcg@10"]
                for q in qids
            ]
        for (dense_arm, key_arm), values in ((tuple(k.split("/")), v) for k, v in results.items()):
            ref = results["fast/fast"]
            res = paired_test(ref, values)
            tag = (
                ""
                if (dense_arm, key_arm) == ("fast", "fast")
                else f" ({res.mean_diff:+.3f} [{res.ci_lo:+.3f}, {res.ci_hi:+.3f}])"
            )
            own = ""
            if dense_arm == key_arm:
                rows = saved[(vote, dense_arm)]
                own = f"{sum(rows[q]['metrics']['ndcg@10'] for q in qids) / len(qids):.3f}"
            print(
                f"| {vote} | {dense_arm} | {key_arm} | {sum(values) / len(values):.3f}{tag} | {own} |"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("what", choices=["ensemble", "ablation", "rrf"])
    parser.add_argument("--dataset", default="fiqa-sub")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.60,
        help="stop before embedding if the list-price estimate (USD) is higher",
    )
    parser.add_argument(
        "--count-only", action="store_true", help="ablation: print what would be embedded and stop"
    )
    args = parser.parse_args()
    if args.what == "rrf":
        rrf_mix(args.dataset)
    elif args.what == "ensemble":
        asyncio.run(ensemble(args.dataset))
    elif args.what == "ablation":
        asyncio.run(ablation(args.questions, args.dataset, args.max_cost, args.count_only))
    else:
        raise SystemExit(f"unknown command {args.what}")


if __name__ == "__main__":
    main()
