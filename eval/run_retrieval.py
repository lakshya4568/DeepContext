"""Run the retrieval ablation for one dataset and append per-query rows to raw_retrieval.jsonl.

Usage (from the repo root):
    uv run python -m eval.run_retrieval --dataset scifact --n-queries 20 --results-dir eval/results/smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from eval import config as C
from eval import pg
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.datasets import Task, build_task
from eval.metrics import rank_metrics
from eval.strict_clients import Trace, stage_scope, trace_scope

log = logging.getLogger("eval.run_retrieval")
RETRIEVAL_CONFIGS = ["dense_only", "bm25_only", "rrf", "rrf_bge"]


def run_tag(dataset: str, n_queries: int | None, max_docs: int | None) -> str:
    return f"{dataset}|q{n_queries or 'all'}|d{max_docs or 'full'}"


def load_done(path: Path, tag: str) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            if r.get("run_tag") == tag and not r.get("error"):
                done.add((r["config"], r["qid"]))
    return done


async def run_dataset(
    task: Task,
    tag: str,
    out_path: Path,
    configs: list[str],
    concurrency: int = 6,
) -> None:
    fp = corpus_fingerprint(task.corpus)[:10]
    adapter = DeepContextAdapter(f"{task.name}_{fp}", configs[0])
    await adapter.open()
    t0 = time.perf_counter()
    stats = await adapter.index(task.corpus)
    log.info("[%s] index: %s (%.1fs)", tag, stats, time.perf_counter() - t0)

    done = load_done(out_path, tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.parent / "dataset_meta.json"
    all_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    all_meta[tag] = {
        **task.meta,
        "n_queries_with_relevance_labels": sum(1 for q in task.qids if task.qrels[q]),
        "index": {"backend": adapter.backend,
                  "storage": (f"PostgreSQL database {pg.EVAL_DB}, schema {adapter.schema}, hnsw.ef_search={pg.HNSW_EF_SEARCH}"
                              if adapter.backend == "postgres" else f"SQLite file {adapter.db_path.name}"),
                  "n_docs": stats.n_docs, "n_parents": stats.n_parents, "n_children": stats.n_children,
                  "approx_embed_tokens": stats.approx_embed_tokens, "embedding_model": C.EMBEDDING_MODEL,
                  "embedding_dim": C.EMBEDDING_DIM, "chunker": "ParentChildChunker (child 300-600 tok, parent 1000-2500 tok, 15% overlap)"},
    }  # fmt: skip
    meta_path.write_text(json.dumps(all_meta, indent=1))
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n_done = 0
    total = len(configs) * len(task.queries)

    async def one(config: str, qid: str) -> None:
        nonlocal n_done
        if (config, qid) in done:
            return
        async with sem:
            row: dict = {"run_tag": tag, "dataset": task.name, "config": config, "qid": qid,
                         "query": task.queries[qid], "relevant": sorted(task.qrels[qid])}  # fmt: skip
            try:
                t = time.perf_counter()
                with trace_scope(Trace()) as trace, stage_scope("retrieval"):
                    # per-call config: retrieve_children is stateless w.r.t. self.config
                    children = await adapter.retrieve_children(
                        task.queries[qid], depth=C.RETRIEVAL_DEPTH, config=config
                    )
                row["wall_s"] = round(time.perf_counter() - t, 4)
                ranked = adapter.rank_documents(children)
                row["ranked"] = [d for d, _ in ranked]
                row["top_scores"] = [round(s, 5) for _, s in ranked[:20]]
                row["metrics"] = rank_metrics(row["ranked"], task.qrels[qid])
                row["trace_summary"] = trace.summary()
            except Exception as exc:  # noqa: BLE001 - recorded, run continues, row is retried next time
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                log.error("[%s/%s/%s] %s", tag, config, qid, row["error"])
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 25 == 0:
                log.info("[%s] %d/%d rows", tag, n_done, total - len(done))

    await asyncio.gather(*(one(c, q) for c in configs for q in task.qids))
    await adapter.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--configs", nargs="*", default=RETRIEVAL_CONFIGS)
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    task = build_task(args.dataset, args.n_queries, args.max_docs)
    log.info("Task %s: %s", task.name, json.dumps(task.meta))
    asyncio.run(
        run_dataset(
            task,
            run_tag(args.dataset, args.n_queries, args.max_docs),
            Path(args.results_dir) / "raw_retrieval.jsonl",
            args.configs,
            args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
