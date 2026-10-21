"""Live (uncached, sequential) latency pass.

Main runs replay cached model calls, so their wall time says nothing about latency. This pass
runs a fixed-seed sample of queries one at a time with the cache bypassed (neither read nor
written, so cached results stay untouched) and records true per-stage latency.

Usage: uv run python -m eval.run_latency --dataset docsqa --n 20 --results-dir eval/results/smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

from eval import config as C
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.datasets import build_task
from eval.run_retrieval import RETRIEVAL_CONFIGS, run_tag
from eval.strict_clients import get_runtime

log = logging.getLogger("eval.run_latency")
ALL_CONFIGS = list(dict.fromkeys(RETRIEVAL_CONFIGS + ["rrf_bge_two_pass", C.ENGINE_CONFIG]))


RETRIEVAL_ONLY_CONFIGS = ["dense_only", "bm25_only", "rrf", "rrf_bge"]


async def retrieval_pass(args: argparse.Namespace) -> None:
    """Live (uncached) latency of the retrieval stack only: query embedding, database search, fusion,
    hosted reranker. No generation, so it does not depend on the answer model's availability."""
    from eval.strict_clients import Trace, trace_scope

    task = build_task(args.dataset, args.n_queries, args.max_docs)
    fp = corpus_fingerprint(task.corpus)[:10]
    adapter = DeepContextAdapter(f"{task.name}_{fp}", "rrf")
    await adapter.open()
    await adapter.index(task.corpus)
    qids = [q for q in task.qids if q not in task.unanswerable]
    qids = sorted(random.Random(C.SAMPLE_SEED).sample(qids, min(args.n, len(qids))))
    out = Path(args.results_dir) / "raw_latency_retrieval.jsonl"
    rt = get_runtime()
    rt.bypass_cache = True
    tag = run_tag(args.dataset, args.n_queries, args.max_docs)
    configs = [c for c in args.configs if c in RETRIEVAL_ONLY_CONFIGS] or RETRIEVAL_ONLY_CONFIGS
    for cfg in configs:
        for qid in qids:
            row = {"run_tag": tag, "dataset": task.name, "config": cfg, "qid": qid}
            try:
                with trace_scope(Trace()) as trace:
                    t0 = time.perf_counter()
                    await adapter.retrieve_children(
                        task.queries[qid], depth=C.RETRIEVAL_DEPTH, config=cfg
                    )
                    row["latency_s"] = time.perf_counter() - t0
                row["trace_summary"] = trace.summary()
            except Exception as exc:  # noqa: BLE001 - recorded in the row, the pass continues
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
        log.info("retrieval latency %s/%s done", args.dataset, cfg)
    rt.bypass_cache = False
    await adapter.close()


async def main_async(args: argparse.Namespace) -> None:
    if args.retrieval_only:
        await retrieval_pass(args)
        return
    task = build_task(args.dataset, args.n_queries, args.max_docs)
    fp = corpus_fingerprint(task.corpus)[:10]
    adapter = DeepContextAdapter(f"{task.name}_{fp}", "rrf")
    await adapter.open()
    await adapter.index(task.corpus)  # reuse; fails loudly if the index is missing/mismatched
    qids = [q for q in task.qids if q not in task.unanswerable]
    qids = sorted(random.Random(C.SAMPLE_SEED).sample(qids, min(args.n, len(qids))))
    out = Path(args.results_dir) / "raw_latency.jsonl"
    rt = get_runtime()
    rt.bypass_cache = True
    tag = run_tag(args.dataset, args.n_queries, args.max_docs)
    for cfg in args.configs:
        for qid in qids:
            row = {"run_tag": tag, "dataset": task.name, "config": cfg, "qid": qid}
            try:
                t0 = time.perf_counter()
                # End-to-end (retrieve -> rerank -> generate); per-stage latencies come from the trace.
                res = await adapter.answer(task.queries[qid], config=cfg)
                row["trace_summary"] = res["trace_summary"]
                row["latency_s"] = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
        log.info("latency pass %s/%s done", args.dataset, cfg)
    rt.bypass_cache = False
    await adapter.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--n", type=int, default=20, help="queries in the latency sample")
    ap.add_argument("--configs", nargs="*", default=ALL_CONFIGS)
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="time only embedding+search+rerank (no generation)",
    )
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
