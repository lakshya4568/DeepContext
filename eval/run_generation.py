"""Generate answers for one dataset/config set and append rows to raw_generation.jsonl.

Usage (repo root):
    uv run python -m eval.run_generation --dataset docsqa --configs rrf_bge rrf_bge_two_pass
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
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.datasets import Task, build_task
from eval.metrics import is_abstention, rank_metrics
from eval.run_retrieval import run_tag

log = logging.getLogger("eval.run_generation")
GEN_CONFIGS = ["rrf_bge", "rrf_bge_two_pass"]


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
    concurrency: int = 4,
    max_queries: int | None = None,
    index_dir: Path = C.INDEX_DIR,
) -> None:
    fp = corpus_fingerprint(task.corpus)[:10]
    adapter = DeepContextAdapter(index_dir / f"{task.name}_{fp}.db", configs[0])
    await adapter.open()
    stats = await adapter.index(task.corpus)
    log.info("[%s] index: %s", tag, stats)

    qids = task.qids[:max_queries] if max_queries else task.qids
    done = load_done(out_path, tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    todo = [(c, q) for c in configs for q in qids if (c, q) not in done]
    n_done = 0

    async def one(config: str, qid: str) -> None:
        nonlocal n_done
        async with sem:
            row: dict = {
                "run_tag": tag, "dataset": task.name, "config": config, "qid": qid,
                "query": task.queries[qid], "reference": task.answers.get(qid, ""),
                "qtype": task.qtypes.get(qid, ""), "unanswerable": qid in task.unanswerable,
                "relevant": sorted(task.qrels[qid]),
            }  # fmt: skip
            try:
                out = await adapter.answer(task.queries[qid], config=config)
                trace_records = out["trace"]
                row.update(out)
                # Keep files commit-sized: the judge input is `contexts_seen`; the full retrieved
                # parents are reproducible from the deterministic retrieval and the call cache.
                row["n_contexts"] = len(row.pop("contexts"))
                row.pop("trace", None)
                row["abstained"] = bool(out["refused"]) or is_abstention(out["answer"])
                gold = set(task.qrels[qid])
                ctx_ids = list(dict.fromkeys(out["context_ids"]))
                row["context_doc_ids"] = ctx_ids
                seen_ids = list(dict.fromkeys(out["context_ids_seen"]))
                row["context_doc_ids_seen"] = seen_ids
                row["gold_in_context"] = bool(gold & set(ctx_ids)) if gold else None
                row["gold_in_seen_context"] = bool(gold & set(seen_ids)) if gold else None
                row["gold_frac_seen"] = (len(gold & set(seen_ids)) / len(gold)) if gold else None
                row["context_metrics"] = (
                    {
                        k: v
                        for k, v in rank_metrics(ctx_ids, task.qrels[qid]).items()
                        if k in ("recall@5", "hit@5", "evidence_all@5")
                    }
                    if gold
                    else {}
                )  # context set is unranked-ish (<=6 parents); keep the @5-style views only
                gi, go = adapter_tokens(trace_records)
                row["gen_in_tokens"], row["gen_out_tokens"] = gi, go
            except Exception as exc:  # noqa: BLE001 - recorded; row is retried on the next run
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                log.error("[%s/%s/%s] %s", tag, config, qid, row["error"])
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 10 == 0:
                log.info("[%s] %d/%d answers", tag, n_done, len(todo))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(c, q) for c, q in todo))
    log.info("[%s] generation finished in %.1fs", tag, time.perf_counter() - t0)
    await adapter.close()


def adapter_tokens(trace: list[dict]) -> tuple[int, int]:
    rows = [r for r in trace if r["kind"] == "llm" and r["stage"] == "generation"]
    return sum(r["in_tokens"] for r in rows), sum(r["out_tokens"] for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-gen", type=int, default=None, help="cap generated queries (first N)")
    ap.add_argument("--configs", nargs="*", default=GEN_CONFIGS)
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    task = build_task(args.dataset, args.n_queries, args.max_docs)
    asyncio.run(
        run_dataset(
            task,
            run_tag(args.dataset, args.n_queries, args.max_docs),
            Path(args.results_dir) / "raw_generation.jsonl",
            args.configs,
            args.concurrency,
            args.max_gen,
        )
    )


if __name__ == "__main__":
    main()
