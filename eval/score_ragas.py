"""Score generated answers with RAGAS (judge != generator, temperature 0, cached).

Reads ``raw_generation.jsonl`` and appends one row per (run, config, query) to
``ragas_scores.jsonl``; resumable and idempotent (judge responses are cached on disk).

Usage:
    uv run python -m eval.score_ragas --results-dir eval/results/smoke [--model gemini-2.5-pro]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from eval import config as C
from eval.judge import score_rows
from eval.strict_clients import get_runtime

log = logging.getLogger("eval.score_ragas")
NO_CONTEXT = "(no context was retrieved)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--model", default=C.JUDGE_MODEL)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--thinking-budget", type=int, default=None, help="judge thinking-token budget")
    ap.add_argument("--limit", type=int, default=None, help="score at most N pending rows")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    rdir = Path(args.results_dir)
    label = args.model + (
        f"@think{args.thinking_budget}" if args.thinking_budget is not None else ""
    )
    raw = [json.loads(x) for x in (rdir / "raw_generation.jsonl").open()]
    out_path = rdir / "ragas_scores.jsonl"
    done = set()
    if out_path.exists():
        for x in out_path.open():
            r = json.loads(x)
            if r.get("judge_model") == label:
                done.add((r["run_tag"], r["config"], r["qid"]))
    todo = [
        r for r in raw
        if not r.get("error") and not r["unanswerable"] and r["reference"].strip()
        and (r["run_tag"], r["config"], r["qid"]) not in done
        and (args.datasets is None or r["dataset"] in args.datasets)
    ]  # fmt: skip
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d rows to score with judge=%s (generator=%s)", len(todo), label, C.GENERATOR_MODEL)
    if args.model == C.GENERATOR_MODEL:
        raise SystemExit("Judge model must differ from the generator model.")

    t0 = time.perf_counter()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        scores = score_rows(
            [
                {
                    "user_input": r["query"],
                    "response": r["answer"] or " ",
                    "retrieved_contexts": (
                        r.get("contexts_seen") or r.get("contexts") or [NO_CONTEXT]
                    ),
                    "reference": r["reference"],
                }
                for r in chunk
            ],
            model=args.model,
            thinking_budget=args.thinking_budget,
        )
        with out_path.open("a") as f:
            for r, sc in zip(chunk, scores, strict=True):
                if all(v != v for v in sc.values()):  # every metric failed: retry next run
                    log.error(
                        "all metrics NaN for %s/%s/%s - not saved",
                        r["dataset"],
                        r["config"],
                        r["qid"],
                    )
                    continue
                f.write(json.dumps({
                    "run_tag": r["run_tag"], "dataset": r["dataset"], "config": r["config"],
                    "qid": r["qid"], "judge_model": label, "generator_model": C.GENERATOR_MODEL,
                    **sc,
                }) + "\n")  # fmt: skip
        get_runtime().cache.flush()
        log.info(
            "scored %d/%d (%.0fs)",
            min(i + args.batch, len(todo)),
            len(todo),
            time.perf_counter() - t0,
        )
    log.info("usage this process:\n%s", get_runtime().usage_report())
    u = get_runtime().usage.get(args.model)
    if u and todo:
        price = C.PRICES_USD_PER_M.get(args.model)
        with (rdir / "judge_usage.jsonl").open("a") as f:
            f.write(json.dumps({
                "judge_model": label, "rows_scored": len(todo), "calls": int(u["calls"]),
                "live_calls": int(u["live_calls"]), "live_in_tokens": int(u["live_in"]),
                "live_out_tokens": int(u["live_out"]),
                "est_list_price_usd": (u["live_in"] * price["in"] + u["live_out"] * price["out"]) / 1e6 if price else None,
                "source": "score_ragas runtime ledger",
            }) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
