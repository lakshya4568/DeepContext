"""Quantify the stock ``enable_thinking`` bug on the configured generator model.

Runs the two-pass generator on real prompts twice - honouring ``enable_thinking=False`` (the
harness default) and ignoring it (the stock client) - and records how often a call hits the
output-token cap (``MAX_TOKENS``) because thinking tokens ate the answer budget.

Usage: uv run --no-sync python -m eval.check_thinking_flag --results-dir eval/results/smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from eval import config as C
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.datasets import build_task
from eval.strict_clients import get_runtime

SPECS = [("docsqa", None, 20), ("multihoprag", 20, 12)]  # (dataset, n_queries arg, queries used)


async def measure(honor: bool) -> dict[str, dict[str, float]]:
    rt = get_runtime()
    rt.honor_enable_thinking = honor
    out: dict[str, dict[str, float]] = {}
    for name, nq, n_use in SPECS:
        task = build_task(name, nq, None)
        ad = DeepContextAdapter(
            f"{task.name}_{corpus_fingerprint(task.corpus)[:10]}", "rrf_bge_two_pass"
        )
        await ad.open()
        await ad.index(task.corpus)
        qids = [q for q in task.qids if q not in task.unanswerable][:n_use]
        sem = asyncio.Semaphore(4)
        c: Counter[str] = Counter()
        think: list[int] = []

        async def one(q: str) -> None:
            async with sem:
                r = await ad.answer(task.queries[q], config="rrf_bge_two_pass")
            c["refused"] += int(r["refused"])
            for t in r["trace"]:
                if t["kind"] == "llm" and t["stage"] == "generation":
                    c["calls"] += 1
                    c["max_tokens"] += int(t["finish_reason"] == "MAX_TOKENS")
                    think.append(t["think_tokens"])

        await asyncio.gather(*(one(q) for q in qids))
        await ad.close()
        out[name] = {
            "queries": len(qids), "generation_calls": c["calls"], "calls_hitting_max_tokens": c["max_tokens"],
            "two_pass_refusals": c["refused"], "think_tokens_mean": sum(think) / max(1, len(think)),
            "think_tokens_max": max(think) if think else 0,
        }  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    res = {
        "generator_model": C.GENERATOR_MODEL,
        "honoured (harness default)": asyncio.run(measure(True)),
        "ignored (stock client)": asyncio.run(measure(False)),
    }
    path = Path(args.results_dir) / "thinking_flag_check.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
