"""RGB (Retrieval-Augmented Generation Benchmark) robustness runs - generation only.

RGB supplies the passages, so retrieval is bypassed and the *generators* are compared: the
system's one-pass prompt (PromptAssembler + one call) vs its two-pass grounded generator, with
5 passages per query at a controlled noise ratio (RGB protocol). Licence: CC BY-NC-SA 4.0
(non-commercial research use).

  en_refine : noise robustness (noise 0..0.8, accuracy) and negative rejection (noise 1.0)
  en_int    : information integration (answer needs 2 sub-answers; noise 0)
  en_fact   : counterfactual robustness (passages carry a wrong answer)

Usage: uv run python -m eval.run_rgb --kind en_refine --n 20 --results-dir eval/results/smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from eval import config as C
from eval.datasets import load_rgb
from eval.metrics import is_abstention, rgb_correct
from eval.strict_clients import Trace, install_strict_patches, stage_scope, trace_scope

log = logging.getLogger("eval.run_rgb")
PASSAGES = 5


def build_passages(row: dict[str, Any], kind: str, noise: float) -> list[str]:
    rng = random.Random(f"{row['id']}|{noise}|{C.SAMPLE_SEED}")
    if kind == "en_int":
        pos = [p for group in row["positive"] for p in group[:3]]
        rng.shuffle(pos)
        docs = pos[:PASSAGES]
    elif kind == "en_fact":
        docs = (list(row["positive_wrong"]) + list(row["negative"]))[:PASSAGES]
        rng.shuffle(docs)
    else:
        n_neg = int(PASSAGES * noise)
        pos = list(row["positive"])
        neg = list(row["negative"])
        rng.shuffle(pos)
        rng.shuffle(neg)
        docs = pos[: PASSAGES - n_neg] + neg[:n_neg]
        rng.shuffle(docs)
    return [str(d) for d in docs]


async def generate(mode: str, query: str, passages: list[str]) -> tuple[str, bool, Trace]:
    from deep_context.core.llm_client import llm_client
    from deep_context.generation.grounded_answer import generate_grounded_answer
    from deep_context.memory.prompt_assembler import PromptAssembler
    from eval.adapter import _NoMemory

    chunks = [
        {"content": p, "document_id": f"p{i}", "document_title": "Passage", "section_path": ""}
        for i, p in enumerate(passages)
    ]
    with trace_scope(Trace()) as tr, stage_scope("generation"):
        if mode == "two_pass":
            g = await generate_grounded_answer(
                query, chunks, model=C.GENERATOR_MODEL, timeout=C.REQUEST_TIMEOUT_S
            )
            return g.answer, g.refused, tr
        assembler = PromptAssembler(None)  # type: ignore[arg-type]
        assembler.memory_manager = _NoMemory()  # type: ignore[assignment]
        messages = await assembler.assemble_messages(query=query, retrieved_chunks=chunks)
        ans, _ = await llm_client.complete(
            messages,
            model=C.GENERATOR_MODEL,
            temperature=0.1,
            max_tokens=1024,
            enable_thinking=False,
        )
        return (ans or "").strip(), False, tr


async def main_async(args: argparse.Namespace) -> None:
    install_strict_patches()
    rows = load_rgb(args.kind, args.n)
    noises = [0.0] if args.kind in ("en_int", "en_fact") else args.noise
    out = Path(args.results_dir) / "raw_rgb.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["kind"], r["id"], r["noise"], r["mode"], r["n_total"]))
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async def one(row: dict[str, Any], noise: float, mode: str) -> None:
        key = (args.kind, str(row["id"]), noise, mode, args.n or len(rows))
        if key in done:
            return
        rec: dict[str, Any] = {"kind": args.kind, "id": str(row["id"]), "noise": noise, "mode": mode,
                               "n_total": args.n or len(rows), "query": row["query"]}  # fmt: skip
        async with sem:
            try:
                passages = build_passages(row, args.kind, noise)
                answer, refused, tr = await generate(mode, row["query"], passages)
                rec["answer"] = answer
                rec["abstained"] = bool(refused) or is_abstention(answer)
                if args.kind == "en_fact":
                    rec["mentions_true"] = rgb_correct(
                        answer,
                        [[row["answer"]] if isinstance(row["answer"], str) else row["answer"]],
                    )
                    rec["mentions_fake"] = rgb_correct(
                        answer,
                        [
                            [row["fakeanswer"]]
                            if isinstance(row["fakeanswer"], str)
                            else row["fakeanswer"]
                        ],
                    )
                else:
                    rec["correct"] = rgb_correct(answer, row["answer"])
                rec["trace_summary"] = tr.summary()
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        async with lock:
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    await asyncio.gather(*(one(r, n, m) for r in rows for n in noises for m in args.modes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="en_refine", choices=["en_refine", "en_int", "en_fact"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--modes", nargs="*", default=["one_pass", "two_pass"])
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "deep_context", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
