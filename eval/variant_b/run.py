"""Command line for variant B.

    uv run --no-sync python -m eval.variant_b.run summarize --dataset docsqa [--mode summary_v2] [--limit-groups 2]
    uv run --no-sync python -m eval.variant_b.run index --dataset docsqa [--mode context] [--recipe prefix]

``summarize`` needs the summariser server (``--url``, default ``$VB_LLM_URL`` or http://127.0.0.1:8000, e.g. an SSH
port-forward to the GPU VM). ``index`` embeds with Gemini (Vertex) and writes the derived schema ``<fast schema>_vb``.
Outputs go to ``eval/results_variant_b/<dataset>/`` (local-only until reviewed: summaries of copyrighted PDFs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from eval import config as C
from eval import pg
from eval.adapter import corpus_fingerprint
from eval.datasets import build_task
from eval.variant_b.llm import DEFAULT_MODEL, DEFAULT_URL, SlmClient
from eval.variant_b.pipeline import add_lexical_context, build_index, run_summaries
from eval.variant_b.summarize import MODES

RESULTS = C.EVAL_DIR / "results_variant_b"
# schema suffix per enrichment mode: EVAL_INDEX_SUFFIX=<suffix> points the retrieval runners at that index
INDEX_SUFFIX = {
    "summary_v1": "_vb",
    "summary_v2": "_vbs2",
    "context": "_vbctx",
    "context_v2": "_vbctx2",
    "questions": "_vbq",
    "keywords": "_vbkw",
}


def _source_schema(dataset: str, corpus: dict[str, dict[str, str]]) -> str:
    return pg.schema_name(f"{dataset}_{corpus_fingerprint(corpus)[:10]}")


async def _main(args: argparse.Namespace) -> None:
    task = build_task(args.dataset)
    schema = _source_schema(task.name, task.corpus)
    out = Path(args.out) if args.out else RESULTS / task.name
    mode = MODES[args.mode]
    if args.mode != "summary_v1" and not args.out:
        out = out / "modes" / args.mode
    if args.command == "summarize":
        llm = SlmClient(args.url, args.model, concurrency=args.concurrency)
        try:
            report = await run_summaries(
                task.name, schema, task.corpus, llm, out,
                mode=mode, use_previous=True if args.chain_previous else None,
                limit_groups=args.limit_groups,
            )  # fmt: skip
        finally:
            await llm.aclose()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        suffix = (args.index_suffix or INDEX_SUFFIX[args.mode]) + (
            "bm" if args.lexical_context else ""
        )
        res = await build_index(
            task.name,
            schema,
            out / "summaries.jsonl",
            suffix=suffix,
            recipe=args.recipe,
            min_words=args.min_words,
        )
        if args.lexical_context:
            res["chunks_with_lexical_context"] = await add_lexical_context(f"{schema}{suffix}")
        print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["summarize", "index"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--out", default=None, help="output directory (default eval/results_variant_b/<dataset>)"
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--limit-groups", type=int, default=None)
    ap.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="summary_v1",
        help="what to generate for each chunk",
    )
    ap.add_argument(
        "--recipe",
        choices=["prefix", "suffix"],
        default="prefix",
        help="index: where the generated text goes",
    )
    ap.add_argument(
        "--index-suffix",
        default=None,
        help="index: schema suffix (default: one per mode, see INDEX_SUFFIX)",
    )
    ap.add_argument(
        "--lexical-context",
        action="store_true",
        help="index: also add the generated text to the keyword-search column (contextual BM25)",
    )
    ap.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="index: embed the chunk alone below this many words",
    )
    ap.add_argument(
        "--chain-previous",
        action="store_true",
        help="chain the previous two summaries into each prompt (the default only for summary_v1)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
