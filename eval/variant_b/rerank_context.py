"""Study: give the reranker the generated context line too (rerank text = context line + chunk).

In the shipped engine the reranker reads only the chunk text, so anything the contextual enrichment adds to the vectors
is invisible to it. This wrapper emulates the alternative without touching ``src/``: it maps each candidate's text
(first 1000 characters, as the engine truncates) to ``context + blank line + text`` before the reranker call. Nothing
stored changes; the returned and answered text is still the plain chunk.

    EVAL_INDEX_SUFFIX=_vbctx EVAL_RERANK_URL=http://127.0.0.1:18001 PYTHONPATH=. uv run --no-sync python \
        -m eval.variant_b.rerank_context --schema docsqa_7c20e221ae_vbctx --dataset docsqa \
        --results-dir eval/results_variant_b/retrieval_docsqa_ctx_rerank --configs rrf_bge
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import asyncpg

from eval import pg, run_retrieval
from eval.strict_clients import VertexRuntime
from eval.variant_b.pipeline import SUMMARISED_OK, _check_ident

PREFIX_LEN = 1000


async def load_context_map(schema: str) -> dict[str, str]:
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    try:
        s = _check_ident(schema)
        rows = await conn.fetch(
            f'SELECT c.content, cs.summary_text, cs.status FROM "{s}".chunks c '
            f'JOIN "{s}".chunk_summaries cs ON cs.chunk_id = c.id WHERE c.level = $1',
            "child",
        )
    finally:
        await conn.close()
    return {
        r["content"][:PREFIX_LEN]: r["summary_text"]
        for r in rows
        if r["status"] in SUMMARISED_OK and r["summary_text"].strip()
    }


def install(context_map: dict[str, str]) -> None:
    original = VertexRuntime.rerank_scores

    async def with_context(self: VertexRuntime, query: str, documents: list[str]) -> list[float]:
        augmented = [f"{context_map[d]}\n\n{d}" if d in context_map else d for d in documents]
        return await original(self, query, augmented)

    VertexRuntime.rerank_scores = with_context  # type: ignore[method-assign]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--schema", required=True, help="schema whose chunk_summaries hold the context lines"
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--configs", nargs="*", default=["rrf_bge"])
    args = ap.parse_args()
    install(asyncio.run(load_context_map(args.schema)))
    sys.argv = [
        "run_retrieval", "--dataset", args.dataset, "--results-dir", args.results_dir,
        "--configs", *args.configs, "--concurrency", "6",
    ]  # fmt: skip
    run_retrieval.main()


if __name__ == "__main__":
    main()
