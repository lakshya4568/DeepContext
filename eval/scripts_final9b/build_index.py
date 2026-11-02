"""Builds (or continues building) the index of one dataset without running any question.

Used for FiQA while the GPU is still writing summaries: each call embeds and stores every 1,500-document window whose
summaries are all in the cache and stops at the first window that is not finished (``attach_summaries`` raises then).
Embeddings are cached by text and chunk inserts are idempotent, so the next call carries on where this one stopped.

    EVAL_STORE=stock EVAL_INGEST=deep EVAL_INDEX_SUFFIX=_maindeep9b PYTHONPATH=.:src \\
        uv run --no-sync python eval/scripts_final9b/build_index.py --dataset fiqa
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from eval import config as C
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.main_summaries import MissingSummariesError, task_for


async def build(dataset: str) -> int:
    task = task_for(dataset)
    adapter = DeepContextAdapter(
        f"{task.name}_{corpus_fingerprint(task.corpus)[:10]}{C.INDEX_SUFFIX}", "rrf"
    )
    await adapter.open()
    try:
        stats = await adapter.index(task.corpus)
    except MissingSummariesError as exc:  # raised at the first window with a missing summary
        print(
            f"stopped, index not finished: {exc.n_missing} of {exc.n_total} chunks of that window have no summary yet"
        )
        return 1
    print(f"index finished: {stats}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    return asyncio.run(build(parser.parse_args().dataset))


if __name__ == "__main__":
    sys.exit(main())
