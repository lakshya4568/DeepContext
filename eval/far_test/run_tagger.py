"""Run the local Qwen over every chunk of a window: who or what is each chunk about?

Writes one JSON line per chunk (``page``, ``index``, ``name`` or null, ``raw``), appended after every batch,
so a stopped run resumes. Needs the model in the local Hugging Face cache; nothing is downloaded.

    HF_HUB_OFFLINE=1 PYTHONPATH=.:src python -m eval.far_test.run_tagger --pdf pdf-got \\
        --out eval/cache/far_test/tags_pdf-got.jsonl --device mps --batch-size 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.far_test.tagging import ChunkRef, Registry, tag_chunks
from eval.far_test.window import WINDOWS, child_texts
from eval.pdf_sets import extract_pages, page_id

TAGS_VERSION = "v3"  # bump when the registry or the prompt changes: old tags are not reused


def tags_path(pdf: str) -> Path:
    from eval import config as C

    return Path(C.CACHE_DIR) / "far_test" / f"tags_{TAGS_VERSION}_{pdf}.jsonl"


def kind_of(pdf: str) -> str:
    return "story" if pdf == "pdf-got" else "thesis"


def chunk_refs(pdf: str) -> tuple[list[str], list[ChunkRef]]:
    """The window's pages (blank ones included, so list index = page - first page) and its chunks."""
    lo, hi = WINDOWS[pdf]
    pages = extract_pages(pdf)[lo - 1 : hi]
    refs = [
        ChunkRef(page=page_id(lo + i), index=n, page_index=i, text=child)
        for i, text in enumerate(pages)
        if text.strip()
        for n, child in enumerate(child_texts(text))
    ]
    return pages, refs


def load_tags(path: Path) -> dict[tuple[str, int], str | None]:
    """``{(page, chunk index): name or None}`` from a tags file."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {(r["page"], int(r["index"])): r["name"] for r in rows}


def make_generate(summarizer: Any, max_new_tokens: int = 8) -> Callable[[list[str]], list[str]]:
    import torch

    def generate(prompts: list[str]) -> list[str]:
        tok = summarizer._tokenizer
        inputs = tok(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536
        ).to(summarizer._device)
        with torch.inference_mode():
            out = summarizer._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        return [tok.decode(o[inputs.input_ids.shape[1] :], skip_special_tokens=True) for o in out]

    return generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--pdf", required=True, choices=sorted(WINDOWS))
    parser.add_argument(
        "--out", type=Path, default=None, help="default: the cache file of this PDF"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="first N chunks only (smoke test)")
    args = parser.parse_args(argv)

    args.out = args.out or tags_path(args.pdf)
    pages, refs = chunk_refs(args.pdf)
    kind = kind_of(args.pdf)
    registry = Registry(pages, kind=kind)
    done = load_tags(args.out)
    todo = sorted(
        (r for r in refs if (r.page, r.index) not in done), key=lambda r: len(r.text), reverse=True
    )
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{args.pdf}: {len(refs)} chunks, {len(done)} already tagged, {len(todo)} to do", flush=True
    )

    from deep_context.ingestion.summarizer import ChunkSummarizer

    summarizer = ChunkSummarizer(device=args.device)
    if not asyncio.run(summarizer._ensure_model_loaded()):
        print("The model did not load; nothing was tagged.")
        return 2
    generate = make_generate(summarizer)
    started = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            for tag in tag_chunks(registry, batch, generate, batch_size=len(batch), kind=kind):
                fh.write(
                    json.dumps(
                        {"page": tag.page, "index": tag.index, "name": tag.name, "raw": tag.raw},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fh.flush()
            finished = start + len(batch)
            elapsed = time.perf_counter() - started
            print(
                f"{finished}/{len(todo)} chunks, {elapsed:.0f}s, {finished / elapsed:.2f} chunks/s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
