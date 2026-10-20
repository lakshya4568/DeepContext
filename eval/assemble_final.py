"""Assemble the retrieval rows of the FINAL (after-fix) results directory from the per-variant runs.

``dense_only`` does not touch the changed code (no word search, no fusion), so its rows are reused
from the original run; ``bm25_only`` comes from the BM25 variant; every fused setup (``rrf``,
``rrf_bge``, ``engine_as_shipped``) comes from the all-fixes variant (BM25 + exact dedup + lexical
vote 0.1). ``rrf_bge_two_pass`` shares ``rrf_bge`` retrieval by construction (it differs only in how
the answer is written), so its rows are copies. RGB uses fixed passages (no retrieval), so its rows
are reused unchanged.

Usage: uv run --no-sync python -m eval.assemble_final
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from eval.make_tables import latest_per_key, read_jsonl

BASE = Path("eval/results")
FIX = Path("eval/results_fix")
OUT = Path("eval/results_final")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = latest_per_key(read_jsonl(BASE / "raw_retrieval.jsonl"))
    v1 = latest_per_key(read_jsonl(FIX / "v1_bm25" / "raw_retrieval.jsonl"))
    v3 = latest_per_key(read_jsonl(FIX / "v3_weighted" / "raw_retrieval.jsonl"))
    rows: list[dict[str, Any]] = []

    def take(src: list[dict[str, Any]], config: str, label: str, as_config: str | None = None) -> None:
        for r in src:
            if r["config"] == config:
                rows.append({**r, "config": as_config or config, "source_variant": label})

    take(base, "dense_only", "original (unchanged by the fixes)")
    take(v1, "bm25_only", "v1_bm25")
    for cfg in ("rrf", "rrf_bge", "engine_as_shipped"):
        take(v3, cfg, "v3_weighted (all fixes)")
    take(v3, "rrf_bge", "v3_weighted (all fixes); shares rrf_bge retrieval", "rrf_bge_two_pass")
    with (OUT / "raw_retrieval.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    shutil.copy(BASE / "dataset_meta.json", OUT / "dataset_meta.json")
    shutil.copy(BASE / "raw_rgb.jsonl", OUT / "raw_rgb.jsonl")
    print(f"wrote {len(rows)} retrieval rows to {OUT / 'raw_retrieval.jsonl'}")


if __name__ == "__main__":
    main()
