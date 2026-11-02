"""Dense retrieval under different ways of using the generated text: a controlled grid, no summariser needed.

Each recipe decides what a child chunk's vector embeds (or how two vectors are combined). Scoring is the dense channel
only: exact cosine between the cached query vectors and the child vectors, best child per document, the same rank
metrics as the harness. ``fast`` must reproduce the stored fast-ingestion dense run; the controls show what any extra
text does (``control_shuffled``: another chunk's text; ``control_duplicate``: the chunk repeated).

    PYTHONPATH=. uv run --no-sync python -m eval.variant_b.recipes --dataset docsqa \
        --enrich b_v1=eval/results_variant_b/docsqa/summaries.jsonl [--enrich ctx=...jsonl] [--split-file questions.json]

The question set used for choosing a recipe must not be the one a claim is made on: docs QA is the development set for
variant B (see FINDINGS_WHY_LOWER.md); held-out sets are scored once with the recipe chosen here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np

from eval import config as C
from eval import pg
from eval.adapter import DeepContextAdapter, corpus_fingerprint
from eval.datasets import build_task
from eval.metrics import rank_metrics
from eval.stats import paired_test
from eval.strict_clients import install_strict_patches, stage_scope
from eval.variant_b.pipeline import SUMMARISED_OK, _check_ident


@dataclass
class ChildText:
    id: str
    ext_doc: str
    title: str | None  # embedding prefix title, as the fast adapter uses it
    content: str


def strip_unclear(summary: str) -> str:
    return "\n".join(
        b for b in summary.splitlines() if not b.strip().lower().startswith("- unclear")
    ).strip()


def first_bullet(summary: str) -> str:
    bullets = [b for b in summary.splitlines() if b.strip()]
    return bullets[0] if bullets else summary


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


async def load_children(schema: str) -> list[ChildText]:
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    try:
        s = _check_ident(schema)
        rows = await conn.fetch(
            f'SELECT c.id::text AS id, c.content, d.title, d.metadata AS meta FROM "{s}".chunks c '
            f'JOIN "{s}".documents d ON d.id = c.document_id WHERE c.level = $1 ORDER BY c.id',
            "child",
        )
    finally:
        await conn.close()
    out = []
    for r in rows:
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else dict(r["meta"])
        ext = meta["external_id"]
        out.append(ChildText(r["id"], ext, r["title"] if r["title"] != ext else None, r["content"]))
    return out


def load_enrichment(path: Path) -> dict[str, tuple[str, str]]:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return {r["id"]: (r["summary"], r["status"]) for r in rows}


def text_recipes(
    children: list[ChildText],
    enrich: dict[str, dict[str, tuple[str, str]]],
    flags: dict[str, dict[str, Any]] | None = None,
    flags_for: str = "",
) -> dict[str, list[str]]:
    """Recipe name -> the text to embed for every child (same order as ``children``)."""
    recipes: dict[str, list[str]] = {"fast": [c.content for c in children]}

    def usable(name: str, c: ChildText) -> str | None:
        text, status = enrich[name].get(c.id, ("", "missing"))
        return text if status in SUMMARISED_OK and text.strip() else None

    for name in enrich:
        recipes[f"{name}:prefix"] = [
            (f"{usable(name, c)}\n\n{c.content}" if usable(name, c) else c.content)
            for c in children
        ]
        recipes[f"{name}:suffix"] = [
            (f"{c.content}\n\n{usable(name, c)}" if usable(name, c) else c.content)
            for c in children
        ]
        recipes[f"{name}:prefix_first_bullet"] = [
            (
                f"{first_bullet(usable(name, c) or '')}\n\n{c.content}"
                if usable(name, c)
                else c.content
            )
            for c in children
        ]
        long_enough = {c.id: len(c.content.split()) >= 100 for c in children}
        recipes[f"{name}:prefix_if_long"] = [
            (
                f"{usable(name, c)}\n\n{c.content}"
                if usable(name, c) and long_enough[c.id]
                else c.content
            )
            for c in children
        ]
        recipes[f"{name}:first_bullet_if_long"] = [
            (
                f"{first_bullet(usable(name, c) or '')}\n\n{c.content}"
                if usable(name, c) and long_enough[c.id]
                else c.content
            )
            for c in children
        ]
        if flags and name == flags_for:
            clean = {c.id: not flags.get(c.id, {}).get("borrowed", False) for c in children}
            recipes[f"{name}:prefix_if_clean"] = [
                (
                    f"{usable(name, c)}\n\n{c.content}"
                    if usable(name, c) and clean[c.id]
                    else c.content
                )
                for c in children
            ]
            recipes[f"{name}:prefix_if_clean_and_long"] = [
                (
                    f"{usable(name, c)}\n\n{c.content}"
                    if usable(name, c) and clean[c.id] and long_enough[c.id]
                    else c.content
                )
                for c in children
            ]
        recipes[f"{name}:prefix_no_unclear"] = [
            (
                f"{strip_unclear(usable(name, c) or '')}\n\n{c.content}".strip()
                if usable(name, c)
                else c.content
            )
            for c in children
        ]
    first = next(iter(enrich))
    # controls: extra text that carries no information about this chunk, of the same size as the real text
    others = [usable(first, c) or "" for c in children]
    shift = 37 % max(1, len(children))
    shuffled = others[shift:] + others[:shift]
    recipes["control_shuffled"] = [
        (f"{s}\n\n{c.content}" if s else c.content) for s, c in zip(shuffled, children, strict=True)
    ]
    recipes["control_duplicate"] = [f"{c.content}\n\n{c.content}" for c in children]
    return recipes


def format_for_embedding(children: list[ChildText], texts: list[str]) -> list[str]:
    from eval.strict_clients import VertexRuntime

    return [
        VertexRuntime.format_embedding_text(t, is_query=False, title=c.title, task=None)
        for c, t in zip(children, texts, strict=True)
    ]


async def embed_texts(runtime: Any, formatted: list[str]) -> np.ndarray:
    with stage_scope("index"):
        vecs = await runtime.embed_many(formatted, model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM)
    runtime.cache.flush()
    return _norm(np.stack([np.asarray(v, dtype=np.float64) for v in vecs]))


def rank_docs(q: np.ndarray, docs: list[str], m: np.ndarray) -> list[str]:
    best: dict[str, float] = {}
    for d, s in zip(docs, m @ q, strict=True):
        if s > best.get(d, -9.0):
            best[d] = float(s)
    return sorted(best, key=lambda d: -best[d])


def score(
    vectors: np.ndarray,
    docs: list[str],
    qvecs: dict[str, np.ndarray],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    return {qid: rank_metrics(rank_docs(q, docs, vectors), qrels[qid]) for qid, q in qvecs.items()}


def score_fused(
    a: np.ndarray,
    b: np.ndarray,
    weight_b: float,
    docs: list[str],
    qvecs: dict[str, np.ndarray],
    qrels: dict[str, dict[str, int]],
    k: int = 60,
) -> dict[str, dict[str, float]]:
    """Two vectors per chunk, searched separately and joined by reciprocal-rank fusion (needs a multi-vector store)."""
    out = {}
    for qid, q in qvecs.items():
        fused: dict[str, float] = {}
        for weight, m in ((1.0, a), (weight_b, b)):
            for rank, d in enumerate(rank_docs(q, docs, m), start=1):
                fused[d] = fused.get(d, 0.0) + weight / (k + rank)
        out[qid] = rank_metrics(sorted(fused, key=lambda d: -fused[d]), qrels[qid])
    return out


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    task = build_task(args.dataset)
    fp = corpus_fingerprint(task.corpus)[:10]
    schema = pg.schema_name(f"{task.name}_{fp}")
    children = await load_children(schema)
    enrich = {}
    for spec in args.enrich:
        name, _, path = spec.partition("=")
        enrich[name] = load_enrichment(Path(path))
    qids = [q for q in sorted(task.queries) if task.qrels.get(q) and q not in task.unanswerable]
    if args.split_file:
        keep = set(json.loads(Path(args.split_file).read_text()))
        qids = [q for q in qids if q in keep]
    adapter = DeepContextAdapter(f"{task.name}_{fp}", "dense_only")
    qvecs = {}
    for qid in qids:
        v = np.asarray(await adapter._query_vector(task.queries[qid]), dtype=np.float64)
        qvecs[qid] = v / np.linalg.norm(v)
    await adapter.close()
    runtime = install_strict_patches()
    docs = [c.ext_doc for c in children]
    flags = None
    if args.flags:
        per_child = json.loads(Path(args.flags).read_text())["per_child"]
        flags = {r["id"]: r for r in per_child}
    recipes = text_recipes(children, enrich, flags, args.flags_for)
    vectors: dict[str, np.ndarray] = {}
    for name, texts in recipes.items():
        vectors[name] = await embed_texts(runtime, format_for_embedding(children, texts))
    # combined vectors: weighted average of the chunk-only vector and the generated-text-only vector
    for name in enrich:
        only = [
            (
                enrich[name].get(c.id, ("", ""))[0]
                if enrich[name].get(c.id, ("", ""))[1] in SUMMARISED_OK
                else ""
            )
            for c in children
        ]
        v_only = await embed_texts(
            runtime,
            format_for_embedding(
                children, [o or c.content for o, c in zip(only, children, strict=True)]
            ),
        )
        vectors[f"{name}:only"] = v_only
        for w in (0.85, 0.7, 0.5):
            vectors[f"{name}:avg_{int(w * 100)}"] = _norm(w * vectors["fast"] + (1 - w) * v_only)
    per = {name: score(v, docs, qvecs, task.qrels) for name, v in vectors.items()}
    for name in enrich:
        for w in (1.0, 0.5):
            per[f"{name}:two_vectors_rrf_{w}"] = score_fused(
                vectors["fast"], vectors[f"{name}:only"], w, docs, qvecs, task.qrels
            )
    base = per["fast"]
    qs = sorted(base)
    rows = []
    for name, res in per.items():
        row: dict[str, Any] = {"recipe": name, "n": len(qs)}
        for key in ("ndcg@10", "hit@5", "mrr", "recall@10"):
            vals = [res[q][key] for q in qs]
            row[key] = statistics.mean(vals)
        d = paired_test([base[q]["ndcg@10"] for q in qs], [res[q]["ndcg@10"] for q in qs])
        row.update(delta_ndcg=d.mean_diff, delta_lo=d.ci_lo, delta_hi=d.ci_hi, p=d.p_bootstrap)
        row["better"] = sum(res[q]["ndcg@10"] > base[q]["ndcg@10"] + 1e-9 for q in qs)
        row["worse"] = sum(res[q]["ndcg@10"] < base[q]["ndcg@10"] - 1e-9 for q in qs)
        rows.append(row)
    # plumbing check: the recomputed fast vectors must equal the stored fast vectors
    conn = await asyncpg.connect(pg.dsn_for(pg.EVAL_DB))
    from pgvector.asyncpg import register_vector

    await register_vector(conn)
    stored = {
        r["id"]: r["embedding"]
        for r in await conn.fetch(
            f'SELECT id::text AS id, embedding FROM "{schema}".chunks WHERE level = $1', "child"
        )
    }
    await conn.close()
    sims = []
    for c, v in zip(children, vectors["fast"], strict=True):
        s = stored[c.id]
        s = np.asarray(s.to_numpy() if hasattr(s, "to_numpy") else s, dtype=np.float64)
        sims.append(float(v @ (s / np.linalg.norm(s))))
    print(
        f"plumbing check: recomputed fast vectors vs stored fast vectors, min cosine {min(sims):.6f}"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| recipe | nDCG@10 | Hit@5 | MRR | change in nDCG@10 vs fast (95% CI) | better / worse |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['recipe']} | {r['ndcg@10']:.3f} | {r['hit@5']:.3f} | {r['mrr']:.3f} | "
            f"{r['delta_ndcg']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}] | {r['better']} / {r['worse']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--enrich", action="append", required=True, help="name=path/to/summaries.jsonl (repeatable)"
    )
    ap.add_argument(
        "--split-file",
        default=None,
        help="JSON list of question ids to score (default: all answerable)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--flags", default=None, help="diagnostics.json with per_child flags (borrowed words)"
    )
    ap.add_argument("--flags-for", default="b_v1", help="which --enrich name the flags belong to")
    args = ap.parse_args()
    args.out = args.out or f"eval/results_variant_b/{args.dataset}/recipes.json"
    rows = asyncio.run(run(args))
    print(table(rows))
    Path(args.out).with_suffix(".md").write_text(table(rows) + "\n")


if __name__ == "__main__":
    main()
