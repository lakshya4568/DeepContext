"""Vectorised drop-in for ``SQLiteStore.search_vector`` (harness-only performance shim).

The stock method re-parses every child embedding from JSON and loops in Python for *each*
query (O(N) JSON decodes per query). That is fine for the product's 1-3k-chunk corpora but turns
a 57k-document benchmark into hours. This shim keeps identical semantics - cosine similarity over
child chunks, same filters, same output fields, stable descending order - and loads the matrix
once. ``tests/test_eval_harness.py`` asserts equivalence against the stock implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deep_context.core.types import RetrievalFilters
from deep_context.storage.sqlite_store import SQLiteStore


@dataclass
class VectorIndex:
    ids: list[str]
    doc_ids: list[str]
    parent_ids: list[str | None]
    contents: list[str]
    section_paths: list[str | None]
    page_numbers: list[int | None]
    titles: list[str]
    source_uris: list[str | None]
    perms: list[list[str]]
    tenants: list[str]
    mat: np.ndarray  # (N, D) float32, rows L2-normalised
    _perm_masks: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)


_STOCK_SEARCH_VECTOR = SQLiteStore.search_vector


async def _build_index(store: SQLiteStore) -> VectorIndex:
    conn = store._get_conn()
    sql = """
        SELECT c.id, c.document_id, c.parent_chunk_id, c.content, c.section_path,
               c.page_number, d.title, d.source_uri, d.permission_scope, d.tenant_id
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.level = 'child' AND c.embedding IS NOT NULL
        ORDER BY c.rowid
    """
    async with conn.execute(sql) as cur:
        rows = await cur.fetchall()
    ids = [r["id"] for r in rows]

    sidecar = Path(f"{store.db_path}.vec.npz")
    mat: np.ndarray | None = None
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as z:
            if list(z["ids"]) == ids:
                mat = z["mat"]
    if mat is None:
        emb: list[np.ndarray] = []
        async with conn.execute(
            "SELECT embedding FROM chunks WHERE level = 'child' AND embedding IS NOT NULL "
            "ORDER BY rowid"
        ) as cur:
            async for r in cur:
                emb.append(np.asarray(json.loads(r["embedding"]), dtype=np.float32))
        mat = np.vstack(emb) if emb else np.zeros((0, 1), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = (mat / norms).astype(np.float32)
        np.savez(sidecar, ids=np.asarray(ids), mat=mat)

    return VectorIndex(
        ids=ids,
        doc_ids=[r["document_id"] for r in rows],
        parent_ids=[r["parent_chunk_id"] for r in rows],
        contents=[r["content"] for r in rows],
        section_paths=[r["section_path"] for r in rows],
        page_numbers=[r["page_number"] for r in rows],
        titles=[r["title"] for r in rows],
        source_uris=[r["source_uri"] for r in rows],
        perms=[json.loads(r["permission_scope"]) for r in rows],
        tenants=[r["tenant_id"] for r in rows],
        mat=mat,
    )


def invalidate_vector_index(store: SQLiteStore) -> None:
    """Call after inserting chunks so the next search rebuilds the matrix."""
    store._eval_vec_index = None  # type: ignore[attr-defined]


async def fast_search_vector(
    self: SQLiteStore,
    query_embedding: list[float],
    filters: RetrievalFilters,
    limit: int = 100,
) -> list[dict[str, Any]]:
    idx: VectorIndex | None = getattr(self, "_eval_vec_index", None)
    if idx is None:
        idx = await _build_index(self)
        self._eval_vec_index = idx  # type: ignore[attr-defined]

    q = np.asarray(query_embedding, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn == 0 or idx.mat.shape[0] == 0 or q.shape[0] != idx.mat.shape[1]:
        return []
    sims = idx.mat @ (q / qn)

    valid = np.array([t == filters.tenant_id for t in idx.tenants], dtype=bool)
    scope = tuple(filters.permission_scope)
    if scope not in idx._perm_masks:
        idx._perm_masks[scope] = np.array(
            [any(p in scope for p in perms) for perms in idx.perms], dtype=bool
        )
    valid &= idx._perm_masks[scope]
    if filters.document_ids:
        allowed = set(filters.document_ids)
        valid &= np.array([d in allowed for d in idx.doc_ids], dtype=bool)

    sims = np.where(valid, sims, -np.inf)
    order = np.argsort(-sims, kind="stable")[:limit]
    out: list[dict[str, Any]] = []
    for i in order:
        if not np.isfinite(sims[i]):
            break
        out.append(
            {
                "id": idx.ids[i],
                "document_id": idx.doc_ids[i],
                "parent_chunk_id": idx.parent_ids[i],
                "content": idx.contents[i],
                "section_path": idx.section_paths[i],
                "page_number": idx.page_numbers[i],
                "document_title": idx.titles[i],
                "source_uri": idx.source_uris[i],
                "score": float(sims[i]),
            }
        )
    return out


def install_fast_vector_search() -> None:
    SQLiteStore.search_vector = fast_search_vector  # type: ignore[method-assign,assignment]
    SQLiteStore._stock_search_vector = _STOCK_SEARCH_VECTOR  # type: ignore[attr-defined]
