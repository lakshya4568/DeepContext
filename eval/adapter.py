"""Thin adapter exposing DeepContext through ``index`` / ``retrieve`` / ``answer``.

Entry points reused from the product (nothing re-implemented except the corpus-loading loop):

* index    : ``DocumentParser`` + ``ParentChildChunker`` (the "fast" ingestion path: no Docling,
             no local summariser), ``llm_client.get_embeddings`` (patched to the strict Vertex
             client), ``SQLiteStore``.
* retrieve : ``SQLiteStore.search_vector`` / ``search_bm25``, ``HybridRetriever`` (RRF k=60),
             ``Reranker`` -> ``EcoHashReranker`` (hosted bge-reranker-v2-m3) + ``protect_consensus``.
* answer   : ``RetrievalEngine._resolve_parent_chunks`` (child -> parent), then either the
             ``PromptAssembler`` single call (one-pass) or ``generate_grounded_answer`` (two-pass).

Configurations (see ``eval.config.CONFIGS``) differ only in the retrieval kind and generation
mode; chunking, embeddings, top-k and seeds are shared.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval import config as C
from eval.fast_vector import invalidate_vector_index
from eval.strict_clients import (
    Trace,
    VertexRuntime,
    install_strict_patches,
    stage_scope,
    trace_scope,
)

log = logging.getLogger("eval.adapter")

_NS = uuid.UUID("6b1c1f0e-6a52-4a7e-9d1e-0c0de0c0ffee")  # deterministic chunk ids


@dataclass
class IndexStats:
    n_docs: int
    n_parents: int
    n_children: int
    reused: bool
    seconds: float
    approx_embed_tokens: int = 0


def corpus_fingerprint(corpus: dict[str, dict[str, str]]) -> str:
    h = hashlib.sha256()
    for doc_id in sorted(corpus):
        d = corpus[doc_id]
        h.update(doc_id.encode())
        h.update(hashlib.sha1((d.get("title", "") + "\x1f" + d.get("text", "")).encode()).digest())
    h.update(json.dumps([C.EMBEDDING_MODEL, C.EMBEDDING_DIM, "v1-title-in-content"]).encode())
    return h.hexdigest()


EVIDENCE_CHAR_BUDGET = 12_000  # both generators hard-cap the evidence they send to the LLM


def visible_contexts(parents: list[dict[str, Any]], mode: str) -> list[tuple[int, str]]:
    """Replicate which slice of each retrieved parent actually reaches the generation prompt.

    one_pass : ``PromptAssembler`` (blocks ``[i] Source: .. | Section: ..\n<content>``, 12k-char
               budget counted per block, last block kept only if >200 chars remain).
    two_pass : ``generate_grounded_answer`` (``"[i] <content>"`` blocks joined by blank lines,
               then ``[:12000]``).
    Returns ``[(1-based index, visible content)]`` for parents with any visible content.
    """
    seen: list[tuple[int, str]] = []
    if mode == "one_pass":
        total = 0
        for idx, c in enumerate(parents, start=1):
            content = str(c.get("content", "")).strip()
            header = f"[{idx}] Source: {c.get('document_title', 'Document')} | Section: {c.get('section_path', '')}\n"
            if total + len(header) + len(content) > EVIDENCE_CHAR_BUDGET:
                remaining = EVIDENCE_CHAR_BUDGET - total
                if remaining > 200:
                    seen.append((idx, content[: max(0, remaining - len(header))]))
                break
            seen.append((idx, content))
            total += len(header) + len(content)
        return seen
    pos = 0
    for idx, c in enumerate(parents, start=1):
        content = str(c.get("content", ""))
        header = f"[{idx}] "
        start = pos + len(header)
        if start >= EVIDENCE_CHAR_BUDGET:
            break
        seen.append((idx, content[: EVIDENCE_CHAR_BUDGET - start]))
        pos = start + len(content) + 2  # "\n\n" separator
    return seen


class _NoMemory:
    """Empty memory manager: avoids a no-op embedding call per query in PromptAssembler."""

    async def get_active_policies(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return []

    async def get_user_preferences(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return []

    async def search_relevant_facts(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return []


class DeepContextAdapter:
    """Benchmark adapter around the DeepContext retrieval + generation stack."""

    def __init__(self, db_path: Path, config: str = "rrf_bge_two_pass") -> None:
        if config not in C.CONFIGS and config != C.ENGINE_CONFIG:
            raise ValueError(f"Unknown config '{config}'. Supported: {list(C.CONFIGS)}")
        self.db_path = Path(db_path)
        self.config = config
        self.runtime: VertexRuntime = install_strict_patches()
        self.store: Any = None

    # ------------------------------------------------------------------ lifecycle
    async def open(self) -> None:
        import deep_context.storage as storage_mod
        from deep_context.storage.sqlite_store import SQLiteStore

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteStore(str(self.db_path))
        await self.store.initialize()
        # The retrieval engine resolves storage through the module-level singleton.
        storage_mod._active_store = self.store

    async def close(self) -> None:
        import deep_context.storage as storage_mod

        if self.store is not None:
            await self.store.close()
        storage_mod._active_store = None
        self.runtime.cache.flush()

    def set_config(self, config: str) -> None:
        if config not in C.CONFIGS and config != C.ENGINE_CONFIG:
            raise ValueError(f"Unknown config '{config}'.")
        self.config = config

    # ------------------------------------------------------------------ index
    async def index(self, corpus: dict[str, dict[str, str]]) -> IndexStats:
        """Fast ingestion of ``{doc_id: {title, text}}``; reuses an existing identical index."""
        from deep_context.core.types import ChunkLevel, Document
        from deep_context.ingestion.chunker import ParentChildChunker
        from deep_context.ingestion.parser import DocumentParser

        t0 = time.perf_counter()
        if self.store is None:
            await self.open()
        meta_path = Path(f"{self.db_path}.meta.json")
        fp = corpus_fingerprint(corpus)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") == fp:
                log.info("Reusing index %s (%d docs).", self.db_path.name, meta["n_docs"])
                return IndexStats(
                    meta["n_docs"], meta["n_parents"], meta["n_children"], True,
                    time.perf_counter() - t0, meta.get("approx_embed_tokens", 0),
                )  # fmt: skip
            raise RuntimeError(
                f"{self.db_path} holds a different corpus (fingerprint mismatch); "
                "delete it or choose another path."
            )

        chunker = ParentChildChunker()
        staged: list[tuple[Document, list[Any], list[Any]]] = []
        for doc_id, d in corpus.items():
            title = (d.get("title") or "").strip()
            text = (d.get("text") or "").strip()
            content = f"{title}\n\n{text}" if title else text
            if not content:
                continue
            sections = DocumentParser.parse(content, doc_type="text")
            parents, children = chunker.chunk_sections(doc_id, sections)
            id_map: dict[str, str] = {}
            for n, p in enumerate(parents):
                new = str(uuid.uuid5(_NS, f"{doc_id}|parent|{n}"))
                id_map[p.id], p.id = new, new
            for n, ch in enumerate(children):
                ch.id = str(uuid.uuid5(_NS, f"{doc_id}|child|{n}"))
                if ch.parent_chunk_id is not None:
                    ch.parent_chunk_id = id_map[ch.parent_chunk_id]
            doc = Document(
                id=doc_id, title=title or doc_id, doc_type="text", metadata={"eval": True}
            )
            staged.append((doc, parents, children))

        # One embedding request per child chunk (gemini-embedding-2 accepts one content per call),
        # run concurrently and cached on disk by formatted text.
        texts: list[str] = []
        owners: list[Any] = []
        for doc, _p, children in staged:
            for ch in children:
                assert ch.level == ChunkLevel.CHILD
                texts.append(
                    self.runtime.format_embedding_text(
                        ch.content,
                        is_query=False,
                        title=doc.title if doc.title != doc.id else None,
                        task=None,
                    )
                )
                owners.append(ch)
        log.info("Embedding %d child chunks for %d docs ...", len(texts), len(staged))
        window = 2000
        approx_tokens = 0
        with stage_scope("index"):
            for start in range(0, len(texts), window):
                vecs = await self.runtime.embed_many(
                    texts[start : start + window], model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM
                )
                for ch, v in zip(owners[start : start + window], vecs, strict=True):
                    ch.embedding = v.tolist()
                self.runtime.cache.flush()
                log.info("  embedded %d/%d", min(start + window, len(texts)), len(texts))
        approx_tokens = sum(max(len(t.split()), len(t) // 4) for t in texts)

        n_parents = n_children = 0
        batch: list[Any] = []
        for doc, parents, children in staged:
            await self.store.insert_document(doc)
            batch.extend(parents)
            batch.extend(children)
            n_parents += len(parents)
            n_children += len(children)
            if len(batch) >= 2000:
                await self.store.insert_chunks(batch)
                batch = []
        if batch:
            await self.store.insert_chunks(batch)
        invalidate_vector_index(self.store)

        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "n_docs": len(staged),
                    "n_parents": n_parents,
                    "n_children": n_children,
                    "embedding_model": C.EMBEDDING_MODEL,
                    "embedding_dim": C.EMBEDDING_DIM,
                    "approx_embed_tokens": approx_tokens,
                }  # fmt: skip
            )
        )
        return IndexStats(
            len(staged), n_parents, n_children, False, time.perf_counter() - t0, approx_tokens
        )

    # ------------------------------------------------------------------ retrieval
    async def _query_vector(self, query: str) -> list[float]:
        from deep_context.core.llm_client import llm_client

        return (
            await llm_client.get_embeddings(
                [query], model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM, is_query=True
            )
        )[0]

    async def retrieve_children(
        self, query: str, depth: int = C.RETRIEVAL_DEPTH, config: str | None = None
    ) -> list[dict[str, Any]]:
        """Ranked child-chunk candidates for ``config`` (component-level, raw query)."""
        from deep_context.core.types import RetrievalFilters
        from deep_context.retrieval.hybrid import HybridRetriever
        from deep_context.retrieval.quality_gates import protect_consensus
        from deep_context.retrieval.reranker import Reranker

        cfg = config or self.config
        kind = C.CONFIGS[cfg][1] if cfg in C.CONFIGS else "rrf_bge"
        filters = RetrievalFilters(tenant_id="default")
        with stage_scope("retrieval"):
            if kind == "dense":
                return await self.store.search_vector(
                    await self._query_vector(query), filters=filters, limit=depth
                )
            if kind == "bm25":
                return await self.store.search_bm25(query, filters=filters, limit=depth)
            candidates = await HybridRetriever(self.store).retrieve_candidates(
                sub_queries=[query],
                filters=filters,
                limit=depth,
                embedding_model=C.EMBEDDING_MODEL,
                embedding_dim=C.EMBEDDING_DIM,
            )
            if kind == "rrf":
                return candidates
            # rrf_bge: mirror RetrievalEngine (pool = min(24, n), BGE blend, consensus guard);
            # candidates beyond the reranked pool keep their RRF order so Recall@100 is defined.
            pool = min(24, max(C.GENERATION_TOP_K, len(candidates)))
        with stage_scope("rerank"):
            reranked = await Reranker.rerank(
                query=query, candidates=candidates, top_k=pool, strategy="ecohash",
                embedding_model=C.EMBEDDING_MODEL, embedding_dim=C.EMBEDDING_DIM,
            )  # fmt: skip
            reranked = protect_consensus(candidates, reranked, top_k=pool)
        seen = {c["id"] for c in reranked}
        return reranked + [c for c in candidates if c["id"] not in seen]

    @staticmethod
    def _child_score(c: dict[str, Any]) -> float:
        for key in ("rerank_score", "rrf_score", "score"):
            if c.get(key) is not None:
                return float(c[key])
        return 0.0

    async def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-``k`` *documents* (best chunk per document, rank order preserved)."""
        children = await self.retrieve_children(query, depth=max(k, C.RETRIEVAL_DEPTH))
        seen: set[str] = set()
        ranked: list[tuple[str, float]] = []
        for ch in children:
            doc_id = ch["document_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ranked.append((doc_id, self._child_score(ch)))
            if len(ranked) >= k:
                break
        return ranked

    # ------------------------------------------------------------------ generation
    async def answer(self, query: str, config: str | None = None) -> dict[str, Any]:
        """Retrieve -> generate. Returns answer, contexts, context_ids and per-stage trace."""
        from deep_context.generation.grounded_answer import generate_grounded_answer
        from deep_context.memory.prompt_assembler import PromptAssembler
        from deep_context.retrieval.engine import retrieval_engine
        from deep_context.retrieval.quality_gates import REFUSAL_TEMPLATE

        cfg = config or self.config
        t0 = time.perf_counter()
        extra: dict[str, Any] = {}
        with trace_scope(Trace()) as trace:
            if cfg == C.ENGINE_CONFIG:
                from deep_context.core.types import RetrievalFilters

                with stage_scope("retrieval"):
                    res = await retrieval_engine.retrieve(
                        query=query,
                        filters=RetrievalFilters(tenant_id="default"),
                        top_k=C.GENERATION_TOP_K,
                        embedding_model=C.EMBEDDING_MODEL,
                        embedding_dim=C.EMBEDDING_DIM,
                        reranker="ecohash",
                    )
                parents = res.parent_chunks
                mode = "two_pass"
                extra = {
                    "engine_sufficient": res.sufficient,
                    "engine_insufficiency_reason": res.insufficiency_reason,
                    "engine_query_shape": res.query_shape.value if res.query_shape else None,
                    "engine_retry_count": res.retry_count,
                    "anachronism_gate_fired": res.insufficiency_reason == "anachronism",
                }
            else:
                mode = C.CONFIGS[cfg][2]
                children = await self.retrieve_children(query, depth=C.RETRIEVAL_DEPTH, config=cfg)
                pool = children[: C.GENERATION_TOP_K * 4]
                parents = (await retrieval_engine._resolve_parent_chunks(self.store, pool))[
                    : C.GENERATION_TOP_K
                ]

            with stage_scope("generation"):
                refused = False
                reason: str | None = None
                if mode == "two_pass":
                    g = await generate_grounded_answer(
                        query, parents, model=C.GENERATOR_MODEL, timeout=C.REQUEST_TIMEOUT_S
                    )
                    answer, refused, reason = g.answer, g.refused, g.reason
                    extra.update(
                        support_passed=g.support_passed, support_confidence=g.support_confidence
                    )
                else:
                    from deep_context.core.llm_client import llm_client

                    assembler = PromptAssembler(self.store)
                    assembler.memory_manager = _NoMemory()  # type: ignore[assignment]
                    messages = await assembler.assemble_messages(
                        query=query, retrieved_chunks=parents
                    )
                    answer, _ = await llm_client.complete(
                        messages, model=C.GENERATOR_MODEL, temperature=0.1, max_tokens=1024,
                        enable_thinking=False,
                    )  # fmt: skip
                    answer = (answer or "").strip()
                    refused = REFUSAL_TEMPLATE in answer

        vis = visible_contexts(parents, mode)
        return {
            "answer": answer,
            "contexts": [p.get("content", "") for p in parents],
            "context_ids": [p["document_id"] for p in parents],
            "contexts_seen": [t for _, t in vis],
            "context_ids_seen": [parents[i - 1]["document_id"] for i, _ in vis],
            "evidence_chars_total": sum(len(p.get("content", "")) for p in parents),
            "evidence_chars_seen": sum(len(t) for _, t in vis),
            "latency_s": time.perf_counter() - t0,
            "refused": refused,
            "refusal_reason": reason,
            "generation_mode": mode,
            "trace_summary": trace.summary(),
            "trace": trace.to_json(),
            **extra,
        }


# --- Spec-style module-level convenience wrappers ----------------------------------------
_DEFAULT: DeepContextAdapter | None = None


async def index(corpus: dict[str, dict[str, str]], db_path: Path, config: str = "rrf") -> None:
    global _DEFAULT
    _DEFAULT = DeepContextAdapter(db_path, config)
    await _DEFAULT.open()
    await _DEFAULT.index(corpus)


async def retrieve(query: str, k: int) -> list[tuple[str, float]]:
    assert _DEFAULT is not None, "call index() first"
    return await _DEFAULT.retrieve(query, k)


async def answer(query: str) -> dict[str, Any]:
    assert _DEFAULT is not None, "call index() first"
    return await _DEFAULT.answer(query)
