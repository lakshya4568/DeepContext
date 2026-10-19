"""Thin adapter exposing DeepContext through ``index`` / ``retrieve`` / ``answer``.

Entry points reused from the product (nothing re-implemented except the corpus-loading loop):

* index    : ``DocumentParser`` + ``ParentChildChunker`` (the "fast" ingestion path: no Docling,
             no local summariser), ``llm_client.get_embeddings`` (patched to the strict Vertex
             client), ``PostgresStore`` (default) or ``SQLiteStore``.
* retrieve : the store's ``search_vector`` / ``search_bm25``, ``HybridRetriever`` (RRF k=60),
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
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval import config as C
from eval import pg
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
        if d.get("title_in_content") == "0":  # default (absent) keeps the original fingerprints
            h.update(b"\x1fnotitle")
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
    """Benchmark adapter around the DeepContext retrieval + generation stack.

    ``backend``: ``postgres`` (default; dedicated ``deepcontext_eval`` database, one schema per
    index, pgvector HNSW + Postgres full-text) or ``sqlite`` (FTS5 BM25 + exact cosine; explicit
    opt-in via ``EVAL_BACKEND=sqlite``). Results from the two backends are not interchangeable.
    """

    def __init__(
        self, index_name: str | Path, config: str = "rrf_bge_two_pass", backend: str | None = None
    ) -> None:
        if config not in C.CONFIGS and config != C.ENGINE_CONFIG:
            raise ValueError(f"Unknown config '{config}'. Supported: {list(C.CONFIGS)}")
        self.backend = (backend or os.environ.get("EVAL_BACKEND", "postgres")).lower()
        if self.backend not in ("postgres", "sqlite"):
            raise ValueError(f"Unknown backend '{self.backend}' (postgres | sqlite).")
        self.index_name = Path(str(index_name)).stem
        self.db_path = C.INDEX_DIR / f"{self.index_name}.db"  # sqlite only
        self.schema = pg.schema_name(self.index_name)  # postgres only
        self.config = config
        self.runtime: VertexRuntime = install_strict_patches()
        self.store: Any = None
        self._ext_by_store: dict[str, str] = {}

    # ------------------------------------------------------------------ ids
    def store_doc_id(self, external_id: str) -> str:
        """Id under which a benchmark document is stored (UUID on Postgres, as-is on SQLite)."""
        return pg.doc_uuid(external_id) if self.backend == "postgres" else external_id

    def _register_ids(self, corpus: dict[str, dict[str, str]]) -> None:
        self._ext_by_store = {self.store_doc_id(d): d for d in corpus}

    def ext_id(self, store_id: str) -> str:
        return self._ext_by_store.get(store_id, store_id)

    # ------------------------------------------------------------------ lifecycle
    async def open(self) -> None:
        import deep_context.storage as storage_mod

        if self.backend == "postgres":
            if not await pg.server_available():
                raise RuntimeError(
                    "PostgreSQL is not reachable with the configured POSTGRES_DSN. Start it "
                    "(brew services start postgresql@15) or run with EVAL_BACKEND=sqlite - "
                    "results from the two backends must not be mixed."
                )
            await pg.ensure_database()
            await pg.ensure_schema(self.schema)
            self.store = pg.make_store_class()(pg.dsn_for(pg.EVAL_DB, self.schema))
        else:
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

    # ------------------------------------------------------------------ index metadata
    async def _read_meta(self) -> dict[str, Any] | None:
        if self.backend == "postgres":
            async with self.store._get_pool().acquire() as c:
                await c.execute(
                    "CREATE TABLE IF NOT EXISTS eval_index_meta (k TEXT PRIMARY KEY, v JSONB NOT NULL)"
                )
                row = await c.fetchrow("SELECT v FROM eval_index_meta WHERE k = 'meta'")
            return json.loads(row["v"]) if row else None
        path = Path(f"{self.db_path}.meta.json")
        return json.loads(path.read_text()) if path.exists() else None

    async def _write_meta(self, meta: dict[str, Any]) -> None:
        if self.backend == "postgres":
            async with self.store._get_pool().acquire() as c:
                await c.execute(
                    "INSERT INTO eval_index_meta (k, v) VALUES ('meta', $1::jsonb) "
                    "ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v",
                    json.dumps(meta),
                )
        else:
            Path(f"{self.db_path}.meta.json").write_text(json.dumps(meta))

    # ------------------------------------------------------------------ index
    async def index(self, corpus: dict[str, dict[str, str]]) -> IndexStats:
        """Fast ingestion of ``{doc_id: {title, text}}``; reuses an existing identical index."""
        from deep_context.core.types import ChunkLevel, Document
        from deep_context.ingestion.chunker import ParentChildChunker
        from deep_context.ingestion.parser import DocumentParser

        t0 = time.perf_counter()
        if self.store is None:
            await self.open()
        self._register_ids(corpus)
        fp = corpus_fingerprint(corpus)
        meta = await self._read_meta()
        if meta is not None:
            if meta.get("fingerprint") == fp:
                log.info(
                    "Reusing %s index %s (%d docs).", self.backend, self.index_name, meta["n_docs"]
                )
                return IndexStats(
                    meta["n_docs"], meta["n_parents"], meta["n_children"], True,
                    time.perf_counter() - t0, meta.get("approx_embed_tokens", 0),
                )  # fmt: skip
            raise RuntimeError(
                f"{self.index_name} holds a different corpus (fingerprint mismatch); "
                "delete it or choose another name."
            )

        chunker = ParentChildChunker()

        def stage(
            ext_doc_id: str, d: dict[str, str]
        ) -> tuple[Document, list[Any], list[Any]] | None:
            doc_id = self.store_doc_id(ext_doc_id)
            title = (d.get("title") or "").strip()
            text = (d.get("text") or "").strip()
            # Default: title + text (BEIR convention). Corpora whose titles are long and shared by
            # every document (PDF pages, doc sections) set title_in_content="0": the title then lives
            # only in Document.title and the embedding prefix (as in the shipped ingestion), so the
            # shipped near-duplicate filter (first 120 characters) is not triggered by the title.
            with_title = bool(title) and d.get("title_in_content") != "0"
            content = f"{title}\n\n{text}" if with_title else text
            if not content:
                return None
            sections = DocumentParser.parse(content, doc_type="text")
            parents, children = chunker.chunk_sections(doc_id, sections)
            id_map: dict[str, str] = {}
            for n, p in enumerate(parents):
                new = str(uuid.uuid5(_NS, f"{ext_doc_id}|parent|{n}"))
                id_map[p.id], p.id = new, new
            for n, ch in enumerate(children):
                ch.id = str(uuid.uuid5(_NS, f"{ext_doc_id}|child|{n}"))
                if ch.parent_chunk_id is not None:
                    ch.parent_chunk_id = id_map[ch.parent_chunk_id]
            doc = Document(
                id=doc_id,
                title=title or ext_doc_id,
                doc_type="text",
                metadata={"eval": True, "external_id": ext_doc_id},
            )
            return doc, parents, children

        # Stream the corpus in doc windows (chunk -> embed -> store -> free) so memory stays flat
        # on 200k-doc corpora. Every insert is idempotent (deterministic uuid5 ids; documents upsert,
        # chunks ON CONFLICT DO NOTHING) and embeddings are cached by text, so an interrupted
        # build resumes cheaply by re-running the same call.
        n_docs = n_parents = n_children = approx_tokens = 0
        items = list(corpus.items())
        window_docs = 1500
        log.info("Indexing %d docs in windows of %d ...", len(items), window_docs)
        for w0 in range(0, len(items), window_docs):
            staged = [
                st
                for ext_doc_id, d in items[w0 : w0 + window_docs]
                if (st := stage(ext_doc_id, d)) is not None
            ]
            # One embedding request per child chunk (gemini-embedding-2 accepts one content per
            # call), run concurrently and cached on disk by formatted text.
            texts: list[str] = []
            owners: list[Any] = []
            for doc, _p, children in staged:
                for ch in children:
                    assert ch.level == ChunkLevel.CHILD
                    texts.append(
                        self.runtime.format_embedding_text(
                            ch.content,
                            is_query=False,
                            title=doc.title if doc.title != doc.metadata["external_id"] else None,
                            task=None,
                        )
                    )
                    owners.append(ch)
            with stage_scope("index"):
                vecs = await self.runtime.embed_many(
                    texts, model=C.EMBEDDING_MODEL, dim=C.EMBEDDING_DIM
                )
            for ch, v in zip(owners, vecs, strict=True):
                ch.embedding = v.tolist()
            self.runtime.cache.flush()
            approx_tokens += sum(max(len(t.split()), len(t) // 4) for t in texts)

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
            n_docs += len(staged)
            del staged, texts, owners, vecs, batch
            log.info(
                "  indexed %d/%d docs (%d child chunks, %.0fs)",
                min(w0 + window_docs, len(items)), len(items), n_children, time.perf_counter() - t0,
            )  # fmt: skip

        if self.backend == "postgres":
            async with self.store._get_pool().acquire() as c:
                await c.execute("ANALYZE documents")
                await c.execute("ANALYZE chunks")
        else:
            invalidate_vector_index(self.store)

        await self._write_meta(
            {
                "fingerprint": fp,
                "backend": self.backend,
                "n_docs": n_docs,
                "n_parents": n_parents,
                "n_children": n_children,
                "n_skipped_empty_docs": len(corpus) - n_docs,
                "embedding_model": C.EMBEDDING_MODEL,
                "embedding_dim": C.EMBEDDING_DIM,
                "approx_embed_tokens": approx_tokens,
            }
        )
        return IndexStats(
            n_docs, n_parents, n_children, False, time.perf_counter() - t0, approx_tokens
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

    def rank_documents(self, children: list[dict[str, Any]]) -> list[tuple[str, float]]:
        """Collapse ranked chunks to ranked *documents* (best chunk wins), external ids."""
        seen: set[str] = set()
        ranked: list[tuple[str, float]] = []
        for ch in children:
            ext = self.ext_id(ch["document_id"])
            if ext in seen:
                continue
            seen.add(ext)
            ranked.append((ext, self._child_score(ch)))
        return ranked

    async def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-``k`` *documents* (best chunk per document, rank order preserved)."""
        children = await self.retrieve_children(query, depth=max(k, C.RETRIEVAL_DEPTH))
        return self.rank_documents(children)[:k]

    async def engine_retrieve(self, query: str) -> tuple[list[tuple[str, float]], dict[str, Any]]:
        """Retrieval stage of the shipped engine (rewriter, classifier, gates, retry, rerank).

        The engine returns at most ``GENERATION_TOP_K`` parent chunks, so the ranked document list
        is at most that long (rank metrics at k > 6 are lower bounds, flagged in the tables).
        """
        from deep_context.core.types import RetrievalFilters
        from deep_context.retrieval.engine import retrieval_engine

        with stage_scope("retrieval"):
            res = await retrieval_engine.retrieve(
                query=query,
                filters=RetrievalFilters(tenant_id="default"),
                top_k=C.GENERATION_TOP_K,
                embedding_model=C.EMBEDDING_MODEL,
                embedding_dim=C.EMBEDDING_DIM,
                reranker="ecohash",
            )
        ranked = self.rank_documents(res.parent_chunks)
        extra = {
            "engine_sufficient": res.sufficient,
            "engine_insufficiency_reason": res.insufficiency_reason,
            "engine_query_shape": res.query_shape.value if res.query_shape else None,
            "engine_retry_count": res.retry_count,
        }
        return ranked, extra

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
            "context_ids": [self.ext_id(p["document_id"]) for p in parents],
            "contexts_seen": [t for _, t in vis],
            "context_ids_seen": [self.ext_id(parents[i - 1]["document_id"]) for i, _ in vis],
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
