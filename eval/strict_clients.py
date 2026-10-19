"""Strict, cached, instrumented model transport for the benchmark harness.

Why this exists (all verified by reading ``src/deep_context/core/llm_client.py`` on the
``feat/rag-quality-generation`` base branch):

* ``LLMClient.get_embeddings`` falls through to deterministic *hash* embeddings whenever the
  Gemini call fails and an NVIDIA key exists, even with ``ALLOW_MOCK_FALLBACK=false``.
* ``LLMClient.complete`` silently fails over Gemini -> Groq -> NVIDIA -> canned mock text.
* ``EcoHashReranker.rerank`` silently falls back to the heuristic reranker on any error.
* This branch only talks to Gemini through an API key; the available credentials are Vertex AI
  Application Default Credentials, and ``gemini-embedding-2`` accepts one content per request.

A benchmark must never score a silent substitute as "DeepContext + Gemini + BGE". This module
replaces those three entry points *at runtime, inside the eval process only* (``src/`` is not
modified) with versions that: use Vertex AI, retry transient errors with exponential backoff,
raise instead of degrading, cache every response on disk (resumable, deterministic reruns),
and record per-call latency / token usage for the system tables.

Documented deviations from stock behaviour (also listed in REPORT.md):
  1. Caller timeouts (5-15 s in the rewriter/generator) are replaced by ``REQUEST_TIMEOUT_S``
     so results do not depend on API load.
  2. Cached calls report the latency measured on the first live call (retry sleeps excluded).
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

from eval import config as C

load_dotenv(C.REPO_ROOT / ".env")
log = logging.getLogger("eval.transport")


# ======================================================================================
# Disk cache
# ======================================================================================
class DiskCache:
    """Tiny key/value store on SQLite (WAL) shared safely between concurrent eval processes.

    Autocommit: every put is its own short transaction, so no process holds the write lock while
    it waits on the network (the earlier batched-commit version caused "database is locked").
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            path, check_same_thread=False, timeout=120, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=120000")
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, value))

    def flush(self) -> None:  # kept for API compatibility; autocommit makes it a no-op
        return None

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0])


def _sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
        h.update(b"\x1f")
    return h.hexdigest()


# ======================================================================================
# Per-query tracing (stage-tagged call records)
# ======================================================================================
@dataclass
class CallRecord:
    stage: str  # index | retrieval | rerank | generation | judge
    kind: str  # llm | embed | rerank
    model: str
    in_tokens: int = 0
    out_tokens: int = 0  # candidates + thinking tokens (billed as output)
    think_tokens: int = 0
    latency_s: float = 0.0  # successful attempt only (retry sleeps excluded)
    cached: bool = False
    retries: int = 0
    finish_reason: str | None = None
    empty_output: bool = False


@dataclass
class Trace:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.records.append(rec)

    def stage_latency(self, stage: str) -> float:
        return sum(r.latency_s for r in self.records if r.stage == stage)

    def tokens(self, stage: str | None = None, kind: str = "llm") -> tuple[int, int]:
        rows = [r for r in self.records if r.kind == kind and (stage is None or r.stage == stage)]
        return sum(r.in_tokens for r in rows), sum(r.out_tokens for r in rows)

    def summary(self) -> dict[str, Any]:
        stages = sorted({r.stage for r in self.records})
        out: dict[str, Any] = {}
        for s in stages:
            rows = [r for r in self.records if r.stage == s]
            out[s] = {
                "calls": len(rows),
                "latency_s": round(sum(r.latency_s for r in rows), 4),
                "in_tokens": sum(r.in_tokens for r in rows),
                "out_tokens": sum(r.out_tokens for r in rows),
                "cached": sum(1 for r in rows if r.cached),
                "retries": sum(r.retries for r in rows),
                "empty_outputs": sum(1 for r in rows if r.empty_output),
            }
        return out

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]


_TRACE: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("eval_trace", default=None)
_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("eval_stage", default="unattributed")


@contextmanager
def trace_scope(trace: Trace | None = None) -> Iterator[Trace]:
    trace = trace or Trace()
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


@contextmanager
def stage_scope(stage: str) -> Iterator[None]:
    token = _STAGE.set(stage)
    try:
        yield
    finally:
        _STAGE.reset(token)


def _record(rec: CallRecord) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.add(rec)


class MockFallbackError(RuntimeError):
    """Raised if any stock mock/hash fallback is reached during a benchmark run."""


class ApiExhaustedError(RuntimeError):
    """Raised when a call still fails after ``MAX_API_ATTEMPTS`` attempts."""


# ======================================================================================
# Transport
# ======================================================================================
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in (408, 409, 429, 500, 502, 503, 504):
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("resource_exhausted", "unavailable", "deadline", "overloaded", "temporarily")
    )


@dataclass
class GenResult:
    text: str
    in_tokens: int
    out_tokens: int
    think_tokens: int
    latency_s: float
    finish_reason: str | None
    cached: bool
    retries: int


class VertexRuntime:
    """Vertex AI (ADC) Gemini + EcoHash reranker with retries, caching and tracing."""

    def __init__(self, cache: DiskCache | None = None) -> None:
        self.cache = cache or DiskCache(C.LLM_CACHE_PATH)
        self._clients: dict[str, Any] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._cooldown_until = 0.0
        self.honor_enable_thinking = os.environ.get("EVAL_HONOR_ENABLE_THINKING", "1") != "0"
        self.bypass_cache = False  # latency passes: neither read nor write the cache
        # per-model usage ledger for this process (live = billed, cached = replayed)
        self.usage: dict[str, dict[str, float]] = {}
        self.gen_thinking_budget: int | None = (
            int(os.environ["EVAL_GEN_THINKING_BUDGET"])
            if os.environ.get("EVAL_GEN_THINKING_BUDGET", "").lstrip("-").isdigit()
            else None
        )

    # ---- plumbing -------------------------------------------------------------------
    def client(self, location: str | None = None) -> Any:
        """Vertex AI client (ADC). ``location`` defaults to GOOGLE_CLOUD_LOCATION."""
        loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if loc not in self._clients:
            from google import genai

            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set (Vertex AI + ADC required).")
            self._clients[loc] = genai.Client(vertexai=True, project=project, location=loc)
        return self._clients[loc]

    def _sem(self, kind: str) -> asyncio.Semaphore:
        """Semaphore for ``kind``, one per running event loop.

        RAGAS runs each ``evaluate()`` call in its own event loop; a semaphore created in an
        earlier loop raises "bound to a different event loop" (which RAGAS then reports as NaN).
        """
        key = f"{kind}:{id(asyncio.get_running_loop())}"
        if key not in self._sems:
            limit = {
                "llm": C.LLM_CONCURRENCY,
                "embed": C.EMBED_CONCURRENCY,
                "rerank": C.RERANK_CONCURRENCY,
            }[kind]
            self._sems[key] = asyncio.Semaphore(limit)
        return self._sems[key]

    async def _with_retries(self, kind: str, call: Any) -> tuple[Any, float, int]:
        """Run ``call()``; retry transient failures. Returns (value, ok_attempt_seconds, retries)."""
        last: BaseException | None = None
        for attempt in range(C.MAX_API_ATTEMPTS):
            wait = self._cooldown_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            async with self._sem(kind):
                t0 = time.perf_counter()
                try:
                    value = await asyncio.wait_for(call(), timeout=C.REQUEST_TIMEOUT_S)
                    return value, time.perf_counter() - t0, attempt
                except Exception as exc:  # noqa: BLE001 - classified below, re-raised if fatal
                    last = exc
                    if not _is_retryable(exc):
                        raise
            delay = min(60.0, 2.0 * 2**attempt) * (0.5 + random.random())
            log.warning(
                "%s call retry %d/%d in %.0fs after %s: %s",
                kind,
                attempt + 1,
                C.MAX_API_ATTEMPTS,
                delay,
                type(last).__name__,
                str(last)[:140],
            )
            if "429" in str(last) or "resource_exhausted" in str(last).lower():
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
            await asyncio.sleep(delay)
        raise ApiExhaustedError(f"{kind} call failed after {C.MAX_API_ATTEMPTS} attempts: {last}")

    # ---- generation -----------------------------------------------------------------
    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        thinking_budget: int | None = None,
        json_mode: bool = False,
        candidate_idx: int = 0,
        stage: str | None = None,
    ) -> GenResult:
        from google.genai import types as T

        system = "\n".join(m["content"] for m in messages if m.get("role") == "system").strip()
        turns = [
            (("user" if m.get("role") == "user" else "model"), m.get("content", ""))
            for m in messages
            if m.get("role") != "system" and m.get("content")
        ] or [("user", "Hello")]
        max_out = min(max_tokens, 8192) if max_tokens else 8192
        key = _sha(
            "gen", model, system, turns, temperature, top_p, max_out, thinking_budget, json_mode,
            candidate_idx,
        )  # fmt: skip
        stage_name = stage or _STAGE.get()

        hit = None if self.bypass_cache else self.cache.get(key)
        if hit is not None:
            d = json.loads(hit)
            res = GenResult(**{**d, "cached": True, "retries": 0})
            self._ledger(model, res)
            self._trace_gen(stage_name, model, res)
            return res

        cfg: dict[str, Any] = dict(
            system_instruction=system or None,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_out,
        )
        if thinking_budget is not None:
            cfg["thinking_config"] = T.ThinkingConfig(thinking_budget=thinking_budget)
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        contents = [T.Content(role=r, parts=[T.Part.from_text(text=c)]) for r, c in turns]

        location = "global" if model.startswith(C.GLOBAL_ENDPOINT_PREFIXES) else None

        async def call() -> Any:
            return await self.client(location).aio.models.generate_content(
                model=model, contents=contents, config=T.GenerateContentConfig(**cfg)
            )

        resp, secs, retries = await self._with_retries("llm", call)
        usage = resp.usage_metadata
        think = int(getattr(usage, "thoughts_token_count", 0) or 0)
        cand = int(getattr(usage, "candidates_token_count", 0) or 0)
        finish = None
        if resp.candidates:
            fr = resp.candidates[0].finish_reason
            finish = getattr(fr, "name", str(fr)) if fr is not None else None
        text = resp.text or ""
        res = GenResult(
            text=text,
            in_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            out_tokens=cand + think,
            think_tokens=think,
            latency_s=secs,
            finish_reason=finish,
            cached=False,
            retries=retries,
        )
        stored = {k: v for k, v in asdict(res).items() if k not in ("cached", "retries")}
        if not self.bypass_cache:
            self.cache.put(key, json.dumps(stored).encode())
        self._ledger(model, res)
        self._trace_gen(stage_name, model, res)
        return res

    def _ledger(self, model: str, res: GenResult) -> None:
        u = self.usage.setdefault(
            model, {"calls": 0, "live_calls": 0, "in": 0, "out": 0, "live_in": 0, "live_out": 0}
        )
        u["calls"] += 1
        u["in"] += res.in_tokens
        u["out"] += res.out_tokens
        if not res.cached:
            u["live_calls"] += 1
            u["live_in"] += res.in_tokens
            u["live_out"] += res.out_tokens

    def usage_report(self) -> str:
        lines = []
        for model, u in sorted(self.usage.items()):
            price = C.PRICES_USD_PER_M.get(model)
            cost = (
                (u["live_in"] * price["in"] + u["live_out"] * price["out"]) / 1e6
                if price
                else float("nan")
            )
            lines.append(
                f"{model}: {int(u['calls'])} calls ({int(u['live_calls'])} live) | live tokens in/out "
                f"{int(u['live_in']):,}/{int(u['live_out']):,} | est. list-price cost ${cost:.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _trace_gen(stage: str, model: str, res: GenResult) -> None:
        _record(
            CallRecord(
                stage=stage, kind="llm", model=model, in_tokens=res.in_tokens,
                out_tokens=res.out_tokens, think_tokens=res.think_tokens,
                latency_s=res.latency_s, cached=res.cached, retries=res.retries,
                finish_reason=res.finish_reason, empty_output=not res.text.strip(),
            )
        )  # fmt: skip

    # ---- embeddings -----------------------------------------------------------------
    @staticmethod
    def format_embedding_text(
        text: str, *, is_query: bool, title: str | None, task: str | None
    ) -> str:
        """Same input format the stock client uses for gemini-embedding-2."""
        t = text.strip() or " "
        if is_query:
            return f"task: {task or 'search result'} | query: {t}"
        return f"title: {title or 'none'} | text: {t}"

    async def embed_formatted(self, formatted: str, *, model: str, dim: int) -> np.ndarray:
        key = _sha("emb", model, dim, formatted)
        stage_name = _STAGE.get()
        hit = None if self.bypass_cache else self.cache.get(key)
        if hit is not None:
            vec = np.frombuffer(hit, dtype=np.float32)
            _record(CallRecord(stage=stage_name, kind="embed", model=model,
                               in_tokens=_approx_tokens(formatted), cached=True))  # fmt: skip
            return vec
        from google.genai import types as T

        async def call() -> Any:
            return await self.client(C.EMBEDDING_LOCATION).aio.models.embed_content(
                model=model,
                contents=[T.Content(parts=[T.Part.from_text(text=formatted)])],
                config=T.EmbedContentConfig(output_dimensionality=dim),
            )

        resp, secs, retries = await self._with_retries("embed", call)
        if not resp.embeddings or resp.embeddings[0].values is None:
            raise ApiExhaustedError("Embedding response contained no vector.")
        vec = np.asarray(resp.embeddings[0].values, dtype=np.float32)
        if vec.shape[0] != dim:
            raise ApiExhaustedError(f"Embedding dim {vec.shape[0]} != requested {dim}.")
        if not self.bypass_cache:
            self.cache.put(key, vec.tobytes())
        _record(CallRecord(stage=stage_name, kind="embed", model=model,
                           in_tokens=_approx_tokens(formatted), latency_s=secs, retries=retries))  # fmt: skip
        return vec

    def embed_formatted_sync(self, formatted: str, *, model: str, dim: int) -> np.ndarray:
        """Blocking twin of ``embed_formatted`` (used by RAGAS' synchronous similarity step)."""
        key = _sha("emb", model, dim, formatted)
        hit = None if self.bypass_cache else self.cache.get(key)
        if hit is not None:
            return np.frombuffer(hit, dtype=np.float32)
        from google.genai import types as T

        last: BaseException | None = None
        for attempt in range(C.MAX_API_ATTEMPTS):
            try:
                resp = self.client(C.EMBEDDING_LOCATION).models.embed_content(
                    model=model,
                    contents=[T.Content(parts=[T.Part.from_text(text=formatted)])],
                    config=T.EmbedContentConfig(output_dimensionality=dim),
                )
                vec = np.asarray(resp.embeddings[0].values, dtype=np.float32)
                if vec.shape[0] != dim:
                    raise ApiExhaustedError(f"Embedding dim {vec.shape[0]} != {dim}.")
                if not self.bypass_cache:
                    self.cache.put(key, vec.tobytes())
                return vec
            except Exception as exc:  # noqa: BLE001
                last = exc
                if not _is_retryable(exc):
                    raise
                time.sleep(min(60.0, 2.0 * 2**attempt) * (0.5 + random.random()))
        raise ApiExhaustedError(f"sync embed failed after retries: {last}")

    async def embed_many(self, formatted: list[str], *, model: str, dim: int) -> list[np.ndarray]:
        return list(
            await asyncio.gather(
                *(self.embed_formatted(f, model=model, dim=dim) for f in formatted)
            )
        )

    # ---- reranking ------------------------------------------------------------------
    async def rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        """Hosted bge-reranker-v2-m3 relevance scores, aligned with ``documents``."""
        from deep_context.core.config import settings

        # EVAL_RERANK_URL switches to a self-hosted bge-reranker-v2-m3 server (POST <url>/rerank,
        # same response shape, no API key). The backend is part of the cache key and of the model
        # label recorded in every trace, so rows produced by different backends stay identifiable.
        selfhosted = os.environ.get("EVAL_RERANK_URL", "").rstrip("/")
        model_label = (
            f"{settings.ecohash_rerank_model}@{'self-hosted' if selfhosted else 'ecohash'}"
        )
        key = (
            _sha("rerank-selfhosted", "BAAI/bge-reranker-v2-m3", query, documents)
            if selfhosted
            else _sha("rerank", settings.ecohash_rerank_model, query, documents)
        )
        stage_name = "rerank"
        hit = None if self.bypass_cache else self.cache.get(key)
        if hit is not None:
            d = json.loads(hit)
            _record(CallRecord(stage=stage_name, kind="rerank", model=model_label,
                               in_tokens=d["tokens"], latency_s=d["latency_s"], cached=True))  # fmt: skip
            return list(d["scores"])

        async def call() -> httpx.Response:
            async with httpx.AsyncClient(timeout=C.REQUEST_TIMEOUT_S) as http:
                if selfhosted:
                    r = await http.post(
                        f"{selfhosted}/rerank", json={"query": query, "documents": documents}
                    )
                    r.raise_for_status()
                    return r
                r = await http.post(
                    settings.ecohash_rerank_url,
                    headers={
                        "Authorization": f"Bearer {settings.ecohash_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.ecohash_rerank_model,
                        "query": query,
                        "documents": documents,
                    },
                )
                r.raise_for_status()
                return r

        r, secs, retries = await self._with_retries("rerank", call)
        data = r.json()
        scores = [0.0] * len(documents)
        returned = 0
        for item in data.get("results", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(scores):
                scores[idx] = float(item.get("relevance_score", 0.0))
                returned += 1
        if returned != len(documents):
            raise ApiExhaustedError(
                f"Reranker returned {returned}/{len(documents)} scores; refusing to continue."
            )
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        if not self.bypass_cache:
            self.cache.put(
                key, json.dumps({"scores": scores, "tokens": tokens, "latency_s": secs}).encode()
            )
        _record(CallRecord(stage=stage_name, kind="rerank", model=model_label,
                           in_tokens=tokens, latency_s=secs, retries=retries))  # fmt: skip
        return scores


def _approx_tokens(text: str) -> int:
    """Same heuristic the repo uses (``count_approx_tokens``); labelled approximate in tables."""
    return max(len(text.split()), len(text) // 4)


# ======================================================================================
# Installation of the strict entry points into the stock modules
# ======================================================================================
_RUNTIME: VertexRuntime | None = None


def get_runtime() -> VertexRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = VertexRuntime()
    return _RUNTIME


def install_strict_patches() -> VertexRuntime:
    """Patch the stock singletons for the lifetime of this process. Idempotent."""
    rt = get_runtime()

    from deep_context.core.config import settings
    from deep_context.core.llm_client import llm_client

    # Single-provider, no-mock configuration (the rewriter picks ``settings.llm_model`` when no
    # NVIDIA key is present, so blanking the key routes it to Gemini as well).
    settings.allow_mock_fallback = False
    settings.nvidia_api_key = ""
    settings.groq_api_key = ""
    settings.llm_provider = "gemini"
    settings.llm_model = C.GENERATOR_MODEL
    settings.embedding_provider = "gemini"
    settings.embedding_model = C.EMBEDDING_MODEL
    settings.embedding_dim = C.EMBEDDING_DIM
    settings.database_type = "sqlite"
    settings.cache_enabled = False

    async def complete(
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        enable_thinking: bool = True,
        timeout: float = 60.0,  # noqa: ARG001 - see module docstring, deviation 1
        max_retries: int = 2,  # noqa: ARG001
    ) -> tuple[str, str | None]:
        target = model or settings.llm_model
        if not target.startswith("gemini-"):
            raise MockFallbackError(
                f"Non-Gemini model '{target}' requested; the harness is single-provider."
            )
        # The stock Gemini branch ignores ``enable_thinking``. With a thinking model's default
        # dynamic thinking, the two-pass extractor (max_tokens=512) spends ~490 tokens thinking
        # and returns truncated JSON. Callers pass enable_thinking=False precisely to avoid that,
        # so we honour it (EVAL_HONOR_ENABLE_THINKING=0 reproduces the stock behaviour).
        budget = rt.gen_thinking_budget
        if not enable_thinking and rt.honor_enable_thinking:
            budget = 0
        res = await rt.generate(
            model=target,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking_budget=budget,
        )
        text, reasoning = llm_client._parse_think_tags(res.text, None)
        return text, reasoning

    async def get_embeddings(
        texts: list[str],
        model: str | None = None,
        dim: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
        is_query: bool = False,
    ) -> list[list[float]]:
        use_model = model or C.EMBEDDING_MODEL
        if "embedding-2" not in use_model:
            raise MockFallbackError(f"Unsupported embedding model '{use_model}' in harness.")
        fmt = [
            rt.format_embedding_text(t, is_query=is_query, title=title, task=task_type)
            for t in texts
        ]
        vecs = await rt.embed_many(fmt, model=use_model, dim=dim or C.EMBEDDING_DIM)
        return [v.tolist() for v in vecs]

    async def get_embedding(text: str, **kw: Any) -> list[float]:
        return (await get_embeddings([text], **kw))[0]

    def _forbidden(*_a: Any, **_k: Any) -> Any:
        raise MockFallbackError("A stock mock/hash fallback was reached during a benchmark run.")

    llm_client.complete = complete  # type: ignore[method-assign]
    llm_client.get_embeddings = get_embeddings  # type: ignore[method-assign]
    llm_client.get_embedding = get_embedding  # type: ignore[method-assign]
    llm_client._mock_embedding = _forbidden  # type: ignore[method-assign]
    llm_client._mock_completion = _forbidden  # type: ignore[method-assign]

    from deep_context.retrieval.ecohash_reranker import EcoHashReranker
    from deep_context.retrieval.reranker import _blend_with_rrf

    async def strict_rerank(
        cls: Any,  # noqa: ARG001
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        api_key: str | None = None,  # noqa: ARG001
        model: str | None = None,  # noqa: ARG001
    ) -> list[dict[str, Any]]:
        # Mirrors EcoHashReranker.rerank (same 1000-char truncation, same blend) minus the
        # silent heuristic fallback.
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates
        docs = [str(c.get("content", ""))[:1000] for c in candidates]
        raw = await rt.rerank_scores(query, docs)
        scored = _blend_with_rrf(candidates, raw, score_type="probability")
        return [item for _, item in scored[:top_k]]

    EcoHashReranker.rerank = classmethod(strict_rerank)  # type: ignore[method-assign,assignment]

    from eval.fast_vector import install_fast_vector_search

    install_fast_vector_search()
    return rt
