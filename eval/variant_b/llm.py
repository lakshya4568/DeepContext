"""Async client for the OpenAI-compatible vLLM server that runs the small summariser model.

Every reply is cached in Postgres (``eval_cache.kv``, namespace ``variant_b_llm``) by its full request, so an
interrupted run resumes without sampling again and a re-run reproduces the same text (sampling at
temperature > 0 is otherwise not repeatable).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from eval.pg_cache import KV, PgKV

log = logging.getLogger("eval.variant_b.llm")

DEFAULT_URL = os.environ.get("VB_LLM_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("VB_LLM_MODEL", "qwen3.5-2b")


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish: str
    cached: bool


class SlmClient:
    """``chat`` with retries, a concurrency cap and the Postgres cache (pass ``cache=MemoryKV()`` to skip it)."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        *,
        concurrency: int = 48,
        cache: KV | None = None,
        timeout_s: float = 300.0,
        attempts: int = 4,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout_s = timeout_s
        self.attempts = attempts
        self._sem = asyncio.Semaphore(concurrency)
        self._cache: KV = PgKV("variant_b_llm") if cache is None else cache
        self._http: httpx.AsyncClient | None = None
        self.calls = 0
        self.cache_hits = 0

    def _key(self, body: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    async def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 300,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": json_schema, "strict": True},
            }
        key = self._key(body)
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            d = json.loads(hit)
            return Reply(d["text"], d["pt"], d["ct"], d["lat"], d["finish"], True)

        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
        last: Exception | None = None
        for attempt in range(self.attempts):
            async with self._sem:
                t0 = time.perf_counter()
                try:
                    r = await self._http.post(self.url, json=body)
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last = e
                else:
                    if r.status_code == 200:
                        data = r.json()
                        choice = data["choices"][0]
                        reply = Reply(
                            choice["message"].get("content") or "",
                            int(data["usage"]["prompt_tokens"]),
                            int(data["usage"]["completion_tokens"]),
                            time.perf_counter() - t0,
                            choice.get("finish_reason") or "",
                            False,
                        )
                        self.calls += 1
                        self._cache.put(
                            key,
                            json.dumps(
                                {
                                    "text": reply.text,
                                    "pt": reply.prompt_tokens,
                                    "ct": reply.completion_tokens,
                                    "lat": reply.latency_s,
                                    "finish": reply.finish,
                                }  # fmt: skip
                            ).encode(),
                        )
                        return reply
                    if r.status_code in (429, 500, 502, 503, 504):
                        last = RuntimeError(f"HTTP {r.status_code}")
                    else:
                        raise RuntimeError(
                            f"summariser server error {r.status_code}: {r.text[:300]}"
                        )
            await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError(
            f"summariser server unreachable after {self.attempts} attempts: {last!r}"
        )

    def flush(self) -> None:
        self._cache.flush()

    async def aclose(self) -> None:
        self.flush()
        if self._http is not None:
            await self._http.aclose()
            self._http = None
