"""Response cache layer for RAG pipelines.

Implements a Redis-backed cache (via ``redis.asyncio``) with an automatic
in-memory fallback so the platform works identically in dev/test environments
without a running Redis server. Cache keys are stable SHA-256 digests of the
sorted JSON payload, namespaced per operation, following the production
agentic-RAG pattern of caching "question -> final answer + citations" at the
pipeline level.

Configuration (see core/config.py):
- CACHE_ENABLED: master switch
- CACHE_URL: redis://... URL; empty string forces the in-memory backend
- CACHE_TTL: default TTL seconds
- CACHE_NAMESPACE: key prefix
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from deep_context.core.config import settings
from deep_context.core.logging import logger


class InMemoryCacheBackend:
    """Process-local TTL cache used when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at and expires_at < time.monotonic():
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        self._store[key] = (time.monotonic() + ttl, value)

    async def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)

    async def aclose(self) -> None:
        self._store.clear()


class RedisCacheBackend:
    """Async Redis backend. Only instantiated when a CACHE_URL is reachable."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._client.setex(key, ttl, value)

    async def delete_prefix(self, prefix: str) -> int:
        count = 0
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor, match=f"{prefix}*", count=100
            )
            if keys:
                await self._client.delete(*keys)
                count += len(keys)
            if cursor == 0:
                break
        return count

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass


class ResponseCache:
    """Namespaced JSON cache with TTL, safe for both sync and async call sites."""

    def __init__(self) -> None:
        self._backend: InMemoryCacheBackend | RedisCacheBackend | None = None
        self._backend_kind: str = "uninitialized"

    def _resolve_backend(self) -> InMemoryCacheBackend | RedisCacheBackend:
        if self._backend is not None:
            return self._backend

        url = settings.cache_url.strip()
        if url:
            try:
                import redis.asyncio as aioredis  # type: ignore[import-not-found]

                client = aioredis.from_url(url, decode_responses=True)
                self._backend = RedisCacheBackend(client)
                self._backend_kind = "redis"
                logger.info("Response cache using Redis at %s", url)
                return self._backend
            except ImportError:
                logger.warning(
                    "redis package not installed; falling back to in-memory cache."
                )
            except Exception as e:
                logger.warning(
                    "Failed to connect to Redis (%s); using in-memory cache.", e
                )
        else:
            logger.info("CACHE_URL not set; response cache using in-memory backend.")

        self._backend = InMemoryCacheBackend()
        self._backend_kind = "memory"
        return self._backend

    @property
    def backend_kind(self) -> str:
        """'redis' or 'memory'; exposed for health diagnostics."""
        if self._backend is None:
            self._resolve_backend()
        return self._backend_kind

    @staticmethod
    def make_key(namespace: str, payload: dict[str, Any]) -> str:
        """Stable cache key: namespace + sha256(sorted canonical JSON)."""
        raw = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{settings.cache_namespace}:{namespace}:{digest}"

    async def get_json(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        if not settings.cache_enabled:
            return None
        backend = self._resolve_backend()
        key = self.make_key(namespace, payload)
        try:
            raw = await backend.get(key)
        except Exception as e:
            logger.debug("Cache get failed for %s: %s", key[:60], e)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
        ttl: int | None = None,
    ) -> None:
        if not settings.cache_enabled:
            return
        backend = self._resolve_backend()
        key = self.make_key(namespace, payload)
        effective_ttl = ttl if ttl is not None else settings.cache_default_ttl
        try:
            await backend.set(
                key, json.dumps(value, default=str), max(1, effective_ttl)
            )
        except Exception as e:
            logger.debug("Cache set failed for %s: %s", key[:60], e)

    async def invalidate_namespace(self, namespace: str) -> int:
        """Drop all cached entries under a namespace (used after re-ingestion)."""
        backend = self._resolve_backend()
        prefix = f"{settings.cache_namespace}:{namespace}:"
        try:
            return await backend.delete_prefix(prefix)
        except Exception as e:
            logger.debug("Cache invalidation failed for %s: %s", namespace, e)
            return 0

    async def aclose(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()
            self._backend = None
            self._backend_kind = "uninitialized"


response_cache = ResponseCache()


def make_query_cache_payload(req: Any) -> dict[str, Any]:
    """Build the cache-key payload for a query/retrieve request.

    Includes every field that can change the answer: query text, tenant,
    permission scope, document filters, model selection, reranker, top_k.
    User IDs are deliberately excluded from the key material to avoid caching
    across personalization boundaries incorrectly — callers pass them via
    embedding/reranker resolution instead.
    """
    return {
        "query": req.query,
        "tenant_id": getattr(req, "tenant_id", "default"),
        "permission_scope": sorted(getattr(req, "permission_scope", ["default"]) or []),
        "document_ids": sorted(getattr(req, "document_ids", None) or []),
        "top_k": getattr(req, "top_k", None),
        "model": getattr(req, "model", None),
        "embedding_model": getattr(req, "embedding_model", None),
        "embedding_dim": getattr(req, "embedding_dim", None),
        "reranker": getattr(req, "reranker", None),
    }
