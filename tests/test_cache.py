"""Tests for the response cache layer (deep_context.cache)."""

from __future__ import annotations

import pytest

from deep_context.cache import ResponseCache, make_query_cache_payload
from deep_context.core.config import settings


@pytest.fixture(autouse=True)
def memory_cache(monkeypatch: pytest.MonkeyPatch) -> ResponseCache:
    """Force the in-memory backend and reset it per test."""
    monkeypatch.setattr(settings, "cache_url", "")
    monkeypatch.setattr(settings, "cache_enabled", True)
    cache = ResponseCache()
    monkeypatch.setattr("deep_context.cache.response_cache", cache)
    return cache


class TestCacheKeys:
    async def test_make_key_is_stable_and_namespaced(self) -> None:
        payload = {"query": "hello", "top_k": 5}
        key1 = ResponseCache.make_key("rag:ask", payload)
        key2 = ResponseCache.make_key("rag:ask", {"top_k": 5, "query": "hello"})
        assert key1 == key2
        assert key1.startswith(f"{settings.cache_namespace}:rag:ask:")

    async def test_different_payloads_produce_different_keys(self) -> None:
        k1 = ResponseCache.make_key("ns", {"q": "a"})
        k2 = ResponseCache.make_key("ns", {"q": "b"})
        assert k1 != k2


class TestCacheRoundTrip:
    async def test_set_then_get(self) -> None:
        cache = ResponseCache()
        await cache.set_json("ns", {"q": "x"}, {"answer": "42"})
        assert await cache.get_json("ns", {"q": "x"}) == {"answer": "42"}

    async def test_miss_returns_none(self) -> None:
        cache = ResponseCache()
        assert await cache.get_json("ns", {"q": "missing"}) is None

    async def test_disabled_cache_never_stores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "cache_enabled", False)
        cache = ResponseCache()
        await cache.set_json("ns", {"q": "x"}, {"a": 1})
        assert await cache.get_json("ns", {"q": "x"}) is None

    async def test_ttl_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "cache_default_ttl", 1)
        cache = ResponseCache()
        await cache.set_json("ns", {"q": "x"}, {"a": 1}, ttl=1)
        # Force expiry by manipulating the monotonic deadline.
        backend = cache._resolve_backend()
        key = cache.make_key("ns", {"q": "x"})
        expires_at, value = backend._store[key]
        backend._store[key] = (expires_at - 10, value)
        assert await cache.get_json("ns", {"q": "x"}) is None

    async def test_invalidate_namespace(self) -> None:
        cache = ResponseCache()
        await cache.set_json("rag:ask", {"q": "x"}, {"a": 1})
        await cache.set_json("rag:retrieve", {"q": "y"}, {"b": 2})
        removed = await cache.invalidate_namespace("rag")
        assert removed >= 2
        assert await cache.get_json("rag:ask", {"q": "x"}) is None


class TestQueryPayload:
    async def test_payload_includes_relevant_fields(self) -> None:
        class Req:
            query = "what is RAG?"
            tenant_id = "t1"
            permission_scope = ["read", "default"]
            document_ids = ["d2", "d1"]
            top_k = 8
            model = None
            embedding_model = None
            embedding_dim = None
            reranker = None

        payload = make_query_cache_payload(Req())
        assert payload["query"] == "what is RAG?"
        # Sorted lists make keys stable regardless of input ordering.
        assert payload["document_ids"] == ["d1", "d2"]
        assert payload["permission_scope"] == ["default", "read"]
