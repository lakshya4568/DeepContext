"""The open 9B model as the answer model: an OpenAI-compatible server behind the same cache and retries as Gemini."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from eval.pg_cache import MemoryKV
from eval.strict_clients import (
    MockFallbackError,
    VertexRuntime,
    get_runtime,
    install_strict_patches,
)

from deep_context.core.llm_client import llm_client


def _reply(text: str = "Paris.") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )


def _runtime(handler: Any) -> VertexRuntime:
    rt = VertexRuntime(cache=MemoryKV())
    rt._transport = httpx.MockTransport(handler)
    return rt


async def test_a_chat_call_is_an_openai_request_and_returns_text_and_token_counts() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen.append(json.loads(request.content))
        return _reply()

    res = await _runtime(handler).generate_selfhosted(
        base_url="http://vllm", model="qwen3.5-9b",
        messages=[{"role": "user", "content": "Capital of France?"}],
        temperature=0.1, top_p=0.95, max_tokens=64, enable_thinking=False,
    )  # fmt: skip
    assert (res.text, res.in_tokens, res.out_tokens, res.finish_reason, res.cached) == (
        "Paris.", 12, 3, "stop", False,
    )  # fmt: skip
    body = seen[0]
    assert body["model"] == "qwen3.5-9b" and body["max_tokens"] == 64 and body["temperature"] == 0.1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "Capital of France?"}]


async def test_an_identical_second_call_comes_from_the_cache_without_a_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _reply()

    rt = _runtime(handler)
    kw: dict[str, Any] = dict(
        base_url="http://vllm", model="qwen3.5-9b", messages=[{"role": "user", "content": "q"}],
        temperature=0.1, top_p=0.95, max_tokens=64, enable_thinking=False,
    )  # fmt: skip
    first = await rt.generate_selfhosted(**kw)
    second = await rt.generate_selfhosted(**kw)
    assert calls == 1 and not first.cached and second.cached and second.text == "Paris."


async def test_the_patched_client_sends_qwen_to_the_server_only_when_its_address_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = get_runtime()
    monkeypatch.setattr(rt, "cache", MemoryKV())
    monkeypatch.setattr(rt, "_transport", httpx.MockTransport(lambda request: _reply("hello")))
    install_strict_patches()
    messages = [{"role": "user", "content": "hi"}]

    monkeypatch.delenv("EVAL_SELFHOSTED_LLM_URL", raising=False)
    with pytest.raises(MockFallbackError):
        await llm_client.complete(messages, model="qwen3.5-9b")

    monkeypatch.setenv("EVAL_SELFHOSTED_LLM_URL", "http://vllm")
    text, _reasoning = await llm_client.complete(
        messages, model="qwen3.5-9b", temperature=0.1, max_tokens=16, enable_thinking=False
    )
    assert text == "hello"
    with pytest.raises(MockFallbackError):  # any other non-Gemini model stays forbidden
        await llm_client.complete(messages, model="meta/llama-3.1-8b-instruct")
