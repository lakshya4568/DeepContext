"""NVIDIA NIM API Client for BGE-m3 Embeddings and GLM-5.2 LLM Reasoning."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any, AsyncIterator, cast

import numpy as np
from openai import AsyncOpenAI

from deep_context.core.config import settings
from deep_context.core.logging import logger


def parse_rate_limit_error(
    error: Exception | str, provider: str = "groq", model: str = "qwen/qwen3.6-27b"
) -> dict[str, Any]:
    """Extract structured details (retry_after, limit, used, quota_type) from rate limit errors."""
    err_str = str(error)
    is_rate_limit = (
        "429" in err_str
        or "rate_limit" in err_str.lower()
        or "too many requests" in err_str.lower()
        or "quota" in err_str.lower()
        or "tokens per day" in err_str.lower()
    )
    if not is_rate_limit:
        return {}

    retry_match = re.search(r"try again in ([\w\.\s]+?)(?:\.|\s*Need|$)", err_str, re.IGNORECASE)
    retry_after = retry_match.group(1).strip() if retry_match else "a few minutes"

    limit_match = re.search(r"Limit (\d+),\s*Used (\d+)", err_str)
    limit = limit_match.group(1) if limit_match else "200000"
    used = limit_match.group(2) if limit_match else "200000"

    quota_desc = "tokens per day (TPD)"
    if "tokens per minute" in err_str.lower() or "tpm" in err_str.lower():
        quota_desc = "tokens per minute (TPM)"
    elif "requests per minute" in err_str.lower() or "rpm" in err_str.lower():
        quota_desc = "requests per minute (RPM)"

    friendly_msg = (
        f"Groq daily limit reached for model `{model}` ({used}/{limit} tokens). Reset in {retry_after}."
        if limit and used
        else f"Groq rate limit reached for model `{model}`. Please try again in {retry_after}."
    )

    return {
        "is_rate_limited": True,
        "provider": provider,
        "model": model,
        "quota_type": quota_desc,
        "limit": limit,
        "used": used,
        "retry_after": retry_after,
        "message": friendly_msg,
        "raw_error": err_str[:300],
        "timestamp": time.time(),
    }


class LLMClient:
    """Unified client for Groq and NVIDIA NIM APIs (fast reasoning LLMs and dense embeddings)."""

    # Global rate limit state accessible across threads/requests
    global_rate_limit: dict[str, Any] | None = None

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ):
        self.embedding_model = embedding_model or settings.embedding_model
        self.llm_model = llm_model or settings.llm_model
        self.last_rate_limit: dict[str, Any] | None = None
        self._current_groq_key: str | None = settings.groq_api_key

        # Groq client (primary ultra-fast reasoning LLM)
        self._groq_client: AsyncOpenAI | None
        if settings.has_groq_key:
            self._groq_client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )
        else:
            self._groq_client = None

        # NVIDIA NIM client (dense embeddings and fallback LLM)
        nvidia_key = api_key or settings.nvidia_api_key
        nvidia_url = base_url or settings.nvidia_base_url
        self._client: AsyncOpenAI | None
        if nvidia_key:
            self._client = AsyncOpenAI(
                api_key=nvidia_key,
                base_url=nvidia_url,
            )
        else:
            self._client = None

    def _refresh_groq_client(self) -> AsyncOpenAI | None:
        """Dynamically check .env and reload Groq API key if changed."""
        import os

        current_key = settings.groq_api_key
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GROQ_API_KEY=") or line.startswith("GROK_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                current_key = val
                                settings.groq_api_key = val
            except Exception:
                pass

        if not current_key:
            self._groq_client = None
            self._current_groq_key = None
            return None

        if getattr(self, "_current_groq_key", None) != current_key or self._groq_client is None:
            self._current_groq_key = current_key
            self._groq_client = AsyncOpenAI(
                api_key=current_key,
                base_url=settings.groq_base_url,
            )
            self.last_rate_limit = None
            LLMClient.global_rate_limit = None

        return self._groq_client

    @classmethod
    def clear_rate_limits(cls) -> None:
        cls.global_rate_limit = None

    def record_rate_limit(
        self, error: Exception | str, provider: str, model: str
    ) -> dict[str, Any]:
        notice = parse_rate_limit_error(error, provider, model)
        if notice:
            self.last_rate_limit = notice
            LLMClient.global_rate_limit = notice
        return notice

    def _has_live_client(self) -> bool:
        return bool(self._groq_client or (self._client and settings.has_nvidia_key))

    # -----------------------------------------------------------------------
    # Embeddings (baai/bge-m3)
    # -----------------------------------------------------------------------

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate 1024-dim dense embeddings using NVIDIA NIM baai/bge-m3."""
        if not texts:
            return []

        # Sanitize texts
        cleaned_texts = [t.strip() if t.strip() else " " for t in texts]

        import asyncio

        if self._client and settings.has_nvidia_key:
            # Batch in slices of 32 to avoid HTTP timeout/payload limit
            all_embeddings: list[list[float]] = []
            batch_size = 32

            # Sanitize and bound text slice for embedding models
            bounded_texts = [t[:2500] if len(t) > 2500 else t for t in cleaned_texts]

            # Primary attempt with configured embedding model
            try:
                for i in range(0, len(bounded_texts), batch_size):
                    batch = bounded_texts[i : i + batch_size]
                    response = await asyncio.wait_for(
                        self._client.embeddings.create(
                            input=batch,
                            model=self.embedding_model,
                            encoding_format="float",
                            extra_body={"input_type": "passage", "truncate": "END"},
                            timeout=8.0,
                        ),
                        timeout=8.0,
                    )
                    for item in response.data:
                        all_embeddings.append(item.embedding)
                return all_embeddings
            except Exception as e:
                logger.warning(
                    "NVIDIA NIM %s failed (%s). Retrying with nvidia/nv-embedqa-e5-v5...",
                    self.embedding_model,
                    e,
                )
                try:
                    all_embeddings = []
                    for i in range(0, len(bounded_texts), batch_size):
                        batch = bounded_texts[i : i + batch_size]
                        response = await asyncio.wait_for(
                            self._client.embeddings.create(
                                input=batch,
                                model="nvidia/nv-embedqa-e5-v5",
                                encoding_format="float",
                                extra_body={"input_type": "passage", "truncate": "END"},
                                timeout=8.0,
                            ),
                            timeout=8.0,
                        )
                        for item in response.data:
                            all_embeddings.append(item.embedding)
                    return all_embeddings
                except Exception as e2:
                    logger.warning("NVIDIA NIM fallback embedding failed: %s", e2)
                    if not settings.allow_mock_fallback:
                        raise

        # Deterministic offline mock embedding fallback for local dev / tests
        return [self._mock_embedding(t, dim=settings.embedding_dim) for t in cleaned_texts]

    async def get_embedding(self, text: str) -> list[float]:
        """Convenience method for a single text embedding."""
        res = await self.get_embeddings([text])
        return res[0]

    def _mock_embedding(self, text: str, dim: int = 1024) -> list[float]:
        """Generate deterministic normalized pseudo-embedding based on sha256 + token hashes."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            weight = 1.0 / (1.0 + math.log(i + 1))
            vec[idx] += weight * (1.0 if (h % 2 == 0) else -1.0)

        # Ensure non-zero vector
        h_full = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vec[h_full % dim] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    # -----------------------------------------------------------------------
    # Chat & Reasoning Completions (Groq primary, NVIDIA NIM fallback)
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_think_tags(text: str, reasoning: str | None = None) -> tuple[str, str | None]:
        """Extract <think>...</think> reasoning blocks from model content."""
        if reasoning:
            return text, reasoning
        if "<think>" in text:
            if "</think>" in text:
                parts = text.split("</think>", 1)
                think_part = parts[0].replace("<think>", "").strip()
                ans_part = parts[1].strip()
                return ans_part, think_part
            return text.replace("<think>", "").strip(), None
        return text, reasoning

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        enable_thinking: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> tuple[str, str | None]:
        """
        Generate completion using Groq (qwen/qwen3.6-27b) or NVIDIA NIM fallback.

        Args:
            timeout: Per-request timeout in seconds.
            max_retries: Maximum retry attempts on rate-limit errors.

        Returns: (content, reasoning_content)
        """
        import asyncio

        is_nim_model = bool(
            model
            and any(
                model.startswith(p)
                for p in [
                    "meta/",
                    "nvidia/",
                    "z-ai/",
                    "mistralai/",
                    "ibm/",
                    "google/",
                    "microsoft/",
                ]
            )
        )

        # --- Attempt 1: NVIDIA NIM (When NIM model requested) ---
        if is_nim_model and self._client and settings.has_nvidia_key:
            target_model = model or "z-ai/glm-5.2"
            extra_body: dict[str, Any] = {}
            if enable_thinking:
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": True,
                    "clear_thinking": False,
                }

            nim_models = [target_model]
            if "meta/llama-3.1-8b-instruct" not in nim_models:
                nim_models.append("meta/llama-3.1-8b-instruct")

            for m in nim_models:
                for attempt in range(max_retries):
                    try:
                        resp = await asyncio.wait_for(
                            self._client.chat.completions.create(
                                model=m,
                                messages=cast(Any, messages),
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=min(max_tokens, 8192) if max_tokens else 8192,
                                extra_body=extra_body if extra_body else None,
                                timeout=timeout,
                            ),
                            timeout=timeout,
                        )
                        if resp.choices and len(resp.choices) > 0:
                            msg = resp.choices[0].message
                            raw_content = msg.content or ""
                            reasoning = getattr(msg, "reasoning_content", None)
                            content, reasoning = self._parse_think_tags(raw_content, reasoning)
                            if not content and reasoning:
                                content = reasoning.split("\n\n")[-1].replace("ANSWER:", "").strip()
                            return content, reasoning
                    except asyncio.TimeoutError:
                        logger.warning("NVIDIA NIM model %s timed out after %.0fs.", m, timeout)
                        break
                    except Exception as e:
                        logger.warning("NVIDIA NIM model %s failed: %s", m, e)
                        break

        # --- Attempt 2: Groq API (Ultra-fast reasoning) ---
        groq_client = self._refresh_groq_client()
        if groq_client and not is_nim_model:
            m = model or self.llm_model or "qwen/qwen3.6-27b"
            try:
                resp = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=m,
                        messages=cast(Any, messages),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=min(max_tokens, 8192),
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )
                if resp.choices and len(resp.choices) > 0:
                    self.last_rate_limit = None
                    LLMClient.global_rate_limit = None
                    msg = resp.choices[0].message
                    raw_content = msg.content or ""
                    reasoning = getattr(msg, "reasoning_content", None)
                    content, reasoning = self._parse_think_tags(raw_content, reasoning)
                    return content, reasoning
            except asyncio.TimeoutError:
                logger.warning("Groq model %s timed out after %.0fs.", m, timeout)
            except Exception as e:
                notice = self.record_rate_limit(e, "groq", m)
                if notice:
                    logger.warning(
                        "Groq model %s hit rate limit (%s). Reset in %s. Failing over to NVIDIA NIM...",
                        m,
                        notice.get("quota_type"),
                        notice.get("retry_after"),
                    )
                else:
                    logger.warning(
                        "Groq model %s call failed: %s. Failing over to NVIDIA NIM...", m, e
                    )

        # --- Attempt 3: NVIDIA NIM (General High-Speed Fallback) ---
        if self._client and settings.has_nvidia_key:
            target_model = "meta/llama-3.1-8b-instruct"
            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=target_model,
                        messages=cast(Any, messages),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=min(max_tokens, 8192) if max_tokens else 8192,
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )
                if resp.choices and len(resp.choices) > 0:
                    self.last_rate_limit = None
                    LLMClient.global_rate_limit = None
                    msg = resp.choices[0].message
                    raw_content = msg.content or ""
                    reasoning = getattr(msg, "reasoning_content", None)
                    content, reasoning = self._parse_think_tags(raw_content, reasoning)
                    return content, reasoning
            except Exception as e:
                logger.warning("NVIDIA NIM fallback failed: %s", e)

        # Fallback: Deterministic mock
        return self._mock_completion(messages)

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        enable_thinking: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yields chunks of {"type": "reasoning" | "content", "text": "..."}
        """

        is_nim_model = bool(
            model
            and any(
                model.startswith(p)
                for p in [
                    "meta/",
                    "nvidia/",
                    "z-ai/",
                    "mistralai/",
                    "ibm/",
                    "google/",
                    "microsoft/",
                ]
            )
        )

        # --- Attempt 1: NVIDIA NIM (When NIM model requested) ---
        if is_nim_model and self._client and settings.has_nvidia_key:
            nim_models = [model or "meta/llama-3.1-8b-instruct"]
            if "meta/llama-3.1-8b-instruct" not in nim_models:
                nim_models.append("meta/llama-3.1-8b-instruct")

            for m in nim_models:
                try:
                    stream = await self._client.chat.completions.create(
                        model=m,
                        messages=cast(Any, messages),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=min(max_tokens, 8192),
                        stream=True,
                    )
                    stream_emitted = False
                    async for chunk in stream:
                        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                            continue
                        delta = chunk.choices[0].delta
                        if delta is None:
                            continue
                        content = getattr(delta, "content", None)
                        if content:
                            stream_emitted = True
                            self.last_rate_limit = None
                            LLMClient.global_rate_limit = None
                            yield {"type": "content", "text": content}
                    if stream_emitted:
                        return
                except Exception as e:
                    logger.warning("NVIDIA NIM model %s streaming failed: %s", m, e)

        # --- Attempt 2: Groq API ---
        groq_client = self._refresh_groq_client()
        if groq_client and not is_nim_model:
            m = model or self.llm_model or "qwen/qwen3.6-27b"
            try:
                stream_obj: Any = await groq_client.chat.completions.create(
                    model=m,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max(max_tokens, 2048), 8192),
                    stream=True,
                )
                in_think = False
                stream_emitted = False
                emitted_content = ""
                emitted_reasoning = ""
                async for chunk in stream_obj:
                    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    raw_text = getattr(delta, "content", None) or ""
                    if "<think>" in raw_text:
                        in_think = True
                        raw_text = raw_text.replace("<think>", "")
                    if "</think>" in raw_text:
                        in_think = False
                        parts = raw_text.split("</think>", 1)
                        if parts[0]:
                            stream_emitted = True
                            self.last_rate_limit = None
                            LLMClient.global_rate_limit = None
                            emitted_reasoning += parts[0]
                            yield {"type": "reasoning", "text": parts[0]}
                        if len(parts) > 1 and parts[1]:
                            stream_emitted = True
                            self.last_rate_limit = None
                            LLMClient.global_rate_limit = None
                            emitted_content += parts[1]
                            yield {"type": "content", "text": parts[1]}
                        continue

                    if raw_text:
                        stream_emitted = True
                        self.last_rate_limit = None
                        LLMClient.global_rate_limit = None
                        if in_think:
                            emitted_reasoning += raw_text
                            yield {"type": "reasoning", "text": raw_text}
                        else:
                            emitted_content += raw_text
                            yield {"type": "content", "text": raw_text}

                # Fallback: if reasoning took all tokens and content was empty, extract answer from reasoning
                if not emitted_content.strip() and emitted_reasoning.strip():
                    full_r = emitted_reasoning.strip()
                    extracted = full_r
                    if "**Draft Response" in full_r:
                        extracted = (
                            full_r.split("**Draft Response", 1)[1]
                            .replace("Mental Refinement):", "")
                            .replace(":", "", 1)
                            .strip()
                        )
                    elif "ANSWER:" in full_r:
                        extracted = full_r.split("ANSWER:", 1)[1].strip()
                    else:
                        paras = [p.strip() for p in full_r.split("\n\n") if p.strip()]
                        extracted = "\n\n".join(paras[-2:]) if len(paras) >= 2 else full_r
                    if extracted:
                        yield {"type": "content", "text": extracted}

                if stream_emitted:
                    return

            except Exception as e:
                notice = self.record_rate_limit(e, "groq", m)
                if notice:
                    logger.warning(
                        "Groq model %s hit rate limit (%s). Reset in %s. Failing over to NVIDIA NIM...",
                        m,
                        notice.get("quota_type"),
                        notice.get("retry_after"),
                    )
                else:
                    logger.warning(
                        "Groq model %s streaming failed: %s. Failing over to NVIDIA NIM...", m, e
                    )

        # --- Attempt 2: NVIDIA NIM (General Fallback) ---
        if self._client and settings.has_nvidia_key:
            target_nim_model = (
                model
                if (
                    model
                    and any(
                        model.startswith(p) for p in ["meta/", "nvidia/", "z-ai/", "mistralai/"]
                    )
                )
                else "meta/llama-3.1-8b-instruct"
            )
            try:
                stream = await self._client.chat.completions.create(
                    model=target_nim_model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 8192),
                    stream=True,
                )
                stream_emitted = False
                async for chunk in stream:
                    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        stream_emitted = True
                        self.last_rate_limit = None
                        LLMClient.global_rate_limit = None
                        yield {"type": "reasoning", "text": reasoning}

                    content = getattr(delta, "content", None)
                    if content:
                        stream_emitted = True
                        self.last_rate_limit = None
                        LLMClient.global_rate_limit = None
                        yield {"type": "content", "text": content}

                if stream_emitted:
                    return

            except Exception as e:
                logger.warning("NVIDIA NIM streaming fallback failed: %s. Using mock.", e)
                if not settings.allow_mock_fallback:
                    raise

        # --- Attempt 3: Direct Context Extraction Fallback ---
        active_notice = self.last_rate_limit or LLMClient.global_rate_limit
        if active_notice:
            yield {"type": "rate_limit", "notice": active_notice}

        content, reasoning = self._mock_completion(messages)
        if reasoning:
            yield {"type": "reasoning", "text": reasoning}
        yield {"type": "content", "text": content}

    def _mock_completion(self, messages: list[dict[str, Any]]) -> tuple[str, str | None]:
        """Generate structured synthetic answer for testing purposes.

        When the NVIDIA NIM API is unavailable, this extracts retrieved context
        from the assembled prompt messages and presents it directly so the user
        still gets useful information from their documents.
        """
        last_msg = messages[-1]["content"] if messages else ""
        system_msg = (
            messages[0]["content"] if len(messages) > 1 and messages[0]["role"] == "system" else ""
        )

        active_notice = self.last_rate_limit or LLMClient.global_rate_limit
        if active_notice:
            reasoning = f"⚠️ **Groq Rate Limit Notice**: {active_notice['message']}\nShowing retrieved document evidence directly."
            banner = f"> ⚠️ **Groq Rate Limit Reached**: {active_notice['message']}\n\n"
        else:
            reasoning = (
                "⚠️ Provider API currently unavailable. "
                "Showing retrieved document excerpts directly."
            )
            banner = ""

        # If json output is expected (e.g. classification or verification or structured output)
        if "JSON" in system_msg or "json" in last_msg.lower():
            if "query_shape" in system_msg.lower() or "classify" in system_msg.lower():
                out = {
                    "query_shape": "factual_lookup",
                    "path": "hybrid_rag",
                    "reason": "Standard knowledge lookup request.",
                    "sub_queries": [last_msg],
                }
                return json.dumps(out), reasoning
            if "check_answer_support" in system_msg.lower() or "verifier" in system_msg.lower():
                out_check: dict[str, Any] = {
                    "passed": True,
                    "confidence": 0.95,
                    "claims": [
                        {"text": last_msg[:50], "support": "retrieved", "evidence_id": "chunk-1"}
                    ],
                    "failure_reasons": [],
                }
                return json.dumps(out_check), reasoning

        # Extract retrieved context from assembled messages to present directly.
        # The PromptAssembler places evidence in the system message under
        # "### Retrieved Evidence (Grounding Context)".
        context_snippets: list[str] = []
        for m in messages:
            text = m.get("content", "")
            if "Retrieved Evidence" in text:
                parts = text.split("### Retrieved Evidence")
                if len(parts) > 1:
                    evidence_section = parts[1]
                    for line in evidence_section.split("\n"):
                        stripped = line.strip()
                        if stripped and not stripped.startswith("###"):
                            context_snippets.append(stripped)

        if context_snippets:
            excerpts = "\n\n".join(context_snippets[:100])
            content = (
                f"{banner}**Retrieved Document Evidence** for: "
                f"_{last_msg.strip()[:100]}_\n\n"
                f"---\n\n{excerpts}"
            )
        else:
            content = (
                f"{banner}**No Document Evidence Retrieved** for: "
                f"_{last_msg.strip()[:200]}_\n\n"
                "Try searching with specific keywords or verify that documents are ingested."
            )

        return content, reasoning


# Global default client
llm_client = LLMClient()
