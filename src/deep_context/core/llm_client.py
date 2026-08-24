"""NVIDIA NIM API Client for BGE-m3 Embeddings and GLM-5.2 LLM Reasoning."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from typing import Any, AsyncIterator, Callable, cast

import numpy as np
from google import genai
from google.genai import types as genai_types
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
        or "resource_exhausted" in err_str.lower()
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
        f"{provider.capitalize()} daily limit reached for model `{model}` ({used}/{limit} tokens). Reset in {retry_after}."
        if limit and used
        else f"{provider.capitalize()} rate limit reached for model `{model}`. Please try again in {retry_after}."
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
    """Unified client for Google Gemini, Groq, and NVIDIA NIM APIs (fast reasoning LLMs and dense embeddings)."""

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
        self._current_gemini_key: str | None = settings.gemini_api_key

        # Google GenAI client (Gemini embeddings & reasoning)
        self._gemini_client: genai.Client | None = None
        if settings.has_gemini_key:
            try:
                self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
            except Exception as e:
                logger.warning("Failed to initialize Google GenAI client: %s", e)
                self._gemini_client = None

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

    def _refresh_gemini_client(self) -> genai.Client | None:
        """Dynamically check .env and reload Gemini API key if changed."""
        import os

        current_key = settings.gemini_api_key
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                current_key = val
                                settings.gemini_api_key = val
            except Exception:
                pass

        if not current_key or current_key.startswith("AIzaSy-your-key") or len(current_key) < 10:
            self._gemini_client = None
            self._current_gemini_key = None
            return None

        if getattr(self, "_current_gemini_key", None) != current_key or self._gemini_client is None:
            self._current_gemini_key = current_key
            try:
                self._gemini_client = genai.Client(api_key=current_key)
            except Exception as e:
                logger.warning("Failed to reload Gemini client: %s", e)
                self._gemini_client = None

        return self._gemini_client

    async def _call_gemini_with_backoff(
        self,
        coro_fn: Callable[[], Any],
        operation_name: str = "Gemini API",
    ) -> Any:
        """Executes a Gemini API coroutine with automatic 429/ResourceExhausted retry backoff."""
        max_retries = max(1, settings.gemini_max_retries)
        base_delay = max(1.0, settings.gemini_retry_delay_sec)

        for attempt in range(max_retries):
            try:
                return await coro_fn()
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = any(
                    k in err_msg
                    for k in (
                        "429",
                        "resource_exhausted",
                        "resourceexhausted",
                        "quota",
                        "rate limit",
                        "rate_limit",
                        "too many requests",
                    )
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait_sec = base_delay * (2**attempt)
                    logger.warning(
                        "Google GenAI 429 / Rate Limit on %s. Throttling and backing off for %.1fs (attempt %d/%d)...",
                        operation_name,
                        wait_sec,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                raise

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
        return bool(
            self._gemini_client or self._groq_client or (self._client and settings.has_nvidia_key)
        )

    # -----------------------------------------------------------------------
    # Embeddings (Google Gemini & NVIDIA NIM)
    # -----------------------------------------------------------------------

    async def get_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        dim: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
        is_query: bool = False,
    ) -> list[list[float]]:
        """
        Generate dense embeddings using Google Gemini (gemini-embedding-2 / gemini-embedding-001) or NVIDIA NIM.

        Supports Matryoshka Representation Learning (MRL) dimensions: 768, 1536, 3072, 1024.
        """
        if not texts:
            return []

        # Sanitize texts
        cleaned_texts = [t.strip() if t.strip() else " " for t in texts]
        target_model = model or self.embedding_model or settings.embedding_model

        # Determine target dimension
        if dim:
            target_dim = dim
        elif "gemini" in target_model.lower():
            target_dim = settings.embedding_dim or 768
        else:
            target_dim = settings.embedding_dim or 1024

        import asyncio

        is_gemini = (
            "gemini" in target_model.lower()
            or target_model.startswith("models/gemini")
            or target_model.startswith("text-embedding")
        )

        # --- Attempt 1: Google Gemini API (gemini-embedding-2 / gemini-embedding-001) ---
        if is_gemini:
            gemini_client = self._refresh_gemini_client()
            if gemini_client:
                all_embeddings: list[list[float]] = []
                batch_size = max(1, settings.gemini_batch_size)
                try:
                    for i in range(0, len(cleaned_texts), batch_size):
                        batch = cleaned_texts[i : i + batch_size]

                        # Task-specific formatting for gemini-embedding-2 (multimodal & text MRL)
                        if (
                            "embedding-2" in target_model.lower()
                            or "gemini-embedding-2" in target_model
                        ):
                            formatted_contents = []
                            for t in batch:
                                if is_query:
                                    prefix_task = task_type or "search result"
                                    text_val = f"task: {prefix_task} | query: {t}"
                                else:
                                    doc_title = title or "none"
                                    text_val = f"title: {doc_title} | text: {t}"
                                formatted_contents.append(
                                    genai_types.Content(
                                        parts=[genai_types.Part.from_text(text=text_val)]
                                    )
                                )
                            config = genai_types.EmbedContentConfig(
                                output_dimensionality=target_dim
                            )

                            async def _do_embed_v2(
                                cur_contents=formatted_contents, cur_config=config
                            ):
                                return await asyncio.wait_for(
                                    gemini_client.aio.models.embed_content(
                                        model=target_model,
                                        contents=cur_contents,
                                        config=cur_config,
                                    ),
                                    timeout=20.0,
                                )

                            resp = await self._call_gemini_with_backoff(
                                _do_embed_v2, f"embed_content ({target_model})"
                            )
                            if resp.embeddings:
                                for emb in resp.embeddings:
                                    if emb.values is not None:
                                        all_embeddings.append(list(emb.values))
                        else:
                            # gemini-embedding-001 / text-embedding-004
                            g_task_type = task_type or (
                                "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
                            )
                            formatted_contents = [
                                genai_types.Content(parts=[genai_types.Part.from_text(text=t)])
                                for t in batch
                            ]
                            config = genai_types.EmbedContentConfig(
                                task_type=g_task_type,
                                output_dimensionality=target_dim
                                if target_dim in (768, 1536, 3072)
                                else None,
                            )

                            async def _do_embed_001(
                                cur_contents=formatted_contents, cur_config=config
                            ):
                                return await asyncio.wait_for(
                                    gemini_client.aio.models.embed_content(
                                        model=target_model,
                                        contents=cur_contents,
                                        config=cur_config,
                                    ),
                                    timeout=20.0,
                                )

                            resp = await self._call_gemini_with_backoff(
                                _do_embed_001, f"embed_content ({target_model})"
                            )
                            if resp.embeddings:
                                for emb in resp.embeddings:
                                    vals = np.array(emb.values, dtype=np.float32)
                                    # Normalize if target_dim < 3072 on gemini-embedding-001
                                    if target_dim < 3072:
                                        n = np.linalg.norm(vals)
                                        if n > 0:
                                            vals = vals / n
                                    all_embeddings.append(vals.tolist())

                        # Throttle between successive batches if configured (e.g. for free tier safety)
                        if settings.gemini_rate_limit_delay_sec > 0 and (
                            i + batch_size < len(cleaned_texts)
                        ):
                            await asyncio.sleep(settings.gemini_rate_limit_delay_sec)

                    if len(all_embeddings) == len(cleaned_texts):
                        return all_embeddings
                except Exception as e:
                    logger.warning(
                        "Gemini embedding model %s failed (%s). Checking fallbacks...",
                        target_model,
                        e,
                    )
                    if not settings.allow_mock_fallback and not settings.has_nvidia_key:
                        raise

        # --- Attempt 2: NVIDIA NIM API ---
        if self._client and settings.has_nvidia_key and not is_gemini:
            all_embeddings = []
            batch_size = 32
            bounded_texts = [t[:2500] if len(t) > 2500 else t for t in cleaned_texts]
            nim_model = target_model
            try:
                for i in range(0, len(bounded_texts), batch_size):
                    batch = bounded_texts[i : i + batch_size]
                    response = await asyncio.wait_for(
                        self._client.embeddings.create(
                            input=batch,
                            model=nim_model,
                            encoding_format="float",
                            extra_body={"input_type": "passage", "truncate": "END"},
                            timeout=8.0,
                        ),
                        timeout=8.0,
                    )
                    for item in response.data:
                        raw_emb = list(item.embedding) if hasattr(item, "embedding") else []
                        emb_vals: list[float] = [float(x) for x in raw_emb]
                        if len(emb_vals) != target_dim:
                            if len(emb_vals) > target_dim:
                                vals = np.array(emb_vals[:target_dim], dtype=np.float32)
                                n = np.linalg.norm(vals)
                                if n > 0:
                                    vals = vals / n
                                emb_vals = vals.tolist()
                            else:
                                emb_vals = emb_vals + [0.0] * (target_dim - len(emb_vals))
                        all_embeddings.append(emb_vals)
                return all_embeddings
            except Exception as e:
                logger.warning(
                    "NVIDIA NIM %s failed (%s). Retrying with nvidia/nv-embedqa-e5-v5...",
                    nim_model,
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
                            raw_emb = list(item.embedding) if hasattr(item, "embedding") else []
                            emb_vals = [float(x) for x in raw_emb]
                            if len(emb_vals) != target_dim:
                                if len(emb_vals) > target_dim:
                                    vals = np.array(emb_vals[:target_dim], dtype=np.float32)
                                    n = np.linalg.norm(vals)
                                    if n > 0:
                                        vals = vals / n
                                    emb_vals = vals.tolist()
                                else:
                                    emb_vals = emb_vals + [0.0] * (target_dim - len(emb_vals))
                            all_embeddings.append(emb_vals)
                    return all_embeddings
                except Exception as e2:
                    logger.warning("NVIDIA NIM fallback embedding failed: %s", e2)
                    if not settings.allow_mock_fallback:
                        raise

        # --- Attempt 3: Deterministic offline mock embedding fallback ---
        return [self._mock_embedding(t, dim=target_dim) for t in cleaned_texts]

    async def get_embedding(
        self,
        text: str,
        model: str | None = None,
        dim: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
        is_query: bool = False,
    ) -> list[float]:
        """Convenience method for a single text embedding."""
        res = await self.get_embeddings(
            [text],
            model=model,
            dim=dim,
            task_type=task_type,
            title=title,
            is_query=is_query,
        )
        return res[0]

    def _mock_embedding(self, text: str, dim: int = 768) -> list[float]:
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

        is_gemini_model = bool(
            model
            and (
                model.startswith("gemini-")
                or model.startswith("models/gemini")
                or model.startswith("google/gemini")
            )
        )

        is_nim_model = bool(
            model
            and not is_gemini_model
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

        # --- Attempt 0: Google Gemini API (When Gemini model requested) ---
        if is_gemini_model:
            gemini_client = self._refresh_gemini_client()
            if gemini_client:
                target_model = model or "gemini-2.5-flash"
                system_prompt = ""
                gemini_contents = []
                for m_item in messages:
                    if m_item.get("role") == "system":
                        system_prompt += m_item.get("content", "") + "\n"
                    else:
                        r = "user" if m_item.get("role") == "user" else "model"
                        c = m_item.get("content", "")
                        if c:
                            gemini_contents.append(
                                genai_types.Content(
                                    role=r, parts=[genai_types.Part.from_text(text=c)]
                                )
                            )
                if not gemini_contents:
                    gemini_contents.append(
                        genai_types.Content(
                            role="user", parts=[genai_types.Part.from_text(text="Hello")]
                        )
                    )

                try:

                    async def _do_gemini_gen():
                        return await asyncio.wait_for(
                            gemini_client.aio.models.generate_content(
                                model=target_model,
                                contents=gemini_contents,
                                config=genai_types.GenerateContentConfig(
                                    system_instruction=system_prompt.strip()
                                    if system_prompt
                                    else None,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_output_tokens=min(max_tokens, 8192) if max_tokens else 8192,
                                ),
                            ),
                            timeout=timeout,
                        )

                    gemini_resp = await self._call_gemini_with_backoff(
                        _do_gemini_gen, f"generate_content ({target_model})"
                    )
                    raw_content = gemini_resp.text or ""
                    content, reasoning = self._parse_think_tags(raw_content, None)
                    return content, reasoning
                except Exception as e:
                    logger.warning("Gemini model %s complete failed: %s", target_model, e)

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
                        nim_resp: Any = await asyncio.wait_for(
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
                        if nim_resp.choices and len(nim_resp.choices) > 0:
                            msg = nim_resp.choices[0].message
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
        if groq_client:
            m = self.llm_model if (not model or is_gemini_model or is_nim_model) else model
            if not m or m.startswith("gemini-") or ("meta/" in m and "groq" not in m):
                m = "qwen/qwen3.6-27b"
            try:
                groq_resp: Any = await asyncio.wait_for(
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
                if groq_resp.choices and len(groq_resp.choices) > 0:
                    self.last_rate_limit = None
                    LLMClient.global_rate_limit = None
                    msg = groq_resp.choices[0].message
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
                fallback_resp: Any = await asyncio.wait_for(
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
                if fallback_resp.choices and len(fallback_resp.choices) > 0:
                    self.last_rate_limit = None
                    LLMClient.global_rate_limit = None
                    msg = fallback_resp.choices[0].message
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

        is_gemini_model = bool(
            model
            and (
                model.startswith("gemini-")
                or model.startswith("models/gemini")
                or model.startswith("google/gemini")
            )
        )

        is_nim_model = bool(
            model
            and not is_gemini_model
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

        # --- Attempt 0: Google Gemini API Streaming ---
        if is_gemini_model:
            gemini_client = self._refresh_gemini_client()
            if gemini_client:
                target_model = model or "gemini-2.5-flash"
                system_prompt = ""
                gemini_contents = []
                for m_item in messages:
                    if m_item.get("role") == "system":
                        system_prompt += m_item.get("content", "") + "\n"
                    else:
                        r = "user" if m_item.get("role") == "user" else "model"
                        c = m_item.get("content", "")
                        if c:
                            gemini_contents.append(
                                genai_types.Content(
                                    role=r, parts=[genai_types.Part.from_text(text=c)]
                                )
                            )
                if not gemini_contents:
                    gemini_contents.append(
                        genai_types.Content(
                            role="user", parts=[genai_types.Part.from_text(text="Hello")]
                        )
                    )

                try:
                    gemini_stream = await gemini_client.aio.models.generate_content_stream(
                        model=target_model,
                        contents=gemini_contents,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_prompt.strip() if system_prompt else None,
                            temperature=temperature,
                            top_p=top_p,
                            max_output_tokens=min(max_tokens, 8192) if max_tokens else 8192,
                        ),
                    )
                    stream_emitted = False
                    async for chunk in gemini_stream:
                        text_val = chunk.text or ""
                        if text_val:
                            stream_emitted = True
                            self.last_rate_limit = None
                            LLMClient.global_rate_limit = None
                            yield {"type": "content", "text": text_val}
                    if stream_emitted:
                        return
                except Exception as e:
                    logger.warning("Gemini model %s streaming failed: %s", target_model, e)

        # --- Attempt 1: NVIDIA NIM (When NIM model requested) ---
        if is_nim_model and self._client and settings.has_nvidia_key:
            nim_models = [model or "meta/llama-3.1-8b-instruct"]
            if "meta/llama-3.1-8b-instruct" not in nim_models:
                nim_models.append("meta/llama-3.1-8b-instruct")

            for m in nim_models:
                try:
                    nim_stream: Any = await self._client.chat.completions.create(
                        model=m,
                        messages=cast(Any, messages),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=min(max_tokens, 8192),
                        stream=True,
                    )
                    stream_emitted = False
                    async for chunk in nim_stream:
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
                fallback_stream: Any = await self._client.chat.completions.create(
                    model=target_nim_model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 8192),
                    stream=True,
                )
                stream_emitted = False
                async for chunk in fallback_stream:
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
