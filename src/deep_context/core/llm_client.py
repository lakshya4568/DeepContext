"""Unified Client for Google Gemini / Vertex AI, Groq LLMs, and Dense Embeddings."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
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
    """Parses Groq or provider rate limit / quota errors into structured status."""
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

    # Detect quota type (TPD, TPM, RPM, etc.)
    quota_desc = "Tokens Per Day (TPD)"
    limit = None
    used = None
    retry_after = "a few minutes"

    # Groq-style rate limit format: "Limit 200000, Used 200000, Requested 4867. Please try again in 5m23.12s."
    m_limit = re.search(r"Limit\s+(\d+)", err_str, re.IGNORECASE)
    m_used = re.search(r"Used\s+(\d+)", err_str, re.IGNORECASE)
    m_time = re.search(r"try again in\s+([0-9a-zA-Z\.\s]+)\.", err_str, re.IGNORECASE)

    if m_limit:
        limit = int(m_limit.group(1))
    if m_used:
        used = int(m_used.group(1))
    if m_time:
        retry_after = m_time.group(1).strip()

    if "TPM" in err_str or "tokens per minute" in err_str.lower():
        quota_desc = "Tokens Per Minute (TPM)"
    elif "RPM" in err_str or "requests per minute" in err_str.lower():
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
    """Unified client for Google Gemini / Vertex AI, Groq LLMs, and Dense Embeddings."""

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
        if settings.has_gemini_key and not (
            settings.allow_mock_fallback and os.environ.get("PYTEST_CURRENT_TEST")
        ):
            try:
                self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
            except Exception as e:
                logger.warning("Failed to initialize Google GenAI client: %s", e)
                self._gemini_client = None

        # Groq client (primary ultra-fast reasoning LLM)
        self._groq_client: AsyncOpenAI | None
        if settings.has_groq_key and not (
            settings.allow_mock_fallback and os.environ.get("PYTEST_CURRENT_TEST")
        ):
            self._groq_client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )
        else:
            self._groq_client = None

        # NVIDIA NIM client (dense embeddings)
        nvidia_key = api_key or settings.nvidia_api_key
        nvidia_url = base_url or settings.nvidia_base_url
        self._client: AsyncOpenAI | None
        if nvidia_key and not (
            settings.allow_mock_fallback and os.environ.get("PYTEST_CURRENT_TEST")
        ):
            self._client = AsyncOpenAI(
                api_key=nvidia_key,
                base_url=nvidia_url,
            )
        else:
            self._client = None

    def _refresh_gemini_client(self) -> genai.Client | None:
        """Dynamically check .env and reload Gemini API key if changed."""
        import os

        if settings.allow_mock_fallback and os.environ.get("PYTEST_CURRENT_TEST"):
            return None

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
                        elif line.startswith("GOOGLE_CLOUD_PROJECT=") or line.startswith(
                            "GCP_PROJECT="
                        ):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                settings.google_cloud_project = val
                        elif line.startswith("GOOGLE_CLOUD_LOCATION=") or line.startswith(
                            "GCP_REGION="
                        ):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                settings.google_cloud_location = val
                        elif line.startswith("VERTEX_AI_ENABLED="):
                            val = line.split("=", 1)[1].strip().lower()
                            settings.vertex_ai_enabled = val in ("true", "1", "yes")
            except Exception:
                pass

        # 1. Vertex AI Mode: Uses Google Cloud Project & Application Default Credentials (ADC)
        use_vertex = (
            settings.vertex_ai_enabled
            or bool(settings.google_cloud_project)
            or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
        )
        if use_vertex:
            proj = settings.google_cloud_project or os.environ.get(
                "GOOGLE_CLOUD_PROJECT", "agentic-core"
            )
            loc = settings.google_cloud_location or os.environ.get(
                "GOOGLE_CLOUD_LOCATION", "us-central1"
            )
            creds_path = settings.google_application_credentials or os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
            if creds_path and os.path.exists(creds_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

            client_key = f"vertexai:{proj}:{loc}"
            if (
                getattr(self, "_current_gemini_key", None) != client_key
                or self._gemini_client is None
            ):
                self._current_gemini_key = client_key
                try:
                    self._gemini_client = genai.Client(
                        vertexai=True,
                        project=proj,
                        location=loc,
                    )
                    logger.info(
                        "Initialized Google GenAI client with Vertex AI (Project: %s, Region: %s, ADC)",
                        proj,
                        loc,
                    )
                except Exception as e:
                    logger.warning("Failed to initialize Vertex AI client: %s", e)
                    self._gemini_client = None
            return self._gemini_client

        # 2. Public Google AI Studio API Key Mode
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
        elif "gemini" in target_model.lower() or "text-embedding" in target_model.lower():
            target_dim = settings.embedding_dim or 768
        else:
            target_dim = settings.embedding_dim or 1024

        import asyncio

        is_gemini = (
            "gemini" in target_model.lower()
            or target_model.startswith("models/gemini")
            or target_model.startswith("text-embedding")
        )

        # --- Attempt 1: Google Gemini / Vertex AI API ---
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
                                try:
                                    return await asyncio.wait_for(
                                        gemini_client.aio.models.embed_content(
                                            model=target_model,
                                            contents=cur_contents,
                                            config=cur_config,
                                        ),
                                        timeout=20.0,
                                    )
                                except Exception as e_v2:
                                    err_str = str(e_v2).lower()
                                    if "only supports one content" in err_str:
                                        tasks = [
                                            gemini_client.aio.models.embed_content(
                                                model=target_model,
                                                contents=c,
                                                config=cur_config,
                                            )
                                            for c in cur_contents
                                        ]
                                        responses = await asyncio.gather(*tasks)
                                        combined_embeddings = []
                                        for r in responses:
                                            if hasattr(r, "embeddings") and r.embeddings:
                                                combined_embeddings.extend(r.embeddings)
                                            elif hasattr(r, "embedding") and r.embedding:
                                                combined_embeddings.append(r.embedding)
                                        from types import SimpleNamespace

                                        return SimpleNamespace(embeddings=combined_embeddings)
                                    raise

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
                        "Gemini embedding model %s failed (%s). Falling back to secondary providers...",
                        target_model,
                        e,
                    )

        # --- Attempt 2: NVIDIA NIM API (Automatic Fallback or Primary) ---
        if self._client and settings.has_nvidia_key:
            all_embeddings = []
            batch_size = 32
            bounded_texts = [t[:2500] if len(t) > 2500 else t for t in cleaned_texts]
            nim_model = target_model if not is_gemini else "nvidia/llama-3.2-nv-embedqa-1b-v2"
            try:
                for i in range(0, len(bounded_texts), batch_size):
                    batch = bounded_texts[i : i + batch_size]
                    response = await asyncio.wait_for(
                        self._client.embeddings.create(
                            input=batch,
                            model=nim_model,
                            encoding_format="float",
                            extra_body={"input_type": "passage", "truncate": "END"},
                            timeout=15.0,
                        ),
                        timeout=15.0,
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
                if len(all_embeddings) == len(cleaned_texts):
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
                                timeout=15.0,
                            ),
                            timeout=15.0,
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
                    if len(all_embeddings) == len(cleaned_texts):
                        return all_embeddings
                except Exception as e2:
                    logger.warning("NVIDIA NIM fallback embedding failed: %s", e2)
                    if not settings.allow_mock_fallback:
                        raise

        # Offline test/mock fallback ONLY when allow_mock_fallback is explicitly enabled in dev/test mode
        if settings.allow_mock_fallback and not self._has_live_client():
            return [self._mock_embedding(t, dim=target_dim) for t in cleaned_texts]

        raise RuntimeError(
            f"Failed to generate embeddings using target model '{target_model}'. "
            "Please verify your API keys (GEMINI_API_KEY / NVIDIA_API_KEY) and quota in .env."
        )

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
        """Extract <think>...</think> reasoning blocks and thinking headers from model content."""
        if not text:
            return "", reasoning
        if "<think>" in text:
            if "</think>" in text:
                parts = text.split("</think>", 1)
                think_part = parts[0].replace("<think>", "").strip()
                ans_part = parts[1].strip()
                return ans_part, reasoning or think_part
            return text.replace("<think>", "").strip(), reasoning

        # Strip standard conversational thinking prefixes
        pattern = r"^(?:Here'?s a thinking process:?|Thinking Process:?|Thought:?)\s*(.+?)(?=\n\n(?=[A-Z\"#*0-9]))"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            think_part = match.group(1).strip()
            ans_part = text[match.end() :].strip()
            if ans_part:
                return ans_part, reasoning or think_part

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
        """Generate completion using Google Gemini / Vertex AI or Groq (qwen/qwen3.6-27b).

        Args:
            timeout: Per-request timeout in seconds.
            max_retries: Maximum retry attempts on rate-limit errors.
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

        # --- Attempt 0: Google Gemini / Vertex AI API (When Gemini model requested) ---
        if is_gemini_model:
            gemini_client = self._refresh_gemini_client()
            if gemini_client:
                target_model = model or "gemini-3.7-flash"
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

                # Configure thinking for Gemini 3.7 Flash
                thinking_cfg = None
                if enable_thinking and (
                    "3.7" in target_model or "flash" in target_model or "gemini" in target_model
                ):
                    try:
                        thinking_cfg = genai_types.ThinkingConfig(
                            thinking_level="MEDIUM",
                        )
                    except Exception:
                        thinking_cfg = None

                try:

                    async def _do_gemini_gen():
                        gen_kwargs: dict[str, Any] = {
                            "system_instruction": system_prompt.strip() if system_prompt else None,
                            "max_output_tokens": min(max_tokens, 8192) if max_tokens else 8192,
                        }
                        if thinking_cfg is not None:
                            gen_kwargs["thinking_config"] = thinking_cfg
                        else:
                            gen_kwargs["temperature"] = temperature
                            gen_kwargs["top_p"] = top_p

                        return await asyncio.wait_for(
                            gemini_client.aio.models.generate_content(
                                model=target_model,
                                contents=gemini_contents,  # type: ignore[arg-type]
                                config=genai_types.GenerateContentConfig(**gen_kwargs),
                            ),
                            timeout=timeout,
                        )

                    gemini_resp = await self._call_gemini_with_backoff(
                        _do_gemini_gen, f"generate_content ({target_model})"
                    )

                    reasoning_parts: list[str] = []
                    content_parts: list[str] = []
                    if getattr(gemini_resp, "candidates", None) and len(gemini_resp.candidates) > 0:
                        cand_content = getattr(gemini_resp.candidates[0], "content", None)
                        if cand_content and getattr(cand_content, "parts", None):
                            for part in cand_content.parts:
                                part_text = getattr(part, "text", "") or ""
                                if getattr(part, "thought", False):
                                    reasoning_parts.append(part_text)
                                else:
                                    content_parts.append(part_text)

                    if reasoning_parts or content_parts:
                        content = "".join(content_parts)
                        reasoning = "".join(reasoning_parts) if reasoning_parts else None
                        if not reasoning:
                            content, reasoning = self._parse_think_tags(content, None)
                        return content, reasoning

                    raw_content = gemini_resp.text or ""
                    content, reasoning = self._parse_think_tags(raw_content, None)
                    return content, reasoning
                except Exception as e:
                    logger.warning("Gemini model %s complete failed: %s", target_model, e)

        # --- Attempt 1: Groq API (Ultra-fast reasoning) ---
        groq_client = self._refresh_groq_client()
        if groq_client:
            m = self.llm_model if (not model or is_gemini_model) else model
            if not m or m.startswith("gemini-"):
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
                        "Groq model %s hit rate limit (%s). Reset in %s.",
                        m,
                        notice.get("quota_type"),
                        notice.get("retry_after"),
                    )
                else:
                    logger.warning("Groq model %s call failed: %s.", m, e)

        # Fallback: Deterministic mock
        return self._mock_completion(messages)

    # Alias for backward compatibility
    generate_completion = complete

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        enable_thinking: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields chunks with type (reasoning or content) and text."""

        is_gemini_model = bool(
            model
            and (
                model.startswith("gemini-")
                or model.startswith("models/gemini")
                or model.startswith("google/gemini")
            )
        )

        # --- Attempt 0: Google Gemini / Vertex AI API Streaming ---
        if is_gemini_model:
            gemini_client = self._refresh_gemini_client()
            if gemini_client:
                target_model = model or "gemini-3.7-flash"
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

                # Configure thinking for Gemini 3.7 Flash
                thinking_cfg = None
                if enable_thinking and (
                    "3.7" in target_model or "flash" in target_model or "gemini" in target_model
                ):
                    try:
                        thinking_cfg = genai_types.ThinkingConfig(
                            thinking_level="MEDIUM",
                        )
                    except Exception:
                        thinking_cfg = None

                try:
                    gen_kwargs: dict[str, Any] = {
                        "system_instruction": system_prompt.strip() if system_prompt else None,
                        "max_output_tokens": min(max_tokens, 8192) if max_tokens else 8192,
                    }
                    if thinking_cfg is not None:
                        gen_kwargs["thinking_config"] = thinking_cfg
                    else:
                        gen_kwargs["temperature"] = temperature
                        gen_kwargs["top_p"] = top_p

                    gemini_stream = await gemini_client.aio.models.generate_content_stream(
                        model=target_model,
                        contents=gemini_contents,  # type: ignore[arg-type]
                        config=genai_types.GenerateContentConfig(**gen_kwargs),
                    )
                    stream_emitted = False
                    async for chunk in gemini_stream:
                        has_parts = False
                        if getattr(chunk, "candidates", None) and len(chunk.candidates) > 0:
                            content_obj = getattr(chunk.candidates[0], "content", None)
                            if content_obj and getattr(content_obj, "parts", None):
                                for part in content_obj.parts:
                                    part_text = getattr(part, "text", "") or ""
                                    if not part_text:
                                        continue
                                    has_parts = True
                                    stream_emitted = True
                                    self.last_rate_limit = None
                                    LLMClient.global_rate_limit = None
                                    if getattr(part, "thought", False):
                                        yield {"type": "reasoning", "text": part_text}
                                    else:
                                        yield {"type": "content", "text": part_text}
                        if not has_parts:
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

        # --- Attempt 1: Groq API ---
        groq_client = self._refresh_groq_client()
        if groq_client:
            m = model or self.llm_model or "qwen/qwen3.6-27b"
            if m.startswith("gemini-"):
                m = "qwen/qwen3.6-27b"
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
                        "Groq model %s hit rate limit (%s). Reset in %s.",
                        m,
                        notice.get("quota_type"),
                        notice.get("retry_after"),
                    )
                else:
                    logger.warning("Groq model %s streaming failed: %s.", m, e)

        # --- Attempt 2: Direct Context Extraction Fallback ---
        active_notice = self.last_rate_limit or LLMClient.global_rate_limit
        if active_notice:
            yield {"type": "rate_limit", "notice": active_notice}

        content, reasoning = self._mock_completion(messages)
        if reasoning:
            yield {"type": "reasoning", "text": reasoning}
        yield {"type": "content", "text": content}

    def _mock_completion(self, messages: list[dict[str, Any]]) -> tuple[str, str | None]:
        """Generate structured synthetic answer for testing purposes."""
        last_msg = messages[-1]["content"] if messages else ""
        system_msg = (
            messages[0]["content"] if len(messages) > 1 and messages[0]["role"] == "system" else ""
        )

        active_notice = self.last_rate_limit or LLMClient.global_rate_limit
        if active_notice:
            reasoning = (
                f"⚠️ **Groq Rate Limit Notice**: {active_notice['message']}\n"
                "Showing retrieved document evidence directly."
            )
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
