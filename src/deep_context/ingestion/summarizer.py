"""LLM-based chunk summarization using lazy-loaded Qwen3 (0.6B/0.8B) local model optimized for Apple Silicon MPS."""

from __future__ import annotations

import asyncio
import gc
from datetime import datetime, timezone
from typing import Any, Callable

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.core.types import Chunk


class ChunkSummarizer:
    """Generates concise semantic summaries for text chunks using a lazy-loaded local LLM."""

    def __init__(
        self,
        model_name: str | None = None,
        max_summary_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or settings.summary_model
        self.max_summary_tokens = max_summary_tokens or settings.summary_max_tokens
        self.batch_size = batch_size or settings.summary_batch_size
        self._target_device_str = device or settings.summary_device

        # Lazy state: Model & tokenizer are initialized only on first generation request
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._is_loaded = False
        self._load_lock = asyncio.Lock()

    def _resolve_device(self) -> str:
        """Auto-detect optimal compute device (MPS on Apple Silicon GPU, CUDA on Nvidia GPU, or CPU)."""
        import torch

        if self._target_device_str != "auto":
            return self._target_device_str

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon Metal Performance Shaders (Mac M1/M2/M3/M4 GPU)
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    async def _ensure_model_loaded(self) -> bool:
        """Loads model and tokenizer into memory if not already initialized."""
        if self._is_loaded:
            return True

        async with self._load_lock:
            if self._is_loaded:
                return True

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device_str = self._resolve_device()
                logger.info(
                    "Lazy-loading summarization model '%s' on GPU device '%s' (MPS/Metal)...",
                    self.model_name,
                    device_str,
                )

                loop = asyncio.get_running_loop()

                def _load() -> tuple[Any, Any, str]:
                    import os

                    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
                    if token:
                        os.environ["HF_TOKEN"] = token
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

                    tok = AutoTokenizer.from_pretrained(
                        self.model_name,
                        token=token,
                        trust_remote_code=True,
                    )
                    # FP16 on GPU (MPS/CUDA) cuts RAM/VRAM usage in half (~600MB vs 1.2GB)
                    dtype = torch.float16 if device_str in ("cuda", "mps") else torch.float32
                    target_device = torch.device(device_str)
                    mdl: Any = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        token=token,
                        trust_remote_code=True,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                    )
                    mdl = mdl.to(target_device)
                    mdl.eval()
                    return tok, mdl, device_str

                self._tokenizer, self._model, self._device = await loop.run_in_executor(None, _load)
                self._is_loaded = True
                logger.info(
                    "Summarization model '%s' successfully loaded on %s.",
                    self.model_name,
                    self._device,
                )
                return True
            except Exception as e:
                logger.error("Failed to load summarization model '%s': %s", self.model_name, e)
                if not settings.allow_mock_fallback:
                    raise
                return False

    def unload(self) -> None:
        """Explicitly free model weights and release GPU (MPS) / RAM memory."""
        if not self._is_loaded:
            return

        try:
            import torch

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
            gc.collect()

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Summarization model '%s' unloaded from memory.", self.model_name)
        except Exception as e:
            logger.warning("Error during model unload: %s", e)

    def _build_prompt(self, chunk_text: str, context_prefix: str | None = None) -> str:
        """Constructs a high-precision, direct instruction prompt for fast semantic summarization."""
        system_msg = (
            "You are a high-speed factual summarizer for a RAG retrieval system. "
            "Write a single concise 1-2 sentence semantic summary (under 60 words) capturing the main concept, key entities, numbers/metrics, and technical terms. "
            "Rules:\n"
            "- Do NOT think or reason. Do NOT output <think> tags.\n"
            "- Do NOT include conversational preambles like 'This text discusses' or 'In this section'.\n"
            "- Output ONLY the factual summary text directly."
        )
        context_hint = f"Topic / Path: {context_prefix}\n" if context_prefix else ""
        user_msg = f"{context_hint}Content:\n{chunk_text.strip()[:3000]}"

        # Prefilling the closed think tag ensures zero thinking tokens are generated by Qwen3
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n"
        )

    def _generate_sync(self, prompt: str) -> tuple[str, int]:
        """Synchronous model generation call with minimal memory overhead."""
        import re

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_summary_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id or self._tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][input_len:]
        raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        summary = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        summary_tokens = len(new_tokens)

        # Clear temporary tensors
        del inputs
        del outputs
        if self._device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        return summary, summary_tokens

    async def summarize_chunk(
        self, chunk_text: str, context_prefix: str | None = None
    ) -> tuple[str, int]:
        """Generates summary and token count for a single chunk."""
        if not chunk_text.strip():
            return "", 0

        loaded = await self._ensure_model_loaded()
        if not loaded or not self._model:
            # Fallback when model is not available in mock/test mode
            words = chunk_text.strip().split()
            fallback_summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
            return fallback_summary, len(fallback_summary.split())

        prompt = self._build_prompt(chunk_text, context_prefix)
        loop = asyncio.get_running_loop()
        try:
            summary, summary_tokens = await loop.run_in_executor(None, self._generate_sync, prompt)
            return summary, summary_tokens
        except Exception as e:
            logger.warning("Chunk summarization failed (%s); using fallback.", e)
            words = chunk_text.strip().split()
            fallback_summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
            return fallback_summary, len(fallback_summary.split())

    async def summarize_batch(
        self, chunks: list[Chunk], progress_callback: Callable[[Chunk], None] | None = None
    ) -> list[tuple[str, int]]:
        """Summarizes a batch of child chunks."""
        results: list[tuple[str, int]] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            tasks = [
                self.summarize_chunk(
                    chunk.content,
                    context_prefix=chunk.section_path,
                )
                for chunk in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            if progress_callback:
                for c in batch:
                    progress_callback(c)
        return results

    async def summarize_chunks(
        self,
        chunks: list[Chunk],
        progress_callback: Callable[[Chunk], None] | None = None,
    ) -> list[Chunk]:
        """Populates summary_text, summary_tokens, summary_model, and generated_at in-place."""
        if not chunks:
            return chunks

        summaries = await self.summarize_batch(chunks, progress_callback=progress_callback)
        now = datetime.now(timezone.utc)

        for chunk, (summary, tokens) in zip(chunks, summaries):
            chunk.summary_text = summary
            chunk.summary_tokens = tokens
            chunk.summary_model = self.model_name
            chunk.generated_at = now

        return chunks
