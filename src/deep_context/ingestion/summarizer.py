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
                    tok.padding_side = "left"
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token or "<|extra_0|>"
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
        if not self._is_loaded and self._model is None:
            return

        try:
            import gc

            import torch

            if self._model is not None:
                del self._model
            if self._tokenizer is not None:
                del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._is_loaded = False

            gc.collect()
            gc.collect()

            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                "Summarization model '%s' successfully unloaded from memory (RAM/GPU freed).",
                self.model_name,
            )
        except Exception as e:
            logger.warning("Error during model unload: %s", e)

    def _build_prompt(
        self,
        chunk_text: str,
        section_path: str | None = None,
        document_title: str | None = None,
        parent_context: str | None = None,
    ) -> str:
        """
        Constructs Anthropic Contextual Retrieval instruction prompt.
        Situates the chunk within the document title and enclosing parent context,
        producing a concise 1-2 sentence situating context with document-identifying info.
        """
        system_msg = (
            "/no_think\n"
            "You are an expert document annotator. "
            "Your task is to provide a brief, succinct context (1 to 2 sentences, 30-60 words maximum) that situates the given text chunk within the broader document for search retrieval. "
            "Identify the specific topic, key entities or subjects, and section context, and clarify ambiguous references. "
            "Do NOT output conversational filler, introductory phrases (e.g. 'The document is about'), or repeated sentences. Output only the succinct factual context."
        )

        doc_header = f"Document: {document_title}\n" if document_title else ""
        sec_header = f"Section: {section_path}\n" if section_path else ""

        parent_snippet = ""
        if parent_context:
            parent_clean = parent_context.strip()
            if len(parent_clean) > 800:
                parent_snippet = f"Enclosing Section Context:\n{parent_clean[:800]}...\n"
            else:
                parent_snippet = f"Enclosing Section Context:\n{parent_clean}\n"

        doc_block = f"<document>\n{doc_header}{sec_header}{parent_snippet}</document>".strip()

        user_msg = (
            f"{doc_block}\n\n"
            "Here is the chunk we want to situate within the whole document:\n"
            f"<chunk>\n{chunk_text.strip()[:2500]}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval of the chunk. "
            "Include a one sentence description of the document with identifying info.\n"
            "Answer only with the succinct context and nothing else."
        )

        # Prefilling the closed think tag ensures zero thinking tokens are generated by Qwen3
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n"
        )

    @staticmethod
    def _clean_and_complete_summary(raw_text: str) -> str:
        """Cleans think tags, conversational preambles, meta-phrasing, deduplicates sentences, and ensures complete terminal sentences."""
        import re

        summary = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        summary = re.sub(
            r"^(?:Here is the summary:?|Summary:?|Context:?|This chunk discusses:?|The situating context is:?|Document:?)\s*",
            "",
            summary,
            flags=re.IGNORECASE,
        ).strip()

        # If model repeats "Summary:" or "Context:", truncate before the repetition
        for rep in ("\nSummary:", " Summary:", "\nContext:", " Context:"):
            rep_idx = summary.find(rep)
            if rep_idx > 30:
                summary = summary[:rep_idx].strip()

        # Strip echoed meta-rules if model repeats prompt instructions
        summary = re.sub(
            r"(?:It resolves ambiguous pronouns.*?context\.\s*|The context centers around.*?context\.\s*)",
            "",
            summary,
            flags=re.IGNORECASE,
        ).strip()

        # Deduplicate identical or near-identical consecutive sentences
        if summary:
            sentences = re.split(r"(?<=[.!?])\s+", summary)
            unique_sentences: list[str] = []
            seen_sentences: set[str] = set()
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_norm = re.sub(r"\W+", " ", s_clean.lower()).strip()
                if s_norm and s_norm not in seen_sentences:
                    seen_sentences.add(s_norm)
                    unique_sentences.append(s_clean)
            summary = " ".join(unique_sentences).strip()

        if not summary:
            return ""

        # Ensure summary doesn't abruptly end mid-sentence: trim to last complete punctuation if truncated
        if summary and summary[-1] not in (".", "!", "?", '"', "'"):
            last_punct = max(summary.rfind("."), summary.rfind("!"), summary.rfind("?"))
            if last_punct > 35:
                summary = summary[: last_punct + 1].strip()
            else:
                summary = summary.rstrip(",;:- ") + "."
        return summary

    def _generate_batch_sync(self, prompts: list[str]) -> list[tuple[str, int]]:
        """Synchronous batch generation executed efficiently in vectorized batches on GPU."""
        import torch

        if not prompts:
            return []

        results: list[tuple[str, int]] = []
        try:
            pad_id = (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else self._tokenizer.eos_token_id
            )
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._device)

            input_seq_len = inputs.input_ids.shape[1]

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_summary_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

            for idx in range(len(prompts)):
                new_tokens = outputs[idx][input_seq_len:]
                raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                summary = self._clean_and_complete_summary(raw_text)
                results.append((summary, len(new_tokens)))

        except Exception as e:
            logger.warning(
                "Vectorized batch generation notice (%s). Falling back to sequential: %s", e, e
            )
            for prompt in prompts:
                try:
                    inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
                    in_len = inputs.input_ids.shape[1]
                    with torch.inference_mode():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=self.max_summary_tokens,
                            do_sample=False,
                            pad_token_id=self._tokenizer.eos_token_id,
                        )
                    new_tokens = outputs[0][in_len:]
                    raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    summary = self._clean_and_complete_summary(raw_text)
                    results.append((summary, len(new_tokens)))
                except Exception as inner_e:
                    logger.warning("Generation error on prompt fallback: %s", inner_e)
                    results.append(("", 0))
                except Exception as inner_e:
                    logger.warning("Generation error on prompt fallback: %s", inner_e)
                    results.append(("", 0))
        finally:
            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
            gc.collect()

        return results

    async def summarize_chunk(
        self,
        chunk_text: str,
        section_path: str | None = None,
        context_prefix: str | None = None,
        document_title: str | None = None,
        parent_context: str | None = None,
    ) -> tuple[str, int]:
        """Generates summary and token count for a single chunk."""
        if not chunk_text.strip():
            return "", 0

        loaded = await self._ensure_model_loaded()
        if not loaded or not self._model:
            words = chunk_text.strip().split()
            fallback_summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
            return fallback_summary, len(fallback_summary.split())

        effective_sec = section_path or context_prefix
        prompt = self._build_prompt(
            chunk_text,
            section_path=effective_sec,
            document_title=document_title,
            parent_context=parent_context,
        )
        loop = asyncio.get_running_loop()
        try:
            async with self._load_lock:
                res = await loop.run_in_executor(None, self._generate_batch_sync, [prompt])
            return res[0]
        except Exception as e:
            logger.warning("Chunk summarization failed (%s); using fallback.", e)
            words = chunk_text.strip().split()
            fallback_summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
            return fallback_summary, len(fallback_summary.split())

    async def summarize_batch(
        self,
        chunks: list[Chunk],
        parent_chunk_map: dict[str, Chunk] | None = None,
        document_title: str | None = None,
        progress_callback: Callable[[Chunk], None] | None = None,
    ) -> list[tuple[str, int]]:
        """Summarizes a batch of child chunks sequentially and safely on MPS/GPU."""
        if not chunks:
            return []

        loaded = await self._ensure_model_loaded()
        if not loaded or not self._model:
            results: list[tuple[str, int]] = []
            for c in chunks:
                words = c.content.strip().split()
                s = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
                results.append((s, len(s.split())))
            return results

        parent_map = parent_chunk_map or {}
        prompts: list[str] = []
        for c in chunks:
            p_chunk = parent_map.get(c.parent_chunk_id) if c.parent_chunk_id else None
            p_text = p_chunk.content if p_chunk else None
            prompts.append(
                self._build_prompt(
                    c.content,
                    section_path=c.section_path,
                    document_title=document_title,
                    parent_context=p_text,
                )
            )
        results = []
        loop = asyncio.get_running_loop()

        async with self._load_lock:
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_chunks = chunks[i : i + self.batch_size]

                batch_res = await loop.run_in_executor(
                    None, self._generate_batch_sync, batch_prompts
                )
                for (summary, tokens), chunk in zip(batch_res, batch_chunks):
                    if not summary:
                        words = chunk.content.strip().split()
                        summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
                        tokens = len(summary.split())
                    results.append((summary, tokens))
                    if progress_callback:
                        progress_callback(chunk)

        return results

    async def summarize_chunks(
        self,
        chunks: list[Chunk],
        parent_chunk_map: dict[str, Chunk] | None = None,
        document_title: str | None = None,
        progress_callback: Callable[[Chunk], None] | None = None,
    ) -> list[Chunk]:
        """Populates summary_text, summary_tokens, summary_model, and generated_at in-place."""
        if not chunks:
            return chunks

        summaries = await self.summarize_batch(
            chunks,
            parent_chunk_map=parent_chunk_map,
            document_title=document_title,
            progress_callback=progress_callback,
        )
        now = datetime.now(timezone.utc)

        for chunk, (summary, tokens) in zip(chunks, summaries):
            chunk.summary_text = summary
            chunk.summary_tokens = tokens
            chunk.summary_model = self.model_name
            chunk.generated_at = now

        return chunks
