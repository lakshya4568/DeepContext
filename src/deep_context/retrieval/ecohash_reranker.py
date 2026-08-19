"""EcoHash Hosted BGE Reranker (bge-reranker-v2-m3) integration."""

from __future__ import annotations

from typing import Any

import httpx

from deep_context.core.config import settings
from deep_context.core.logging import logger
from deep_context.retrieval.reranker import CrossEncoderReranker, _blend_with_rrf


class EcoHashReranker:
    """Reranker using hosted BGE-reranker-v2-m3 via EcoHash API.

    Offloads cross-encoder inference to the hosted EcoHash endpoint to avoid
    local GPU/CPU memory and computation constraints while providing full
    transformer-based semantic cross-attention scoring.
    """

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        api_key: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates

        key = api_key or settings.ecohash_api_key
        if not key:
            logger.warning(
                "ECOHASH_API_KEY not configured. Falling back to heuristic CrossEncoderReranker."
            )
            return await CrossEncoderReranker.rerank(query, candidates, top_k=top_k)

        target_model = model or settings.ecohash_rerank_model or "bge-reranker-v2-m3"
        url = settings.ecohash_rerank_url or "https://api.ecohash.com/v1/rerank"
        documents = [str(c.get("content", ""))[:1000] for c in candidates]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": target_model,
                        "query": query,
                        "documents": documents,
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            raw_scores = [0.0] * len(candidates)
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score", 0.0)
                if isinstance(idx, int) and 0 <= idx < len(raw_scores):
                    raw_scores[idx] = float(score)

            scored = _blend_with_rrf(candidates, raw_scores)
            return [item for _, item in scored[:top_k]]
        except Exception as e:
            logger.warning(
                "EcoHash rerank API call failed (%s). Falling back to heuristic CrossEncoderReranker.",
                e,
            )
            return await CrossEncoderReranker.rerank(query, candidates, top_k=top_k)
