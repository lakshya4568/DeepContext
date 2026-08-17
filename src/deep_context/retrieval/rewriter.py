"""Query rewriter & sub-query decomposer."""

from __future__ import annotations

import json
import re

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import QueryShape


def safe_parse_json_list(text: str) -> list[str]:
    """Robustly extracts a list of strings from LLM output, handling markdown, thinking, or truncations."""
    clean = text.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    match = re.search(r"\[\s*\"[^\"]+\"\s*(?:,\s*\"[^\"]+\"\s*)*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass

    items = re.findall(r"\"([^\"\n\r]{3,100})\"", text)
    if items:
        return [it.strip() for it in items if it.strip() and not it.lower().startswith("sub-query") and not it.lower().startswith("search query")]

    return []


class QueryRewriter:
    """Decomposes or rewrites complex, multi-part, or paraphrased queries into focused search queries."""

    @classmethod
    async def rewrite_or_decompose(cls, query: str, shape: QueryShape) -> list[str]:
        q_lower = query.lower()
        sub_queries: list[str] = [query]

        # 1. Appendix / Heraldry / Lineage routing
        if any(w in q_lower for w in ("appendix", "words of house", "house words", "sigil", "sworn houses")):
            appendix_q = f"appendix {query}"
            if appendix_q not in sub_queries:
                sub_queries.append(appendix_q)

        # Target ultra-fast LLM for rewrite
        target_model = "meta/llama-3.1-8b-instruct" if settings.has_nvidia_key else settings.llm_model

        # 2. Multi-hop decomposition
        if shape in (QueryShape.MULTI_HOP, QueryShape.AGGREGATION) or " and " in q_lower or "?" in query[10:]:
            try:
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert search query optimizer for the fantasy novel A Game of Thrones (A Song of Ice and Fire) and corpus documents.\n"
                            "Decompose the complex user query into 2-3 focused sub-queries targeting distinct entities, events, and scenes.\n"
                            'Return ONLY a valid JSON array of strings, e.g. ["sub-query 1", "sub-query 2"].'
                        ),
                    },
                    {"role": "user", "content": f"Query: {query}"},
                ]
                content, _ = await llm_client.complete(
                    prompt,
                    model=target_model,
                    max_tokens=512,
                    temperature=0.0,
                    enable_thinking=False,
                    timeout=5.0,
                    max_retries=1,
                )
                items = safe_parse_json_list(content)
                for item in items:
                    if item and item not in sub_queries:
                        sub_queries.append(item)
            except Exception as e:
                logger.debug("LLM query rewrite fallback to heuristic: %s", e)
                if " and " in query:
                    parts = [p.strip() for p in re.split(r"\band\b", query) if len(p.strip()) > 3]
                    for p in parts:
                        if p not in sub_queries:
                            sub_queries.append(p)

        # 3. Paraphrase expansion for descriptive queries
        all_words = re.findall(r"\w+", query)
        has_proper_nouns = any(w[0].isupper() for w in all_words[1:]) if len(all_words) > 1 else False
        if not has_proper_nouns and len(sub_queries) == 1:
            try:
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert search query expander for the fantasy novel A Game of Thrones (A Song of Ice and Fire) and corpus documents.\n"
                            "Identify the core scene, entity, or event described in the question and produce 2-3 specific search queries targeting the canon names, characters, objects, and terms.\n"
                            'Return ONLY a JSON array of strings, e.g. ["search query 1", "search query 2"].'
                        ),
                    },
                    {"role": "user", "content": f"Query: {query}"},
                ]
                content, _ = await llm_client.complete(
                    prompt,
                    model=target_model,
                    max_tokens=512,
                    temperature=0.0,
                    enable_thinking=False,
                    timeout=5.0,
                    max_retries=1,
                )
                items = safe_parse_json_list(content)
                for item in items:
                    if item and item not in sub_queries:
                        sub_queries.append(item)
            except Exception as e:
                logger.debug("LLM paraphrase rewrite fallback: %s", e)

        return sub_queries[:4]
