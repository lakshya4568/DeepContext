"""Agentic Planner for multi-step reasoning implementing FR18."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from deep_context.core.llm_client import llm_client
from deep_context.core.types import Citation, RetrievalFilters
from deep_context.retrieval.engine import retrieval_engine
from deep_context.storage.base import StorageInterface


class AgenticPlanner:
    """Executes multi-hop iterative retrieval, sub-query execution, and evidence synthesis."""

    def __init__(self, storage: StorageInterface):
        self.storage = storage

    async def execute_plan(
        self,
        query: str,
        filters: RetrievalFilters,
        max_iterations: int = 3,
    ) -> tuple[str, list[dict[str, Any]], str | None]:
        """
        Plans and executes a multi-step retrieval and synthesis workflow.
        Returns (final_answer, citations, reasoning).
        """
        # Step 1: Generate Plan
        plan_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an agentic RAG search planner. Decompose the user question into 2-3 specific, concise search queries "
                    "for hybrid retrieval (exact quotes, character names, or key phrases). Do NOT output conversational instructions like 'Search for...'. "
                    'Return ONLY a JSON array of search strings. Example: ["Ser Kevan ferocious enough", "Tyrion vanguard wildlings"]'
                ),
            },
            {"role": "user", "content": f"Question: {query}"},
        ]
        plan_content, reasoning = await llm_client.complete(plan_prompt, temperature=0.2)

        steps: list[str] = []
        try:
            import re

            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", plan_content).strip()
            match = re.search(r"\[[\s\S]*?\]", cleaned)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    steps = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            steps = []

        search_queries = [query]
        for s in steps:
            if s and s not in search_queries:
                search_queries.append(s)

        # Fallback queries from question phrases / keywords
        clean_q = query.replace("“", '"').replace("”", '"')
        quoted = re.findall(r'"([^"]+)"', clean_q)
        for q_phrase in quoted:
            if q_phrase and q_phrase not in search_queries:
                search_queries.append(q_phrase)
        words = [
            w
            for w in re.findall(r"\w+", clean_q.lower())
            if len(w) > 2 and w not in {"who", "what", "where", "when", "how", "did", "and", "the"}
        ]
        if len(words) >= 2:
            key_phrase = " ".join(words[:4])
            if key_phrase not in search_queries:
                search_queries.append(key_phrase)

        all_retrieved_chunks: list[dict[str, Any]] = []
        all_citations: list[Citation] = []

        # Step 2: Iterative Retrieval execution
        for step in search_queries[: max_iterations + 1]:
            res = await retrieval_engine.retrieve(query=step, filters=filters, top_k=4)
            all_retrieved_chunks.extend(res.parent_chunks)
            all_citations.extend(res.citations)

        # Deduplicate chunks
        seen_ids: set[str] = set()
        deduped_chunks: list[dict[str, Any]] = []
        for c in all_retrieved_chunks:
            cid = str(c.get("chunk_id") or c.get("id") or "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                deduped_chunks.append(c)

        # Step 3: Synthesis with token budgeting (max 12,000 chars / ~3,000 tokens)
        context_blocks = []
        char_budget = 12000
        total_chars = 0

        for idx, c in enumerate(deduped_chunks, start=1):
            title = c.get("document_title", "Document")
            sec = c.get("section_path", "")
            content = c.get("content", "").strip()
            block = f"[{idx}] Source: {title} | Section: {sec}\n{content}"
            if total_chars + len(block) > char_budget:
                remaining = char_budget - total_chars
                if remaining > 200:
                    context_blocks.append(block[:remaining] + "...\n[Remaining context trimmed]")
                break
            context_blocks.append(block)
            total_chars += len(block)

        context_text = "\n\n".join(context_blocks)

        synthesis_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a grounded multi-hop reasoning agent. Answer the user query using the retrieved context. "
                    "Ensure every claim is supported by the context."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nUser Query: {query}",
            },
        ]

        answer, synth_reasoning = await llm_client.complete(synthesis_prompt, temperature=0.4)
        citations_dict = [c.to_dict() for c in all_citations]

        combined_reasoning = (reasoning or "") + ("\n" + synth_reasoning if synth_reasoning else "")

        return answer, citations_dict, combined_reasoning

    async def execute_plan_stream(
        self,
        query: str,
        filters: RetrievalFilters,
        max_iterations: int = 3,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Executes multi-hop iterative retrieval, streaming sub-query status, citations,
        real-time thinking tokens, and final answer tokens.
        """
        # Step 1: Generate Plan
        yield {
            "type": "status",
            "message": "🧠 Planning focused search queries for multi-hop retrieval...",
        }
        plan_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an agentic RAG search planner. Decompose the user question into 2-3 specific, concise search queries "
                    "for hybrid retrieval (exact quotes, character names, or key phrases). Do NOT output conversational instructions like 'Search for...'. "
                    'Return ONLY a JSON array of search strings. Example: ["Ser Kevan ferocious enough", "Tyrion vanguard wildlings"]'
                ),
            },
            {"role": "user", "content": f"Question: {query}"},
        ]
        plan_content, reasoning = await llm_client.complete(plan_prompt, temperature=0.2)
        if reasoning:
            yield {"type": "reasoning", "delta": f"[Search Plan Strategy]\n{reasoning}\n\n"}

        steps: list[str] = []
        try:
            import re

            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", plan_content).strip()
            match = re.search(r"\[[\s\S]*?\]", cleaned)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    steps = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            steps = []

        search_queries = [query]
        for s in steps:
            if s and s not in search_queries:
                search_queries.append(s)

        # Fallback queries from question phrases / keywords
        clean_q = query.replace("“", '"').replace("”", '"')
        quoted = re.findall(r'"([^"]+)"', clean_q)
        for q_phrase in quoted:
            if q_phrase and q_phrase not in search_queries:
                search_queries.append(q_phrase)
        words = [
            w
            for w in re.findall(r"\w+", clean_q.lower())
            if len(w) > 2 and w not in {"who", "what", "where", "when", "how", "did", "and", "the"}
        ]
        if len(words) >= 2:
            key_phrase = " ".join(words[:4])
            if key_phrase not in search_queries:
                search_queries.append(key_phrase)

        all_retrieved_chunks: list[dict[str, Any]] = []
        all_citations: list[Citation] = []

        # Step 2: Iterative Retrieval execution
        for i, step in enumerate(search_queries[: max_iterations + 1], 1):
            yield {
                "type": "status",
                "message": f"📚 [{i}/{min(len(search_queries), max_iterations + 1)}] Retrieving: '{step}'...",
            }
            res = await retrieval_engine.retrieve(query=step, filters=filters, top_k=4)
            all_retrieved_chunks.extend(res.parent_chunks)
            all_citations.extend(res.citations)

        # Deduplicate chunks
        seen_ids: set[str] = set()
        deduped_chunks: list[dict[str, Any]] = []
        for c in all_retrieved_chunks:
            cid = str(c.get("chunk_id") or c.get("id") or "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                deduped_chunks.append(c)

        # Emit citations immediately
        citations_dict = [c.to_dict() for c in all_citations]
        yield {"type": "citations", "citations": citations_dict}

        # Step 3: Synthesis with token budgeting (max 12,000 chars / ~3,000 tokens)
        context_blocks = []
        char_budget = 12000
        total_chars = 0

        for idx, c in enumerate(deduped_chunks, start=1):
            title = c.get("document_title", "Document")
            sec = c.get("section_path", "")
            content = c.get("content", "").strip()
            block = f"[{idx}] Source: {title} | Section: {sec}\n{content}"
            if total_chars + len(block) > char_budget:
                remaining = char_budget - total_chars
                if remaining > 200:
                    context_blocks.append(block[:remaining] + "...\n[Remaining context trimmed]")
                break
            context_blocks.append(block)
            total_chars += len(block)

        context_text = "\n\n".join(context_blocks)

        synthesis_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a grounded multi-hop reasoning agent. Answer the user query using the retrieved context. "
                    "Ensure every claim is supported by the context."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nUser Query: {query}",
            },
        ]

        yield {
            "type": "status",
            "message": "🧠 Synthesizing grounded multi-hop response in real-time...",
        }
        emitted_content = ""
        emitted_reasoning = ""
        async for token_chunk in llm_client.stream_complete(
            synthesis_prompt, model=model, temperature=0.4, enable_thinking=True
        ):
            chunk_type = token_chunk.get("type", "content")
            chunk_text = token_chunk.get("text", "")
            if chunk_type == "reasoning":
                emitted_reasoning += chunk_text
                yield {"type": "reasoning", "delta": chunk_text}
            else:
                emitted_content += chunk_text
                yield {"type": "content", "delta": chunk_text}

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
                yield {"type": "content", "delta": extracted}
