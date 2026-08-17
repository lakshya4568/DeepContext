"""Vectorless / PageIndex-style hierarchical tree navigator implementing FR6."""

from __future__ import annotations

from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.storage.base import StorageInterface


class TreeNavigator:
    """Navigates hierarchical document trees using LLM reasoning at each level."""

    def __init__(self, storage: StorageInterface):
        self.storage = storage

    async def navigate(self, query: str, document_id: str, max_hops: int = 4) -> list[str]:
        """
        Traverses tree from root to leaf node(s), returning relevant chunk IDs.
        """
        current_node_id: str | None = None
        leaf_chunk_ids: list[str] = []

        for _ in range(max_hops):
            child_nodes = await self.storage.get_child_tree_nodes(
                document_id=document_id, parent_node_id=current_node_id
            )
            if not child_nodes:
                break

            # If only 1 node, descend directly
            if len(child_nodes) == 1:
                node = child_nodes[0]
                if node.chunk_id:
                    leaf_chunk_ids.append(node.chunk_id)
                current_node_id = node.id
                continue

            # LLM selects most relevant branch based on titles & summaries
            nodes_desc = "\n".join(
                f"[{idx}] {n.title}: {n.summary or ''}" for idx, n in enumerate(child_nodes)
            )
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are navigating a document table of contents. Pick the index (integer) of the section "
                        "most likely to contain the answer to the user question. Return ONLY a single integer."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nSections:\n{nodes_desc}"},
            ]

            chosen_idx = 0
            try:
                import re

                content, _ = await llm_client.complete(prompt, max_tokens=100, temperature=0.0)
                match = re.search(r"\b\d+\b", content)
                if match:
                    idx_val = int(match.group())
                    if 0 <= idx_val < len(child_nodes):
                        chosen_idx = idx_val
            except Exception as e:
                logger.debug("Tree navigation LLM selection fallback to 0: %s", e)
                chosen_idx = 0

            chosen_node = child_nodes[chosen_idx]
            if chosen_node.chunk_id:
                leaf_chunk_ids.append(chosen_node.chunk_id)
            current_node_id = chosen_node.id

        return leaf_chunk_ids
