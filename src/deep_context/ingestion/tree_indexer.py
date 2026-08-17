"""Hierarchical tree indexer for vectorless document navigation (FR6)."""

from __future__ import annotations

import uuid

from deep_context.core.types import Chunk, DocumentTreeNode
from deep_context.ingestion.parser import ParsedSection


class DocumentTreeIndexer:
    """Builds hierarchical DocumentTreeNode records for vectorless/PageIndex-style navigation."""

    @classmethod
    def build_tree_nodes(
        cls,
        document_id: str,
        sections: list[ParsedSection],
        parent_chunks: list[Chunk],
    ) -> list[DocumentTreeNode]:
        """
        Constructs a tree hierarchy from parsed sections and maps leaves to parent chunks.
        """
        nodes: list[DocumentTreeNode] = []

        # Create root node
        root_id = str(uuid.uuid4())
        root_node = DocumentTreeNode(
            id=root_id,
            document_id=document_id,
            parent_node_id=None,
            title="Root Table of Contents",
            summary=f"Table of contents with {len(sections)} main sections.",
            chunk_id=None,
            node_order=0,
        )
        nodes.append(root_node)

        # Create section nodes
        for idx, sec in enumerate(sections, start=1):
            sec_id = str(uuid.uuid4())
            # Find matching parent chunk if any
            matching_chunk_id = None
            for p in parent_chunks:
                if sec.content[:80] in p.content:
                    matching_chunk_id = p.id
                    break
            if not matching_chunk_id and parent_chunks:
                matching_chunk_id = parent_chunks[min(idx - 1, len(parent_chunks) - 1)].id

            summary = (
                f"Section covers: {sec.title}. " + sec.content[:150].replace("\n", " ") + "..."
            )

            node = DocumentTreeNode(
                id=sec_id,
                document_id=document_id,
                parent_node_id=root_id,
                title=sec.title,
                summary=summary,
                chunk_id=matching_chunk_id,
                node_order=idx,
            )
            nodes.append(node)

        return nodes
