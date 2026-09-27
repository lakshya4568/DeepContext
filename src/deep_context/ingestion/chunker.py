"""Structure-aware hierarchical parent-child chunker implementing FR2."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from deep_context.core.config import settings
from deep_context.core.types import (
    Chunk,
    ChunkLevel,
    DocumentElementType,
    DocumentNode,
    DocumentTree,
    ParsedSection,
)
from deep_context.ingestion.parser import count_approx_tokens


def generate_stable_chunk_id(parent_or_doc_id: str, prefix: str, index: int | str) -> str:
    """Generate deterministic UUIDv5 for chunk based on document/parent ID and sequence index."""
    seed = f"{parent_or_doc_id}:{prefix}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


class ParentChildChunker:
    """Structure-aware parent-child chunker preserving sections, tables, captions, and figures."""

    def __init__(
        self,
        child_min_tokens: int | None = None,
        child_max_tokens: int | None = None,
        parent_min_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        overlap_pct: float | None = None,
    ):
        self.child_min_tokens = child_min_tokens or settings.child_chunk_min_tokens
        self.child_max_tokens = child_max_tokens or settings.child_chunk_max_tokens
        self.parent_min_tokens = parent_min_tokens or settings.parent_chunk_min_tokens
        self.parent_max_tokens = parent_max_tokens or settings.parent_chunk_max_tokens
        self.overlap_pct = overlap_pct or settings.chunk_overlap_percentage

    def chunk_tree(self, tree: DocumentTree) -> tuple[list[Chunk], list[Chunk]]:
        """
        Creates structure-aware parent chunks and child chunks from a normalized DocumentTree.
        Preserves natural boundaries: sections, paragraphs, tables, and figures with captions.
        """
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []
        now = datetime.now(timezone.utc)

        # Filter out DOCUMENT root node
        content_nodes = [
            n
            for n in sorted(tree.nodes, key=lambda x: x.reading_order)
            if n.node_type != DocumentElementType.DOCUMENT
        ]
        if not content_nodes:
            # Fallback for empty document
            parent_id = generate_stable_chunk_id(tree.document_id, "parent", 0)
            p = Chunk(
                id=parent_id,
                document_id=tree.document_id,
                level=ChunkLevel.PARENT,
                content=tree.title or "(Empty Document)",
                token_count=count_approx_tokens(tree.title or ""),
                section_path=tree.title or "Document",
                created_at=now,
            )
            c = Chunk(
                id=generate_stable_chunk_id(parent_id, "child", 0),
                document_id=tree.document_id,
                parent_chunk_id=parent_id,
                level=ChunkLevel.CHILD,
                content=tree.title or "(Empty Document)",
                token_count=count_approx_tokens(tree.title or ""),
                section_path=tree.title or "Document",
                created_at=now,
            )
            return [p], [c]

        # 1. Group nodes into logical sections/parent units
        parent_node_groups: list[list[DocumentNode]] = []
        current_group: list[DocumentNode] = []
        current_tokens = 0

        for node in content_nodes:
            # If node is a top-level heading and current group has content, check boundary
            is_major_heading = node.node_type in (
                DocumentElementType.TITLE,
                DocumentElementType.HEADING,
                DocumentElementType.SECTION,
            )
            n_tokens = count_approx_tokens(node.text)

            if is_major_heading and current_tokens >= self.parent_min_tokens and current_group:
                parent_node_groups.append(current_group)
                current_group = [node]
                current_tokens = n_tokens
            elif current_tokens + n_tokens > self.parent_max_tokens and current_group:
                parent_node_groups.append(current_group)
                current_group = [node]
                current_tokens = n_tokens
            else:
                current_group.append(node)
                current_tokens += n_tokens

        if current_group:
            parent_node_groups.append(current_group)

        # 2. For each parent group, generate parent chunk and child chunks
        for p_idx, group in enumerate(parent_node_groups):
            parent_id = generate_stable_chunk_id(tree.document_id, "parent", p_idx)
            pages = [
                n.provenance.page_number for n in group if n.provenance.page_number is not None
            ]
            start_page = min(pages) if pages else None
            end_page = max(pages) if pages else None

            # Format section path
            first_heading = next(
                (
                    n
                    for n in group
                    if n.node_type
                    in (
                        DocumentElementType.TITLE,
                        DocumentElementType.HEADING,
                        DocumentElementType.SECTION,
                    )
                ),
                None,
            )
            if first_heading and first_heading.section_path:
                sec_path = first_heading.section_path
            elif start_page and end_page and start_page != end_page:
                sec_path = f"Pages {start_page}–{end_page}"
            elif start_page:
                sec_path = f"Page {start_page}"
            else:
                sec_path = group[0].section_path or tree.title or "Document"

            parent_text_parts: list[str] = []
            for n in group:
                if n.node_type == DocumentElementType.TABLE and n.table_data:
                    cap = f"*{n.table_data.caption}*\n" if n.table_data.caption else ""
                    parent_text_parts.append(f"{cap}{n.table_data.markdown}")
                elif (
                    n.node_type in (DocumentElementType.FIGURE, DocumentElementType.CHART)
                    and n.figure_data
                ):
                    fig_desc = (
                        f"Figure: {n.figure_data.caption}"
                        if n.figure_data.caption
                        else f"[{n.node_type.value}]"
                    )
                    if n.figure_data.enrichment and n.figure_data.enrichment.description:
                        fig_desc += f"\nDescription: {n.figure_data.enrichment.description}"
                    parent_text_parts.append(fig_desc)
                elif n.node_type == DocumentElementType.EQUATION and n.equation_data:
                    parent_text_parts.append(n.text)
                elif n.text:
                    parent_text_parts.append(n.text)

            parent_content = "\n\n".join(parent_text_parts).strip()
            parent_tokens = count_approx_tokens(parent_content)

            group_asset_ids = [
                n.asset_id or (n.figure_data.asset_id if n.figure_data else None)
                for n in group
                if (n.asset_id or (n.figure_data and n.figure_data.asset_id))
            ]

            parent_meta: dict[str, Any] = {
                "document_id": tree.document_id,
                "node_ids": [n.id for n in group],
                "element_types": list({n.node_type.value for n in group}),
                "page_range": [start_page, end_page] if start_page else None,
                "asset_ids": [aid for aid in group_asset_ids if aid],
            }

            parent_chunk = Chunk(
                id=parent_id,
                document_id=tree.document_id,
                parent_chunk_id=None,
                level=ChunkLevel.PARENT,
                content=parent_content,
                token_count=parent_tokens,
                section_path=sec_path,
                page_number=start_page,
                metadata=parent_meta,
                created_at=now,
            )
            parent_chunks.append(parent_chunk)

            # Generate child chunks preserving elements
            children = self._split_nodes_into_children(
                parent_id=parent_id,
                tree=tree,
                nodes=group,
                default_section_path=sec_path,
                created_at=now,
            )
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _split_nodes_into_children(
        self,
        parent_id: str,
        tree: DocumentTree,
        nodes: list[DocumentNode],
        default_section_path: str,
        created_at: datetime,
    ) -> list[Chunk]:
        """Splits nodes into structure-aware child chunks, keeping tables and figures intact with captions."""
        children: list[Chunk] = []

        curr_text_buffer: list[str] = []
        curr_node_ids: list[str] = []
        curr_element_types: list[str] = []
        curr_tokens = 0
        curr_start_page: int | None = None
        curr_end_page: int | None = None
        curr_sec_path = default_section_path

        def flush_buffer() -> None:
            nonlocal \
                curr_text_buffer, \
                curr_node_ids, \
                curr_element_types, \
                curr_tokens, \
                curr_start_page, \
                curr_end_page, \
                curr_sec_path
            if not curr_text_buffer:
                return

            text_content = "\n\n".join(curr_text_buffer).strip()
            if not text_content:
                curr_text_buffer = []
                curr_node_ids = []
                curr_element_types = []
                curr_tokens = 0
                return

            c_meta: dict[str, Any] = {
                "document_id": tree.document_id,
                "node_ids": list(curr_node_ids),
                "element_types": list(set(curr_element_types)),
                "page_range": [curr_start_page, curr_end_page] if curr_start_page else None,
            }

            chunk = Chunk(
                id=generate_stable_chunk_id(parent_id, "child", len(children)),
                document_id=tree.document_id,
                parent_chunk_id=parent_id,
                level=ChunkLevel.CHILD,
                content=text_content,
                token_count=count_approx_tokens(text_content),
                section_path=curr_sec_path,
                page_number=curr_start_page,
                metadata=c_meta,
                created_at=created_at,
            )
            children.append(chunk)

            curr_text_buffer = []
            curr_node_ids = []
            curr_element_types = []
            curr_tokens = 0
            curr_start_page = None
            curr_end_page = None

        for node in nodes:
            n_type = node.node_type
            page = node.provenance.page_number
            node_sec = node.section_path or default_section_path

            # 1. TABLE: Keep table and its caption together
            if n_type == DocumentElementType.TABLE and node.table_data:
                flush_buffer()
                t_data = node.table_data
                cap_str = f"Table: {t_data.caption}\n\n" if t_data.caption else ""
                table_full_text = f"{cap_str}{t_data.markdown}".strip()
                t_tokens = count_approx_tokens(table_full_text)

                t_meta: dict[str, Any] = {
                    "document_id": tree.document_id,
                    "node_ids": [node.id],
                    "element_types": ["table"],
                    "has_table": True,
                    "num_rows": t_data.num_rows,
                    "num_cols": t_data.num_cols,
                    "table_caption": t_data.caption,
                    "page_range": [min(t_data.page_numbers), max(t_data.page_numbers)]
                    if t_data.page_numbers
                    else ([page, page] if page else None),
                }

                # Check if oversized table requiring row split
                if t_tokens > self.child_max_tokens and len(t_data.rows) > 3:
                    # Deliberately split table rows while preserving headers and caption reference
                    headers = t_data.headers
                    header_line = "| " + " | ".join(headers) + " |" if headers else ""
                    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |" if headers else ""
                    data_rows = (
                        t_data.rows[1:]
                        if headers and t_data.rows and t_data.rows[0] == headers
                        else t_data.rows
                    )

                    row_batch_size = max(
                        2, len(data_rows) // max(2, t_tokens // self.child_max_tokens + 1)
                    )
                    split_idx = 1
                    total_splits = (len(data_rows) + row_batch_size - 1) // row_batch_size

                    for i in range(0, len(data_rows), row_batch_size):
                        batch = data_rows[i : i + row_batch_size]
                        b_lines = ["| " + " | ".join(r) + " |" for r in batch]
                        table_part_md = "\n".join(
                            ([header_line, sep_line] if header_line else []) + b_lines
                        )
                        part_content = (
                            f"{cap_str}(Part {split_idx} of {total_splits})\n{table_part_md}"
                            if cap_str
                            else table_part_md
                        )

                        part_meta = dict(t_meta)
                        part_meta["split_index"] = split_idx
                        part_meta["total_splits"] = total_splits

                        children.append(
                            Chunk(
                                id=generate_stable_chunk_id(parent_id, "child", len(children)),
                                document_id=tree.document_id,
                                parent_chunk_id=parent_id,
                                level=ChunkLevel.CHILD,
                                content=part_content,
                                token_count=count_approx_tokens(part_content),
                                section_path=node_sec,
                                page_number=page,
                                metadata=part_meta,
                                created_at=created_at,
                            )
                        )
                        split_idx += 1
                else:
                    children.append(
                        Chunk(
                            id=generate_stable_chunk_id(parent_id, "child", len(children)),
                            document_id=tree.document_id,
                            parent_chunk_id=parent_id,
                            level=ChunkLevel.CHILD,
                            content=table_full_text,
                            token_count=t_tokens,
                            section_path=node_sec,
                            page_number=page,
                            metadata=t_meta,
                            created_at=created_at,
                        )
                    )
                continue

            # 2. FIGURE or CHART: Keep figure, caption, and enrichment together
            if (
                n_type in (DocumentElementType.FIGURE, DocumentElementType.CHART)
                and node.figure_data
            ):
                flush_buffer()
                f_data = node.figure_data
                is_chart = n_type == DocumentElementType.CHART or f_data.figure_type == "chart"
                label_name = "Chart" if is_chart else "Figure"
                parts = [f"{label_name}: {f_data.caption}" if f_data.caption else f"[{label_name}]"]

                if f_data.enrichment and f_data.enrichment.description:
                    parts.append(f"Visual Interpretation: {f_data.enrichment.description}")
                if f_data.ocr_text:
                    parts.append(f"Extracted Text: {f_data.ocr_text}")

                fig_content = "\n\n".join(parts)
                a_ids = (
                    [f_data.asset_id]
                    if f_data.asset_id
                    else ([node.asset_id] if node.asset_id else [])
                )
                f_meta: dict[str, Any] = {
                    "document_id": tree.document_id,
                    "node_ids": [node.id],
                    "element_types": [n_type.value],
                    "has_figure": True,
                    "has_chart": is_chart,
                    "figure_type": f_data.figure_type,
                    "asset_id": f_data.asset_id or node.asset_id,
                    "asset_ids": a_ids,
                    "caption": f_data.caption,
                    "enrichment_status": f_data.enrichment.status
                    if f_data.enrichment
                    else "not_processed",
                    "page_range": [page, page] if page else None,
                }

                children.append(
                    Chunk(
                        id=generate_stable_chunk_id(parent_id, "child", len(children)),
                        document_id=tree.document_id,
                        parent_chunk_id=parent_id,
                        level=ChunkLevel.CHILD,
                        content=fig_content,
                        token_count=count_approx_tokens(fig_content),
                        section_path=node_sec,
                        page_number=page,
                        metadata=f_meta,
                        created_at=created_at,
                    )
                )
                continue

            # 2.5. EQUATION: Keep equation, LaTeX, symbols, and readable text together
            if n_type == DocumentElementType.EQUATION and node.equation_data:
                flush_buffer()
                eq_data = node.equation_data
                eq_parts = [node.text]
                if eq_data.variables:
                    eq_parts.append(f"Symbols/Variables: {', '.join(eq_data.variables)}")
                if eq_data.symbolic_repr:
                    eq_parts.append(f"Symbolic Form: {eq_data.symbolic_repr}")
                eq_content = "\n\n".join(eq_parts)

                eq_a_ids = (
                    [eq_data.asset_id]
                    if eq_data.asset_id
                    else ([node.asset_id] if node.asset_id else [])
                )
                eq_meta: dict[str, Any] = {
                    "document_id": tree.document_id,
                    "node_ids": [node.id],
                    "element_types": ["equation"],
                    "has_equation": True,
                    "latex": eq_data.latex,
                    "equation_number": eq_data.equation_number,
                    "variables": eq_data.variables,
                    "asset_id": eq_data.asset_id or node.asset_id,
                    "asset_ids": eq_a_ids,
                    "page_range": [page, page] if page else None,
                }
                children.append(
                    Chunk(
                        id=generate_stable_chunk_id(parent_id, "child", len(children)),
                        document_id=tree.document_id,
                        parent_chunk_id=parent_id,
                        level=ChunkLevel.CHILD,
                        content=eq_content,
                        token_count=count_approx_tokens(eq_content),
                        section_path=node_sec,
                        page_number=page,
                        metadata=eq_meta,
                        created_at=created_at,
                    )
                )
                continue

            # 3. TEXT / PARAGRAPH / CODE / LIST_ITEM
            node_txt = node.text
            if not node_txt:
                continue

            node_tok = count_approx_tokens(node_txt)

            # Check if this single text element is oversized
            if node_tok > self.child_max_tokens:
                flush_buffer()
                # Split along sentence or paragraph boundaries
                sub_units = [
                    u.strip() for u in re.split(r"(?<=\n\n)|(?<=\. )", node_txt) if u.strip()
                ]
                if not sub_units:
                    sub_units = [node_txt]

                sub_buf: list[str] = []
                sub_tok = 0
                sub_split_idx = 1
                for u in sub_units:
                    u_tok = count_approx_tokens(u)
                    if sub_tok + u_tok > self.child_max_tokens and sub_buf:
                        p_txt = " ".join(sub_buf).strip()
                        children.append(
                            Chunk(
                                id=generate_stable_chunk_id(parent_id, "child", len(children)),
                                document_id=tree.document_id,
                                parent_chunk_id=parent_id,
                                level=ChunkLevel.CHILD,
                                content=p_txt,
                                token_count=count_approx_tokens(p_txt),
                                section_path=node_sec,
                                page_number=page,
                                metadata={
                                    "document_id": tree.document_id,
                                    "node_ids": [node.id],
                                    "element_types": [n_type.value],
                                    "split_index": sub_split_idx,
                                    "page_range": [page, page] if page else None,
                                },
                                created_at=created_at,
                            )
                        )
                        sub_split_idx += 1
                        sub_buf = [u]
                        sub_tok = u_tok
                    else:
                        sub_buf.append(u)
                        sub_tok += u_tok

                if sub_buf:
                    p_txt = " ".join(sub_buf).strip()
                    children.append(
                        Chunk(
                            id=generate_stable_chunk_id(parent_id, "child", len(children)),
                            document_id=tree.document_id,
                            parent_chunk_id=parent_id,
                            level=ChunkLevel.CHILD,
                            content=p_txt,
                            token_count=count_approx_tokens(p_txt),
                            section_path=node_sec,
                            page_number=page,
                            metadata={
                                "document_id": tree.document_id,
                                "node_ids": [node.id],
                                "element_types": [n_type.value],
                                "split_index": sub_split_idx,
                                "page_range": [page, page] if page else None,
                            },
                            created_at=created_at,
                        )
                    )
                continue

            # Accumulate into current buffer
            if curr_tokens + node_tok > self.child_max_tokens and curr_text_buffer:
                flush_buffer()

            curr_text_buffer.append(node_txt)
            curr_node_ids.append(node.id)
            curr_element_types.append(n_type.value)
            curr_tokens += node_tok
            if page:
                if curr_start_page is None:
                    curr_start_page = page
                curr_end_page = page
            curr_sec_path = node_sec

        flush_buffer()
        return children

    def chunk_sections(
        self, document_id: str, sections: list[ParsedSection]
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Backward-compatible method taking parsed sections and returning (parent_chunks, child_chunks).
        """
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []
        now = datetime.now(timezone.utc)

        parent_groups: list[list[ParsedSection]] = []
        current_group: list[ParsedSection] = []
        current_tokens = 0

        for sec in sections:
            sec_tokens = count_approx_tokens(sec.content)
            if current_tokens + sec_tokens > self.parent_max_tokens and current_group:
                parent_groups.append(current_group)
                current_group = [sec]
                current_tokens = sec_tokens
            else:
                current_group.append(sec)
                current_tokens += sec_tokens

        if current_group:
            parent_groups.append(current_group)

        for p_idx, group in enumerate(parent_groups):
            parent_id = generate_stable_chunk_id(document_id, "parent", p_idx)
            parent_text = "\n\n".join(s.content for s in group)
            parent_tokens = count_approx_tokens(parent_text)

            start_page = group[0].page_number if group else None
            end_page = group[-1].page_number if group else None

            if start_page and end_page and start_page != end_page:
                section_path = f"Pages {start_page}–{end_page}"
            elif start_page:
                section_path = f"Page {start_page}"
            else:
                section_path = group[0].section_path if group else "Document"

            node_ids = [s.metadata.get("node_id") for s in group if s.metadata.get("node_id")]
            elem_types = [s.metadata.get("node_type") for s in group if s.metadata.get("node_type")]

            parent_chunk = Chunk(
                id=parent_id,
                document_id=document_id,
                parent_chunk_id=None,
                level=ChunkLevel.PARENT,
                content=parent_text,
                token_count=parent_tokens,
                section_path=section_path,
                page_number=start_page,
                embedding=None,
                metadata={
                    "document_id": document_id,
                    "node_ids": node_ids,
                    "element_types": elem_types,
                    "page_range": [start_page, end_page] if start_page else None,
                },
                created_at=now,
            )
            parent_chunks.append(parent_chunk)

            children = self._split_parent_into_children(
                parent_id=parent_id,
                document_id=document_id,
                parent_text=parent_text,
                section_path=section_path,
                sections=group,
                created_at=now,
            )
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _split_parent_into_children(
        self,
        parent_id: str,
        document_id: str,
        parent_text: str,
        section_path: str,
        sections: list[ParsedSection],
        created_at: datetime,
    ) -> list[Chunk]:
        """Split parent text into overlapping child chunks with exact page resolution."""
        units = [p.strip() for p in re.split(r"(?<=\n\n)|(?<=\. )", parent_text) if p.strip()]
        if not units:
            units = [parent_text]

        children: list[Chunk] = []
        current_units: list[str] = []
        current_tokens = 0
        overlap_tokens = int(self.child_max_tokens * self.overlap_pct)

        def resolve_page_and_node(text: str) -> tuple[int | None, list[str]]:
            sample = text[:80].strip()
            for s in sections:
                if sample in s.content:
                    nid = [s.metadata["node_id"]] if "node_id" in s.metadata else []
                    return s.page_number, nid
            if sections:
                nid = [sections[0].metadata["node_id"]] if "node_id" in sections[0].metadata else []
                return sections[0].page_number, nid
            return None, []

        for unit in units:
            u_tokens = count_approx_tokens(unit)
            if current_tokens + u_tokens > self.child_max_tokens and current_units:
                child_content = " ".join(current_units).strip()
                page_num, node_ids = resolve_page_and_node(child_content)
                has_tbl = "|" in child_content and "\n" in child_content
                has_fig = "figure:" in child_content.lower() or "fig." in child_content.lower()

                child_chunk = Chunk(
                    id=generate_stable_chunk_id(parent_id, "child", len(children)),
                    document_id=document_id,
                    parent_chunk_id=parent_id,
                    level=ChunkLevel.CHILD,
                    content=child_content,
                    token_count=count_approx_tokens(child_content),
                    section_path=f"Page {page_num}" if page_num else section_path,
                    page_number=page_num,
                    metadata={
                        "document_id": document_id,
                        "node_ids": node_ids,
                        "has_table": has_tbl,
                        "has_figure": has_fig,
                        "page_range": [page_num, page_num] if page_num else None,
                    },
                    created_at=created_at,
                )
                children.append(child_chunk)

                overlap_units: list[str] = []
                acc = 0
                for rev_u in reversed(current_units):
                    acc += count_approx_tokens(rev_u)
                    overlap_units.insert(0, rev_u)
                    if acc >= overlap_tokens:
                        break

                current_units = overlap_units + [unit]
                current_tokens = sum(count_approx_tokens(x) for x in current_units)
            else:
                current_units.append(unit)
                current_tokens += u_tokens

        if current_units:
            child_content = " ".join(current_units).strip()
            page_num, node_ids = resolve_page_and_node(child_content)
            has_tbl = "|" in child_content and "\n" in child_content
            has_fig = "figure:" in child_content.lower() or "fig." in child_content.lower()

            child_chunk = Chunk(
                id=generate_stable_chunk_id(parent_id, "child", len(children)),
                document_id=document_id,
                parent_chunk_id=parent_id,
                level=ChunkLevel.CHILD,
                content=child_content,
                token_count=count_approx_tokens(child_content),
                section_path=f"Page {page_num}" if page_num else section_path,
                page_number=page_num,
                metadata={
                    "document_id": document_id,
                    "node_ids": node_ids,
                    "has_table": has_tbl,
                    "has_figure": has_fig,
                    "page_range": [page_num, page_num] if page_num else None,
                },
                created_at=created_at,
            )
            children.append(child_chunk)

        return children


RecursiveChunker = ParentChildChunker
