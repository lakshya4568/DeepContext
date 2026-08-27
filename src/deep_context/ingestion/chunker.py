"""Parent-child hierarchical chunker implementing FR2."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from deep_context.core.config import settings
from deep_context.core.types import Chunk, ChunkLevel
from deep_context.ingestion.parser import ParsedSection, count_approx_tokens


class ParentChildChunker:
    """Creates searchable child chunks (150-300 tokens) linked to context parent chunks (1000-2500 tokens)."""

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

    def chunk_sections(
        self, document_id: str, sections: list[ParsedSection]
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Takes parsed sections and returns (parent_chunks, child_chunks).
        Child chunks have parent_chunk_id set to their enclosing parent chunk's ID.
        """
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []

        now = datetime.now(timezone.utc)

        # 1. Group sections into parent chunks
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

        # 2. For each parent group, create Parent Chunk and Child Chunks
        for group in parent_groups:
            parent_id = str(uuid.uuid4())
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

            parent_chunk = Chunk(
                id=parent_id,
                document_id=document_id,
                parent_chunk_id=None,
                level=ChunkLevel.PARENT,
                content=parent_text,
                token_count=parent_tokens,
                section_path=section_path,
                page_number=start_page,
                embedding=None,  # Parents not embedded by default
                created_at=now,
            )
            parent_chunks.append(parent_chunk)

            # Generate child chunks from this parent with exact page tracking
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
        """Split parent text into overlapping child chunks (150-300 tokens) with exact page resolution."""
        # Split by paragraphs or sentence boundaries
        units = [p.strip() for p in re.split(r"(?<=\n\n)|(?<=\. )", parent_text) if p.strip()]
        if not units:
            units = [parent_text]

        children: list[Chunk] = []
        current_units: list[str] = []
        current_tokens = 0

        overlap_tokens = int(self.child_max_tokens * self.overlap_pct)

        def resolve_page_number(text: str) -> int | None:
            """Find which section/page this text segment belongs to."""
            sample = text[:80].strip()
            for s in sections:
                if sample in s.content:
                    return s.page_number
            return sections[0].page_number if sections else None

        for unit in units:
            u_tokens = count_approx_tokens(unit)
            if current_tokens + u_tokens > self.child_max_tokens and current_units:
                child_content = " ".join(current_units).strip()
                page_num = resolve_page_number(child_content)
                child_chunk = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    parent_chunk_id=parent_id,
                    level=ChunkLevel.CHILD,
                    content=child_content,
                    token_count=count_approx_tokens(child_content),
                    section_path=f"Page {page_num}" if page_num else section_path,
                    page_number=page_num,
                    created_at=created_at,
                )
                children.append(child_chunk)

                # Keep overlap for next window
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
            page_num = resolve_page_number(child_content)
            child_chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                parent_chunk_id=parent_id,
                level=ChunkLevel.CHILD,
                content=child_content,
                token_count=count_approx_tokens(child_content),
                section_path=f"Page {page_num}" if page_num else section_path,
                page_number=page_num,
                created_at=created_at,
            )
            children.append(child_chunk)

        return children


# Convenience Alias matching standard RAG naming conventions
RecursiveChunker = ParentChildChunker
