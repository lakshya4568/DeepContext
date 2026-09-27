"""Domain models, enums, and types for the Deep Context Platform."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Retrieval & Ingestion Enums & Models
# ---------------------------------------------------------------------------


class QueryShape(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    HOW_TO = "how_to"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    NAVIGATION = "navigation"


class RetrievalMode(str, Enum):
    HYBRID = "hybrid"


class ChunkLevel(str, Enum):
    PARENT = "parent"
    CHILD = "child"


class RoutingPath(str, Enum):
    HYBRID_RAG = "hybrid_rag"
    AGENTIC_PLANNER = "agentic_planner"


@dataclass
class RetrievalFilters:
    tenant_id: str = "default"
    permission_scope: list[str] = field(default_factory=lambda: ["default"])
    document_ids: list[str] | None = None
    date_range: tuple[str, str] | None = None
    doc_types: list[str] | None = None
    section_prefix: str | None = None


class DocumentElementType(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    CHART = "chart"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    CODE = "code"
    OTHER = "other"


@dataclass
class Provenance:
    source_uri: str | None = None
    page_number: int | None = None
    page_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # (left, top, right, bottom)
    char_span: tuple[int, int] | None = None
    raw_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "bbox": list(self.bbox) if self.bbox else None,
            "char_span": list(self.char_span) if self.char_span else None,
            "raw_ref": self.raw_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Provenance:
        if not data:
            return cls()
        bbox = tuple(data["bbox"]) if data.get("bbox") else None
        char_span = tuple(data["char_span"]) if data.get("char_span") else None
        return cls(
            source_uri=data.get("source_uri"),
            page_number=data.get("page_number"),
            page_end=data.get("page_end"),
            bbox=bbox,  # type: ignore[arg-type]
            char_span=char_span,  # type: ignore[arg-type]
            raw_ref=data.get("raw_ref"),
        )


@dataclass
class TableCellData:
    row_idx: int
    col_idx: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    is_header: bool = False
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_idx": self.row_idx,
            "col_idx": self.col_idx,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "text": self.text,
            "is_header": self.is_header,
            "bbox": list(self.bbox) if self.bbox else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableCellData:
        bbox = tuple(data["bbox"]) if data.get("bbox") else None
        return cls(
            row_idx=data["row_idx"],
            col_idx=data["col_idx"],
            row_span=data.get("row_span", 1),
            col_span=data.get("col_span", 1),
            text=data.get("text", ""),
            is_header=data.get("is_header", False),
            bbox=bbox,  # type: ignore[arg-type]
        )


@dataclass
class TableDataModel:
    num_rows: int
    num_cols: int
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    cells: list[TableCellData] = field(default_factory=list)
    caption: str | None = None
    markdown: str = ""
    html: str | None = None
    page_numbers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "headers": self.headers,
            "rows": self.rows,
            "cells": [c.to_dict() for c in self.cells],
            "caption": self.caption,
            "markdown": self.markdown,
            "html": self.html,
            "page_numbers": self.page_numbers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TableDataModel | None:
        if not data:
            return None
        cells = [TableCellData.from_dict(c) for c in data.get("cells", [])]
        return cls(
            num_rows=data.get("num_rows", 0),
            num_cols=data.get("num_cols", 0),
            headers=data.get("headers", []),
            rows=data.get("rows", []),
            cells=cells,
            caption=data.get("caption"),
            markdown=data.get("markdown", ""),
            html=data.get("html"),
            page_numbers=data.get("page_numbers", []),
        )


@dataclass
class MultimodalEnrichment:
    status: str = "not_processed"  # "not_processed" | "completed" | "failed" | "skipped"
    model_name: str | None = None
    description: str | None = None  # Distinguish generated descriptions from source text
    extracted_labels: list[str] = field(default_factory=list)
    extracted_values: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    limitations: str | None = None
    processed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_name": self.model_name,
            "description": self.description,
            "extracted_labels": self.extracted_labels,
            "extracted_values": self.extracted_values,
            "confidence": self.confidence,
            "limitations": self.limitations,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MultimodalEnrichment | None:
        if not data:
            return None
        proc_at = datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None
        return cls(
            status=data.get("status", "not_processed"),
            model_name=data.get("model_name"),
            description=data.get("description"),
            extracted_labels=data.get("extracted_labels", []),
            extracted_values=data.get("extracted_values", []),
            confidence=data.get("confidence"),
            limitations=data.get("limitations"),
            processed_at=proc_at,
        )


@dataclass
class FigureDataModel:
    asset_id: str | None = None
    caption: str | None = None
    figure_type: str = "image"  # "image" | "chart" | "diagram" | "unknown"
    chart_metadata: dict[str, Any] = field(default_factory=dict)
    ocr_text: str | None = None
    enrichment: MultimodalEnrichment | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "caption": self.caption,
            "figure_type": self.figure_type,
            "chart_metadata": self.chart_metadata,
            "ocr_text": self.ocr_text,
            "enrichment": self.enrichment.to_dict() if self.enrichment else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigureDataModel | None:
        if not data:
            return None
        enrichment = (
            MultimodalEnrichment.from_dict(data["enrichment"]) if data.get("enrichment") else None
        )
        return cls(
            asset_id=data.get("asset_id"),
            caption=data.get("caption"),
            figure_type=data.get("figure_type", "image"),
            chart_metadata=data.get("chart_metadata", {}),
            ocr_text=data.get("ocr_text"),
            enrichment=enrichment,
        )


@dataclass
class DocumentNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    parent_id: str | None = None
    node_type: DocumentElementType = DocumentElementType.PARAGRAPH
    reading_order: int = 0
    text: str = ""
    raw_text: str | None = None
    section_path: str | None = None
    children_ids: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    table_data: TableDataModel | None = None
    figure_data: FigureDataModel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "parent_id": self.parent_id,
            "node_type": self.node_type.value
            if hasattr(self.node_type, "value")
            else str(self.node_type),
            "reading_order": self.reading_order,
            "text": self.text,
            "raw_text": self.raw_text,
            "section_path": self.section_path,
            "children_ids": self.children_ids,
            "provenance": self.provenance.to_dict(),
            "table_data": self.table_data.to_dict() if self.table_data else None,
            "figure_data": self.figure_data.to_dict() if self.figure_data else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentNode:
        return cls(
            id=data["id"],
            document_id=data.get("document_id", ""),
            parent_id=data.get("parent_id"),
            node_type=DocumentElementType(data.get("node_type", "paragraph")),
            reading_order=data.get("reading_order", 0),
            text=data.get("text", ""),
            raw_text=data.get("raw_text"),
            section_path=data.get("section_path"),
            children_ids=data.get("children_ids", []),
            provenance=Provenance.from_dict(data.get("provenance")),
            table_data=TableDataModel.from_dict(data.get("table_data")),
            figure_data=FigureDataModel.from_dict(data.get("figure_data")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ParsedSection:
    title: str
    content: str
    section_path: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.title, str) and "\x00" in self.title:
            self.title = self.title.replace("\x00", "")
        if isinstance(self.content, str) and "\x00" in self.content:
            self.content = self.content.replace("\x00", "")
        if isinstance(self.section_path, str) and "\x00" in self.section_path:
            self.section_path = self.section_path.replace("\x00", "")


@dataclass
class DocumentTree:
    document_id: str
    title: str = ""
    source_uri: str | None = None
    doc_type: str = "pdf"
    nodes: list[DocumentNode] = field(default_factory=list)
    root_node_id: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def node_map(self) -> dict[str, DocumentNode]:
        return {n.id: n for n in self.nodes}

    def get_node(self, node_id: str) -> DocumentNode | None:
        return self.node_map.get(node_id)

    def get_children(self, node_id: str) -> list[DocumentNode]:
        node = self.get_node(node_id)
        if not node:
            return []
        if node.children_ids:
            return [self.node_map[cid] for cid in node.children_ids if cid in self.node_map]
        return [n for n in self.nodes if n.parent_id == node_id]

    def get_parent(self, node_id: str) -> DocumentNode | None:
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return None
        return self.get_node(node.parent_id)

    def get_ancestors(self, node_id: str) -> list[DocumentNode]:
        ancestors: list[DocumentNode] = []
        curr = self.get_parent(node_id)
        while curr:
            ancestors.append(curr)
            curr = self.get_parent(curr.id)
        return ancestors

    def render_canonical_text(self) -> str:
        """Renders elements in reading order with headings, captions, tables, and page markers."""
        sorted_nodes = sorted(self.nodes, key=lambda n: n.reading_order)
        blocks: list[str] = []
        for n in sorted_nodes:
            if n.node_type == DocumentElementType.DOCUMENT:
                continue
            if n.node_type in (DocumentElementType.HEADING, DocumentElementType.SECTION):
                blocks.append(f"\n## {n.text}\n")
            elif n.node_type == DocumentElementType.TABLE and n.table_data:
                caption = f"\n*{n.table_data.caption}*\n" if n.table_data.caption else ""
                blocks.append(f"{caption}{n.table_data.markdown}")
            elif (
                n.node_type in (DocumentElementType.FIGURE, DocumentElementType.CHART)
                and n.figure_data
            ):
                caption = (
                    f"\n*Figure Caption: {n.figure_data.caption}*\n"
                    if n.figure_data.caption
                    else ""
                )
                enrichment = (
                    f"\n[Visual Interpretation: {n.figure_data.enrichment.description}]\n"
                    if n.figure_data.enrichment and n.figure_data.enrichment.description
                    else ""
                )
                blocks.append(
                    f"{caption}{enrichment}".strip() or f"[{n.node_type.value.capitalize()}]"
                )
            elif n.text:
                blocks.append(n.text)
        return "\n\n".join(b for b in blocks if b.strip())

    def to_parsed_sections(self) -> list[ParsedSection]:
        """Convert tree into backward-compatible ParsedSections for existing consumers."""
        sections: list[ParsedSection] = []
        sorted_nodes = sorted(self.nodes, key=lambda n: n.reading_order)

        current_title = self.title or "Document"
        current_path = current_title
        current_page: int | None = None
        current_content_lines: list[str] = []
        current_meta: dict[str, Any] = {}

        for n in sorted_nodes:
            if n.node_type == DocumentElementType.DOCUMENT:
                continue
            if n.node_type in (
                DocumentElementType.HEADING,
                DocumentElementType.SECTION,
                DocumentElementType.TITLE,
            ):
                if current_content_lines:
                    meta = dict(self.raw_metadata)
                    meta.update(current_meta)
                    sections.append(
                        ParsedSection(
                            title=current_title,
                            content="\n\n".join(current_content_lines).strip(),
                            section_path=current_path,
                            page_number=current_page,
                            metadata=meta,
                        )
                    )
                    current_content_lines = []
                current_title = n.text
                current_path = n.section_path or n.text
                current_page = n.provenance.page_number
                current_meta = {"node_id": n.id, "node_type": n.node_type.value}
            elif n.node_type == DocumentElementType.CODE and (
                n.metadata.get("symbol") or n.metadata.get("block_index")
            ):
                if current_content_lines:
                    meta = dict(self.raw_metadata)
                    meta.update(current_meta)
                    sections.append(
                        ParsedSection(
                            title=current_title,
                            content="\n\n".join(current_content_lines).strip(),
                            section_path=current_path,
                            page_number=current_page,
                            metadata=meta,
                        )
                    )
                    current_content_lines = []
                kind = n.metadata.get("kind") or "Block"
                sym = n.metadata.get("symbol") or n.metadata.get("first_line", "code")
                sec_title = f"{kind} {sym}".strip()
                code_meta = dict(self.raw_metadata)
                code_meta.update(n.metadata)
                sections.append(
                    ParsedSection(
                        title=sec_title,
                        content=n.text,
                        section_path=n.section_path or sec_title,
                        page_number=n.provenance.page_number or 1,
                        metadata=code_meta,
                    )
                )
            else:
                if n.provenance.page_number and current_page is None:
                    current_page = n.provenance.page_number
                if n.node_type == DocumentElementType.TABLE and n.table_data:
                    cap = f"*{n.table_data.caption}*\n" if n.table_data.caption else ""
                    current_content_lines.append(f"{cap}{n.table_data.markdown}")
                elif (
                    n.node_type in (DocumentElementType.FIGURE, DocumentElementType.CHART)
                    and n.figure_data
                ):
                    fig_text = (
                        f"Figure: {n.figure_data.caption}"
                        if n.figure_data.caption
                        else f"[{n.node_type.value}]"
                    )
                    if n.figure_data.enrichment and n.figure_data.enrichment.description:
                        fig_text += f"\nDescription: {n.figure_data.enrichment.description}"
                    current_content_lines.append(fig_text)
                elif n.text:
                    current_content_lines.append(n.text)

        if current_content_lines:
            meta = dict(self.raw_metadata)
            meta.update(current_meta)
            sections.append(
                ParsedSection(
                    title=current_title,
                    content="\n\n".join(current_content_lines).strip(),
                    section_path=current_path,
                    page_number=current_page,
                    metadata=meta,
                )
            )
        return sections or [
            ParsedSection(
                title=self.title or "Document",
                content="",
                section_path="Document",
                metadata=dict(self.raw_metadata),
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "doc_type": self.doc_type,
            "nodes": [n.to_dict() for n in self.nodes],
            "root_node_id": self.root_node_id,
            "raw_metadata": self.raw_metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentTree:
        nodes = [DocumentNode.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            document_id=data["document_id"],
            title=data.get("title", ""),
            source_uri=data.get("source_uri"),
            doc_type=data.get("doc_type", "pdf"),
            nodes=nodes,
            root_node_id=data.get("root_node_id", ""),
            raw_metadata=data.get("raw_metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(timezone.utc),
        )


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    title: str = ""
    source_uri: str | None = None
    section_path: str | None = None
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "section_path": self.section_path,
            "page_number": self.page_number,
        }


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    title: str = ""
    source_uri: str | None = None
    doc_type: str = "markdown"  # 'pdf' | 'markdown' | 'code' | 'html' | 'text'
    permission_scope: list[str] = field(default_factory=lambda: ["default"])
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if isinstance(self.title, str) and "\x00" in self.title:
            self.title = self.title.replace("\x00", "")
        if isinstance(self.source_uri, str) and "\x00" in self.source_uri:
            self.source_uri = self.source_uri.replace("\x00", "")


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    parent_chunk_id: str | None = None
    level: ChunkLevel = ChunkLevel.CHILD
    content: str = ""
    token_count: int = 0
    section_path: str | None = None
    page_number: int | None = None
    embedding: list[float] | None = None
    summary_text: str | None = None
    summary_tokens: int | None = None
    summary_model: str | None = None
    generated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if isinstance(self.content, str) and "\x00" in self.content:
            self.content = self.content.replace("\x00", "")
        if isinstance(self.section_path, str) and "\x00" in self.section_path:
            self.section_path = self.section_path.replace("\x00", "")
        if isinstance(self.summary_text, str) and "\x00" in self.summary_text:
            self.summary_text = self.summary_text.replace("\x00", "")
        if isinstance(self.summary_model, str) and "\x00" in self.summary_model:
            self.summary_model = self.summary_model.replace("\x00", "")

    @property
    def text(self) -> str:
        """Alias for content to match standard RAG conventions."""
        return self.content

    @text.setter
    def text(self, val: str) -> None:
        self.content = val.replace("\x00", "") if isinstance(val, str) else val

    @property
    def node_ids(self) -> list[str]:
        return self.metadata.get("node_ids", [])

    @property
    def element_types(self) -> list[str]:
        return self.metadata.get("element_types", [])


@dataclass
class RetrievalResult:
    sufficient: bool
    parent_chunks: list[dict[str, Any]] = field(default_factory=list)  # [{content, citation, ...}]
    citations: list[Citation] = field(default_factory=list)
    query_shape: QueryShape | None = None
    retry_count: int = 0
    insufficiency_reason: str | None = None


# ---------------------------------------------------------------------------
# Typed Memory Models
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    POLICY = "policy"
    PREFERENCE = "preference"
    FACT = "fact"
    EPISODE = "episode"
    DISCARD = "discard"


class WriteDecision(str, Enum):
    WRITE = "write"
    REJECT = "reject"
    STAGE = "stage"  # inferred preference awaiting corroboration


@dataclass
class Observation:
    raw_text: str
    tenant_id: str = "default"
    user_id: str | None = None
    source: str = "user_stated"  # 'user_stated' | 'tool_output' | 'inferred' | 'operator'


@dataclass
class ExistingMemory:
    id: str
    content: str
    confidence: float
    created_at: datetime


@dataclass
class PromotionResult:
    decision: WriteDecision
    memory_type: MemoryType
    atomic_claim: str | None = None
    confidence: float | None = None
    expires_at: datetime | None = None
    superseded_id: str | None = None
    reject_reason: str | None = None


# ---------------------------------------------------------------------------
# Verification Models
# ---------------------------------------------------------------------------


class ClaimSupport(str, Enum):
    RETRIEVED = "retrieved"
    COMPUTED = "computed"
    INFERENCE = "inference"
    UNSUPPORTED = "unsupported"


@dataclass
class Claim:
    text: str
    support: ClaimSupport
    evidence_id: str | None = None


@dataclass
class SupportCheckResult:
    passed: bool
    claims: list[Claim] = field(default_factory=list)
    coverage_ratio: float | None = None
    confidence: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Router & Plan Models
# ---------------------------------------------------------------------------


@dataclass
class RouterDecision:
    path: RoutingPath
    query_shape: QueryShape
    reason: str
    estimated_tokens: int = 0
    requires_aggregation: bool = False


# ---------------------------------------------------------------------------
# Pydantic Schemas for API Requests & Responses
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    title: str
    content: str
    doc_type: str = "markdown"  # 'pdf' | 'markdown' | 'code' | 'html' | 'text'
    source_uri: str | None = None
    tenant_id: str = "default"
    permission_scope: list[str] = Field(default_factory=lambda: ["default"])
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    embedding_dim: int | None = None
    generate_summaries: bool | None = None
    enrich_multimodal: bool | None = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    parent_chunks_count: int
    child_chunks_count: int
    retrieval_mode: RetrievalMode
    summaries_generated_count: int = 0
    embedding_model: str | None = None
    embedding_dim: int | None = None
    tree_nodes_count: int = 0


class RetrieveRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    user_id: str | None = None
    permission_scope: list[str] = Field(default_factory=lambda: ["default"])
    document_ids: list[str] | None = None
    top_k: int = 8
    embedding_model: str | None = None
    embedding_dim: int | None = None
    reranker: str | None = None


class RetrieveResponse(BaseModel):
    sufficient: bool
    parent_chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    query_shape: QueryShape
    retry_count: int
    insufficiency_reason: str | None = None
    embedding_model: str | None = None
    reranker: str | None = None
    cache_hit: bool = False


class QueryRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    user_id: str | None = None
    permission_scope: list[str] = Field(default_factory=lambda: ["default"])
    document_ids: list[str] | None = None
    force_path: RoutingPath | None = None
    model: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    reranker: str | None = None
    stream: bool = False


class UserPreferenceRequest(BaseModel):
    user_id: str = "default"
    embedding_model: str | None = None
    embedding_dim: int | None = None
    reranker: str | None = None
    llm_model: str | None = None


class UserPreferenceResponse(BaseModel):
    user_id: str
    embedding_model: str
    embedding_dim: int
    reranker: str
    llm_model: str
    preferences: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    path_taken: RoutingPath
    query_shape: QueryShape
    reasoning: str | None = None
    support_check_passed: bool = True
    support_confidence: float = 1.0
    latency_ms: int = 0
    token_cost: int = 0
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# Needle In A Haystack Diagnostic Benchmark Models
# ---------------------------------------------------------------------------


class HaystackGenerateRequest(BaseModel):
    needle: str = "The secret passcode for project Apollo is DELTA-998822."
    needle_query: str = "What is the secret passcode for project Apollo?"
    total_words: int = 15000  # Default ~50 pages, scalable up to 1000 pages
    depth_percent: float = 50.0  # 0% = top, 50% = middle, 100% = bottom
    topic: str = "Distributed Systems Architecture and Cloud Engineering"


class StageDiagnostic(BaseModel):
    stage_name: str
    needle_found: bool
    needle_rank: int | None = None
    score: float | None = None
    details: str


class HaystackBenchmarkRequest(BaseModel):
    document_id: str | None = None
    needle: str = "DELTA-998822"
    query: str = "What is the secret passcode for project Apollo?"
    top_k: int = 8


class HaystackBenchmarkResponse(BaseModel):
    document_id: str
    document_title: str
    total_parent_chunks: int
    total_child_chunks: int
    query: str
    needle: str
    stages: list[StageDiagnostic]
    retrieved_parent_chunk: dict[str, Any] | None = None
    passed: bool
    answer: str
    reasoning: str | None = None
    latency_ms: int
