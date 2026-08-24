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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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


class IngestResponse(BaseModel):
    document_id: str
    title: str
    parent_chunks_count: int
    child_chunks_count: int
    retrieval_mode: RetrievalMode
    summaries_generated_count: int = 0
    embedding_model: str | None = None
    embedding_dim: int | None = None


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
