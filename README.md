<div align="center">

# ⚡ Deep Context Platform

### _Bare-Metal Agentic RAG, Recursive Language Models (RLM), and Typed Long-Term Memory_

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector%20HNSW-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Google Gemini GenAI](https://img.shields.io/badge/Google%20GenAI-Gemini%202.5%20%2B%20Embedding--2-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Groq Fast Inference](https://img.shields.io/badge/Groq-Qwen%203.6%2027B%20%2F%20Llama%203.3-F05032?logo=fastly&logoColor=white)](https://groq.com)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Llama%203.1%20%2F%20GLM--5.2-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![Tests](<https://img.shields.io/badge/Tests-43%20Passed%20(100%25)-brightgreen>)](https://github.com)
[![Zero Frameworks](<https://img.shields.io/badge/Frameworks-Zero%20(No%20LangChain%20%2F%20LlamaIndex)-black>)](https://github.com)

**A high-performance, framework-free Agentic Retrieval-Augmented Generation (RAG) platform.**  
Built entirely from scratch with raw Python, pure SQL (`asyncpg` + `pgvector`), multi-provider LLMs, typed durable memory, sandboxed recursive language modeling (RLM), and a zero-dependency Vanilla web interface.

[Quickstart](#-quickstart) • [Architecture](#-architecture) • [Features](#-core-features) • [API Contracts](#-api--streaming-contracts) • [CLI Manual](#-cli-reference) • [Documentation](#-documentation)

---

</div>

## 🌟 Why Deep Context Platform?

Most modern RAG systems suffer from three critical flaws:

1. **Framework Bloat & Fragility:** Heavy abstraction layers (LangChain, LlamaIndex, CrewAI) obscure SQL execution, add latency, and complicate production debugging.
2. **Context Blindness & Haystack Loss:** Traditional fixed chunking either loses macro context (small chunks) or dilutes embedding precision (large chunks), failing on complex 1,000-page documents.
3. **Lack of Infinite-Context Recursion:** When a query requires reading or aggregating over millions of tokens across an entire repository or book, standard top-$k$ retrieval fails.

**Deep Context Platform solves this from first principles:**

- **100% Hand-Crafted Core:** Zero LangChain, zero LlamaIndex, zero LangGraph. Pure, reviewable, high-speed Python 3.12 and raw SQL.
- **Anthropic Contextual Retrieval Standard:** Ingests documents with local GPU/MPS Qwen3-0.6B contextual summaries prepended to raw text (`summary_text + "\n\n" + raw_content`), cutting retrieval failure rates by up to 67%.
- **Hierarchical Parent-Child Resolution & Multi-Chunk Synthesis:** Precise 300-token child chunks for dense vector search; automatically expands to 1,500-token parent sections during LLM synthesis across distant chapters.
- **Decoupled Zero Data-Loss Ingestion Pipeline:** Checkpoint 1 atomic PostgreSQL writes preserve document hierarchy and Qwen3 summaries even on upstream rate limits, with on-demand deferred embedding resumption (`/v1/documents/{id}/embed-stream`).
- **Google Cloud Vertex AI & ADC Integration:** Native support for Google Cloud Vertex AI with OAuth 2.0 Application Default Credentials (`gcloud auth application-default login`), drawing directly from Google Cloud credits.
- **Multi-Strategy Hybrid Retrieval:** Combines weighted BM25 full-text indexing (prioritizing summary terms), `pgvector` HNSW dense vector search, Reciprocal Rank Fusion (RRF $k=60$), and neural cross-encoder rerankers.
- **Matryoshka Representation Learning (MRL):** Native support for `gemini-embedding-2` and `text-embedding-004` with flexible output dimensions (768d, 1536d, 3072d) for up to 75% vector storage savings.
- **Recursive Language Model (RLM) Engine:** Implements the MIT / Prime-Intellect architecture. When context exceeds window limits, the agent operates in a sandboxed Python REPL, spawning subagents and searching the corpus recursively.
- **4-Store Typed Memory with Promotion Gate:** Durable memory partitioned into **Policy**, **Preference**, **Semantic Fact**, and **Episodic Summary**, governed by a strict 4-stage promotion gate and compiled via an 8-layer prompt assembler.
- **Grounding Verification Gate:** Deterministic Natural Language Inference (NLI) and quote overlap checks that score evidence support before emitting final answers.

---

## 🏛 Architecture Overview

```
                                 ┌────────────────────────────────────────────────────────┐
                                 │                   USER / WEB UI / CLI                  │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │                FastAPI Application Layer               │
                                 │       • SSE Token Streaming    • REST Endpoints        │
                                 │       • Deferred Embedding Streaming Resumption        │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │            Agentic Router & Query Classifier           │
                                 │          (Factual Lookup / Multi-Hop / Aggregation)    │
                                 └───────┬───────────────────┼────────────────────┬───────┘
                                         │                   │                    │
                    ┌────────────────────┘                   │                    └────────────────────┐
                    ▼                                        ▼                                         ▼
   ┌─────────────────────────────────┐     ┌───────────────────────────────────┐     ┌──────────────────────────────────┐
   │  Anthropic Contextual Hybrid    │     │       Agentic Planner Loop        │     │       RLM Recursion Engine       │
   │  1. Weighted BM25 (Summary+Text)│     │  • Iterative Sub-Query Generation │     │  • Sandboxed Python REPL Kernel  │
   │  2. Dense Vector (Vertex/Gemini)│     │  • Multi-Hop Retrieval            │     │  • Regex & Keyword Search APIs   │
   │  3. Reciprocal Rank Fusion (RRF)│     │  • Chunk Deduplication            │     │  • Async Subagent Spawn & Mailbox│
   │  4. Multi-Strategy Precision    │     │  • Bounded Token Budgeting        │     │  • Structured Answer Synthesis   │
   │     Reranker (Cross/EcoHash)    │     └─────────────────┬─────────────────┘     └────────────────┬─────────────────┘
   │  5. Child -> Parent Resolution  │                       │                                        │
   └────────────────┬────────────────┘                       │                                        │
                    │                                        │                                        │
                    └────────────────────────────────────────┼────────────────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │           Evidence-Sufficiency & Support Gate          │
                                 │             • NLI Claim Verification                   │
                                 │             • Grounding & Citation Traceability        │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │               4-Store Typed Memory Layer               │
                                 │  • Policy  • User Preferences  • Facts  • Episodes     │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │                Storage & Vector Engine                 │
                                 │   • PostgreSQL + pgvector (HNSW)   • SQLite + FTS5     │
                                 │   • Checkpoint 1 Zero-Loss Writes  • search_tsv Trigger│
                                 └───────────────────────────┬────────────────────────────┘
```

---

## ⚡ Core Features

### 1. Multi-Provider LLM & Embedding Matrix

The platform features dynamic model routing, automatic failover, and dynamic `.env` hot-reloading:

| Provider | Supported Models | Capabilities & Auth |
| :--- | :--- | :--- |
| **Google Cloud Vertex AI** | `gemini-3.7-flash`, `gemini-embedding-2` | Fast reasoning with medium thinking effort & multimodal text embeddings via ADC / Cloud Credits |
| **Groq Cloud** | `qwen/qwen3.6-27b` | Sub-second ultra-fast reasoning token streaming (`<think>` blocks) |
| **Local Neural Models** | `Qwen/Qwen3-0.6B` (FP16) | On-device contextual chunk summarization with Apple Silicon Metal (MPS) and NVIDIA CUDA hardware acceleration |

### 2. Multi-Strategy Precision Reranking

Configurable on the fly or persisted per user in typed memory:

- **`cross_encoder` (Default):** Exact quote match + token Jaccard overlap + lexical boost.
- **`ecohash` (Hosted Neural):** BGE-reranker-v2-m3 cross-encoder via EcoHash API with calibrated probabilities.
- **`local_cross_encoder`:** Quantized INT8 BGE-reranker-v2-m3 running on ONNX Runtime with sigmoid normalization.

### 3. 4-Store Typed Memory System

Durable memory is partitioned to prevent cross-contamination:

- **`memory_policy`:** Immutable runtime safety and behavior constraints.
- **`memory_preference`:** User-scoped persistent preferences (e.g. preferred embedding model, output dimension, reranker, and LLM).
- **`memory_fact`:** Verified semantic world/user facts promoted through confidence scoring.
- **`memory_episode`:** Session history, outcome summaries, and interaction traces.

### 4. 100% Vanilla Web Studio

Located at `src/deep_context/ui/index.html`:

- **Zero npm, zero webpack, zero React:** Pure HTML5, CSS3, and JavaScript.
- **Live Thought Drawer:** Real-time collapsible display for model reasoning tokens (`<think>`).
- **Needle-in-a-Haystack 5-Stage Diagnostic Lab:** Step-by-step visibility into every retrieval layer.
- **1,000-Page Document Hub:** Drag-and-drop batch upload with streaming PDF extraction.

### 5. Response Cache Layer (Redis or In-Memory)

Whole-answer caching at the RAG pipeline level (`question → answer + citations`), comparable to production agentic-RAG cache services:

- **Stable keys:** SHA-256 of canonical JSON over query + tenant + permissions + filters + model config.
- **Redis-backed** when `CACHE_URL` is set; automatic in-memory fallback otherwise.
- **TTL-based expiry** (`CACHE_TTL`, default 300s) and namespace invalidation on document deletion.
- Only support-checked answers are cached; `/v1/retrieve` and `/v1/query` responses expose a `cache_hit` flag.

### 6. Internal Scheduler (Airflow-style job table)

A lightweight persistence-backed scheduler for ingestion and index maintenance:

- **`jobs` table** in SQLite/Postgres with cron-like schedules (`*/15 * * * *`) or `every:<seconds>` shorthand.
- **Built-in tasks:** `cleanup_orphaned_docs`, `reindex_corpus`, `refresh_embeddings`.
- **Retries with backoff**, failure recording, and manual tick endpoint for cron-wrapped deploys.

### 7. Corrective Agentic RAG State Machine

A hand-built LangGraph-equivalent corrective loop following the CRAG pattern (arXiv:2401.15884):

```
retrieve → grade_documents ──relevant──→ generate_answer → END
                │ irrelevant (rewrite_count < max)
          rewrite_question → retrieve → grade_documents
                │ exhausted
            abstain (safe fallback)
```

- Deterministic relevance grading against `AGENTIC_GRADE_THRESHOLD`.
- Query rewriting reuses the existing LLM rewriter with a deterministic offline fallback.
- Generation reuses the grounded two-pass generator and evidence verifier.

---

## 🚀 Quickstart

### 1. Prerequisites

- **Python 3.12+**
- **uv** (Modern Python package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Database (Choose Option A or Option B)**:
  - **Option A (Recommended for Highest Retrieval Quality): PostgreSQL with `pg_search` (ParadeDB BM25), `pgvector`, and `pg_trgm`**
    - **1-Click Docker (Windows, macOS, Linux)**:
      ```bash
      docker run -d --name deepcontext-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=awems -p 5432:5432 paradedb/paradedb:latest
      ```
    - **macOS Native (Homebrew)**:
      ```bash
      brew install postgresql@16 pgvector
      echo "shared_preload_libraries = 'pg_search'" >> /opt/homebrew/var/postgresql@16/postgresql.conf
      brew services restart postgresql@16
      ```
    - **Linux Native (Ubuntu/Debian)**:
      ```bash
      sudo apt-get install -y postgresql-16-paradedb
      ```
  - **Option B (Zero-Config Local Mode)**: SQLite with FTS5 (set `DATABASE_TYPE=sqlite` in `.env`).

### 2. Installation & Environment Setup

Clone the repository and sync dependencies:

```bash
git clone https://github.com/lakshya4568/DeepContext.git
cd DeepContext

# Create virtual environment and sync dependencies
uv sync --extra dev
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
# Google Gemini API (Embeddings & Reasoning)
GEMINI_API_KEY=your_gemini_api_key_here

# Groq API (Ultra-Fast Reasoning & Streaming)
GROQ_API_KEY=your_groq_api_key_here

# NVIDIA NIM API (Optional Fallback / Enterprise LLMs)
NVIDIA_API_KEY=your_nvidia_api_key_here

# Database Configuration ('postgres' or 'sqlite')
DATABASE_TYPE=postgres
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/deep_context

# Default Embedding & Model Settings
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIM=768
RERANKER_STRATEGY=cross_encoder
LLM_MODEL=qwen/qwen3.6-27b
```

### 4. Launch the Web Studio

Start the high-performance async server:

```bash
uv run uvicorn deep_context.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser to access the **Deep Context Studio**.

---

## 💻 CLI Reference

The platform provides a complete command-line interface via `deep-context`:

```bash
# 1. Ingest a document (PDF, Markdown, Code, TXT) with Gemini embeddings
uv run deep-context ingest path/to/document.pdf -e gemini-embedding-2 -d 768

# 2. Batch ingest an entire folder of files
uv run deep-context ingest-folder ./documents/ -e gemini-embedding-2 -d 768

# 3. Test hybrid retrieval across ingested documents
uv run deep-context retrieve "What is the core finding?" -k 5 -r cross_encoder

# 4. Run full grounded query synthesis
uv run deep-context query "Explain the architecture" -m gemini-3.7-flash

# 5. Manage user preferences in durable memory
uv run deep-context set-preference --user user_42 -e gemini-embedding-2 -d 768 -r ecohash
uv run deep-context preferences user_42

# 6. Run a Recursive Language Model (RLM) session in sandboxed REPL
uv run deep-context rlm "Scan all 195 chunks and identify every occurrence of Ser Kevan"

# 7. Run the corrective agentic RAG state machine (grade -> rewrite loop -> generate)
uv run deep-context agentic-query "What is reward hacking?" --max-rewrites 2

# 8. Run the internal scheduler (Ctrl+C to stop)
uv run deep-context scheduler

# 9. List registered scheduled jobs and their state
uv run deep-context jobs
```

---

## 📡 API & Streaming Contracts

### `POST /v1/query/stream` (Server-Sent Events)

Streams real-time status, citations, live thinking tokens, and final answer content:

```bash
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Identify who is speaking: \"They seem ferocious enough\"",
    "user_id": "user_42",
    "model": "qwen/qwen3.6-27b",
    "embedding_model": "gemini-embedding-2",
    "embedding_dim": 768,
    "reranker": "cross_encoder"
  }'
```

**Stream Event Structure:**

```json
data: {"type": "status", "stage": "retrieval", "message": "📚 Running BM25 + Dense Vector hybrid search..."}
data: {"type": "citations", "citations": [{"chunk_id": "...", "document_title": "Eval 1.pdf", "page_number": 616}]}
data: {"type": "reasoning", "delta": "Analyzing the scene at the Golden Tooth..."}
data: {"type": "content", "delta": "The line is spoken by Ser Kevan Lannister..."}
data: {"type": "done", "latency_ms": 1420, "path_taken": "hybrid_rag", "support_check_passed": true}
```

### `POST /v1/preferences` (User Memory)

Persists embedding and reranker settings to user-specific durable memory:

```bash
curl -X POST http://localhost:8000/v1/preferences \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_42",
    "embedding_model": "gemini-embedding-2",
    "embedding_dim": 768,
    "reranker": "ecohash",
    "llm_model": "gemini-3.7-flash"
  }'
```

### `POST /v1/haystack/benchmark`

Runs automated 5-stage needle-in-a-haystack verification over a multi-thousand-chunk corpus.

### `POST /v1/agentic-rag` (Corrective RAG State Machine)

Runs the corrective loop: retrieve → grade → (rewrite & retry) → generate, with a full execution trace:

```bash
curl -X POST http://localhost:8000/v1/agentic-rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is reward hacking?", "max_rewrites": 2, "top_k": 6}'
```

Response includes `answer`, `citations`, `grade_result` (`relevant`/`irrelevant`), `rewrite_count`,
`abstained`, `support_passed`, and a per-node `trace`.

### Scheduler Control

```bash
GET  /v1/scheduler/jobs          # List jobs + registered task callables
POST /v1/scheduler/jobs          # Register/update a job (cron or "every:<seconds>")
POST /v1/scheduler/tick          # Execute all due jobs once
POST /v1/scheduler/defaults      # Install built-in maintenance jobs
```

### Cache Diagnostics

```bash
GET  /v1/cache/status            # Backend kind (redis|memory), TTL, namespace
POST /v1/cache/invalidate?namespace=rag   # Drop cached ask/retrieve entries
```

---

## 🧪 Testing & Verification

Run the comprehensive test suite (100% offline with zero external network dependencies required):

```bash
# Run all automated unit and integration tests
uv run pytest

# Check code style and formatting
uv run ruff check src tests
uv run ruff format --check src tests

# Static type checking
uv run mypy src
```

---

## 📂 Repository Layout

```text
.
├── pyproject.toml                     # Dependency manifest & build definitions
├── src/deep_context/
│   ├── agentic/                       # Agentic planner & query shape classifier
│   │   ├── planner.py                 # Multi-hop query decomposition & iterative retrieval
│   │   └── router.py                  # Intelligent execution path router
│   ├── api/                           # FastAPI endpoints & Server-Sent Events (SSE)
│   │   ├── app.py                     # App factory & lifecycle handlers
│   │   └── routes_rag.py              # Ingest, stream query, preferences, haystack APIs
│   ├── cli/                           # Command-line interface (Typer + Rich)
│   │   └── main.py                    # CLI commands for ingestion, query, preferences
│   ├── core/                          # Core primitives, config, and LLM client
│   │   ├── config.py                  # Pydantic BaseSettings with .env hot-reloading
│   │   ├── llm_client.py              # Unified client for Gemini, Groq, and NVIDIA NIM
│   │   ├── logging.py                 # Structured application logging
│   │   └── types.py                   # Domain models, enums, and request schemas
│   ├── ingestion/                     # Document loading, parsing, and chunking
│   │   ├── chunker.py                 # Hierarchical parent-child token chunker
│   │   ├── parser.py                  # PDF, Markdown, TXT, Code structural parser
│   │   ├── pipeline.py                # End-to-end ingestion pipeline
│   │   └── tree_indexer.py            # Vectorless hierarchical document tree indexer
│   ├── memory/                        # 4-Store typed memory & prompt assembly
│   │   ├── prompt_assembler.py        # 8-layer prompt compiler
│   │   ├── promotion_gate.py          # 4-stage promotion gate for durable memory
│   │   └── stores.py                  # Policy, Preference, Fact, Episode stores
│   ├── retrieval/                     # Hybrid search, fusion, and reranking
│   │   ├── classifier.py              # Query classifier (factual/multi-hop/aggregation)
│   │   ├── engine.py                  # Central retrieval engine & sufficiency gate
│   │   ├── hybrid.py                  # BM25 + Dense Vector + Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py                # CrossEncoder, Gemini Semantic, Gemini LLM rerankers
│   │   └── tree_navigator.py          # Vectorless DAG tree traversal
│   ├── rlm/                           # Recursive Language Model engine
│   │   ├── host_bridge.py             # Subagent tree, recursion depth & mailbox passing
│   │   ├── kernel.py                  # Sandboxed Python REPL execution environment
│   │   └── orchestrator.py            # Multi-turn RLM session orchestrator
│   ├── storage/                       # Database storage drivers
│   │   ├── base.py                    # StorageInterface abstract base class
│   │   ├── postgres_store.py          # PostgreSQL + pgvector (HNSW) implementation
│   │   └── sqlite_store.py            # SQLite + FTS5 implementation
│   ├── ui/                            # 100% Vanilla Web Interface
│   │   └── index.html                 # Grounded Studio, Haystack Lab, Preference Manager
│   └── verification/                  # Grounding & evidence verification
│       └── checker.py                 # Anti-hallucination evidence verifier
├── tests/                             # 43 automated test suites
├── docs/                              # Formal specifications, PRD, and design docs
│   ├── PRD.md                         # Product requirements & functional specs
│   ├── ARCHITECTURE.md                # System architecture & component boundaries
│   ├── DATA_MODEL.sql                 # PostgreSQL 15 + pgvector DDL schema
│   ├── TECH_STACK.md                  # Concrete tool choices & switch criteria
│   └── VERIFICATION_AND_SOURCES.md    # Primary source verification & benchmarks
└── workflows/                         # Step-by-step pipeline specifications
```

---

## 📜 License

This project is licensed under the Apache 2.0 License. Built for high-reliability, verifiable, and transparent context engineering.
