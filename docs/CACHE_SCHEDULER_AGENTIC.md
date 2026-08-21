# Cache, Scheduler, and Corrective Agentic RAG

This document describes three production-hardening layers added to DeepContext:
a response cache, an internal job scheduler, and a corrective agentic RAG state
machine. All three are hand-built (no LangGraph / Airflow / Celery dependencies)
and follow the patterns validated by the CRAG paper and production agentic-RAG
reference architectures.

---

## 1. Response Cache Layer (`src/deep_context/cache.py`)

Caches whole RAG responses ("question → answer + citations") at the pipeline
level, the same strategy used by production agentic-RAG cache services.

### Backends

| Backend                 | When used                                            | Notes                                                          |
| ----------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| Redis (`redis.asyncio`) | `CACHE_URL` is set (e.g. `redis://localhost:6379/0`) | Uses `SETEX` for TTL expiry; SCAN-based namespace invalidation |
| In-memory               | Default / fallback                                   | Process-local dict with monotonic-clock TTL; zero config       |

The backend is resolved lazily on first use. If the `redis` package is not
installed or the connection fails, the cache degrades gracefully to in-memory.

### Key design

Keys are `{CACHE_NAMESPACE}:{operation}:{sha256(canonical_json)}` where the JSON
payload includes every field that can change the answer:

- `query`, `tenant_id`, sorted `permission_scope`, sorted `document_ids`
- `top_k`, `model`, `embedding_model`, `embedding_dim`, `reranker`

Sorted lists make keys stable regardless of input ordering. User IDs are
deliberately excluded from key material — personalization is resolved inside
the pipeline via embedding/reranker preferences that _are_ in the key.

### Behavior

- `/v1/retrieve` caches sufficient retrieval results.
- `/v1/query` caches only answers that passed the evidence-support gate
  (RLM-engine answers are never cached).
- Responses carry a `cache_hit: bool` field.
- Deleting documents (`DELETE /v1/documents[/{id}]`) invalidates the `rag`
  namespace so stale citations cannot survive re-ingestion.
- `POST /v1/cache/invalidate?namespace=rag` clears ask + retrieve entries.

### Configuration

```bash
CACHE_ENABLED=true                      # master switch
CACHE_URL=redis://localhost:6379/0      # empty = in-memory backend
CACHE_TTL=300                           # default TTL seconds
CACHE_NAMESPACE=deepcontext             # key prefix
```

---

## 2. Internal Scheduler (`src/deep_context/scheduler.py`)

A persistence-backed polling scheduler providing Airflow-like semantics
(DAG-style sequencing, retries, failure tracking) without external services.

### Job model (`jobs` table)

```
name TEXT PRIMARY KEY
schedule_cron TEXT        -- "*/15 * * * *" or "every:<seconds>"
next_run_at TIMESTAMP
status TEXT               -- idle | running | failed
max_retries INT, retries INT, last_error TEXT, last_run_at TIMESTAMP
```

### Built-in tasks

| Task                    | Default schedule | Purpose                                  |
| ----------------------- | ---------------- | ---------------------------------------- |
| `cleanup_orphaned_docs` | every hour       | Remove chunks whose document row is gone |
| `reindex_corpus`        | daily            | Rebuild FTS index entries                |
| `refresh_embeddings`    | every 6h         | Re-embed child chunks missing embeddings |

### Execution semantics

1. Each tick selects jobs where `next_run_at <= now AND status != 'running'`.
2. Jobs transition to `running`; success resets retries and schedules the next
   cron slot; failure increments retries and reschedules soon
   (`2 × poll interval`) until `max_retries` is exhausted.
3. Unknown task names fail immediately with an explanatory error.

### Running it

```bash
# Standalone process
uv run python -m deep_context.scheduler
# or via CLI
uv run deep-context scheduler

# Embedded in the API lifespan (starts/stops with FastAPI)
SCHEDULER_ENABLED=true uv run deep-context serve

# Cron-wrapped single tick (no long-running process needed)
curl -X POST http://localhost:8000/v1/scheduler/tick
```

### Comparison to Airflow

You get dependency-free scheduling, persisted state, and retries. Airflow adds
a rich UI, backfill, SLAs, and distributed workers — unnecessary for one-node
ingestion maintenance. The manual-tick endpoint covers cron/systemd deployments.

---

## 3. Corrective Agentic RAG State Machine (`src/deep_context/agentic/graph.py`)

Implements the CRAG pattern (arXiv:2401.15884) and LangGraph's agentic RAG
reference topology as plain Python functions over an explicit `RAGState`
dataclass:

```
START → retrieve → grade_documents ──relevant──→ generate_answer → END
                        │ irrelevant & rewrite_count < max_rewrites
                  rewrite_query → retrieve → grade_documents
                        │ exhausted
                    abstain (safe refusal)
```

### Nodes

- **retrieve** — delegates to the existing hybrid retrieval engine (BM25 +
  dense + RRF + reranking), using the rewritten query when present.
- **grade_documents** — deterministic term-overlap scoring against
  `AGENTIC_GRADE_THRESHOLD` (default 0.30). Binary relevant/irrelevant per
  document, mirroring LangGraph's `GradeDocuments` gate but fully offline and
  reproducible. An LLM grader can be swapped in behind the same interface.
- **rewrite_query** — reuses `QueryRewriter.rewrite_or_decompose`; falls back
  to deterministic keyword expansion when the LLM rewriter is unavailable.
- **generate_answer** — delegates to the two-pass grounded generator and
  evidence verifier; only graded-relevant documents are passed as context.
- **abstain** — safe fallback returning the standard insufficiency refusal,
  never answering from weak context.

### Usage

```bash
# CLI
uv run deep-context agentic-query "What is reward hacking?" --max-rewrites 2

# API
curl -X POST http://localhost:8000/v1/agentic-rag \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "max_rewrites": 2}'
```

Every run returns a per-node execution `trace` for observability, and loop
exhaustion events are recorded in `events_trace`.

### Configuration

```bash
AGENTIC_MAX_REWRITES=2          # bounded corrective loop
AGENTIC_GRADE_THRESHOLD=0.30    # min term-overlap ratio for relevance
```

---

## Testing

- `tests/test_cache.py` — key stability, TTL expiry, invalidation, disabled mode.
- `tests/test_scheduler.py` — cron parser, job lifecycle, retries, maintenance tasks.
- `tests/test_agentic_graph.py` — grading gates, rewrite loop, abstention, trace.
- `tests/test_ops_api.py` — endpoint contracts for all new routes.

All tests are offline and deterministic (in-memory cache backend, mock LLM).
