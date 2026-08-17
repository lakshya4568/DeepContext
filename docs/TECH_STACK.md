# Tech Stack

Every choice below states the alternative considered and the concrete condition under which you should switch —
the goal is to stop you from either over-engineering on day one or getting stuck on a default past the point it
still fits.

---

## 1. Language & runtime: Python end to end

**Choice:** Python 3.12+ for the host, the kernel, retrieval, memory, and skills.

**Alternative considered:** mirror Prime Agent's actual split (TypeScript host + Python kernel).

**Why Python-only wins here:** the TS/Python split exists in the reference implementation because Prime Agent is
a general-purpose CLI/TUI product shipping to many users' machines, where a fast, single-binary TypeScript CLI
matters. You're building a backend service you deploy yourself. A second language buys you nothing here and costs
you a serialization boundary, a second dependency-management story, and a second set of things that can break. The
authority-separation *property* (§4 of `ARCHITECTURE.md`) is preserved with two Python processes instead.

**Switch condition:** if this ever grows into a product you ship to other people's machines as a CLI/TUI (i.e. you
are rebuilding something closer to Prime Agent itself), revisit — a compiled or fast-starting host process
becomes worth the complexity.

---

## 2. Web framework: FastAPI

**Choice:** FastAPI for the host's HTTP surface.

**Why:** async-native, typed request/response models (which pay off directly for the typed host-request pattern
in §4), automatic OpenAPI docs, and it's already the framework named in your own career roadmap — no new tool to
learn for this piece.

**Alternative considered:** Flask (simpler, but sync-first — fights the async I/O model this platform needs) and
Litestar (comparable to FastAPI, smaller ecosystem). Neither is wrong; FastAPI is the default because it's the one
you already know.

---

## 3. Vector storage: pgvector (Postgres) by default

**Choice:** Postgres with the `pgvector` extension as the single source of truth for both structured data
(sessions, memory, trace log) and vector search.

**Alternatives considered:** Qdrant, Milvus, Weaviate (all named in the source brief as viable).

**Why pgvector first:** one database for typed memory *and* vectors means transactional writes (e.g., "insert this
chunk and its embedding, or neither") without a distributed-transaction problem across two systems. For a
solo-operated system at the scale of "your own projects/papers/codebase," pgvector's HNSW index is fast enough,
and you avoid running and monitoring a second stateful service.

**Concrete switch criteria** (don't switch preemptively — wait for one of these to actually be true):
- Vector search P95 latency exceeds your budget (§4 of `PRD.md`) *after* you've confirmed the index is tuned
  (correct `lists`/`m`/`ef_construction` for your row count), not before.
- Corpus size moves into tens of millions of vectors with high write throughput, where a purpose-built vector DB's
  sharding story starts to matter.
- You need multi-region replication semantics Postgres doesn't give you cleanly.

If/when you do switch, the retrieval skill's `retrieve()` interface (see `skills/rag-retrieval/`) is written so
the swap is internal to that skill, not a change to every caller.

---

## 4. Full-text / BM25: Postgres `tsvector` first, Meilisearch/Elasticsearch later

**Choice:** Postgres built-in full-text search (`tsvector`/`tsquery`/`ts_rank`) for the BM25-style leg of hybrid
retrieval, fused with vector search via Reciprocal Rank Fusion.

**Why:** same one-database argument as §3. Postgres FTS is not a state-of-the-art BM25 implementation, but it's
good enough to catch exact identifiers, function names, and error codes that pure vector search misses — which is
the actual job this leg of hybrid retrieval has, per the source brief's own reasoning.

**Switch condition:** move to Meilisearch or Elasticsearch when you need faceted search, typo tolerance, or
multi-language stemming that Postgres FTS handles poorly, or when FTS query latency becomes the bottleneck at
scale.

---

## 5. Reranking: cross-encoder, self-hosted, small

**Choice:** a self-hosted cross-encoder reranker (e.g., a BGE- or Qwen3-Reranker-family model in the ~0.5–4B
range) called from the retrieval skill between first-stage retrieval and generation.

**Source note:** the source brief cites a 2026 AIMultiple benchmark reporting one reranker raising Hit@1 from
62.67% to 83%. This is a **single benchmark, single reranker, single dataset** result — treat it as "reranking
plausibly matters a lot here," not as a number that will reproduce on your corpus. Measure Hit@k on your own
labeled subset (§4 of `PRD.md`) before and after adding a reranker; that's the number that should drive whether
you keep it in the hot path.

**Alternative considered:** a hosted reranker API (Cohere, Voyage). Reasonable if you'd rather not host a model at
all; self-hosted is the default here mainly to keep the number of external dependencies (and the number of things
with a per-call bill) small while you're building.

**Switch condition:** move to a hosted reranker if self-hosting becomes an operational burden, or if you need a
language/domain the self-hosted model handles poorly.

---

## 6. Model providers: Claude and GPT-class, behind one thin interface

**Choice:** an abstraction (`llm_call(messages, model, tools=...)`) that both the host and the kernel's `rlm_spawn`
stub go through, backed initially by Claude and Groq-hosted models — matching your own existing multi-provider
usage.

**Why an abstraction at all:** the RLM engine spawns children that "inherit the parent model... unless the call
requests another configured model" (verified Prime Agent behavior) — you want the ability to send cheap/fast
subtasks to a smaller or faster model (e.g. a Groq-hosted model) while keeping the root on a stronger model,
without rewriting the orchestrator when you do.

**Note on RLM-specific model choice:** per the verified paper, the *strongest* published RLM results used
frontier models (GPT-5-class) and a small purpose-trained model (RLM-Qwen3-8B) — not an arbitrary small model
prompted into the pattern with no adaptation. If you route RLM subtasks to a small/cheap model with no tuning,
expect the "inconsistent, task-dependent" behavior documented in `CRITICAL_ASSESSMENT_AND_SCOPE.md` to be more
pronounced, not less.

---

## 7. RLM kernel sandboxing

**Choice, in order of how seriously you should take isolation as this grows:**

1. **Local dev / your own trusted documents only:** subprocess with a restricted `PYTHONPATH`, scratch-directory-
   only filesystem access, and network egress limited to an explicit allowlist.
2. **Anything touching a document or repo you didn't author yourself:** container-based isolation (Docker with
   dropped capabilities and a read-only root filesystem) at minimum.
3. **Production / untrusted input at any real scale:** `gVisor`/`nsjail`-style syscall-level isolation, or a
   hosted code-execution sandbox (e.g. E2B, Modal sandboxes) — this is what the RLM paper's own implementation
   uses ("code execution happens in isolated Sandboxes").

**Why this is a real requirement and not boilerplate:** Prime Agent's own docs are explicit that the kernel
process "runs model-generated Python and project commands with the worker's operating-system permissions... It is
a durable control environment, not a security sandbox." If the reference implementation says this about itself,
treat it as doubly true for a from-scratch build that hasn't had the same amount of production hardening.

---

## 8. Agent orchestration framework: LangGraph for the agentic-planner path, hand-rolled for the RLM path

**Choice:** use LangGraph for the multi-step "agentic RAG" path (FR18) — it's already in your stated stack and is
a good fit for an explicit graph of planner → tool-call → evaluate → loop-or-answer. Do **not** try to force the
RLM engine's async subagent/message-passing model (§5.3 of `ARCHITECTURE.md`) into the same graph abstraction —
LangGraph's execution model assumes a graph walk, not open-ended async children that reply on their own schedule.
Implement the RLM host bridge as its own small async component instead.

**Switch condition:** if a future LangGraph version adds first-class support for detached, message-passing
subagent nodes, revisit — but don't wait for that to ship the RLM engine.

---

## 9. Observability

**Choice:** structured JSON logging to start (every request gets a trace ID that threads through retrieval,
memory, and RLM calls, landing in the `events_trace` table — see `docs/DATA_MODEL.sql`), with Prometheus + Grafana
added once there's more than one moving deployment to compare.

**Why not full observability tooling from day one:** for a single-operator system, the trace table itself (queryable
with plain SQL) answers "why did this answer come out the way it did" faster than standing up a metrics stack
would, in the timeframe of the Phase 1 MVP.

---

## 10. Summary table

| Layer | Default | Alternative | Switch when |
|---|---|---|---|
| Language | Python 3.12+ | TypeScript host + Python kernel (Prime-Agent-style) | You're shipping a CLI to other people's machines |
| Web framework | FastAPI | Litestar | Rarely — FastAPI fits this use case well |
| Vector store | Postgres + pgvector | Qdrant / Milvus / Weaviate | Tuned pgvector still misses latency budget, or corpus reaches tens of millions of vectors |
| Full-text | Postgres `tsvector` | Meilisearch / Elasticsearch | Need faceting, typo tolerance, multi-language stemming |
| Reranker | Self-hosted cross-encoder (BGE/Qwen3-Reranker family) | Hosted (Cohere/Voyage) | Self-hosting becomes an operational burden |
| Orchestration (agentic path) | LangGraph | Custom state machine | LangGraph's assumptions stop fitting your control flow |
| Orchestration (RLM path) | Hand-rolled async host bridge | — | N/A — don't force this into a graph framework |
| Sandboxing | Subprocess (dev) → container (untrusted docs) → gVisor/hosted (production) | — | Escalate as soon as input isn't 100% your own |
| Observability | Structured logs + trace table | Prometheus + Grafana | More than one deployment to compare |
