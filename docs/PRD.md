# Product Requirements Document
## Deep Context Platform — Agentic Hybrid RAG + Typed Memory + RLM Engine

| | |
|---|---|
| **Status** | Draft v1.0 |
| **Owner** | You |
| **Last updated** | 17 August 2026 |
| **Related docs** | `ARCHITECTURE.md`, `TECH_STACK.md`, `CRITICAL_ASSESSMENT_AND_SCOPE.md`, `docs/DATA_MODEL.sql` |

---

## 1. Product summary

**Name:** Deep Context Platform (working name)

**One-line description:** An agent backend that answers questions and executes tasks over your own documents and
codebases with three coordinated engines — hybrid RAG for everyday retrieval, typed long-term memory for
continuity across sessions, and an RLM (Recursive Language Model) engine for the minority of tasks that require
reading and reasoning over more material than fits in any model's context window — fronted by a FastAPI service
and exposed to agents as a set of composable skills.

**Why it exists:** No single retrieval paradigm is best for every query (see the source verdict in
`source-material/`). A system that only does vector RAG is weak at multi-hop reasoning and global aggregation. A
system that only does RLM is slow, expensive, and bad at freshness, permissions, and personalization. This
platform routes each request to the cheapest mechanism that can answer it correctly, escalating only when it has to.

**What it is not:** It is not a from-scratch clone of Prime Agent, and it does not require a TypeScript daemon,
a custom CLI, or a background service. It borrows the *ideas* that hold up under scrutiny (host/kernel authority
separation, async subagents, a bounded REPL, typed memory) and implements them as a Python/FastAPI service, because
that is the stack you already know and the one your broader roadmap is built on.

---

## 2. Problem statement

You are building agentic systems (an autonomous multi-agent workforce platform, an AI study agent, a career
roadmap that starts with a deployed Agentic RAG Assistant) that all share the same underlying need: **give an LLM
reliable, current, permission-scoped access to a body of knowledge that is larger than its context window, without
sacrificing latency or cost on the 90% of queries that are simple.**

Three failure modes show up repeatedly if you don't design for this up front:

1. **Under-retrieval:** naive top-k vector search misses evidence, and the model either hallucinates or gives an
   answer it can't actually support.
2. **Over-retrieval:** stuffing 50 chunks into every prompt to be safe increases cost, dilutes attention, and can
   make answers *worse*, not better ("context rot").
3. **No memory:** every conversation starts from zero, so the agent re-asks things it was already told, forgets
   durable facts about your projects, and can't build on prior work.

A fourth failure mode is specific to very large or very dense inputs (a 1,000-page legal bundle, an entire
codebase, a dataset where "did not miss anything" is the actual requirement): normal retrieval — even agentic,
even reranked — only ever surfaces a *subset* of the corpus. For that narrow but real class of task, this platform
adds an RLM engine as a deliberate, expensive, opt-in escalation path, not the default.

---

## 3. Goals and non-goals

### Goals

- **G1 — Route by query shape, not by hype.** Simple QA uses hybrid retrieval. Multi-step tasks use the agentic
  planner. Global-aggregation-over-huge-input tasks use the RLM engine. The classifier's job is to avoid paying
  RLM-level cost and latency for a question hybrid retrieval could have answered.
- **G2 — Typed memory, not one undifferentiated vector store.** Policies, preferences, facts, and episodes have
  different retrieval semantics (exact lookup vs. hybrid search) and different write disciplines (a policy is
  never inferred; a fact is written through a promotion gate with confidence and TTL).
- **G3 — Every generated claim is either retrieved, computed, or explicitly marked as the model's own inference.**
  Corrective retrieval (grade evidence, re-query if weak) is not optional middleware, it's the default path.
- **G4 — Skills are the unit of reuse**, both for this platform's own subsystems and for what you ship to Claude
  or Prime-Agent-style harnesses. A skill you write once should work in both.
- **G5 — Ship something real in weeks, not months.** The MVP (Phase 1, see §10) must be a working, deployable
  agentic RAG assistant on its own, with the RLM engine and deeper memory types added incrementally.

### Non-goals (explicitly out of scope for v1)

- **NG1 — Not a multi-tenant SaaS.** Tenant/permission scoping is designed into the data model (so it isn't
  painful to add later) but auth, billing, and admin consoles are out of scope for v1.
- **NG2 — Not a general-purpose agent daemon.** No persistent background service, no cross-terminal session
  reattachment, no scheduler UI. A FastAPI process that starts, serves requests, and can be redeployed is enough.
- **NG3 — Not training an RLM-native model.** The paper's RLM-Qwen3-8B result (a model *post-trained* to use the
  RLM pattern, beating its base model by 28.3%) is a research direction, not a v1 requirement. v1 prompts an
  off-the-shelf strong model (Claude/GPT-class) into the RLM pattern via scaffolding, and accepts the resulting
  inconsistency described in `CRITICAL_ASSESSMENT_AND_SCOPE.md`.
- **NG4 — Not a security sandbox vendor.** Code execution isolation (§8, NFR3) uses off-the-shelf sandboxing
  (container or `gVisor`/`nsjail`-style isolation, or a hosted code-execution API); this platform does not build
  its own sandbox technology.
- **NG5 — Not GraphRAG by default.** A graph layer is a named, optional extension point (§9), not a v1 component —
  it's expensive to build and maintain and the source verdict itself calls it "usually excessive for ordinary
  document QA."

---

## 4. Success metrics

Directional targets — replace the placeholders with real numbers once you have a labeled eval set from your own
documents/codebase. The point of writing them now is to force the eval set to exist before "it feels like it's
working" becomes the only signal you have.

| Metric | Target | How it's measured |
|---|---|---|
| Answer groundedness | ≥ 90% of answers pass the `check_answer_support` verifier (every claim traceable to a retrieved chunk, a computed value, or an explicit "I'm inferring this") | Automated verifier skill (§7, `verification`) run on a held-out eval set of ≥ 50 questions |
| Retrieval recall@10 (hybrid path) | ≥ 85% on a labeled subset of your own corpus | Standard IR eval: does the known-correct chunk appear in the top 10 pre-rerank candidates |
| Router precision | ≥ 90% of queries land on the mechanism a human would pick (simple QA → hybrid; multi-step → agentic; "read everything" → RLM) | Manually labeled 30–50 query test set, compared against classifier output |
| P50 / P95 latency, hybrid path | < 3s / < 8s | Measured end-to-end from request to first token |
| P50 latency, RLM path | Reported, not gated — RLM is latency-expensive by design; the metric that matters is that it's *never silently invoked* for a query the hybrid path could have handled (see router precision, above) |
| Memory promotion precision | ≥ 95% of auto-promoted facts are still valid 30 days later, sampled manually | Spot-check sample of `memory_fact` rows with `source = 'inferred'` |
| Cost per resolved query | Tracked, not targeted at v1 — establish a baseline in Phase 1, then optimize in Phase 2+ | Token + reranker + RLM sub-call cost, logged per request in the trace table |

---

## 5. Target users and personas

- **You, as the primary developer/operator.** The system is designed to be run and extended by one person first;
  multi-user niceties come later. This shapes several decisions below (SQLite-compatible local dev path, no
  required message bus for v1, etc. — see `TECH_STACK.md`).
- **A downstream agent** (LangGraph graph, Claude Code session, or a Prime-Agent-style harness) that calls this
  platform's skills as tools. The platform is a *backend for agents*, not only a chat UI.
- **A future teammate or reviewer** who needs to understand why a given answer was produced — hence the emphasis
  on citations, trace logs, and an explicit evidence-sufficiency check rather than "it looked right."

---

## 6. Use cases and user stories

- *As the developer of the Autonomous Workforce agentic system*, I want manager/worker agents to call a shared
  retrieval skill so that project knowledge (specs, prior decisions, code) is consistent across agents instead of
  each agent re-deriving it.
- *As a student building a study agent*, I want the system to remember that I'm working through calculus from
  first principles and prefer worked examples, without me repeating that preference every session.
- *As a developer debugging a production issue*, I want to ask "has this exact stack trace happened before?" and
  get an answer grounded in episodic memory of past debugging sessions, not a fresh guess.
- *As a researcher (or a student doing a literature-heavy project)*, I want to ingest 40 papers and ask a question
  that requires checking every one of them (e.g. "which of these use a held-out test set?") and get an answer that
  is actually computed by reading all 40, not sampled from the 5 the vector index happened to retrieve — this is
  the RLM engine's reason for existing.
- *As the platform operator*, I want a corrective-retrieval loop so that when the first retrieval pass is weak,
  the system rewrites the query and tries again (or escalates) instead of confidently answering from thin evidence.

---

## 7. Functional requirements

Requirements are grouped by subsystem. Each maps to one or more skills in `skills/` and one or more workflow docs
in `workflows/`.

### 7.1 Retrieval & RAG

- **FR1.** Hybrid retrieval combining BM25/full-text and dense vector search, fused with Reciprocal Rank Fusion
  (RRF), is the default retrieval mechanism.
- **FR2.** Parent-child chunking is the default ingestion strategy (child 300–600 tokens for matching, parent
  1,000–2,500 tokens sent to the model), with structure-aware splitting (headings/sections/tables/AST for code)
  taking precedence over fixed-size splitting wherever the source format has real structure.
- **FR3.** A cross-encoder reranker sits between first-stage retrieval (top 50–100) and generation (top 5–10).
- **FR4.** A corrective-retrieval step grades evidence sufficiency before generation; on a weak grade, the system
  rewrites the query and retries at least once before it is allowed to answer with a stated uncertainty instead of
  a confident guess.
- **FR5.** Metadata filters (tenant, document, date range, permission scope) are enforced at the retrieval layer,
  not only in the prompt.
- **FR6.** A vectorless/tree navigation mode (PageIndex-style hierarchical navigation) is available as an
  alternative retrieval strategy for long, structurally rich documents (contracts, filings, manuals), selectable
  per-document at ingestion time.

### 7.2 Typed memory

- **FR7.** Four memory types are modeled distinctly, not merged into one vector index: **policy** (exact-lookup,
  never inferred), **preference** (exact-lookup, user-scoped), **semantic fact** (hybrid search, has source +
  confidence + TTL + supersession status), and **episodic summary** (hybrid search, one per completed task/session).
- **FR8.** A promotion gate sits between "the agent observed something" and "it becomes durable memory": classify
  type → check scope → extract atomic claim → check for contradictions with existing memory → assign
  confidence/TTL/provenance → write or reject. Nothing is written to durable memory without passing through it.
- **FR9.** A prompt-assembly step rebuilds context every turn in a fixed order (system → policies → preferences →
  conversation summary → task state → retrieved facts → retrieved documents → current query) rather than
  monotonically appending history.
- **FR10.** A trace log (append-only) records every tool call, retrieval, and model decision for replay, audit,
  and later episodic-memory extraction — this is separate from, and not directly injected into, the prompt.

### 7.3 RLM engine

- **FR11.** A bounded, sandboxed Python REPL (kernel) is available as a tool for the subset of tasks the router
  classifies as "global analysis" — the corpus/codebase/document set is loaded as a variable, not pasted into the
  prompt.
- **FR12.** Subagent spawning (`rlm_spawn`) is **asynchronous**: it returns an admission handle immediately and
  does not block the parent. Results arrive via an explicit message/collection call, matching the verified
  behavior of the reference implementation (see `VERIFICATION_AND_SOURCES.md`), not a synchronous return value.
- **FR13.** Recursion depth defaults to **1** (root can spawn children; children cannot spawn grandchildren) and
  is a config value, not a hardcoded limit — raising it is a deliberate, budget-aware decision, matching the
  reference implementation's own current default.
- **FR14.** REPL output shown to the model per turn is capped (default 8,192 characters, configurable) to force
  the model to search/filter/delegate rather than dump the entire corpus into its own context — this specific
  default is taken directly from Prime Intellect's published RLM ablation.
- **FR15.** The model's final answer is written to a structured variable (`content`, `ready`) rather than returned
  as normal text, so it can be edited/corrected across multiple turns before the rollout ends — matching the
  "diffusion-style" answer construction in the source paper.
- **FR16.** A router explicitly decides RLM-vs-not per request (see `workflows/03_rlm_recursion_pipeline.md`) and
  logs its reasoning; RLM is never the silent default.

### 7.4 Skills & orchestration

- **FR17.** Every reusable capability (retrieval, memory, RLM orchestration, code execution, verification,
  refinement) is packaged as a `SKILL.md` skill per the Agent Skills standard, loadable by Claude, Claude Code, or
  a Prime-Agent-style harness without modification.
- **FR18.** An agentic planner selects among hybrid retrieval, the RLM engine, and direct tool calls (SQL, web
  search, code execution) based on the query classifier's output, with explicit iteration and tool-call limits to
  prevent runaway loops.
- **FR19.** A refinement pipeline (analogous to Prime Agent's `/refine`) can propose small, evidence-linked changes
  to *supplemental* state (memory entries, skill descriptions, prompt notes) but never rewrites the base system
  prompt, and every change is reversible.

### 7.5 Verification

- **FR20.** An evidence-support checker scores whether a generated answer is actually backed by the evidence it
  cites, and this score gates whether the answer is returned as-is or flagged/re-attempted.
- **FR21.** Code-producing tasks (if any skill generates code) run through a test/lint verifier before being
  presented as done.

### 7.6 Interfaces

- **FR22.** A FastAPI HTTP interface exposes retrieval, memory, and RLM operations as endpoints usable by any
  agent framework (LangGraph, a Claude Code MCP server, or a plain HTTP client).
- **FR23.** Structured logging and a minimal metrics endpoint exist from day one (NFR4 depends on this).

---

## 8. Non-functional requirements

- **NFR1 — Latency budgets differ by path.** Interactive hybrid-retrieval queries target the P50/P95 numbers in
  §4. RLM-path queries are exempted from a latency SLA by design — the router's job (FR16) is to keep them off the
  interactive path unless the query genuinely requires them.
- **NFR2 — Resource isolation between the kernel and the host.** The process that runs model-generated Python
  (the "kernel") must not share the credentials or write-access of the process that owns retrieval permissions,
  memory writes, and provider API keys (the "host"). See `ARCHITECTURE.md` §4 for the concrete boundary.
- **NFR3 — Security: this is not a sandbox by default, say so.** Following the reference implementation's own
  posture (confirmed in `VERIFICATION_AND_SOURCES.md`: *"a durable control environment, not a security
  sandbox"*), the kernel process by itself is **lifecycle isolation, not a security boundary**. Anything that runs
  code from untrusted sources (an uploaded document with embedded instructions, a public repo) must run inside an
  actual sandbox (container with dropped capabilities, `gVisor`, or a hosted code-execution service) — this is a
  hard requirement for any deployment beyond your own trusted machine, not a nice-to-have.
- **NFR4 — Observability.** Every retrieval, memory write, and RLM sub-call is logged to the trace table with
  enough detail to reconstruct why an answer was produced.
- **NFR5 — Reliability.** A crashed kernel process must not corrupt durable memory or leave the host in an
  inconsistent state — the host is the source of truth; the kernel is disposable and restartable.
- **NFR6 — Cost predictability.** Every path (hybrid, agentic, RLM) reports token + reranker + sub-call cost per
  request so the cost-per-query metric in §4 is measurable from day one, not retrofitted.

---

## 9. Data model overview

See `docs/DATA_MODEL.sql` for the full, runnable Postgres schema. Core tables:

- `documents`, `chunks` — ingestion output, with parent/child linkage and metadata (source, section path, page,
  permissions, timestamp)
- `memory_policy`, `memory_preference`, `memory_fact`, `memory_episode` — the four typed memory stores, each with
  its own scope/TTL/confidence columns where applicable
- `sessions`, `agents` — session and agent-run bookkeeping
- `rlm_children` — subagent admission handles and status, matching the async spawn/collect model (FR12)
- `events_trace` — the append-only trace log (FR10, NFR4)
- `skills_registry` — installed skills, versions, and enablement scope

An optional graph extension (`entities`, `relationships`) is included but commented out — see NG5.

---

## 10. Release plan

### Phase 1 — MVP (weeks 1–3): the thing that should ship first

- Structure-aware ingestion + parent-child chunking (FR2)
- Hybrid retrieval (BM25 + vector + RRF) with a cross-encoder reranker (FR1, FR3)
- Corrective retrieval loop (FR4)
- `memory_preference` and `memory_fact` only (defer `memory_policy`/`memory_episode` to Phase 2 — see FR7)
- FastAPI interface (FR22) + structured logging (FR23)
- **No RLM engine yet.** This phase is, deliberately, a complete and useful agentic RAG assistant on its own —
  it is compatible with a "deploy an Agentic RAG Assistant" first milestone using FastAPI + Postgres/pgvector.

### Phase 2 — Typed memory + agentic routing (weeks 4–6)

- Full four-type memory system + promotion gate (FR7–FR9)
- Query classifier + agentic planner for multi-step queries (FR18)
- Trace log + evidence-support verifier (FR10, FR20)
- Vectorless/tree navigation for structured documents (FR6)

### Phase 3 — RLM engine (weeks 7–10, the highest-risk phase — see `CRITICAL_ASSESSMENT_AND_SCOPE.md`)

- Sandboxed kernel + host/kernel boundary (FR11, NFR2, NFR3)
- Async subagent spawn/collect (FR12), depth-1 recursion (FR13)
- REPL output cap + structured answer contract (FR14, FR15)
- Router integration (FR16) — RLM only reachable through the classifier, never default

### Phase 4 — Self-improvement & hardening (weeks 11+)

- Refinement pipeline (FR19)
- Skill registry with versioning
- Expanded eval harness against the metrics in §4

---

## 11. Risks and assumptions

Full discussion in `CRITICAL_ASSESSMENT_AND_SCOPE.md`. Summary:

- **Assumption:** a strong off-the-shelf model (Claude/GPT-class) can be prompted into the RLM pattern well enough
  to be useful without RLM-specific fine-tuning. **Risk:** Prime Intellect's own ablations show this is
  inconsistent and task-dependent today (§2 of the verification doc) — budget for prompt iteration, not a
  one-shot integration.
- **Assumption:** one person can build and maintain all four phases. **Risk:** Phase 3 alone (sandboxing + async
  orchestration + a new failure-recovery surface) is a multi-week effort by itself; treat Phases 1–2 as the
  deliverable if the timeline is tight, and Phase 3 as a stretch goal.
- **Assumption:** Postgres/pgvector is sufficient at your data scale. **Risk:** re-evaluate only if/when ingestion
  volume or query latency actually demands a dedicated vector DB — see `TECH_STACK.md` for the switch criteria.

## 12. Open questions

- Does the hero/demo surface for this (if any) need a live streaming backend, or is a scripted/cached demo
  acceptable for the first internal milestone? (This mirrors the open decision point already flagged in your
  landing-page brief for the model-comparison tool — same underlying FastAPI/SSE streaming question, worth
  deciding once and reusing the answer.)
- Which corpus do you actually have ready to ingest first (your own project docs? a specific course's material?)
  — the Phase 1 eval set (§4) depends on this being a real, not hypothetical, dataset.
