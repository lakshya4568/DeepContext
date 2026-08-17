# Deep Architecture — Deep Context Platform

This document is the technical companion to `PRD.md`. It specifies components, boundaries, data flow, and the
concrete design decisions that came out of checking Prime Agent and the RLM paper directly (see
`VERIFICATION_AND_SOURCES.md`) rather than trusting a secondhand description of either.

---

## 1. Design principles

These five principles are referenced by name throughout the rest of this document.

1. **Route before you pay.** Every request passes a cheap classifier before it reaches an expensive mechanism.
   Hybrid retrieval is the default; the agentic planner and the RLM engine are escalations, not entry points.
2. **Separate authority from execution.** The process that holds credentials, writes to durable memory, and calls
   model providers ("the host") is never the same process that runs model-generated code ("the kernel"). The
   kernel talks to the host through a small, typed set of requests. This is adapted directly from Prime Agent's
   TypeScript-host/Python-kernel split (see §4) — implemented here in pure Python, because the *separation*, not
   the *language boundary*, is what buys you the safety property.
3. **Async subagents, not blocking recursion.** Spawning a subagent returns a handle immediately. Results come
   back as messages. This is a correction to the naive `results = [sub_lm(x) for x in xs]` mental model — see
   §5 and `VERIFICATION_AND_SOURCES.md` §2.
4. **Nothing durable without a gate.** No text a user or model produces becomes long-term memory without passing
   through the promotion gate (§6). No answer is returned without an evidence-sufficiency check (§7).
5. **Every capability is a skill.** Internal subsystems are exposed as the same `SKILL.md`-packaged units this
   platform ships externally (`skills/`). There is no special internal API that skills don't also get.

---

## 2. System at a glance

```text
                                   Client (LangGraph graph, Claude Code / MCP,
                                    Prime-Agent-style harness, plain HTTP)
                                                    │
                                          FastAPI ingress (FR22)
                                                    │
                                        Query & Task Classifier
                                    (routes on shape, not on hype)
                                                    │
                ┌───────────────────────────────────┼────────────────────────────────────┐
                │                                    │                                    │
         Simple QA path                     Multi-step task path                 Global-analysis path
                │                                    │                                    │
        Hybrid Retrieval                     Agentic Planner                        RLM Engine
    BM25 + vector + RRF                  tool selection, iteration                Host + Kernel
    + cross-encoder rerank                limits, multi-hop loop              (see §4, §5)
                │                                    │                                    │
                └────────────────────► Evidence-Sufficiency Check ◄────────────────────────┘
                                                    │
                                          Sufficient?  ── No ──► Rewrite query / widen filters /
                                                    │                escalate mechanism (max 1 retry
                                                   Yes               before "I don't have enough
                                                    │                 evidence" is a valid answer)
                                        Generate answer + citations
                                                    │
                                          Memory Extraction (background)
                                                    │
                                          Promotion Gate (§6)
                                                    │
                        ┌───────────────────────────┼───────────────────────────┐
                        │                            │                           │
                Short-term / session          Long-term typed memory      Episodic summary
                    state (Redis or               (Postgres:                  + trace log
                     in-process)               policy/pref/fact)           (append-only)
```

This is the same shape as the routing sketch in the source brief, with one structural change: the evidence
sufficiency check is now a **single shared gate** all three paths pass through, rather than something the RAG
path does and the RLM path has to remember to also do. Section 7 explains why that matters.

A rendered version of this diagram is in `diagrams/system_architecture.mermaid`.

---

## 3. Layered view

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Presentation                                                              │
│   FastAPI HTTP routes  ·  optional CLI (Typer/Rich) for local dev         │
├─────────────────────────────────────────────────────────────────────────┤
│ Orchestration ("the host" — owns authority, never runs model-generated    │
│ code directly)                                                            │
│   Query classifier · Agentic planner · RLM host bridge · Session/goal     │
│   manager · Skill registry · Verification engine                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Execution                                                                 │
│   Sandboxed kernel (RLM's Python REPL)  ·  Skill runtime (imports skill   │
│   packages)  ·  Tool adapters (retrieval, memory, shell-in-sandbox, SQL)  │
├─────────────────────────────────────────────────────────────────────────┤
│ Knowledge & Memory                                                        │
│   Ingestion & chunking · Hybrid index (BM25 + vector) · Reranker ·        │
│   Vectorless/tree index · Typed memory stores · Trace log                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Infrastructure                                                            │
│   Postgres + pgvector  ·  Redis (session cache)  ·  Object storage        │
│   (raw documents)  ·  Model providers (Claude / GPT-class APIs)           │
└─────────────────────────────────────────────────────────────────────────┘
```

The load-bearing line is between **Orchestration** and **Execution**. Everything above it can be trusted with
credentials. Everything at or below it should be treated as running arbitrary, possibly-adversarial code (because
eventually it will: a document you ingest could contain a prompt injection, and a query answered by the RLM
kernel is, by construction, running model-written Python against your data).

---

## 4. The host/kernel boundary (adapted from Prime Agent)

**What the reference implementation actually does** (verified, not assumed — see
`VERIFICATION_AND_SOURCES.md` §2): Prime Agent's control plane — session management, provider calls, persistence,
scheduling, safety policy — is TypeScript. The model's only default tool is a persistent IPython kernel, described
in their own docs as *"the model-facing programming surface."* Anything the kernel needs that requires authority
(sending an agent message, changing a goal, compacting context) goes through `rlm.host_request(...)`, which the
TypeScript host validates before acting. Their own architecture doc is explicit that *"workers and kernels are
separate processes for lifecycle and failure containment, not security sandboxes"* — i.e., even in the reference
implementation, process separation buys you crash isolation, not a security boundary by itself.

**What this platform does instead:** the same separation, implemented as two Python processes (or a process and a
sandboxed subprocess/container) rather than a TypeScript/Python pair, because your stack is Python end to end and
introducing a second language for the control plane would add real cost for no benefit at this scale.

```text
┌─────────────────────────────┐        typed request         ┌─────────────────────────────┐
│           HOST               │◄─────────────────────────────│           KERNEL              │
│  (FastAPI process)            │                              │  (sandboxed subprocess or      │
│                               │──────────────────────────────►│   container; runs model-       │
│  • Holds provider API keys    │        validated result       │   generated Python)             │
│  • Owns Postgres/Redis writes │                              │                               │
│  • Enforces permission scopes │                              │  • Persistent namespace across  │
│  • Runs the promotion gate    │                              │    turns (variables, imports)   │
│  • Talks to model providers   │                              │  • `retrieve()`, `memory.*()`,  │
│  • Never executes model-      │                              │    `rlm_spawn()` exposed as      │
│    generated code directly    │                              │    thin client stubs that just  │
│                               │                              │    forward to the host           │
└─────────────────────────────┘                              └─────────────────────────────┘
```

Concretely: functions like `retrieve()`, `memory.save_fact()`, and `rlm_spawn()` that the model calls *inside* the
kernel are thin stubs. They serialize the call, send it to the host over a local queue/socket, and the host is the
one that actually touches Postgres, checks tenant/permission scope, and calls the model provider for a sub-LLM. The
kernel process itself never holds a database credential or an API key. If the kernel is compromised (a malicious
document tricks the model into writing destructive code), the blast radius is the sandbox, not your data.

**Practical implementation note (NFR3):** for a solo/local build, "the kernel is sandboxed" can start as simple as
running the REPL in a subprocess with a restricted `PYTHONPATH`, no filesystem access outside a scratch directory,
and network egress disabled except to an explicit allowlist (mirroring this very platform's own
`network_configuration` model, if you're building this inside a Claude-tool-enabled environment). Before this
touches any document you didn't write yourself, upgrade to real container isolation (Docker with dropped
capabilities, gVisor, or a hosted sandbox like E2B/Modal). Don't skip this step because the reference
implementation itself skips it — they tell you not to, in their own trust-model docs.

---

## 5. The RLM engine

### 5.1 What "RLM" means here, precisely

Per the verified paper abstract: an RLM **treats a long prompt as part of an external environment** rather than
loading it into the model's context, and lets the model **programmatically examine, decompose, and recursively
call itself over snippets**. It is not a specific product — it's an inference pattern. This platform's RLM engine
is one implementation of that pattern, structured after what actually ships in Prime Agent, corrected for the
async-not-blocking subagent model.

### 5.2 Core loop

```text
flowchart
    task[Task + working context] --> parent[Parent model turn]
    parent -->|writes Python| kernel[Persistent kernel: corpus lives here as a variable]
    kernel <-->|search / filter / transform| data[Loaded corpus: files, DB rows, repo tree]
    kernel -->|await rlm_spawn(subtask)| spawn[Admission handle returned immediately]
    spawn -.->|async, non-blocking| children[Child sessions run independently]
    children -->|agent_message.send(...)| parent
    kernel -->|REPL stdout, capped at 8192 chars/turn| parent
    parent -->|writes to| answer["answer = {content, ready}"]
    answer -->|ready == True| final[Return content as the answer]
    answer -->|ready == False| parent
```

### 5.3 Five concrete mechanisms, each taken from a verified source

1. **Corpus-as-variable, not corpus-in-prompt.** At the start of an RLM session, the target material (document
   set, repo, dataset) is loaded into the kernel's namespace as a Python object — not pasted into the model's
   prompt. The model's *prompt* stays small and stable across the whole session; only its *working state* grows.

2. **Bounded REPL output (default 8,192 characters/turn, configurable).** This is not an arbitrary safety limit —
   it's the mechanism that *forces* the model to search/filter/aggregate instead of trying to read the whole
   corpus by printing it. Confirmed as Prime Intellect's own production default.

3. **Async subagent spawn.** `await rlm_spawn(task_spec)` returns `{child_id, name, session_dir, model}`
   immediately. It does **not** block, and it does **not** return the child's answer. The parent can spawn several
   children in the same turn and end its turn without waiting — exactly matching the verified behavior of
   `rlm(...)` in Prime Agent. Batched, explicitly-parallel sub-calls (`llm_batch()` in the research
   implementation) are the right primitive when you already know you want N parallel workers on N independent
   chunks, rather than N sequential `await`s.

4. **Structured, editable answer.** The model never "returns" its final answer as a normal text completion from
   inside the RLM loop. It writes to `answer["content"]`, can revise that value across multiple turns (string
   replace, targeted edits, re-verification), and the rollout only ends when it explicitly sets
   `answer["ready"] = True`. This buys cheap self-correction (e.g. for verbatim-reproduction or exact-aggregation
   tasks) without a second model call.

5. **Depth defaults to 1.** Root can spawn children. Children cannot spawn grandchildren unless you explicitly
   raise `max_recursion_depth`. This matches the reference implementation's *current, shipped* default — Prime
   Intellect's own research post lists arbitrary depth as future work, not something to assume is battle-tested.

### 5.4 When the router sends a query here

See `workflows/03_rlm_recursion_pipeline.md` for the full decision procedure. Summary condition:

```text
route_to_rlm = (
    estimated_corpus_tokens > model_effective_context_budget
    OR task_requires_global_aggregation        # "every row", "every section", "don't miss anything"
    OR hybrid_retrieval_has_already_failed_twice
)
```

If none of these hold, the query stays on the hybrid or agentic path. This is the single most important guardrail
in the whole system, because — per the verified ablation study — the RLM path is **always slower**, and its
accuracy benefit is **inconsistent and task-dependent today**, not a uniform upgrade. Sending every query through
it "to be safe" is the most common way this architecture would go wrong in practice.

### 5.5 Session/child lifecycle

- A child's admission handle is retained in a `rlm_children` table (see `docs/DATA_MODEL.sql`), scoped to its
  parent session, and survives context compaction — the parent can list active/completed children
  (`list_subagents()`) and re-message a retained one instead of always spawning fresh.
- A child is deleted (`delete_subagent()`) once its context is no longer needed, freeing whatever resources it
  held. Nothing about a child is assumed to live forever by default.

---

## 6. Typed memory subsystem

Plain RAG is retrieval; it has no write path. Memory requires one. Four types, each with different retrieval
semantics and write discipline (this is unchanged from the source brief's taxonomy — it held up well under
verification and is a genuinely good design):

| Type | Retrieval | Write path | Notes |
|---|---|---|---|
| **Policy** | Exact key lookup by tenant/user | Never inferred — set explicitly by an operator | e.g. "refunds above ₹5,000 require approval" |
| **Preference** | Exact key lookup by user | Can be inferred, but requires explicit confirmation or repeated signal (≥2 observations) before promotion | e.g. "prefers concise answers", "codes in TypeScript" |
| **Semantic fact** | Hybrid (BM25 + vector) | Through the promotion gate (§6.1); always carries source, confidence, TTL, scope | e.g. "this project uses PostgreSQL and Redis" |
| **Episodic summary** | Hybrid, keyed loosely by task similarity | Written automatically at the end of a completed task/session | e.g. "debugged JWT filter ordering bug; fix: auth filter before authz filter" |

A fifth store, the **trace log**, is append-only and is explicitly *not* one of the four semantic memory types —
it's for replay, audit, and as the raw material episodic summaries get extracted from. It is never injected
directly into a prompt.

### 6.1 Promotion gate

```text
observation
  → classify type (policy / preference / fact / episode / discard)
  → check user/tenant scope
  → extract the atomic claim (not the raw sentence)
  → compare against existing memory for that scope
  → contradiction found? → resolve (supersede old, lower confidence, or reject new)
  → assign confidence + provenance + TTL
  → write (or reject, with reason logged to the trace)
```

Example the source brief got right and worth keeping verbatim as a design contract:

```text
Observed: "I might use MongoDB for this prototype."
Wrong promotion:  "User uses MongoDB."                     (overconfident, no TTL)
Right promotion:  "User is evaluating MongoDB for project X;
                    confidence 0.55; expires in 30 days."
```

### 6.2 Prompt assembly

Rebuilt every turn, in this fixed order, rather than an ever-growing append log:

```text
1. System / developer instructions
2. Active policies (exact lookup)
3. User preferences (exact lookup)
4. Short conversation summary
5. Current task state
6. Retrieved durable facts (hybrid search, top-k)
7. Retrieved documents/evidence (from the retrieval pipeline)
8. Current user query
```

This bounds context growth and is what keeps a long-running session from degrading the way a naively-appended
transcript would.

---

## 7. Evidence sufficiency: the shared gate

Both the hybrid path and the RLM path terminate at the same check before an answer is returned:

```text
def sufficient(evidence, draft_answer) -> bool:
    # 1. Every non-trivial claim in draft_answer must be traceable to
    #    a retrieved chunk, a computed value, or explicitly flagged
    #    as the model's own inference.
    # 2. Coverage: for aggregation-style queries, evidence must span
    #    the full retrieved/loaded set, not a sampled subset.
    # 3. Confidence: the verifier's own score must clear a threshold.
    ...
```

On failure: rewrite the query and retry retrieval **once**. On a second failure, the answer is allowed to say "I
don't have enough evidence to answer this confidently" — which is a correct, shippable answer, not a bug to be
engineered away. Silently answering anyway is the actual bug this gate exists to prevent.

---

## 8. Concurrency model

- The **host** is a normal async FastAPI app — I/O-bound work (DB queries, provider calls) is `async`/non-blocking
  throughout.
- The **kernel** runs in its own process/container per active RLM session; a session's kernel is not shared across
  requests from different users.
- **Subagents are not threads.** Each is a full child session (its own context, its own kernel if it needs one),
  communicating with its parent only through the typed message-passing interface (§5.3). This is deliberately
  heavier than a thread pool because it's what gives you crash isolation: a child that hangs, loops, or errors
  cannot corrupt the parent's state.
- Budget enforcement (max children, max depth, max wall-clock time per session) lives in the **host**, not the
  kernel, for the same authority-separation reason as everything else in §4 — a compromised kernel should not be
  able to raise its own budget.

---

## 9. Extension points (deliberately not built in v1)

- **GraphRAG / entity-relationship layer.** Schema stub included in `docs/DATA_MODEL.sql` (commented out). Add
  only when a real multi-hop, entity-heavy use case shows up — see NG5 in the PRD.
- **Multiple vector backends.** The data model and retrieval skill are written against pgvector but the
  `retrieve()` interface doesn't leak that — swapping in Qdrant/Milvus later is a retrieval-skill-internal change,
  not an API change. See `TECH_STACK.md` for the switch criteria.
- **Model-agnostic provider layer.** The host talks to model providers through a thin abstraction so Claude- and
  GPT-class APIs are both first-class, matching your own multi-provider roadmap (Groq/Claude APIs).

---

## 10. What changed vs. the source brief's sketch, and why

For anyone comparing this to `source-material/perplexity-verdict-aug-2026.md`:

- The host/kernel authority boundary (§4) is new — the source sketch had "Persistent REPL / Kernel" as one box
  with safeguards listed underneath it, without separating *who enforces the safeguards* from *what runs the
  code*. That distinction is the actual security-relevant one.
- The RLM subagent model is now async/message-passing (§5.3), not the synchronous list-comprehension pseudocode
  the source sketch used. This is a correction, not a style preference — it changes how you'd actually implement
  `rlm_spawn`.
- Recursion depth is now an explicit, small default (1) instead of an implied "spawn subagents as needed" — see
  `VERIFICATION_AND_SOURCES.md` for why.
- The evidence-sufficiency check is now one shared gate both paths pass through (§7), rather than something
  described only under the RAG corrective-retrieval section. RLM output needs exactly the same "did we actually
  support this claim" check that RAG output does — nothing about recursive delegation makes a claim more true.


