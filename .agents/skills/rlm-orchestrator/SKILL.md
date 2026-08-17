---
name: rlm-orchestrator
description: Implements the Deep Context Platform's Recursive Language Model (RLM) engine — the host/kernel authority boundary, async subagent spawn-and-message-passing, depth-1-default recursion, the bounded REPL, and the structured answer contract. Use this skill only when a task genuinely requires reading or aggregating over more material than fits in any model's context window (e.g. "does any file in this repo do X," full literature-review aggregation, legal-discovery-style "read everything" tasks) — never as a default retrieval upgrade. Also use it when implementing, calling, or debugging rlm_spawn(), agent_messages, the sandboxed kernel, or the router's RLM-vs-hybrid decision (FR11–FR16). If you're unsure whether a query needs this vs. skills/rag-retrieval/, read docs/CRITICAL_ASSESSMENT_AND_SCOPE.md §1 first.
---

# RLM Orchestrator Skill

Implements FR11–FR16 of `docs/PRD.md`, §4–§5 of `docs/ARCHITECTURE.md`, and
`workflows/03_rlm_recursion_pipeline.md`. **Read this whole SKILL.md before writing any RLM code** — the two
most common ways an from-scratch RLM build goes wrong (treating this as a default upgrade, and treating the
kernel process boundary as a security sandbox) are both addressed below, and both are cheap to get right up
front and expensive to retrofit.

## Before you build: is this the right tool for the task in front of you?

**Almost certainly not, unless the task is specifically "read everything."** Per
`docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §1, none of the Deep Context Platform's own stated projects (a study
agent, an autonomous workforce system, a month-1 deployed Agentic RAG Assistant) need this skill — they're fully
served by `skills/rag-retrieval/` and `skills/typed-memory/` alone. Reach for this skill when, and only when,
one of the router conditions in `workflows/03_rlm_recursion_pipeline.md` genuinely fires:

```text
route_to_rlm = (
    estimated_corpus_tokens > model_effective_context_budget
    OR task_requires_global_aggregation
    OR hybrid_retrieval_has_already_failed_twice
)
```

## What "RLM" means here, precisely — and what's actually verified

Per the verified paper abstract (`docs/VERIFICATION_AND_SOURCES.md` §3): an RLM treats a long prompt as part of
an **external environment** rather than loading it into the model's context, and lets the model programmatically
examine, decompose, and recursively call itself over snippets. It is a pattern, not a specific product.

Three corrections this skill bakes in, because the naive mental model gets them wrong:

1. **Subagent spawn is async, not blocking.** `await rlm_spawn(task_spec)` returns an admission handle
   *immediately* and never returns the child's answer directly — replies arrive later via
   `agent_message.send(...)`. This is verified, not assumed (`docs/VERIFICATION_AND_SOURCES.md` §2). Do not
   implement `rlm_spawn` as `results = [sub_lm(x) for x in xs]`.
2. **Recursion depth defaults to 1.** Root can spawn children; children cannot spawn grandchildren unless
   `max_recursion_depth` is explicitly raised. Deeper recursion is future work in the source research, not a
   shipped, load-bearing feature — treat it as an explicit, budget-gated opt-in, not a default.
3. **The RLM scaffold is not a uniform accuracy upgrade.** Prime Intellect's own production ablation found it
   made GPT-5-mini *worse* at math-python and made DeepDive *worse* without a hand-written strategy prompt, and
   it is **always slower**. Budget real time for prompt iteration per task type; do not assume a strict upgrade
   over `skills/rag-retrieval/` without measuring both on the actual task first.

## The host/kernel boundary — the part that's easy to skip and shouldn't be

The process that runs model-generated Python (the **kernel**) must never hold the credentials or write-access of
the process that owns retrieval permissions, memory writes, and provider API keys (the **host**). This is
adapted from Prime Agent's verified TypeScript-host/Python-kernel split, implemented here as two Python
processes (or a process + sandboxed subprocess/container) — see `docs/ARCHITECTURE.md` §4 for the full diagram.

**This is not boilerplate.** Prime Agent's own docs are explicit that their kernel process "runs
model-generated Python and project commands with the worker's operating-system permissions... It is a durable
control environment, not a security sandbox." If the reference implementation says this about *itself*, treat
it as doubly true for a from-scratch build with no production hardening yet behind it.

Sandboxing tiers (`docs/TECH_STACK.md` §7) — escalate as soon as input isn't 100% your own:

1. **Local dev / your own trusted documents:** subprocess, restricted `PYTHONPATH`, scratch-directory-only
   filesystem, network egress on an explicit allowlist.
2. **Anything touching a document/repo you didn't author:** container isolation (Docker, dropped capabilities,
   read-only root filesystem) at minimum.
3. **Production / untrusted input at scale:** `gVisor`/`nsjail`-style syscall isolation, or a hosted sandbox
   (E2B, Modal) — what the RLM paper's own implementation actually uses.

## Five concrete mechanisms (each taken from a verified source — see `references/verified_facts.md`)

1. **Corpus-as-variable, not corpus-in-prompt.** The target material loads into the kernel's namespace as a
   Python object at session start. The model's prompt stays small and stable; only kernel-side working state
   grows.
2. **Bounded REPL output** — default 8,192 characters/turn, configurable. Confirmed as Prime Intellect's own
   production default. This forces search/filter/delegate behavior; don't raise it to work around a hard case
   (see `workflows/03_rlm_recursion_pipeline.md`'s failure-modes section).
3. **Async subagent spawn** — `await rlm_spawn(task_spec)` → `{child_id, name, session_dir, model}` immediately,
   non-blocking, no synchronous answer. Prefer `llm_batch()`-style explicit parallel dispatch when you already
   know you want N workers on N independent chunks.
4. **Structured, editable answer.** The model writes to `answer["content"]`, can revise it across turns, and the
   rollout ends only when it sets `answer["ready"] = True`.
5. **Depth defaults to 1** — see correction #2 above.

## Files in this skill

- `references/verified_facts.md` — the corrections table from `docs/VERIFICATION_AND_SOURCES.md`, condensed to
  what you need while writing orchestrator code (don't re-derive these from memory; the granular benchmark
  numbers in the source paper's body were explicitly *not* independently verified — cite only the abstract-level
  figures).
- `scripts/rlm_host_bridge.py` — reference implementation of the async host bridge: session admission, the
  typed host-request contract (`retrieve()`, `memory.*()`, `rlm_spawn()`, `agent_message.send()` as thin kernel
  stubs), depth enforcement, and budget checks. Deliberately **not** built on LangGraph — see
  `docs/TECH_STACK.md` §8 for why the async subagent/message-passing model doesn't fit a graph-walk abstraction.

## What NOT to do

- Don't implement `rlm_spawn` as a blocking call. It breaks the parallelism and crash-isolation properties that
  are the entire point of the async model.
- Don't let recursion depth be enforced only on the kernel side — a compromised or confused kernel should not be
  able to raise its own budget (`docs/ARCHITECTURE.md` §8). Enforce `max_recursion_depth` on the host, on every
  `rlm_spawn` call.
- Don't skip the evidence-sufficiency gate for RLM answers because "it already read everything." The same gate
  in `skills/rag-retrieval/references/retrieval_algorithm.md` applies here — nothing about recursive delegation
  makes a claim more true on its own.
- Don't route a query here "to be safe." Per the verified ablation, that's the most common way this architecture
  goes wrong in practice — see `docs/ARCHITECTURE.md` §5.4.
