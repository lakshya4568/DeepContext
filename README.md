# Deep Context Platform — RAG + RLM Architecture Package

Agent backend design: hybrid RAG for everyday retrieval, typed long-term memory for continuity across sessions,
and an RLM (Recursive Language Model) engine for the minority of tasks that require reading more material than
fits in any model's context window — fronted by FastAPI, exposed to agents as composable `SKILL.md` skills.

Verified against the actual `PrimeIntellect-ai/prime-agent` repository and the RLM paper (arXiv:2512.24601) on
17 August 2026 — see `docs/VERIFICATION_AND_SOURCES.md` for exactly what was confirmed, corrected, and left
unverified. **Read `docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` before you start building** — it's the honest answer
to "should I actually build all of this," not just "here's the full architecture you asked for."

## Where to start

| If you want to... | Read |
|---|---|
| Understand what this platform does and why, requirements, phased release plan | `docs/PRD.md` |
| Understand *whether you should build all of it*, and in what order | `docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` |
| Understand the technical design: components, host/kernel boundary, memory, RLM engine | `docs/ARCHITECTURE.md` |
| See exactly what was verified vs. corrected vs. not re-checked, and against which sources | `docs/VERIFICATION_AND_SOURCES.md` |
| Pick concrete tools (language, DB, reranker, sandboxing, orchestration) and know when to switch | `docs/TECH_STACK.md` |
| Get the runnable Postgres schema | `docs/DATA_MODEL.sql` |
| Follow the exact step-by-step pipelines | `workflows/` |
| Get Claude/Prime-Agent/LangGraph-loadable `SKILL.md` packages for every subsystem | `skills/` |
| See the system diagrams | `diagrams/` |
| Read the original research brief this package extends | `docs/source-verdict-aug-2026.md` |

## Directory map

```text
├── README.md                          — this file
├── docs/
│   ├── PRD.md                         — product requirements: goals, FR1–FR23, NFR1–NFR6, phased release plan
│   ├── ARCHITECTURE.md                — technical design: layers, host/kernel boundary, RLM engine, memory
│   ├── TECH_STACK.md                  — concrete tool choices + explicit switch criteria for each
│   ├── DATA_MODEL.sql                 — full runnable Postgres 15 + pgvector schema
│   ├── CRITICAL_ASSESSMENT_AND_SCOPE.md — honest scope recommendation; read this first
│   ├── VERIFICATION_AND_SOURCES.md    — what was fact-checked against primary sources, and how
│   └── source-verdict-aug-2026.md     — the original Perplexity research brief this package extends
├── workflows/
│   ├── 01_ingestion_pipeline.md       — raw document → searchable chunks (FR2, FR5, FR6)
│   ├── 02_retrieval_pipeline.md       — the default hybrid retrieval path (FR1, FR3–FR6)
│   └── 03_rlm_recursion_pipeline.md   — the RLM escalation path (FR11–FR16)
├── diagrams/
│   ├── system_architecture.mermaid    — rendered version of ARCHITECTURE.md §2
│   └── rlm_core_loop.mermaid          — rendered version of ARCHITECTURE.md §5.2
└── skills/
    ├── rag-retrieval/                 — hybrid retrieval: retrieve() contract + reference implementation
    ├── typed-memory/                  — four-store memory + promotion gate
    ├── rlm-orchestrator/              — host/kernel boundary, async subagent spawn, RLM core loop
    ├── verification/                  — evidence-sufficiency gate + fact-checking methodology
    ├── code-execution/                — sandboxing tiers and policy for the RLM kernel
    └── refinement/                    — bounded corrective-retrieval retry + /refine-analogous loop
        (each skill/ folder is a self-contained SKILL.md package per the open Agent Skills
         standard — loadable by Claude, Claude Code, or a Prime-Agent-style harness as-is)
```

## The one-paragraph version

Route every request through a cheap classifier first (`docs/ARCHITECTURE.md` §1, principle 1). Simple questions
go to hybrid retrieval (BM25 + vector + RRF + reranking). Multi-step tasks go to an agentic planner. Only tasks
that genuinely require reading an entire large corpus — where missing one section invalidates the answer — go
to the RLM engine, which is slower and less consistent by design (`docs/VERIFICATION_AND_SOURCES.md` §2) and
should never be the default. Every path terminates at the same evidence-sufficiency gate before returning an
answer. Nothing becomes durable memory without passing through the promotion gate. **Phase 1 alone (hybrid
retrieval + corrective retry, no RLM) is a complete, deployable Agentic RAG Assistant** — see
`docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §5 for the recommended build order.

## Recommended build order

1. **Phase 1 (weeks 1–3):** ingestion + hybrid retrieval + reranking + corrective retry + FastAPI interface.
   Ship this alone as a first milestone — `skills/rag-retrieval/` covers it in full.
2. **Phase 2 (weeks 4–6):** full typed memory + promotion gate (`skills/typed-memory/`), query classifier +
   agentic planner, evidence verifier (`skills/verification/`).
3. **Phase 3 (weeks 7–10, highest risk — build only when you hit the actual need):** the RLM engine
   (`skills/rlm-orchestrator/`, `skills/code-execution/`) — fully specified here so the *design* cost is zero
   when that day comes, per `docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §1.
4. **Phase 4 (weeks 11+):** `skills/refinement/`'s supplemental-state loop, skill registry versioning, expanded
   eval harness against `docs/PRD.md` §4 metrics.
