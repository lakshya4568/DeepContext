---
name: typed-memory
description: Implements the Deep Context Platform's four-store typed memory system (policy, preference, semantic fact, episodic summary) and the promotion gate that governs every write into durable memory. Use this skill whenever the agent needs to read persistent memory before answering, whenever an observation from a conversation or tool output might need to become a durable fact or preference, whenever resolving a contradiction between a new observation and existing memory, whenever assembling the per-turn prompt context, or whenever asked to design, implement, or debug memory writes, TTL expiry, or confidence scoring. Do not write directly to memory_fact or memory_preference from application code — always go through the promotion gate described here.
---

# Typed Memory Skill

Implements FR7–FR10 of `docs/PRD.md` and §6 of `docs/ARCHITECTURE.md`. The core idea this skill protects: plain
RAG is retrieval with no write path. Memory needs one, and an undisciplined write path is worse than none — see
`docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §3.2 on the promotion gate degrading into "vector RAG with extra steps."

## The four stores — never merge them

| Type | Table | Retrieval | Write path |
|---|---|---|---|
| **Policy** | `memory_policy` | Exact key lookup by tenant/user | Never inferred — set explicitly by an operator/admin path |
| **Preference** | `memory_preference` | Exact key lookup by user | Can be inferred, but requires ≥2 corroborating observations (or explicit user confirmation) before promotion |
| **Semantic fact** | `memory_fact` | Hybrid (BM25 + vector) | Through the promotion gate below; always carries `source`, `confidence`, `expires_at`, `superseded_by` |
| **Episodic summary** | `memory_episode` | Hybrid, keyed loosely by `task_type` similarity | Written automatically at the end of a completed task/session |

A fifth store, `events_trace`, is append-only and is **not** one of the four memory types — it's raw material
for episodic extraction and audit/replay. Never inject it directly into a prompt.

## The promotion gate (FR8) — the one rule that matters most

**Nothing is written to `memory_fact` or `memory_preference` without passing through this gate.** If you don't
have time to implement the full gate, ship three memory types with the gate fully implemented rather than four
types with a gate that's a rubber stamp — a schema that implies more rigor than the code delivers is worse than
an honestly smaller system (`docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §3.2).

```text
observation
  → classify type (policy / preference / fact / episode / discard)
  → check user/tenant scope
  → extract the atomic claim (not the raw sentence)
  → compare against existing memory for that scope
  → contradiction found? → resolve (supersede old, lower confidence, or reject new)
  → assign confidence + provenance + TTL
  → write (or reject, with reason logged to events_trace)
```

The design-contract example worth keeping verbatim (from `docs/ARCHITECTURE.md` §6.1):

```text
Observed: "I might use MongoDB for this prototype."
Wrong promotion:  "User uses MongoDB."                     (overconfident, no TTL)
Right promotion:  "User is evaluating MongoDB for project X;
                    confidence 0.55; expires in 30 days."
```

See `scripts/promotion_gate.py` for a reference implementation of every step above, and
`references/memory_schema.md` for the full column-level contract each store enforces.

## Prompt assembly (FR9)

Rebuilt every turn in this **fixed order** — not an ever-growing append log, which is what actually degrades a
long-running session:

```text
1. System / developer instructions
2. Active policies (exact lookup)
3. User preferences (exact lookup)
4. Short conversation summary
5. Current task state
6. Retrieved durable facts (hybrid search, top-k)
7. Retrieved documents/evidence (from skills/rag-retrieval/)
8. Current user query
```

## Contradiction resolution

When a new observation conflicts with an existing `memory_fact`:

1. **Prefer higher-confidence, more-recent, more-specific evidence** — but never silently discard; always set
   `superseded_by` on the old row rather than deleting it, so provenance is auditable.
2. If confidence is genuinely ambiguous (neither claim clearly wins), lower both confidences rather than picking
   one arbitrarily, and flag for the next promotion-gate pass to re-resolve once more evidence arrives.
3. Log the resolution to `events_trace` with `event_type = 'memory_write'` and the reasoning — this is what
   makes the `memory promotion precision` metric (`docs/PRD.md` §4, ≥95% of auto-promoted facts still valid 30
   days later) measurable rather than aspirational.

## TTL expiry

- `memory_fact.expires_at` — `NULL` means no TTL (rare; reserve for genuinely durable facts like "this project
  uses PostgreSQL"). Anything speculative, time-bound, or stated with hedging language ("might", "probably",
  "for now") gets a real TTL, not `NULL` by default.
- A background sweep (or a lazy check at read time) should treat `expires_at < now()` the same as
  `superseded_by IS NOT NULL` — excluded from retrieval, but not deleted, per the audit-trail principle above.

## Files in this skill

- `references/memory_schema.md` — column-level contract for all four tables plus `events_trace`, extracted from
  `docs/DATA_MODEL.sql` with the write-discipline notes that don't fit in SQL comments.
- `scripts/promotion_gate.py` — reference implementation of the classify → scope → extract → contradict →
  assign → write pipeline above.

## What NOT to do

- Don't let application code write to `memory_fact`/`memory_preference` directly, "just this once," bypassing
  the gate — the schema allows it (nothing in SQL stops you), but the discipline is enforced in code, and that
  discipline is the entire value of having four typed stores instead of one vector table
  (`docs/DATA_MODEL.sql` design notes).
- Don't skip the ≥2-observation requirement for inferred preferences to make the system "feel smarter" faster —
  a wrongly-inferred preference is worse than no preference, because it's silently wrong on every subsequent turn
  until corrected.
- Don't inject `events_trace` rows directly into a prompt — extract them into an episodic summary first.
