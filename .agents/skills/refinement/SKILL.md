---
name: refinement
description: Implements the Deep Context Platform's bounded corrective-retrieval and self-improvement refinement loop — deciding whether to retry a query once, escalate to the RLM engine, or return "insufficient evidence" honestly, plus the analogous /refine-style process for proposing small, reversible changes to supplemental state (memory entries, skill descriptions, prompt notes). Use this skill whenever an initial retrieval or answer attempt fails the evidence-sufficiency gate and you need to decide what happens next, or whenever proposing a change to a skill, memory entry, or prompt note based on observed evidence. Never use this skill to justify rewriting a base system prompt — that is explicitly out of scope.
---

# Refinement Skill

Implements FR4 (corrective retrieval) and FR19 (the `/refine`-analogous supplemental-state refinement pipeline)
of `docs/PRD.md`. Two related loops live here — keep them distinct.

## Loop 1: bounded corrective retrieval

This is the retry logic both `skills/rag-retrieval/` and `skills/rlm-orchestrator/` invoke when
`skills/verification/`'s evidence-sufficiency gate fails:

```text
attempt 1: retrieve → sufficient? → yes: answer. no: continue.
attempt 2 (the ONE allowed retry): rewrite query → retrieve → sufficient? → yes: answer. no: continue.
after attempt 2:
    if the query looked like "read everything" / aggregation → escalate to skills/rlm-orchestrator/
    else → return "I don't have enough evidence to answer this confidently"
```

**The retry bound is exactly 1, not a starting point to raise.** An unbounded "keep trying different queries"
loop is how a retrieval pipeline turns into an expensive infinite loop with nothing to show for it — this
warning appears independently in `workflows/02_retrieval_pipeline.md` and
`workflows/03_rlm_recursion_pipeline.md` because it's the single easiest discipline to erode under the pressure
of "just one more try might get it."

**Query rewriting on retry should target the specific failure**, not restate the same query with different
words. Use the `failure_reasons` from `skills/verification/scripts/check_answer_support.py`:

- "no evidence retrieved" → broaden filters (wider date range, drop an over-specific document_id constraint) or
  decompose into simpler sub-queries.
- "aggregation coverage below threshold" → this usually means the query needs `skills/rlm-orchestrator/`, not
  another hybrid-retrieval attempt — escalate rather than retrying the same mechanism.
- "claims unsupported" → the retrieved evidence exists but doesn't actually address the question; rewrite
  toward the specific missing sub-topic, not a paraphrase of the original query.

## Loop 2: supplemental-state refinement (`/refine`-analogous, FR19)

Proposes small, evidence-linked changes to state that is explicitly **not** the base system prompt:

- Memory entries (e.g. a `memory_fact` row whose confidence should be adjusted given new evidence, or a
  `memory_preference` that a corroborating observation should update)
- Skill descriptions (e.g. a skill's SKILL.md frontmatter `description` under-triggers or over-triggers — see
  `/mnt/skills/examples/skill-creator/` for the full description-optimization loop this platform's own skills
  should eventually go through)
- Prompt notes / supplemental context (not the system prompt itself)

```text
observe evidence of a gap or error
  → propose a specific, minimal change (not a rewrite)
  → link the change to the evidence that motivated it
  → the change must be REVERSIBLE — record what it replaces, not just the new value
  → apply
  → log to events_trace (event_type = 'memory_write' or a dedicated 'refinement' type)
```

**Hard boundary: this loop never rewrites the base system prompt.** If a proposed refinement would require
changing core instructions rather than supplemental state, that's a signal for a human to review a prompt
change deliberately, not something this loop applies automatically. This mirrors Prime Agent's own `/refine`
scoping and is a deliberate safety boundary, not an oversight — an agent that can rewrite its own core
instructions based on its own evidence of what "should" change is a fundamentally different (and much riskier)
system than one that can only adjust its supplemental memory and skill metadata.

## Relationship to `skills/verification/`

Refinement consumes verification's output (the `SupportCheckResult.failure_reasons`) but does not duplicate the
scoring logic — `skills/verification/` decides *whether* an answer/claim is sufficient; `skills/refinement/`
decides *what to do next* given that a check failed. Keep this division; merging them tends to produce a single
component that both grades its own homework and decides how hard to try again, which erodes the honesty of the
gate over time.

## What NOT to do

- Don't raise the retry bound past 1 to chase a marginally better answer on a hard query — escalate to RLM or
  return "insufficient evidence" instead, per Loop 1.
- Don't let Loop 2 touch the base system prompt under any framing ("just a small clarification," "just adding
  one sentence") — route that need to a human-reviewed prompt-change process instead.
- Don't apply a Loop 2 change without recording what it replaces — irreversible "refinements" are indistinguishable
  from silent drift over a long-running deployment.
