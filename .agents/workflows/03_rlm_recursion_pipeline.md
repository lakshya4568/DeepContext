---
description: RLM Recursion Pipeline
---

# Workflow: RLM Recursion Pipeline

The escalation path for tasks that require reading _everything_ in a large corpus, not just the top-k most
similar chunks. Implements FR11–FR16. This is the least mature, highest-risk pipeline in the platform — read
`docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` §2 and `docs/VERIFICATION_AND_SOURCES.md` before treating anything here
as a settled, drop-in upgrade over hybrid retrieval.

## Trigger (the router decision, FR16)

This pipeline is reached from exactly three places, per `docs/ARCHITECTURE.md` §5.4 — never as a default:

```text
route_to_rlm = (
    estimated_corpus_tokens > model_effective_context_budget
    OR task_requires_global_aggregation        # "every file", "every section", "don't miss anything"
    OR hybrid_retrieval_has_already_failed_twice   # see 02_retrieval_pipeline.md step 9
)
```

The classifier call that produces this decision is the **same call** described in step 1 of
`02_retrieval_pipeline.md` — one classifier, two consumers, not two separate classification passes. Log the
router's reasoning (which condition fired) to `events_trace` with `event_type = 'router_decision'` every time,
including the times it decides _not_ to route here — that's the data `router precision` (PRD §4) is measured
against.

## Steps

```text
1. Admit an RLM session
     → create a `sessions` row (docs/DATA_MODEL.sql) with parent_session_id = the calling session's id
     → allocate a sandboxed kernel process/container per TECH_STACK.md §7, tier chosen by trust level of the
       input (your own docs → subprocess; anything you didn't author → container minimum)
     → set budgets: max_turns, max_tokens, max_wall_clock_seconds, max_recursion_depth (default 1) — these live
       on the HOST (agents.budgets / session-level override), never on the kernel, per ARCHITECTURE.md §8

2. Load the corpus as a variable, not into the prompt
     → the target material (document set, repo tree, dataset) is fetched by the host and injected into the
       kernel's persistent namespace as a Python object before the first model turn
     → the model's SYSTEM/user prompt stays small and stable for the whole session; only kernel-side working
       state grows — this is the property that makes RLM different from "just use a bigger context window"

3. Parent model turn (runs in a loop, one kernel-tool-call per turn)
     → the model writes Python against the loaded corpus: search, filter, grep, slice, transform
     → kernel stdout returned to the model is capped at 8,192 characters/turn (default, configurable) — this
       is deliberate: it forces search/filter/delegate behavior instead of "print the whole corpus and hope"
     → any host-authority action (retrieve(), memory.*(), rlm_spawn(), agent_message.send()) goes through the
       typed host-request bridge (ARCHITECTURE.md §4) — the kernel process itself never touches Postgres or a
       provider API key directly

4. (Optional, per-turn) Spawn subagents for independent sub-problems
     → await rlm_spawn(task_spec) returns {child_id, name, session_dir, model} IMMEDIATELY — non-blocking,
       and it does NOT return the child's answer synchronously (verified behavior, see
       docs/VERIFICATION_AND_SOURCES.md §2 — this corrects the naive
       `results = [sub_lm(x) for x in xs]` mental model)
     → insert a row into rlm_children (parent_session_id, child_session_id, name, model, depth) — depth is
       parent_depth + 1 and MUST be rejected by the host if it would exceed max_recursion_depth (default 1)
     → the parent can spawn several children in the same turn and end its turn without awaiting any of them —
       use this for genuinely independent sub-chunks (e.g. one child per document in a 40-paper literature
       review), not for steps that depend on each other's output
     → prefer `llm_batch()`-style explicit parallel dispatch when you already know you want N workers on N
       known-independent chunks, rather than N sequential `await rlm_spawn()` calls

5. Children run independently and reply by message, not return value
     → each child is a full session (own context, own kernel if it needs one) — NOT a thread; a child that
       hangs, loops, or errors cannot corrupt the parent's state (ARCHITECTURE.md §8)
     → a child's ONLY way to hand results back is `agent_message.send(receiver_role='parent', content=...)`,
       written to `agent_messages` — the parent picks these up between its own turns, not via a blocking return
     → children cannot spawn grandchildren unless max_recursion_depth was explicitly raised above 1 for this
       session — this is the current, shipped default of the reference implementation, not an arbitrary limit
       this platform invented

6. Parent collects messages and updates its structured answer
     → the model does not "return" its final answer as a normal text completion from inside this loop
     → it writes/edits `answer["content"]` across multiple turns (string replace, targeted correction,
       re-verification against a child's reply) and the rollout only ends when it explicitly sets
       `answer["ready"] = True`
     → this buys cheap self-correction for verbatim-reproduction or exact-aggregation tasks without a second
       full model call — treat premature `ready = True` (before all spawned children have replied, for an
       aggregation task) as a bug the kernel harness should guard against, not just the model's discipline

7. Evidence-sufficiency check (the SAME shared gate as the hybrid path — ARCHITECTURE.md §7)
     → coverage requirement for aggregation queries: evidence must span the FULL loaded/spawned set, not a
       sampled subset — this is the one place the RLM path's sufficiency check is stricter than the hybrid
       path's, because "read everything" was the reason this pipeline was invoked in the first place
     sufficient?
       Yes → return content as the answer, with citations back to source chunk/file/child identifiers
       No  → same corrective behavior as 02_retrieval_pipeline.md step 9: at most one more loop before
             returning "insufficient evidence" rather than a guess — RLM does not get an exemption from this
             just because it's the expensive path

8. Teardown
     → completed children: mark rlm_children.status = 'completed', completed_at = now()
     → children whose context is no longer needed: delete_subagent() frees kernel/session resources —
       nothing about a child is assumed to live forever by default (ARCHITECTURE.md §5.5)
     → parent session: mark sessions.status = 'completed', write one memory_episode row summarizing the task
       and outcome (this is what step 8's "episodic summary" write actually is, in FR7/FR9 terms)
     → every retrieval, spawn, message, and host-request in this whole pipeline should already be sitting in
       events_trace by this point (NFR4) — teardown does not need to write anything extra for audit purposes
```

## Why this pipeline is a deliberate, expensive escalation and not a general upgrade

Restated from `docs/CRITICAL_ASSESSMENT_AND_SCOPE.md` because it's the single easiest thing to forget once the
pipeline exists and works: per Prime Intellect's own production ablation, the RLM scaffold **increases latency
in all cases** and **actively hurt performance on some tasks** for some models (math-python, DeepDive without a
hand-written strategy prompt). None of your current stated projects (study agent, autonomous workforce, a
month-1 deployed Agentic RAG Assistant) have described a requirement that needs this pipeline yet. Build it —
it's specified here and in `skills/rlm-orchestrator/` in full — but don't let a query reach it unless one of the
three router conditions above genuinely fired.

## Failure modes to design for

- **A child never replies.** The parent's turn budget (`max_turns`) and wall-clock budget
  (`max_wall_clock_seconds`) must expire the _parent's_ wait, not hang indefinitely — a child that's stuck is
  the child's problem, not a reason to block the whole session. Mark the child `status = 'error'` on timeout and
  let the parent proceed with whatever it already has, or fail the sufficiency check honestly.
- **Recursion depth silently exceeds the config.** Enforce `max_recursion_depth` as a host-side check on every
  `rlm_spawn` call (compare requested depth against the config, not against what the kernel claims) — a
  compromised or confused kernel should not be able to raise its own budget (ARCHITECTURE.md §8).
- **`answer["ready"] = True` fires before all spawned children have replied**, on an aggregation task. Guard
  this explicitly in the harness: if the task classification was "global aggregation" and any spawned child is
  still `status IN ('admitted', 'running')`, refuse to accept `ready = True` and surface that back to the model
  as a kernel-level message, not just trust the model's own bookkeeping.
- **The 8,192-character REPL cap gets raised "temporarily" to work around a hard case.** This is very tempting
  and is exactly the failure mode the cap exists to prevent — it's the mechanism that forces search/filter
  behavior instead of "print everything." If a task keeps hitting the cap, that's a signal to spawn a child for
  the sub-problem, not to raise the cap.
