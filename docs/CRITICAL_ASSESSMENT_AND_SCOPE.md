# Critical Assessment & Realistic Scope

This document exists because "here's the full architecture you asked for" and "here's whether you should actually
build all of it" are different questions, and the second one deserves a straight answer rather than getting
buried inside 2,000 lines of design docs. Read this before you start Phase 1.

---

## 1. The honest headline

**You almost certainly don't need the RLM engine for the projects described in your own roadmap.** The study
agent, the autonomous workforce system, and a month-1 "deployed Agentic RAG Assistant" are all well served by
Phase 1 + Phase 2 (hybrid retrieval, reranking, corrective retrieval, typed memory) on their own. RLM earns its
keep specifically when a task requires reading *everything* in a large, static corpus and missing one section
invalidates the answer — legal discovery, "does any file in this repo do X," full literature-review aggregation.
None of your stated current projects have described that requirement yet.

That doesn't mean skip it — the request was explicitly for the full architecture, and Phase 3 is here, complete
and usable, whenever you do hit that requirement. It means: **don't let Phase 3 block or delay shipping Phase 1.**
If you build in the order this document recommends, you'll have a working, demoable, genuinely useful system
after Phase 1 alone.

---

## 2. Is RLM actually production-ready? A direct read of the evidence you asked me to check

You specifically asked me to check RLM's prior iterations on GitHub and the wider internet before building on it.
Here's the unvarnished version of what that check found (full detail in `VERIFICATION_AND_SOURCES.md`):

- **It's genuinely new.** The originating blog post is from October 2025; the full paper was submitted 31 December
  2025 and last revised in May 2026. Prime Agent, the flagship product built around it, is likewise a matter of
  months old relative to today's date (17 August 2026).
- **The paper's own headline results are strong** — median improvements of 26% over context compaction, 130% over
  CodeAct-with-sub-calls, and 13% over Claude Code on GPT-5, across four long-context tasks, at comparable cost.
  That's a real, citable, abstract-level result, not marketing copy.
- **But Prime Intellect's own production ablation (their words, their blog, published the same month as the
  paper) found the opposite result on some tasks.** Using the RLM scaffold made GPT-5-mini *worse* at math-python
  (they attribute this to the RLM enabling the same Python-tool behavior the baseline already had, so the RLM
  wrapper is pure overhead there). It made DeepDive *worse* unless the model was given a hand-written strategy
  prompt ("environment tips"), in which case it improved a lot. Different models (GLM-4.6, GLM-4.5-Air,
  INTELLECT-3) showed different, sometimes opposite, sensitivities to the same tips.
- **It is always slower.** "In all cases, the RLM increases the time required to finish the job significantly" —
  verified, direct from Prime Intellect's own writeup.
- **The gains that do exist come from prompting an existing model into an unfamiliar pattern, not from a model
  built for it.** The one purpose-trained model in the paper (RLM-Qwen3-8B) beat its base model by 28.3% — a big
  number, but it's also the paper's explicit acknowledgment that an off-the-shelf model doing this via scaffolding
  alone is leaving a lot on the table. Prime Intellect frames RL-training a model specifically for RLM usage as
  *future work*, not something you get by adopting the scaffold today.
- **Recursion depth beyond 1 is not a proven, shipped feature.** It's explicitly future work in the same source.

**What this means practically:** if you build the RLM engine, budget real time for prompt iteration
("environment tips" analogues) per task type, expect inconsistent results across models, and do not assume it
will be a strict upgrade over hybrid retrieval for a given task without measuring both on that task first. This is
exactly why `ARCHITECTURE.md` §5.4 makes the router's job to keep queries *off* this path by default — the
evidence supports treating RLM as a specialist tool you reach for deliberately, not a general upgrade.

---

## 3. Where "build it from scratch" will actually cost you time

You asked for this "from scratch," which is a reasonable choice for learning — building the host/kernel boundary
yourself will teach you more about agent security than reading about it ever will. Just go in with eyes open about
where the time actually goes. In rough order of how much they'll surprise you:

1. **The sandbox, not the orchestrator.** Writing `rlm_spawn()` is an afternoon. Making it *safe* to run
   model-generated Python against a document you didn't write yourself is the part that eats weeks if you do it
   properly (§7 of `TECH_STACK.md`). Most from-scratch RLM builds will be tempted to skip this because it's not
   the "interesting" part — resist that; it's the part that turns a cool demo into something you can trust.
2. **The promotion gate degrading into "vector RAG with extra steps."** It is very easy to build the four memory
   types, then never actually implement contradiction resolution or TTL expiry, at which point you have one big
   fact table with a schema that implies more rigor than the code delivers. If you don't have time to implement
   the full gate, ship three memory types with the gate fully implemented rather than four types with a gate
   that's a rubber stamp.
3. **The classifier being a single brittle LLM call wearing a system diagram's clothing.** "Route by query shape"
   sounds like solid engineering right up until it's implemented as one prompt asking a model to output
   `simple | multi_step | global`. That's fine as a start — just track router precision (§4 of `PRD.md`) from day
   one so you notice when it's wrong, instead of assuming the diagram is describing what the code does.
4. **Evaluation data you don't have yet.** Every metric in `PRD.md` §4 needs a labeled set of real questions
   against real documents. Building that set is not a follow-up task — do it in parallel with Phase 1, or the
   metrics section of this package is aspirational rather than actionable.

---

## 4. What I'd push back on if you asked me to just agree with the original brief

The source brief (the Perplexity research doc) is solid work and its RAG-paradigm comparison held up well under
verification. Two places worth a second look before you treat it as settled:

- **It frames RLM and agentic hybrid RAG as cleanly complementary** ("the strongest system can use RLM as one of
  its tools"). That's true in principle, but it understates the real cost: RLM as "one of the tools" still means
  building and maintaining an entire second execution surface (sandboxed kernel, async subagent lifecycle,
  structured-answer contract) for a mechanism you'll invoke rarely. "It's just one more tool" undersells the
  fixed cost of having it at all. Decide deliberately whether that fixed cost is worth it for what you're actually
  building, rather than including it because a complete architecture "should" have it.
- **Some of the RAG-side numbers are single-source and shouldn't be repeated as settled facts** — flagged
  specifically in `VERIFICATION_AND_SOURCES.md` §4 (the PageIndex FinanceBench figure, the reranker Hit@1 jump).
  They're plausible and well-sourced for what they are; they're not independently reproduced results, and a
  report or presentation that cites them as if they were would be overclaiming.

---

## 5. Recommended actual scope, if you're building this alone against a real timeline

- **Do build:** Phase 1 in full. It's a complete, independently useful system and the best match for a "deployed
  Agentic RAG Assistant" first milestone.
- **Do build:** Phase 2's promotion gate and query classifier, even in a simplified form — this is where the
  system starts feeling meaningfully different from "RAG chatbot."
- **Build if and when you hit the actual need:** Phase 3 (RLM engine). Treat the fact that it's fully specified
  here as removing the *design* cost when that day comes, not as a signal that it belongs in the first thing you
  ship.
- **Deprioritize by default:** GraphRAG, vectorless/tree navigation for anything other than a genuinely
  structured document type you actually have. Both are real, useful, and correctly scoped as extension points
  (§9 of `ARCHITECTURE.md`) rather than v1 requirements.
