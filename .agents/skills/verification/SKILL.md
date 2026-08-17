---
name: verification
description: Implements the Deep Context Platform's evidence-support checker — the shared gate that scores whether a generated answer is actually backed by retrieved evidence, computed values, or explicit model inference, before the answer is returned. Also implements the methodology for fact-checking third-party claims (a library's behavior, a benchmark number, a competitor's documented feature) against primary sources before they're written into a document or shipped in code. Use this skill whenever an answer needs to pass the evidence-sufficiency gate shared by skills/rag-retrieval/ and skills/rlm-orchestrator/, whenever a document cites a specific benchmark or claims something about a third-party project's behavior, or whenever asked to fact-check, verify, or re-check sources before committing to a claim.
---

# Verification Skill

Implements FR20–FR21 of `docs/PRD.md` and the evidence-sufficiency gate at `docs/ARCHITECTURE.md` §7. Two
related but distinct jobs live in this skill: **(A)** checking whether an already-generated answer is actually
grounded in evidence, and **(B)** checking whether a claim you're about to write into a document is actually
true, before you write it. `docs/VERIFICATION_AND_SOURCES.md` is the worked example of job (B) — read it once to
see the standard this skill holds itself to.

## (A) The evidence-sufficiency gate

This is the **same gate** both `skills/rag-retrieval/` and `skills/rlm-orchestrator/` terminate at before
returning an answer — it is deliberately not duplicated per-path (`docs/ARCHITECTURE.md` §7's whole point is
that RLM output needs exactly the same "did we actually support this claim" check that RAG output does; nothing
about recursive delegation makes a claim more true).

```python
def sufficient(evidence, draft_answer) -> bool:
    # 1. Every non-trivial claim in draft_answer must be traceable to a retrieved chunk,
    #    a computed value, or explicitly flagged as the model's own inference.
    # 2. Coverage: for aggregation-style queries, evidence must span the full
    #    retrieved/loaded set, not a sampled subset.
    # 3. Confidence: the verifier's own score must clear a threshold.
    ...
```

On failure: the caller (whichever pipeline invoked this) rewrites the query and retries **once**. On a second
failure, "I don't have enough evidence to answer this confidently" is a correct, shippable answer — not a bug to
engineer away. Silently answering anyway is the actual bug this gate exists to prevent. This target maps
directly to the `docs/PRD.md` §4 metric: **≥90% of answers pass `check_answer_support`** on a held-out eval set
of ≥50 questions.

See `scripts/check_answer_support.py` for a reference implementation of claim extraction, evidence-linking, and
scoring.

## (B) Fact-checking claims before they're written down

The standard this platform holds itself to, restated as a repeatable process (not just a one-time exercise):

1. **Go to the primary source, not an aggregator or your training data.** `docs/VERIFICATION_AND_SOURCES.md` was
   produced by fetching the actual GitHub repo and the actual arXiv paper directly, not by summarizing a
   secondhand blog post about them.
2. **Distinguish "confirmed" from "corrected" from "not verified."** Use a three-way split, not a binary
   true/false — see the correction table pattern in `docs/VERIFICATION_AND_SOURCES.md` §2 and
   `skills/rlm-orchestrator/references/verified_facts.md`'s condensed version of it.
3. **Cite at the abstraction level you actually verified.** If you confirmed a paper's abstract but not its
   full body, cite the abstract-level headline numbers — don't repeat granular figures (e.g. a specific
   per-benchmark percentage) that only appear deeper in a document you didn't personally read in full.
4. **Flag single-source claims as single-source.** A benchmark number from one vendor's blog post, cited once,
   is "plausible and well-sourced for what it is" — not an independently reproduced result. Say so explicitly
   wherever it's used (`docs/TECH_STACK.md` does this for the reranker Hit@1 figure and the PageIndex
   FinanceBench number).
5. **Record what you didn't re-check, not just what you did.** `docs/VERIFICATION_AND_SOURCES.md` §4 is an
   explicit list of claims from the source brief that this verification pass did *not* re-fetch, with the
   reasoning for why. This is what keeps a "verified" document honest about its own boundaries.

## Copyright and citation discipline

When verification involves fetching real web sources (as `docs/VERIFICATION_AND_SOURCES.md` did), paraphrase
findings in your own words rather than reproducing source text at length, and keep any direct quotation short
and clearly attributed. The goal of this skill is accurate claims, not reproduced prose.

## Files in this skill

- `scripts/check_answer_support.py` — reference implementation of job (A): claim extraction from a draft answer,
  linking each claim to evidence (or flagging it as inference), and the aggregation-coverage check for
  aggregation-shaped queries.

## What NOT to do

- Don't treat "the model sounded confident" as evidence of groundedness — that's exactly the failure mode this
  gate exists to catch.
- Don't let the corrective retry loop this gate triggers run unbounded — one retry, then an honest
  "insufficient evidence" answer, per `skills/rag-retrieval/` and `skills/rlm-orchestrator/`'s shared bound.
- Don't cite a claim as verified because it "sounds right" or matches training-data intuition — job (B) above
  exists specifically because secondhand descriptions of a fast-moving project (like Prime Agent) go stale or
  were wrong from the start (see the corrections table in `docs/VERIFICATION_AND_SOURCES.md` §2).
