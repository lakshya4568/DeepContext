# RLM / Prime Agent — Verified Facts Quick Reference

Condensed from `docs/VERIFICATION_AND_SOURCES.md` (full detail and primary sources there). Every claim below was
re-checked directly against the GitHub repository and the arXiv paper on 17 August 2026 — this file exists so
you don't have to re-derive these while writing orchestrator code, and so you don't accidentally cite the
unverified granular numbers.

## Confirmed as accurate

- Prime Agent is real, MIT-licensed, actively maintained (16.2k stars / 1.7k forks at time of writing),
  built by Prime Intellect.
- It is genuinely built around a persistent IPython kernel as the model's primary tool, with subagents,
  self-improvement, and skills.
- The RLM paper is real (MIT/Stanford-adjacent researchers); the core idea — treat a long prompt as an external
  environment variable, recursively call sub-LMs over it via code — is accurately summarized.
- "Skills as `SKILL.md` packages" is the open Agent Skills standard (`agentskills.io/specification`), not
  Prime-Agent-specific.

## Corrected after checking the source (do not repeat the original, incorrect claims)

| Don't say | Say instead |
|---|---|
| "Prime Agent is Python end-to-end" | The control plane is **TypeScript**; Python/IPython is only the model-facing execution surface. |
| "`rlm(...)` calls block and return the child's result" | `rlm_spawn` is **async**, returns an admission handle immediately, never returns the answer directly — replies arrive via `agent_message.send(...)`. |
| "RLM subagents can recurse arbitrarily deep" | Default recursion depth is **exactly 1**. Deeper recursion is explicit future work in Prime Intellect's own research post, not shipped/proven. |
| "RLM is uniformly better than RAG / agentic RAG / a big context window" | Prime Intellect's own ablation found the scaffold **actively hurts** some tasks (math-python; DeepDive without hand-written tips) for some models, and **always increases latency**. |
| Granular benchmark numbers like "91.33% on BrowseComp+" | **Not independently verified** — only confirmed at the paper's abstract level (see below). Don't cite the granular per-benchmark figures without reading the full paper yourself. |

## Citable, abstract-level headline results (arXiv:2512.24601, Zhang/Kraska/Khattab, submitted 31 Dec 2025, v3 11
May 2026)

- Median improvement of **26%** over context compaction, **130%** over CodeAct-with-sub-calls, and **13%** over
  Claude Code on GPT-5, across four long-context tasks, at comparable cost.
- A purpose-trained **RLM-Qwen3-8B beats base Qwen3-8B by 28.3%** on average — the paper's own acknowledgment
  that scaffolding an off-the-shelf model leaves real gains on the table; RL-training a model specifically for
  RLM usage is explicit future work, not something you get from adopting the scaffold today.

## Confirmed technical defaults (from Prime Intellect's Jan 2026 blog post, "Recursive Language Models: the
paradigm of 2026")

- Default REPL output cap: **8,192 characters/turn**.
- Answer contract: `answer = {"content": ..., "ready": bool}`.
- Recursion depth: **fixed at 1** as of this writing.
- `llm_batch()` exists as the explicit-parallel-dispatch primitive for N known-independent sub-calls.

## Primary sources (fetched directly, 17 Aug 2026)

- `github.com/PrimeIntellect-ai/prime-agent` (repo root)
- `packages/coding-agent/docs/architecture.md`, `rlm.md`, `skills.md`
- `primeintellect.ai/blog/rlm` — "Recursive Language Models: the paradigm of 2026", Sebastian Müller, 1 Jan 2026
- `arxiv.org/abs/2512.24601` — "Recursive Language Models", Zhang/Kraska/Khattab

## What was NOT re-verified this pass

The RAG-side comparisons (chunking strategies, reranker leaderboard, PageIndex/vectorless RAG, GraphRAG) were
not re-fetched — see `docs/VERIFICATION_AND_SOURCES.md` §4. Treat single-source figures cited in
`docs/TECH_STACK.md` (e.g. the reranker Hit@1 jump) as plausible-but-not-independently-reproduced.
