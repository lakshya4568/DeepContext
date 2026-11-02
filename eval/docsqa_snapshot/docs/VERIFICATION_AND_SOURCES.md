# Verification & Sources

This package extends a prior research brief (`source-material/perplexity-verdict-aug-2026.md`) that compared
RAG paradigms and proposed an initial architecture referencing **Prime Agent**
(`github.com/PrimeIntellect-ai/prime-agent`). Before building on top of that brief, every claim about Prime Agent
and about the Recursive Language Models (RLM) research was re-checked directly against the GitHub repository and
the arXiv paper on **17 August 2026**. This document records what was confirmed, what was corrected, and what
remains unverified so you know exactly how much to trust each part of the architecture that follows.

---

## 1. What the original brief got right

- Prime Agent is a real, actively maintained, MIT-licensed open-source project (16.2k GitHub stars, 1.7k forks at
  time of writing) built by Prime Intellect.
- It genuinely is built around a **persistent IPython kernel** as the model's primary tool, with subagents,
  self-improvement, and skills, roughly as described.
- The **RLM paper** is real, from MIT/Stanford-adjacent researchers, and the core idea — treat a long prompt as an
  external environment variable and recursively call sub-LMs over it via code — is accurately summarized.
- The "skills as `SKILL.md` packages" concept is real and, notably, is **not** Prime-Agent-specific — it is the
  open **Agent Skills standard** (`agentskills.io/specification`), the same standard this package's `skills/`
  directory is written against.

## 2. What was corrected after checking the source

| Original claim | What the repo/paper actually say | Why it matters |
|---|---|---|
| "Prime Agent uses a persistent IPython kernel and Python-based harness" (implying the whole system is Python) | The **control plane is TypeScript** (`package.json`, `tsconfig.json`, `biome.json` at repo root). Python/IPython is only the **model-facing execution surface**. The daemon, supervisor, session runtime, provider calls, scheduling, and persistence are all TypeScript (`AgentSessionRuntime`, `AgentSession`, `Supervisor`). IPython talks back to the TypeScript host through **typed host requests** for anything that needs authority (writing files, sending messages, changing goals). | This is the single most important architectural correction. It means the "REPL-as-tool" idea and the "who owns credentials and state" idea are separable decisions. Section 4 of `ARCHITECTURE.md` adopts the *pattern* (host owns authority, kernel is sandboxed and makes typed requests back) without adopting TypeScript, because your stack is Python/FastAPI. |
| RLM subagents implied as synchronous/blocking (`results = [sub_lm.analyze(s) for s in sections]`) | Prime Agent's `rlm(...)` call is **asynchronous and non-blocking**: it returns an admission handle immediately (`rlm_child_id`, `name`, `session_dir`, `model`) and **never returns the child's answer directly**. Replies only arrive later via explicit `agent_message.send(...)` calls or files. Multiple children can be spawned in parallel without awaiting any of them. | A blocking recursive call is easy to reason about but does not match how the reference implementation actually gets its parallelism and resilience (a stuck child can't hang the parent). `rlm-orchestrator` skill and `ARCHITECTURE.md` §5 use the async, message-passing model instead. |
| Recursion is implied as freely nestable | Prime Agent's **default recursion depth is exactly 1** (root → children; children cannot spawn grandchildren) unless explicitly raised. Prime Intellect's own RLM research blog (Jan 2026) lists arbitrary recursion depth as **future work**, not a shipped, load-bearing feature. | Deep, unbounded recursion is the part of "RLM" people picture first and the part that is least production-proven today. The PRD scopes the MVP to depth-1 recursion and treats deeper recursion as an explicit, budget-gated opt-in. |
| RLM cited as uniformly better ("more consistent... than traditional RAG, agentic RAG... or a giant context window") | Prime Intellect's own ablations (GPT-5-mini, GLM-4.6, GLM-4.5-Air, INTELLECT-3 across DeepDive, math-python, Oolong, verbatim-copy) found the RLM **scaffold actively hurts performance on some tasks** (notably math-python, and DeepDive without hand-written "environment tips"), that **it always increases latency**, and that current gains come from prompting an *existing* model into the pattern, not from a model trained for it — training an RLM-native model is explicitly framed as future work. | This is the basis for the risk register in `CRITICAL_ASSESSMENT_AND_SCOPE.md`. RLM is a promising but immature and prompt-sensitive paradigm as of today, not a drop-in accuracy upgrade. |
| Specific benchmark figures ("GPT-5 RLM scoring 91.33% on BrowseComp+... OOLONG 44%→56.5%") | **Not independently verified.** The paper abstract (arXiv:2512.24601, Zhang/Kraska/Khattab) instead states, verbatim: RLMs beat GPT-5 baselines by a **median of 26% vs. compaction, 130% vs. CodeAct-with-sub-calls, and 13% vs. Claude Code** across four long-context tasks at comparable cost, and a purpose-trained **RLM-Qwen3-8B beats base Qwen3-8B by 28.3%** on average. The 91.33%/70.47%/51% figures may exist inside the 43-page paper body/appendix, but this pass only confirmed the abstract — treat the granular per-benchmark numbers as **unconfirmed** until you've read the full PDF. | Cite the abstract-level numbers (they are official and directly quotable); don't repeat the granular figures without reading the source paper yourself. |

## 3. Primary sources (fetched directly, 17 Aug 2026)

- Repo root — `github.com/PrimeIntellect-ai/prime-agent`
- `packages/coding-agent/docs/architecture.md` — daemon/worker/kernel process boundaries, the prompt execution
  sequence diagram, and the "not a security sandbox" statement
- `packages/coding-agent/docs/rlm.md` — the RLM programming model, the four core invariants, the host bridge,
  the trust model
- `packages/coding-agent/docs/skills.md` — the Agent Skills standard, Python-backed skill packaging
  (`pyproject.toml` + `src/<import_name>/__init__.py`), and confirmation that Prime Agent can load skill
  directories from **Claude Code** (`~/.claude/skills`) directly
- `primeintellect.ai/blog/rlm` ("Recursive Language Models: the paradigm of 2026", Sebastian Müller, 1 Jan 2026) —
  the production ablation study, environment tips, `llm_batch()`, the `answer = {"content", "ready"}` output
  contract, the 8,192-character default REPL output cap, and the explicit statement that recursion depth is
  currently fixed at 1
- `arxiv.org/abs/2512.24601` ("Recursive Language Models", Alex L. Zhang, Tim Kraska, Omar Khattab — submitted
  31 Dec 2025, v3 11 May 2026) — the paper abstract and headline results quoted above

## 4. Claims from the original brief that this package did not re-verify

The RAG-side comparisons (chunking strategies, reranker leaderboard, PageIndex/vectorless RAG, GraphRAG,
memory-system taxonomy) were **not** re-fetched during this pass, because the explicit verification ask was about
RLM and Prime Agent specifically, and the original brief is dated the same month as this package (August 2026)
with its own working source links (Redis, NVIDIA, AIMultiple, Oracle, VectifyAI/PageIndex). Treat those figures —
e.g. PageIndex's 98.7% FinanceBench number, or the reranker Hit@1 improvement from 62.67%→83% — as **single-source,
vendor- or benchmark-specific claims**, not independently reproduced results. `TECH_STACK.md` flags each one
inline where it's used.

## 5. What this means for how you should use this package

Treat `ARCHITECTURE.md`, the `skills/`, and the reference code as a **well-grounded starting design**, not a
guarantee. Two things are worth doing yourself before you commit real build time:

1. Read the full RLM paper (PDF/HTML linked above) if the granular benchmark numbers matter to a report or
   presentation you're producing — don't cite 91.33%-style figures from the brief without confirming them yourself.
2. Prime Agent ships real, running code for all of this. If you want to see the async subagent / typed-host-request
   pattern working rather than just specified, cloning `prime-agent` and reading `packages/coding-agent/src/` will
   teach you more in an afternoon than any document can.
