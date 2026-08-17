---
name: code-execution
description: Guidance for safely executing model-generated Python within the Deep Context Platform's RLM kernel sandbox — process isolation tiers, filesystem/network restrictions, and REPL output truncation. Use this skill when building or modifying the kernel's code-execution path, when a task involves running untrusted or model-generated code against a document or repo you did not author yourself, or when deciding which sandboxing tier is appropriate for a given deployment (local dev vs. untrusted input vs. production). This is a policy/checklist skill, not a code library — it governs skills/rlm-orchestrator/'s kernel process.
---

# Code Execution Skill

Implements NFR2–NFR3 of `docs/PRD.md` and the sandboxing tiers in `docs/TECH_STACK.md` §7. This skill exists to
prevent the single most tempting shortcut in the whole platform: skipping real sandboxing because "the reference
implementation does too."

## The one fact this skill exists to enforce

Prime Agent's own docs are explicit that their kernel process "runs model-generated Python and project commands
with the worker's operating-system permissions... It is a durable control environment, **not a security
sandbox**." Process separation (host vs. kernel, `docs/ARCHITECTURE.md` §4) buys you crash isolation and
authority separation — it does not, by itself, buy you protection from a malicious document that tricks the
model into writing destructive code. If the reference implementation says this about itself, it is doubly true
for a from-scratch build with no production hardening behind it yet.

## Escalate as soon as input isn't 100% your own — the three tiers

| Tier | When | Mechanism |
|---|---|---|
| 1 | Local dev / your own trusted documents only | Subprocess, restricted `PYTHONPATH`, scratch-directory-only filesystem access, network egress limited to an explicit allowlist |
| 2 | Anything touching a document or repo you didn't author yourself | Container isolation — Docker with dropped capabilities and a read-only root filesystem, at minimum |
| 3 | Production / untrusted input at any real scale | `gVisor`/`nsjail`-style syscall-level isolation, or a hosted code-execution sandbox (E2B, Modal) — this is what the RLM paper's own implementation actually uses ("code execution happens in isolated Sandboxes") |

**Don't skip Tier 2 for Tier-2 situations because Tier 1 "seems to work fine."** Whether a document is
"trusted" is a property of who wrote it, not of whether it has misbehaved yet — a prompt-injection payload in an
ingested PDF is indistinguishable from ordinary text until the kernel executes code influenced by it.

## What the sandbox must restrict, regardless of tier

- **Filesystem:** scratch-directory-only by default. No access to the host's credentials, config, or other
  sessions' data. Container/production tiers additionally need a read-only root filesystem.
- **Network:** egress disabled except an explicit allowlist (model provider APIs, and nothing else, unless a
  specific task genuinely needs more).
- **Credentials:** the kernel process never holds a database credential or a provider API key — those live only
  on the host side of the boundary (`docs/ARCHITECTURE.md` §4). Everything the kernel needs that requires
  authority goes through the typed host-request bridge (`skills/rlm-orchestrator/scripts/rlm_host_bridge.py`).
- **Output:** REPL stdout shown back to the model is capped (default 8,192 characters/turn,
  `skills/rlm-orchestrator/`) — this is a correctness mechanism (forces search/filter behavior) as much as a
  safety one, since an uncapped output is also a channel for a malicious document to flood the model's context.
- **Lifecycle:** a crashed or hung kernel process must not corrupt durable memory or leave the host in an
  inconsistent state (NFR5) — the host is the source of truth; the kernel is disposable and restartable.

## When code-producing tasks need a second check (FR21)

If any skill *generates* code (not just executes model-written exploration code in the RLM kernel), that
generated code runs through a test/lint verifier before being presented as done — see `skills/verification/`.
This is a separate concern from sandboxing: sandboxing protects you while code runs; the FR21 verifier checks
whether the code that's about to be shipped is actually correct.

## What NOT to do

- Don't treat host/kernel process separation alone as a security boundary — it's lifecycle isolation. Say so
  explicitly in any deployment doc, the way `docs/PRD.md` NFR3 does, rather than letting "sandboxed" imply more
  than it delivers.
- Don't grant the kernel process broader filesystem or network access "temporarily" to unblock a specific task —
  that's the access a malicious document would also get.
- Don't defer sandbox hardening past Tier 1 because the demo works — escalate at the point input stops being
  100% your own, not at the point something goes wrong.
