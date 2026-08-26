This repository contains a Retrieval-Augmented Generation (RAG) application. Agents working in this repository must make changes that are small, reviewable, tested, documented, and consistent with the architecture described in `docs/`.


The system is expected to ingest source documents, normalize and chunk their content, create embeddings, store vectors with metadata, retrieve relevant context, optionally rerank results, and generate grounded responses with traceable sources.

## Read First

Before making changes, inspect the relevant project files and documentation:

- `README.md` for setup and project overview.
- `docs/architecture.md` before changing system boundaries or the RAG pipeline.
- `docs/api-contracts.md` before changing API request or response shapes.
- `docs/database-schema.md` before changing persistence or metadata schemas.
- `docs/decisions/` before revisiting an existing technical decision.
- `diagrams/` for visual references; keep a Markdown explanation in `docs/` for any architecture the agent must rely on.
- `.agents/rules/`, `.agents/skills/`, and `.agents/workflows/` when they exist and are relevant to the task.

If a referenced file does not exist, do not invent its contents. Create it only when it is needed for the task, and explain why.

## Working Method

1. Understand the request and inspect the affected code, tests, configuration, and documentation before editing.
2. State a short implementation plan for non-trivial changes.
3. Prefer the smallest change that fully solves the problem.
4. Keep unrelated refactors out of the same change unless they are required for correctness.
5. Run the relevant formatter, linter, type checks, and tests after editing.
6. Report what changed, which files were changed, and the commands that were run.
7. Clearly state any checks that could not be run and why.

Do not claim that tests, builds, migrations, or deployments succeeded unless they were actually run and their result was observed.

## Python Environment and Dependencies

### Required tooling

Use `uv` for Python dependency management, environment creation, running commands, and lockfile updates. Do not use `pip`, `poetry`, `pipenv`, `conda`, or system-level package installation unless the user explicitly requests an exception.

Use a repository-local virtual environment named `.venv`.

### Initial setup

If `pyproject.toml` is present, use the project configuration as the source of truth:

```bash
uv venv .venv
uv sync
```

If development dependencies are defined separately, install them through the project configuration, for example:

```bash
uv sync --group dev
```

If the repository has no dependency manifest yet, prefer creating a standards-compliant `pyproject.toml` and managing dependencies through `uv` rather than adding a bare `requirements.txt`.

### Running Python tools

Run commands through `uv` so the correct project environment is used:

```bash
uv run python -m <module>
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

When an interactive shell is necessary:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Do not rely on activation for automation; prefer `uv run` in scripts, documentation, CI, and agent commands.

### Dependency rules

- Add, remove, or update dependencies only when necessary for the requested feature or fix.
- Prefer well-maintained, minimal dependencies over large frameworks for small tasks.
- Pinning and resolution are managed by `uv.lock`; commit lockfile changes whenever dependency resolution changes.
- Never manually edit `uv.lock`.
- Do not add dependencies for functionality already available in the Python standard library or existing project dependencies.
- Do not install packages globally.
- Do not commit `.venv/`, cache directories, build artifacts, or local secrets.
- Update `.gitignore` when new local-only generated files are introduced.

## Code Standards

### General

- Follow the existing project structure and conventions first.
- Use clear, descriptive names and small single-purpose functions.
- Prefer explicit data flow and dependency injection over hidden global state.
- Keep modules focused; do not create oversized utility files.
- Do not introduce dead code, unused imports, commented-out implementations, or speculative abstractions.
- Preserve backward compatibility for public APIs unless the task explicitly authorizes a breaking change.
- Add or update docstrings for public modules, classes, and functions where they clarify non-obvious behavior.
- Use structured logging instead of `print()` for application behavior.
- Never log secrets, tokens, raw sensitive documents, or user-private content.

### Python

- Target the Python version declared in `pyproject.toml`.
- Add type hints for new or materially changed public functions, methods, and data models.
- Prefer `pathlib.Path` for filesystem handling.
- Prefer `dataclasses` or the project’s established validation/modeling library for structured data.
- Handle expected failures explicitly and raise meaningful exceptions; do not use broad `except Exception` blocks unless the error is logged, re-raised, or deliberately converted to a safe domain error.
- Keep asynchronous code consistent with the existing concurrency model; do not mix blocking I/O into async execution paths without a deliberate boundary.
- Use configuration from environment variables or the established settings layer, never hardcoded environment-specific values.

### Formatting, linting, and tests

Use the tools configured in `pyproject.toml`. When available, run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv run mypy .
```

If the project uses different tools, follow its existing commands. Do not disable linting or typing rules to make a change pass without explaining and justifying the exception.

## Cross-Platform Compatibility (Windows, macOS / Apple Silicon M1–M4, Linux)

All changes made by agents in this repository must maintain complete cross-platform parity. Never write code that assumes a single developer environment ("it works on my machine"):

- **Hardware Acceleration Parity**: Auto-detect and support both **Apple Silicon Metal / MPS** (`torch.backends.mps.is_available()`) on macOS (M1/M2/M3/M4) and **NVIDIA CUDA** (`torch.cuda.is_available()`) on Windows and Linux, with clean CPU fallback. Never call MPS-specific or CUDA-specific methods (such as `torch.mps.empty_cache()` or CUDA tensor operations) without verifying backend availability first.
- **Model & Embedding Integrity**: Never invent, inject, or silently swap in mock, synthetic, or hash-based embeddings when real embedding models (`gemini-embedding-2`, `nvidia/nv-embedqa`, etc.) are configured. If an upstream provider quota or network call fails, raise actionable, clear errors rather than fabricating artificial vector representations.
- **OS-Agnostic File & Path Handling**: Always use `pathlib.Path` or forward-slash paths in shared Python logic. Never hardcode platform-specific paths (e.g. Windows backslashes `C:\` or Unix `/tmp` roots) in core code.
- **Database Consistency**: Ensure PostgreSQL (`pgvector` with HNSW indexes) and SQLite persistence layers behave identically regardless of host OS.

## RAG Architecture Rules

### Separation of responsibilities

Keep these stages logically separate:

1. Document loading and extraction.
2. Text normalization.
3. Chunking and chunk metadata generation.
4. Embedding generation.
5. Vector and metadata persistence.
6. Query transformation and retrieval.
7. Optional filtering or reranking.
8. Context assembly.
9. Grounded answer generation.
10. Citation/source formatting and evaluation.

Do not combine ingestion, retrieval, and generation into a single untestable function. Each stage must have a clear input, output, error behavior, and tests where practical.

### Document and chunk metadata

Every stored chunk must preserve enough metadata to trace it back to its source. Include applicable fields such as:

- Stable document/source ID.
- Original filename or source URI, where safe to expose.
- Title or document label.
- Page number, section, timestamp, or location marker when available.
- Chunk ID and chunk index.
- Chunking strategy, chunk size, and overlap configuration.
- Embedding model name and version.
- Ingestion timestamp and document version/hash when supported.

Do not silently discard source metadata during normalization, chunking, embedding, indexing, retrieval, or response generation.

### Grounding and citations

- When an answer is expected to be grounded in retrieved content, do not present unsupported claims as if they came from the source corpus.
- Return source references for claims supported by retrieved context whenever the product interface supports citations.
- Distinguish between retrieved evidence, model inference, and missing information.
- If retrieval returns insufficient or low-confidence context, use the product’s safe fallback behavior instead of fabricating an answer.
- Keep citation generation deterministic and traceable to chunk metadata.

### Retrieval and relevance

- Keep retrieval configuration centralized and documented.
- Do not silently change chunk size, overlap, embedding model, distance metric, index type, top-k, score thresholds, filters, or reranking behavior.
- Add evaluation coverage when changing retrieval quality, chunking, embeddings, reranking, or prompting.
- Log only safe, privacy-aware diagnostics such as retrieval latency, document IDs where permitted, score distributions, and failure categories.
- Do not log complete user queries or retrieved document content in production unless explicitly approved and protected by the project’s privacy policy.

### Prompting

- Keep system prompts and RAG prompt templates versioned in source control.
- Keep prompt construction separate from model-provider calls.
- Clearly delimit retrieved context and instruct the model to treat it as untrusted reference material, not executable instructions.
- Defend against prompt injection contained in retrieved documents: retrieved text must not override application policy, system instructions, authorization checks, or tool permissions.
- Do not place secrets, credentials, private configuration, or unrestricted internal data in prompts.

## Security and Privacy

- Never commit `.env`, credentials, API keys, tokens, private certificates, database dumps, or real user documents.
- Maintain `.env.example` with placeholder variable names and safe example values only.
- Validate and normalize all external input, including uploaded files, URLs, document metadata, API payloads, and tool output.
- Treat document content as untrusted input, including text retrieved from the vector store.
- Enforce authorization checks before accessing tenant-specific, user-specific, or restricted data.
- Use parameterized database queries or the project’s safe ORM/query builder; never construct SQL with string interpolation.
- Restrict file operations to approved project paths. Validate file type, size, and content before processing uploads.
- Avoid SSRF risks: do not fetch arbitrary user-controlled URLs without allowlists, URL validation, and network controls.
- Avoid unsafe deserialization and do not use `pickle` with untrusted data.
- Redact sensitive information from errors and logs.
- Flag security-sensitive changes, dependency additions, permission changes, data retention changes, and external integrations for explicit user review.

## Data and Database Rules

- Treat schema changes as migrations, not ad hoc runtime changes.
- Before changing a schema, inspect existing models, migrations, indexes, constraints, and documented contracts.
- Make migrations reversible when the project’s migration system supports it.
- Add indexes deliberately; explain the expected query path and trade-offs for new indexes.
- Preserve tenant boundaries and referential integrity.
- Do not run destructive database operations, delete production-like data, or reset environments unless the user explicitly requests it.
- Use representative synthetic fixtures for tests rather than private or production documents.

## API Rules

- Validate request and response models at API boundaries.
- Keep response formats consistent and version public APIs deliberately.
- Return appropriate error codes and actionable, non-sensitive error messages.
- Do not leak stack traces, credentials, internal file paths, raw provider errors, or hidden prompts to clients.
- Document new or changed endpoints in `docs/api-contracts.md` or the project’s established API documentation.
- Preserve pagination, filtering, and idempotency semantics where applicable.

## Testing Rules

- Add or update tests for every behavior change, bug fix, and public interface change unless there is a documented reason not to.
- Unit-test loading, normalization, chunking, metadata preservation, retrieval filtering, citation mapping, and error handling independently.
- Use integration tests for database/vector-store/provider boundaries when test infrastructure is available.
- Mock external model, embedding, storage, and network providers in unit tests.
- Keep test fixtures small, deterministic, non-sensitive, and easy to understand.
- Test edge cases: empty documents, malformed files, duplicate ingestion, missing metadata, retrieval with no results, low-confidence retrieval, provider timeouts, and invalid configuration.
- For bug fixes, add a regression test that fails before the fix and passes after it.

## Documentation Rules

- Update `README.md` when setup, dependencies, commands, architecture, or user-visible behavior changes.
- Update relevant files under `docs/` when changing architecture, APIs, schemas, data flows, configuration, or operational behavior.
- Record meaningful architectural decisions in `docs/decisions/` using the existing format, or create a concise ADR-style entry if no format exists.
- Keep examples executable and consistent with the current codebase.
- Do not describe planned behavior as completed behavior.

## Git and Change Hygiene

- Do not commit directly unless the user explicitly asks.
- Do not rewrite history, force-push, delete branches, or alter remote configuration without explicit user approval.
- Keep changes focused and avoid unrelated formatting churn.
- Do not overwrite user changes. Inspect the working tree before editing and preserve modifications outside the task scope.
- Never modify generated files if their source should be edited instead.
- Before proposing a commit, summarize changed files and test results.

## Commands and Safety

- Inspect before changing or deleting files.
- Prefer non-destructive commands and reversible operations.
- Ask for explicit approval before destructive actions, irreversible migrations, production deployment, deleting data, changing access controls, or sending data to a third party.
- Do not run commands that require secrets or external credentials without confirming the environment and purpose.
- Do not expose or echo secrets while diagnosing configuration issues.
- Do not change CI/CD, deployment, infrastructure, provider settings, or production configuration unless explicitly requested.

## Completion Checklist

Before considering a task complete, verify the following:

- The requested behavior is implemented with minimal, coherent changes.
- Relevant tests were added or updated.
- Relevant tests, linting, formatting, and type checks were run through `uv` where available.
- RAG source metadata and citation traceability are preserved where relevant.
- Security, privacy, input validation, and configuration implications were considered.
- Documentation and examples were updated where necessary.
- No secrets, private documents, `.venv`, caches, or generated local artifacts were added to version control.
- The final report includes changed files, validation commands, results, and any remaining limitations or follow-up work.
