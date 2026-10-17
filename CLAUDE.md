# Deep Context Platform

Framework-free agentic hybrid RAG (FastAPI, asyncpg + pgvector, SQLite fallback). Python >=3.12, package `src/deep_context`.
Read `AGENTS.md` first: it holds the working rules. Architecture, API and schema docs are in `docs/`.

## Commands (always via uv)
- Setup: `uv sync --extra dev`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- Types: `uv run mypy src`
- Single test: `uv run pytest tests/test_retrieval.py -k <name>` (asyncio_mode=auto)
- CLI: `uv run deep-context --help`

## Layout (src/deep_context)
- `ingestion/`, `multimodal/`: Docling parsing, chunking, asset store
- `storage/`, `migrations/` (repo root): Postgres/SQLite stores and numbered SQL migrations (never edit applied ones)
- `retrieval/`: hybrid BM25 + vector, RRF, reranker
- `agentic/`, `generation/`, `verification/`: CRAG planner, grounded generation, evidence checks
- `memory/`, `api/`, `cli/`, `ui/`

## Gotchas
- No LangChain, LlamaIndex or LangGraph. Keep it raw asyncio and parameterized SQL.
- `.env` holds API keys (Gemini, Groq, NVIDIA, HF) and `POSTGRES_DSN`. Never print or commit it.
- `*.db`, `*.db-wal`, `eval_results*.json` and `data/` are local artifacts. Don't edit or commit them.
- Changing the embedding provider or dimension (`EMBEDDING_*`) needs a re-index and a migration.
- Many tests need Postgres or API keys. Run only the relevant test file, not the full suite.
- Ruff line length is 100 (E501 ignored). Target py312.
