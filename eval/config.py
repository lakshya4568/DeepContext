"""Central configuration for the DeepContext public-benchmark harness.

Everything that affects a result (models, seeds, depths, prices) lives here so a run is
reproducible from the repo state plus the environment variables documented in REPORT.md.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
CACHE_DIR = Path(os.environ.get("EVAL_CACHE_DIR", EVAL_DIR / "cache"))
DATA_DIR = CACHE_DIR / "data"  # downloaded public datasets (gitignored)
INDEX_DIR = CACHE_DIR / "index"  # per-dataset SQLite indexes (gitignored)
RESULTS_DIR = EVAL_DIR / "results"
LLM_CACHE_PATH = CACHE_DIR / "llm_cache.sqlite"

# --- Determinism -----------------------------------------------------------------------
SAMPLE_SEED = 42  # corpus / query subsampling
BOOTSTRAP_SEED = 0  # bootstrap resampling
BOOTSTRAP_RESAMPLES = 10_000

# --- Models ----------------------------------------------------------------------------
# Gemini 3.x only (no 2.x models). Judge must differ from the generator.
GENERATOR_MODEL = os.environ.get("EVAL_GENERATOR_MODEL", "gemini-3.8-flash")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gemini-3.7-flash")
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
# gemini-embedding-2 is not served from us-central1 on this project (404); "global" works.
EMBEDDING_LOCATION = os.environ.get("EVAL_EMBEDDING_LOCATION", "global")
# Gemini 3.x models are served from the global endpoint on this project (404 in us-central1).
GLOBAL_ENDPOINT_PREFIXES = ("gemini-3",)
RERANKER_NAME = "bge-reranker-v2-m3 (hosted, EcoHash API)"

# --- Retrieval depths ------------------------------------------------------------------
RETRIEVAL_DEPTH = 100  # ranked chunks per config; enough for Recall@100
RERANK_DEPTH = int(os.environ.get("EVAL_RERANK_DEPTH", "0"))  # 0 = score all fused candidates (shipped)
GENERATION_TOP_K = 6  # same top_k the shipped /v1/query hybrid path uses
RANK_K_HIT = (1, 3, 5, 8)
RANK_K_RECALL = (5, 10, 100)
RANK_K_NDCG = (5, 10)

# --- Configurations --------------------------------------------------------------------
# id -> (display name, retrieval kind, generation mode)
CONFIGS: dict[str, tuple[str, str, str]] = {
    "dense_only": ("Dense only", "dense", "one_pass"),
    "bm25_only": ("Lexical only (Postgres FTS ts_rank; FTS5 BM25 on SQLite)", "bm25", "one_pass"),
    "rrf": ("RRF hybrid", "rrf", "one_pass"),
    "rrf_bge": ("RRF + BGE rerank", "rrf_bge", "one_pass"),
    "rrf_bge_two_pass": ("RRF + BGE + two-pass gen", "rrf_bge", "two_pass"),
}
# Reference row: the retrieval engine exactly as shipped (rewriter, classifier, anachronism
# gate, hop-coverage retry, parent resolution) + two-pass generation.
ENGINE_CONFIG = "engine_as_shipped"

# --- List prices (USD per 1M tokens) ---------------------------------------------------
# Source: Gemini API pricing page (standard paid tier), checked 2026-10-18. Vertex AI billing
# can differ; these are used only for the *estimated* cost-per-1k-queries column.
PRICES_USD_PER_M = {
    # Gemini API list price, standard paid tier, valid through 2026-12-31 (doubles from 2027-01-01).
    "gemini-3.7-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-embedding-2": {"in": 0.20, "out": 0.0},
}

# --- Runtime ---------------------------------------------------------------------------
LLM_CONCURRENCY = int(os.environ.get("EVAL_LLM_CONCURRENCY", "8"))
EMBED_CONCURRENCY = int(os.environ.get("EVAL_EMBED_CONCURRENCY", "64"))
RERANK_CONCURRENCY = int(os.environ.get("EVAL_RERANK_CONCURRENCY", "6"))
MAX_API_ATTEMPTS = 8
REQUEST_TIMEOUT_S = float(os.environ.get("EVAL_REQUEST_TIMEOUT_S", "180"))
