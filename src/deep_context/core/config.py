"""Configuration settings for Deep Context Platform."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Endpoints
    nvidia_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "NVIDIA_API_KEY", "NVIDIA_API", "nvidia_api_key", "nvidia_api"
        ),
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )

    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GROQ_API_KEY",
            "GROK_API_KEY",
            "groq_api_key",
            "grok_api_key",
            "GROQ_API",
            "GROK_API",
        ),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="GROQ_BASE_URL",
    )

    # Google Gemini API
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "gemini_api_key",
            "google_api_key",
            "GEMINI_API",
            "GOOGLE_API",
        ),
    )
    gemini_base_url: str = Field(
        default="",
        alias="GEMINI_BASE_URL",
    )

    # EcoHash API (for hosted BGE reranker)
    ecohash_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ECOHASH_API_KEY",
            "ecohash_api_key",
            "ECOHASH_KEY",
        ),
    )
    ecohash_rerank_url: str = Field(
        default="https://api.ecohash.com/v1/rerank",
        alias="ECOHASH_RERANK_URL",
    )
    ecohash_rerank_model: str = Field(
        default="bge-reranker-v2-m3",
        alias="ECOHASH_RERANK_MODEL",
    )

    # Models & Embedding Configuration
    embedding_provider: str = Field(
        default="auto", alias="EMBEDDING_PROVIDER"
    )  # 'auto' | 'gemini' | 'nvidia' | 'mock'
    embedding_model: str = Field(default="gemini-embedding-2", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")  # 768 for gemini-embedding-2 MRL
    reranker_strategy: str = Field(
        default="cross_encoder", alias="RERANKER_STRATEGY"
    )  # 'cross_encoder' | 'ecohash' | 'local_cross_encoder' | 'rrf'

    # Reranker blend tuning (see reranker.py _blend_with_rrf docstring for regression history).
    # 0.60/0.40 is the empirically validated default: it produced 87.1% Hit@5 on the GoT
    # 36-query benchmark. A 0.70/0.30 experiment regressed Hit@5 to 61.3% across every
    # category and must not be reintroduced as the default without a fresh A/B on the
    # frozen eval script.
    reranker_blend_rrf_weight: float = Field(default=0.60, alias="RERANKER_BLEND_RRF_WEIGHT")
    reranker_consensus_top1_count: int = Field(default=3, alias="RERANKER_CONSENSUS_TOP1_COUNT")
    reranker_consensus_top2_count: int = Field(default=6, alias="RERANKER_CONSENSUS_TOP2_COUNT")
    reranker_consensus_boost_tier1: float = Field(
        default=0.15, alias="RERANKER_CONSENSUS_BOOST_TIER1"
    )
    reranker_consensus_boost_tier2: float = Field(
        default=0.0, alias="RERANKER_CONSENSUS_BOOST_TIER2"
    )

    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")  # 'groq' | 'nvidia' | 'gemini'
    llm_model: str = Field(default="qwen/qwen3.6-27b", alias="LLM_MODEL")

    # Database
    database_type: str = Field(default="postgres", alias="DATABASE_TYPE")  # 'sqlite' | 'postgres'
    sqlite_db_path: str = Field(default="deep_context.db", alias="SQLITE_DB_PATH")
    postgres_dsn: str = Field(
        default="postgresql://proximus@127.0.0.1:5432/awems",
        alias="POSTGRES_DSN",
    )

    # Retrieval parameters
    default_retrieval_mode: str = Field(default="hybrid", alias="DEFAULT_RETRIEVAL_MODE")
    reranker_model: str = Field(default="ecohash_hybrid", alias="RERANKER_MODEL")
    rrf_k: int = Field(default=60, alias="RRF_K")
    default_top_k: int = Field(default=8, alias="DEFAULT_TOP_K")
    first_stage_limit: int = Field(default=100, alias="FIRST_STAGE_LIMIT")
    max_retrieval_retries: int = Field(default=1, alias="MAX_RETRIEVAL_RETRIES")

    # Ingestion parameters
    parent_chunk_min_tokens: int = 1000
    parent_chunk_max_tokens: int = 2500
    child_chunk_min_tokens: int = 300
    child_chunk_max_tokens: int = 600
    chunk_overlap_percentage: float = 0.15

    # RLM parameters
    max_recursion_depth: int = Field(default=1, alias="MAX_RECURSION_DEPTH")
    max_repl_chars_per_turn: int = Field(default=8192, alias="MAX_REPL_CHARS_PER_TURN")
    max_rlm_turns: int = Field(default=30, alias="MAX_RLM_TURNS")
    max_rlm_wall_clock_seconds: int = Field(default=3600, alias="MAX_RLM_WALL_CLOCK_SECONDS")

    # Verification parameters
    confidence_threshold: float = 0.75
    min_aggregation_coverage: float = 0.95

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Response cache layer (Redis-backed when available, in-memory fallback otherwise)
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_url: str = Field(default="", alias="CACHE_URL")  # e.g. redis://localhost:6379/0
    cache_default_ttl: int = Field(default=300, alias="CACHE_TTL")  # seconds
    cache_namespace: str = Field(default="deepcontext", alias="CACHE_NAMESPACE")

    # Internal scheduler (job table + polling loop for ingestion/index maintenance)
    scheduler_enabled: bool = Field(default=False, alias="SCHEDULER_ENABLED")
    scheduler_poll_interval: int = Field(default=10, alias="SCHEDULER_POLL_INTERVAL")
    scheduler_max_retries: int = Field(default=3, alias="SCHEDULER_MAX_RETRIES")

    # Agentic corrective-RAG state machine
    agentic_max_rewrites: int = Field(default=2, alias="AGENTIC_MAX_REWRITES")
    agentic_grade_threshold: float = Field(
        default=0.30, alias="AGENTIC_GRADE_THRESHOLD"
    )  # min term-overlap ratio for a doc to count as relevant

    # Summarization parameters (Qwen3 0.6B/0.8B local model)
    summary_enabled: bool = Field(default=False, alias="SUMMARY_ENABLED")
    summary_model: str = Field(default="Qwen/Qwen3-0.6B", alias="SUMMARY_MODEL")
    summary_max_tokens: int = Field(default=80, alias="SUMMARY_MAX_TOKENS")
    summary_batch_size: int = Field(default=8, alias="SUMMARY_BATCH_SIZE")
    summary_device: str = Field(
        default="auto", alias="SUMMARY_DEVICE"
    )  # 'auto' | 'mps' | 'cuda' | 'cpu'

    # Fallback/Test helper
    allow_mock_fallback: bool = Field(default=True, alias="ALLOW_MOCK_FALLBACK")

    @property
    def has_gemini_key(self) -> bool:
        k = self.gemini_api_key.strip()
        return bool(k and not k.startswith("AIzaSy-your-key") and len(k) > 10)

    @property
    def has_groq_key(self) -> bool:
        k = self.groq_api_key.strip()
        return bool(k and not k.startswith("gsk_your_key") and len(k) > 10)

    @property
    def has_nvidia_key(self) -> bool:
        k = self.nvidia_api_key.strip()
        return bool(k and not k.startswith("nvapi-your-key") and len(k) > 10)

    @property
    def has_ecohash_key(self) -> bool:
        k = self.ecohash_api_key.strip()
        return bool(k and len(k) > 5)

    @property
    def has_valid_api_key(self) -> bool:
        return (
            self.has_gemini_key or self.has_groq_key or self.has_nvidia_key or self.has_ecohash_key
        )


settings = Settings()
