from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Paths -----------------------------------------------------------
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"

    # --- Chunking ----------------------------------------------------------
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 60

    # --- Embeddings / retrieval -------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    dense_top_k: int = 25
    sparse_top_k: int = 25
    rrf_k: int = 60  # reciprocal-rank-fusion damping constant
    fused_top_k: int = 15  # candidates handed to the reranker
    final_top_k: int = 5  # chunks handed to the generator after reranking

    # --- Generation ----------------------------------------------------------
    # "hf_local"  -> transformers pipeline running Llama-3.1-8B-Instruct locally
    # "anthropic" -> Anthropic Messages API
    # "openai"    -> OpenAI chat completions API
    llm_backend: str = Field(default="hf_local")
    hf_generation_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token: str | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"  # verify current id at https://docs.claude.com
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    generation_max_tokens: int = 800
    generation_temperature: float = 0.1

    # --- Hallucination checking --------------------------------------------
    entailment_threshold: float = 0.5

    # --- API -----------------------------------------------------------------
    api_key: str | None = Field(default=None, description="If set, required as X-API-Key header")
    cors_allow_origins: list[str] = ["*"]
    rate_limit: str = "30/minute"
    redis_url: str | None = None  # e.g. redis://redis:6379/0 — enables response caching

    # --- Misc --------------------------------------------------------------
    log_level: str = "INFO"
    device: str = "cpu"  # "cuda" if a GPU is available


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
