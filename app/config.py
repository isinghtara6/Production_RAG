"""
Centralized, validated application configuration.

All runtime knobs come from environment variables (12-factor). Nothing is
hardcoded so the same image can be promoted local -> staging -> production
without code changes. Validation happens at process startup so a
misconfigured deployment fails fast instead of misbehaving at request time.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service identity ---
    app_name: str = "rag-service"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_version: str = "v1"

    # --- API integrity ---
    # "key_id:sha256(secret)" pairs. Secrets are never stored in plaintext.
    api_keys: str = ""
    require_request_signing: bool = False
    signature_tolerance_seconds: int = 300
    max_request_body_bytes: int = 5 * 1024 * 1024

    # --- Rate limiting ---
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    ingest_rate_limit_requests: int = 10
    ingest_rate_limit_window_seconds: int = 60

    # --- RAG pipeline ---
    embedding_provider: Literal["hash", "sentence_transformers"] = "hash"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    top_k: int = 5
    min_relevance_score: float = 0.15

    generation_provider: Literal["extractive", "anthropic", "openai", "gemini"] = "extractive"
    generation_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    vector_store_backend: Literal["numpy", "faiss"] = "numpy"
    vector_store_path: str = "./data/vector_store"
    metadata_db_path: str = "./data/metadata.sqlite3"

    allowed_origins: str = "http://localhost:3000"

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size_tokens", 350)
        if v >= size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return v

    @property
    def api_key_map(self) -> dict[str, str]:
        """Parse 'id:hash,id:hash' into {id: hash}. Blank config -> empty (auth denies all)."""
        pairs = [p for p in self.api_keys.split(",") if p.strip()]
        out: dict[str, str] = {}
        for p in pairs:
            if ":" not in p:
                continue
            key_id, secret_hash = p.split(":", 1)
            out[key_id.strip()] = secret_hash.strip()
        return out

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are cached (singleton) for the process lifetime; tests can
    call get_settings.cache_clear() to reload with different env vars."""
    return Settings()
