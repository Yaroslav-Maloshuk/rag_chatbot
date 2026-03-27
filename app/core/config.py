from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    log_level: str = "INFO"

    cors_origins: str = "*"

    database_url: str = "postgresql+asyncpg://rag:rag@postgres:5432/ragdb"
    database_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    redis_url: str = "redis://redis:6379/2"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    upload_dir: str = "/app/data/uploads"
    max_upload_size_mb: int = 200
    large_file_threshold_mb: int = 8

    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_tokenizer_name: str = "bert-base-uncased"
    max_context_tokens: int = 3200
    enable_ocr_fallback: bool = True
    ocr_language: str = "auto"
    ocr_language_candidates: str = (
        "eng,script/Latin,script/Cyrillic,script/Arabic,script/Devanagari,"
        "script/HanS,script/HanT,script/Japanese,script/Korean,script/Thai,"
        "script/Hebrew,script/Greek"
    )
    ocr_good_text_min_chars: int = 120
    ocr_dpi: int = 200

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    llm_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    model_device: str = "auto"
    llm_max_new_tokens: int = 384
    llm_temperature: float = 0.0

    retrieval_top_k: int = 7
    retrieval_candidate_k: int = 40
    min_relevance_score: float = 0.05
    retrieval_min_results: int = 3
    text_search_config: str = "simple"
    enable_hybrid_search: bool = True
    hybrid_vector_weight: float = 0.8
    hybrid_bm25_weight: float = 0.2

    enable_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    enable_cache: bool = True
    cache_ttl_seconds: int = 300

    enable_metrics: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        raw = (self.cors_origins or "").strip()
        if not raw:
            return ["*"]

        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:  # noqa: BLE001
                pass

        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
