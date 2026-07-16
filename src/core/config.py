"""
Centralized configuration management.
All settings loaded from environment variables.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Keys
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None

    # Weaviate
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # Reranking
    reranker_mode: str = Field(
        default="app",
        description="Reranker mode: weaviate, app, or none",
    )
    app_reranker_model: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        description="SentenceTransformers CrossEncoder used when RERANKER_MODE=app",
    )
    weaviate_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Weaviate reranker-transformers model used when RERANKER_MODE=weaviate",
    )
    app_reranker_device: str = Field(
        default="cpu",
        description="Device for app-level CrossEncoder reranker",
    )
    app_reranker_batch_size: int = Field(
        default=8,
        description="Batch size for app-level CrossEncoder reranker",
    )
    warmup_reranker: bool = Field(
        default=True,
        description="Warm up the app-level reranker during API startup",
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=20,
        description="Candidates fetched per query before final reranking",
    )
    retrieval_top_k_reranker: int = Field(
        default=5,
        description="Final number of retrieved documents after reranking",
    )
    retrieval_alpha: float = Field(
        default=0.5,
        description="Hybrid search weight (0=keyword/BM25, 1=vector). Default 0.5 for balanced.",
    )

    # Runtime behavior
    warmup_docling: bool = Field(
        default=False,
        description="Warm up the Docling PDF parser during API startup",
    )
    max_upload_size_mb: int = Field(
        default=100,
        description="Maximum uploaded PDF size in MB",
    )

    # Postgres Persistence
    pg_user: str = Field(default="postgres", description="Postgres user")
    pg_password: str = Field(default="postgres", description="Postgres password")
    pg_database: str = Field(default="agentic-rag", description="Postgres DB name")
    pg_host: str = Field(default="localhost", description="Postgres host")
    pg_port: int = Field(default=5432, description="Postgres port")
    pg_pool_max_size: int = Field(default=5, description="Postgres pool max size")

    @property
    def pg_url(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"

    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def data_raw_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
