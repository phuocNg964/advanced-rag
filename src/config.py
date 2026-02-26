"""
Centralized configuration management.
All settings loaded from environment variables.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Keys
    gemini_api_key: str
    openai_api_key: str | None = None
    unstructured_api_key: str | None = None
    
    # Weaviate
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    
    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent
    
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
