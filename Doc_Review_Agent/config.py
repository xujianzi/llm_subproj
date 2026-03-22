from functools import lru_cache
import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration pulled from environment variables."""

    env: str = Field(default=os.getenv("ENV", "dev"))
    chroma_path: str = Field(default=os.getenv("CHROMA_PATH", "storage/chroma"))
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    search_enabled: bool = Field(
        default=os.getenv("SEARCH_ENABLED", "true").lower() == "true"
    )
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", 800)))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", 100)))
    rerank_enabled: bool = Field(
        default=os.getenv("RERANK_ENABLED", "false").lower() == "true"
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
