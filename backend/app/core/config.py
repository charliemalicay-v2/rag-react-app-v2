import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    chroma_api_key: Optional[str] = None
    chroma_tenant: str = "default"
    chroma_database: str = "default"
    chroma_host: str = "api.chroma.app"
    chroma_collection_name: str = "smartmove-collection"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    backend_cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])
    agent_model_name: str = "minimax-m3:cloud"
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
