from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    semantic_scholar_api_key: Optional[str] = None
    arxiv_max_results: int = 50
    # Use a widely available CPU-friendly model by default
    model_embed: str = "all-MiniLM-L6-v2"
    # Some accounts may not have access to all Gemini variants; prefer 8B as default
    model_gemini: str = "gemini-1.5-flash-8b"
    top_k: int = 20
    downloads_dir: str = "app/static/downloads"
    slowapi_rate_limit: str = "5/minute"
    retrieval_cache_ttl_seconds: int = 900
    retrieval_cache_max_entries: int = 64
    semantic_retry_attempts: int = 2
    semantic_retry_delay_seconds: float = 1.5
    openalex_enabled: bool = True
    openalex_mailto: str = "literature-review@example.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    @property
    def downloads_path(self) -> Path:
        path = Path(self.downloads_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
