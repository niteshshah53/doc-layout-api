"""
app/config.py
-------------
Centralised configuration using pydantic-settings.
All values can be overridden via environment variables or a .env file.

WHY THIS PATTERN:
  - No hardcoded values scattered across files
  - Easy to change behaviour per environment (dev / docker / cloud)
  - Type-safe: pydantic validates types at startup, not at runtime
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),
    )
    # ── API ──────────────────────────────────────────────────────────────
    app_name: str = "Document Layout Detection API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Model ────────────────────────────────────────────────────────────
    model_config_path: str = "docling-project/docling-layout-heron"
    model_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "cuda"          # "cuda" or "cpu" — override via env var

    # ── Inference ────────────────────────────────────────────────────────
    max_image_size: int = 1024    # resize longest edge to this before inference
    max_file_size_mb: int = 10    # reject uploads larger than this

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

@lru_cache()          # ← singleton: parse settings only once per process
def get_settings() -> Settings:
    return Settings()
