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

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── API ──────────────────────────────────────────────────────────────
    app_name: str = "Document Layout Detection API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Model ────────────────────────────────────────────────────────────
    # Which PubLayNet config to use.
    # layoutparser hosts these on their model zoo, but the Dropbox URLs are
    # often corrupted with YAML syntax errors.
    #
    # Recommended: Use COCO-Detection models from Detectron2 directly:
    #   "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    #
    # Or use layoutparser (if your version works):
    #   "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    #
    # Override via env: MODEL_CONFIG_PATH="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    model_config_path: str = (
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    model_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "cuda"          # "cuda" or "cpu" — override via env var

    # ── Inference ────────────────────────────────────────────────────────
    max_image_size: int = 1024    # resize longest edge to this before inference
    max_file_size_mb: int = 10    # reject uploads larger than this

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"         # load from .env if present
        env_file_encoding = "utf-8"


@lru_cache()          # ← singleton: parse settings only once per process
def get_settings() -> Settings:
    return Settings()
