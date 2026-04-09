"""
app/main.py
-----------
FastAPI application factory and startup/shutdown lifecycle.

THE LIFESPAN PATTERN (contextmanager):
  FastAPI (and Starlette) use a lifespan context manager to handle
  startup and shutdown events. This replaced the older @app.on_event
  approach in FastAPI 0.93+.

  WHY IT MATTERS:
    - Model is loaded ONCE when the process starts, held in memory
    - All requests share the same model instance (no reload per request)
    - On shutdown, we can cleanly release GPU memory (important for CUDA)
    - If model loading fails, the app refuses to start → fail fast,
      not fail silently on first request
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.model import init_model
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model into memory.
    Shutdown: release GPU memory.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info("Loading model...")
    init_model()
    
    # Check if model loaded successfully
    from app.model import get_model_info
    model_info = get_model_info()
    
    if model_info.loaded:
        logger.info(f"✓ Model loaded: {model_info.model_type} model is ready")
    else:
        logger.error(f"✗ Model failed to load: {model_info.error}")
        logger.error("The API will start but inference requests will fail (503 Service Unavailable)")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        if model_info.loaded:
            logger.info(
                f"GPU memory after model load: "
                f"{torch.cuda.memory_allocated() / 1e9:.2f}GB allocated"
            )

    logger.info("Application ready ✓")

    yield  # ← application runs while we're yielded here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared ✓")


def create_app() -> FastAPI:
    """
    Application factory function.

    WHY A FACTORY:
      Instead of a module-level `app = FastAPI(...)`, we use a function.
      This makes it easy to create multiple app instances in tests,
      or to parametrise the app differently per environment.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-ready REST API for document layout segmentation. "
            "Detects text blocks, titles, tables, figures, and lists "
            "in document images using a PubLayNet-trained Detectron2 model."
        ),
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc UI (alternative)
        openapi_url="/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    # Allows browser-based clients to call the API.
    # In production: replace "*" with your actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": f"{settings.app_name}",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
