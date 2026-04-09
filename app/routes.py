"""
app/routes.py
-------------
FastAPI route definitions. ONLY handles HTTP concerns:
  - parsing the request
  - calling the inference pipeline
  - returning the response or a structured error

WHAT DOES NOT BELONG HERE:
  - model loading logic     → app/model.py
  - inference logic         → app/inference.py
  - config                  → app/config.py
  - data schemas            → app/schemas.py

This separation means you can unit-test inference.py without
starting an HTTP server.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.config import get_settings
from app.inference import run_full_pipeline
import app.model  # Import module to access model functions
from app.schemas import ErrorResponse, HealthResponse, PredictionResponse

router = APIRouter()
settings = get_settings()

# ── Health check ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns service health status including which model is loaded (primary/fallback). "
        "Used by Docker HEALTHCHECK and load balancers."
    ),
    tags=["Monitoring"],
)
async def health_check() -> HealthResponse:
    model_info = app.model.get_model_info()
    model_info_dict = None
    
    if model_info.loaded:
        model_info_dict = {
            "type": model_info.model_type,
            "error": model_info.error,
        }
    
    return HealthResponse(
        status="ok",
        model_loaded=model_info.loaded,
        model_info=model_info_dict,
        version=settings.app_version,
    )


# ── Predict ───────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or file type"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Model inference failed"},
    },
    summary="Detect document layout",
    description=(
        "Upload a document image (PNG, JPEG, TIFF, BMP, WEBP). "
        "Returns detected layout blocks (text, title, table, figure, list) "
        "with confidence scores and bounding box coordinates."
    ),
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Document image file. Max size controlled by MAX_FILE_SIZE_MB env var.",
    ),
) -> PredictionResponse:
    # ── 1. Validate content type (fast check before reading bytes) ────────
    allowed_content_types = {
        "image/jpeg", "image/jpg", "image/png",
        "image/tiff", "image/bmp", "image/webp",
    }
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. "
                   f"Allowed: {', '.join(sorted(allowed_content_types))}",
        )

    # ── 2. Read bytes & enforce size limit ────────────────────────────────
    image_bytes = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {len(image_bytes) / 1e6:.1f}MB exceeds "
                   f"limit of {settings.max_file_size_mb}MB.",
        )

    logger.info(
        f"Received file: {file.filename!r} | "
        f"type={file.content_type} | "
        f"size={len(image_bytes) / 1e3:.1f}KB"
    )

    # ── 3. Check if model is loaded ───────────────────────────────────────
    model_info = app.model.get_model_info()
    if not model_info.loaded:
        error_detail = (
            "Model is not loaded. Server may still be initializing. "
            "Check /health endpoint for status."
        )
        if model_info.error:
            error_detail += f" Error: {model_info.error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail,
        )

    # ── 4. Run inference pipeline ─────────────────────────────────────────
    try:
        result = run_full_pipeline(image_bytes)
    except ValueError as e:
        # ValueError = user error (bad image, unsupported format)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Unexpected errors = our problem
        logger.exception(f"Inference failed for file {file.filename!r}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Check server logs.",
        )

    logger.info(
        f"Prediction complete: {result.num_blocks} blocks | "
        f"{result.inference_time_ms}ms"
    )
    return result
