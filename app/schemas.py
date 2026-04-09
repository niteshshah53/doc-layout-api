"""
app/schemas.py
--------------
Pydantic models that define the shape of our API's inputs and outputs.

WHY PYDANTIC SCHEMAS MATTER:
  1. Automatic validation — FastAPI validates request/response against these.
     Invalid data → 422 Unprocessable Entity, not a 500 server crash.
  2. Auto-generated OpenAPI docs (Swagger UI) — FastAPI reads these schemas
     to build the /docs page. No manual documentation needed.
  3. Type safety — your IDE will catch bugs at write time, not runtime.
  4. Serialization — Pydantic handles JSON serialization cleanly.

SEPARATION FROM BUSINESS LOGIC:
  Schemas live in their own file. inference.py and routes.py import them.
  This prevents circular imports and keeps each module's job clear.
"""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box in pixel coordinates.
    Coordinates are relative to the ORIGINAL (pre-resize) image.
    """
    x1: float = Field(..., description="Left edge (pixels)")
    y1: float = Field(..., description="Top edge (pixels)")
    x2: float = Field(..., description="Right edge (pixels)")
    y2: float = Field(..., description="Bottom edge (pixels)")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class LayoutBlock(BaseModel):
    """
    A single detected layout element (text block, title, table, etc.)
    """
    label: str = Field(
        ...,
        description="Detected class label",
        examples=["Text", "Title", "Table", "Figure", "List"],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from the model (0.0 – 1.0)",
    )
    bbox: BoundingBox = Field(
        ...,
        description="Bounding box in original image coordinates",
    )


class PredictionResponse(BaseModel):
    """
    Full response from the /predict endpoint.
    """
    num_blocks: int = Field(..., description="Total number of detected layout blocks")
    image_size: dict[str, int] = Field(
        ...,
        description="Original image dimensions",
        examples=[{"width": 2480, "height": 3508}],
    )
    inference_time_ms: float = Field(
        ...,
        description="Time taken for model inference (excludes pre/post processing)",
    )
    blocks: list[LayoutBlock] = Field(
        default_factory=list,
        description="Detected layout blocks, sorted by reading order (top-to-bottom, left-to-right)",
    )


class ErrorResponse(BaseModel):
    """
    Standardised error response shape.
    Using a schema (not bare dicts) means errors are also documented in Swagger.
    """
    error: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(default=None, description="Additional context")


class HealthResponse(BaseModel):
    """Response for the /health endpoint — used by Docker/k8s health checks."""
    status: str
    model_loaded: bool
    model_info: dict | None = Field(
        default=None,
        description=(
            "Details about the loaded model. Contains 'type' (primary/fallback) "
            "and optionally 'error' if fallback was loaded due to primary failure."
        ),
        examples=[
            {"type": "primary", "error": None},
            {"type": "fallback", "error": "Primary model load failed: [details]"},
        ]
    )
    version: str
