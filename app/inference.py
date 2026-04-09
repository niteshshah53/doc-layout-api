"""
app/inference.py
----------------
The core inference pipeline: image bytes → structured predictions.

PIPELINE STAGES:
  1. decode_image()      — raw bytes → PIL Image (validate format)
  2. preprocess()        — resize to safe dimensions, convert to numpy
  3. run_inference()     — layoutparser detection with torch.no_grad()
  4. postprocess()       — convert layoutparser Layout → clean Python dicts

WHY THIS SEPARATION:
  Each stage is independently testable and replaceable.
  If you swap the model later (e.g. your thesis model), you only change
  run_inference(). The pre/post processing stays the same.
"""

import io
import time
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from loguru import logger

from app.config import get_settings
from app.model import get_model
from app.schemas import BoundingBox, LayoutBlock, PredictionResponse


# ── Allowed formats ───────────────────────────────────────────────────────────
ALLOWED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP", "WEBP"}


def decode_image(image_bytes: bytes) -> Image.Image:
    """
    Decode raw bytes to a PIL Image.
    Validates format early — fail fast before touching the model.

    Raises:
        ValueError: if the bytes are not a valid/supported image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()                          # checks file integrity
        image = Image.open(io.BytesIO(image_bytes))  # re-open after verify (verify closes it)
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"Cannot decode image: {e}") from e

    if image.format not in ALLOWED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{image.format}'. "
            f"Allowed: {', '.join(ALLOWED_FORMATS)}"
        )

    # Ensure RGB — models expect 3-channel input
    # RGBA (PNG with transparency), L (grayscale), P (palette) → RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        logger.debug(f"Converted image mode to RGB")

    return image


def preprocess(image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize image so its longest edge ≤ max_image_size, then convert to numpy.

    Returns:
        (numpy_array, original_size) — we return original_size so
        postprocessing can scale bounding boxes back to original coordinates.

    WHY RESIZE:
        Large images (e.g. 4000x3000) would slow inference 10x+.
        The model was trained on document images ~800-1200px wide.
        Resizing to max_image_size gives the best speed/accuracy trade-off.
    """
    settings = get_settings()
    original_size = image.size    # (width, height)
    max_size = settings.max_image_size

    # Aspect-ratio-preserving resize
    w, h = original_size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"Resized image: {original_size} → ({new_w}, {new_h})")

    # layoutparser expects a plain numpy uint8 array, NOT a tensor
    numpy_image = np.array(image)
    return numpy_image, original_size


def run_inference(numpy_image: np.ndarray) -> Any:
    """
    Run the layout detection model on a preprocessed numpy image.

    torch.no_grad() context:
        Disables gradient computation — we are doing inference, not training.
        This reduces memory usage ~50% and speeds up inference.
        This is the CORRECT place to apply it (the forward pass), not model loading.
    """
    model = get_model()

    t_start = time.perf_counter()
    with torch.no_grad():
        layout = model.detect(numpy_image)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(f"Inference completed in {elapsed_ms:.1f}ms | Blocks detected: {len(layout)}")
    return layout, elapsed_ms


def postprocess(
    layout: Any,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> list[LayoutBlock]:
    """
    Convert layoutparser Layout object → list of LayoutBlock (our Pydantic schema).

    COORDINATE SCALING:
        The model ran on the resized image. If we resized, we need to
        scale bounding box coordinates back to the original image dimensions
        so the caller can use them meaningfully.

    Args:
        layout:         layoutparser Layout object from run_inference()
        original_size:  (width, height) of the original image
        resized_size:   (width, height) after preprocessing
    """
    orig_w, orig_h = original_size
    res_w, res_h = resized_size

    # Scale factors (1.0 if no resize happened)
    scale_x = orig_w / res_w
    scale_y = orig_h / res_h

    blocks = []
    for block in layout:
        # layoutparser Rectangle has .coordinates = (x_1, y_1, x_2, y_2)
        x1, y1, x2, y2 = block.block.coordinates

        # Scale back to original image coordinates
        bbox = BoundingBox(
            x1=round(x1 * scale_x, 2),
            y1=round(y1 * scale_y, 2),
            x2=round(x2 * scale_x, 2),
            y2=round(y2 * scale_y, 2),
        )

        blocks.append(
            LayoutBlock(
                label=block.type,
                score=round(float(block.score), 4),
                bbox=bbox,
            )
        )

    # Sort by reading order: top-to-bottom, then left-to-right
    blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))
    return blocks


def run_full_pipeline(image_bytes: bytes) -> PredictionResponse:
    """
    Orchestrates the full pipeline end-to-end.
    This is what the API endpoint calls — it owns the full flow
    and bubbles up any ValueError from validation.
    """
    # Stage 1: Decode & validate
    image = decode_image(image_bytes)
    original_size = image.size

    # Stage 2: Preprocess
    numpy_image, _ = preprocess(image)
    resized_size = (numpy_image.shape[1], numpy_image.shape[0])  # (W, H) from (H, W, C)

    # Stage 3: Inference
    layout, inference_time_ms = run_inference(numpy_image)

    # Stage 4: Postprocess
    blocks = postprocess(layout, original_size, resized_size)

    return PredictionResponse(
        num_blocks=len(blocks),
        image_size={"width": original_size[0], "height": original_size[1]},
        inference_time_ms=round(inference_time_ms, 2),
        blocks=blocks,
    )
