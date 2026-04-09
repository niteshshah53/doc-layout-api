"""
tests/test_inference.py
-----------------------
Unit tests for the inference pipeline.

TESTING PHILOSOPHY:
  We test each pipeline stage independently using mocks where needed.
  This means tests are fast (no GPU needed) and catch regressions early.

  - test_decode_image_*      → Stage 1: image decoding & validation
  - test_preprocess_*        → Stage 2: resize & numpy conversion
  - test_postprocess_*       → Stage 3: coordinate scaling
  - test_full_pipeline_*     → Integration: mock the model, test the flow

Run with:
  pytest tests/ -v
  pytest tests/ -v --tb=short   # shorter tracebacks
"""

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from app.inference import decode_image, postprocess, preprocess
from app.schemas import BoundingBox, LayoutBlock


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_image_bytes(width: int = 800, height: int = 1000, fmt: str = "PNG") -> bytes:
    """Create a minimal valid image in memory."""
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_rgba_image_bytes() -> bytes:
    img = Image.new("RGBA", (400, 600), color=(200, 200, 200, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── decode_image() tests ──────────────────────────────────────────────────────

class TestDecodeImage:
    def test_valid_png(self):
        result = decode_image(make_image_bytes(fmt="PNG"))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_valid_jpeg(self):
        result = decode_image(make_image_bytes(fmt="JPEG"))
        assert result.mode == "RGB"

    def test_rgba_converted_to_rgb(self):
        """RGBA images (PNG with transparency) must be converted to RGB."""
        result = decode_image(make_rgba_image_bytes())
        assert result.mode == "RGB"

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="Cannot decode image"):
            decode_image(b"this is not an image")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError):
            decode_image(b"")


# ── preprocess() tests ────────────────────────────────────────────────────────

class TestPreprocess:
    def test_returns_numpy_array(self):
        image = Image.new("RGB", (400, 600))
        arr, orig_size = preprocess(image)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert arr.ndim == 3          # H x W x C
        assert arr.shape[2] == 3      # 3 channels (RGB)

    def test_original_size_preserved(self):
        image = Image.new("RGB", (400, 600))
        _, orig_size = preprocess(image)
        assert orig_size == (400, 600)

    def test_large_image_is_resized(self):
        """Images larger than max_image_size should be downscaled."""
        image = Image.new("RGB", (3000, 4000))
        arr, _ = preprocess(image)
        # longest edge should be ≤ max_image_size (1024 by default)
        assert max(arr.shape[0], arr.shape[1]) <= 1024

    def test_small_image_not_upscaled(self):
        """Small images should NOT be upscaled."""
        image = Image.new("RGB", (200, 300))
        arr, _ = preprocess(image)
        assert arr.shape[1] == 200   # width unchanged
        assert arr.shape[0] == 300   # height unchanged

    def test_aspect_ratio_preserved_on_resize(self):
        """Resizing should preserve aspect ratio."""
        image = Image.new("RGB", (2000, 1000))  # 2:1 ratio
        arr, _ = preprocess(image)
        ratio = arr.shape[1] / arr.shape[0]     # new W / new H
        assert abs(ratio - 2.0) < 0.05          # allow small rounding


# ── postprocess() tests ───────────────────────────────────────────────────────

class TestPostprocess:
    def _make_mock_layout(self, boxes: list) -> list:
        """Create mock layoutparser block objects."""
        blocks = []
        for label, score, coords in boxes:
            block = MagicMock()
            block.type = label
            block.score = score
            block.block.coordinates = coords   # (x1, y1, x2, y2)
            blocks.append(block)
        return blocks

    def test_no_resize_coordinates_unchanged(self):
        """If original == resized, coordinates should be unchanged."""
        layout = self._make_mock_layout([
            ("Text", 0.95, (10.0, 20.0, 200.0, 80.0)),
        ])
        blocks = postprocess(layout, original_size=(800, 1000), resized_size=(800, 1000))
        assert len(blocks) == 1
        assert blocks[0].bbox.x1 == pytest.approx(10.0)
        assert blocks[0].bbox.y1 == pytest.approx(20.0)

    def test_coordinate_scaling_applied(self):
        """Coordinates must be scaled back to original image dimensions."""
        # Original: 1600x2000, Resized: 800x1000 (scale = 2x)
        layout = self._make_mock_layout([
            ("Title", 0.90, (50.0, 100.0, 300.0, 150.0)),  # in resized coords
        ])
        blocks = postprocess(layout, original_size=(1600, 2000), resized_size=(800, 1000))
        assert blocks[0].bbox.x1 == pytest.approx(100.0)   # 50 * 2
        assert blocks[0].bbox.y1 == pytest.approx(200.0)   # 100 * 2

    def test_reading_order_sort(self):
        """Blocks should be sorted top-to-bottom, then left-to-right."""
        layout = self._make_mock_layout([
            ("Text",  0.8, (100.0, 500.0, 400.0, 600.0)),   # bottom block
            ("Title", 0.9, (100.0, 50.0,  400.0, 120.0)),   # top block
        ])
        blocks = postprocess(layout, original_size=(800, 1000), resized_size=(800, 1000))
        assert blocks[0].label == "Title"   # top block first
        assert blocks[1].label == "Text"

    def test_empty_layout_returns_empty_list(self):
        blocks = postprocess([], original_size=(800, 1000), resized_size=(800, 1000))
        assert blocks == []

    def test_output_types(self):
        layout = self._make_mock_layout([
            ("Table", 0.75, (10.0, 10.0, 100.0, 100.0)),
        ])
        blocks = postprocess(layout, original_size=(800, 1000), resized_size=(800, 1000))
        b = blocks[0]
        assert isinstance(b, LayoutBlock)
        assert isinstance(b.label, str)
        assert isinstance(b.score, float)
        assert isinstance(b.bbox, BoundingBox)
