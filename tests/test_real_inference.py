"""Opt-in real Docling Heron inference regression test.

Run with:
    RUN_REAL_MODEL_TESTS=1 pytest tests/test_real_inference.py -v
"""

import io
import os

import pytest
from PIL import Image, ImageDraw

from app.inference import run_full_pipeline
from app.model import init_model

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="set RUN_REAL_MODEL_TESTS=1 to download/use Docling Heron weights",
)


def test_real_nonblank_document_page():
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 920, 180), outline="black", width=4)
    draw.text((110, 110), "Annual Report", fill="black")
    for y in range(260, 1000, 45):
        draw.line((100, y, 900, y), fill="black", width=2)
    draw.rectangle((100, 1080, 900, 1280), outline="black", width=4)

    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    init_model()
    result = run_full_pipeline(encoded.getvalue())

    assert result.image_size == {"width": 1000, "height": 1400}
    assert result.inference_time_ms > 0
    assert result.num_blocks == len(result.blocks)
    assert all(block.label for block in result.blocks)
    assert all(0.0 <= block.score <= 1.0 for block in result.blocks)
