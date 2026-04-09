"""
tests/test_routes.py
--------------------
Integration tests for the FastAPI HTTP layer.

We use FastAPI's TestClient (which wraps httpx) to test the full
HTTP request/response cycle WITHOUT starting a real server.

MOCKING STRATEGY:
  We mock run_full_pipeline() at the routes level so tests:
  - Don't need a real model loaded
  - Run in milliseconds (no inference)
  - Still test all HTTP logic: status codes, headers, validation, errors

Run with:
  pytest tests/test_routes.py -v
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.schemas import BoundingBox, LayoutBlock, PredictionResponse


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Create a test client with a fresh app instance.
    Model loading is mocked at the session level (conftest.py)
    so the lifespan will succeed with a mock model.
    """
    app = create_app()
    with TestClient(app) as c:
        yield c


def make_png_bytes(width: int = 400, height: int = 600) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_mock_response() -> PredictionResponse:
    return PredictionResponse(
        num_blocks=2,
        image_size={"width": 400, "height": 600},
        inference_time_ms=45.2,
        blocks=[
            LayoutBlock(
                label="Title",
                score=0.95,
                bbox=BoundingBox(x1=10, y1=20, x2=300, y2=60),
            ),
            LayoutBlock(
                label="Text",
                score=0.88,
                bbox=BoundingBox(x1=10, y1=80, x2=390, y2=500),
            ),
        ],
    )


# ── Health check tests ────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_response_shape(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_status_ok(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"


# ── Predict endpoint tests ────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_png_returns_200(self, client):
        with patch("app.routes.run_full_pipeline", return_value=make_mock_response()):
            response = client.post(
                "/api/v1/predict",
                files={"file": ("test.png", make_png_bytes(), "image/png")},
            )
        assert response.status_code == 200

    def test_response_contains_blocks(self, client):
        with patch("app.routes.run_full_pipeline", return_value=make_mock_response()):
            data = client.post(
                "/api/v1/predict",
                files={"file": ("test.png", make_png_bytes(), "image/png")},
            ).json()
        assert "blocks" in data
        assert data["num_blocks"] == 2
        assert data["blocks"][0]["label"] == "Title"

    def test_invalid_content_type_returns_400(self, client):
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.pdf", b"fake pdf bytes", "application/pdf")},
        )
        assert response.status_code == 400

    def test_bad_image_bytes_returns_400(self, client):
        with patch(
            "app.routes.run_full_pipeline",
            side_effect=ValueError("Cannot decode image"),
        ):
            response = client.post(
                "/api/v1/predict",
                files={"file": ("bad.png", b"not an image", "image/png")},
            )
        assert response.status_code == 400

    def test_model_crash_returns_500(self, client):
        with patch(
            "app.routes.run_full_pipeline",
            side_effect=RuntimeError("CUDA out of memory"),
        ):
            response = client.post(
                "/api/v1/predict",
                files={"file": ("test.png", make_png_bytes(), "image/png")},
            )
        assert response.status_code == 500

    def test_missing_file_returns_422(self, client):
        """FastAPI returns 422 when a required field is missing."""
        response = client.post("/api/v1/predict")
        assert response.status_code == 422

    def test_inference_time_in_response(self, client):
        with patch("app.routes.run_full_pipeline", return_value=make_mock_response()):
            data = client.post(
                "/api/v1/predict",
                files={"file": ("test.png", make_png_bytes(), "image/png")},
            ).json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] > 0
