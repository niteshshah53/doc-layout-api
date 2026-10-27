"""Focused tests for Docling model loading, labels, thresholds, and failures."""

from unittest.mock import patch

import pytest
import torch

from app import model


class FakeProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros((1, 3, 10, 10))}

    def post_process_object_detection(self, outputs, target_sizes, threshold):
        assert target_sizes.tolist() == [[10, 20]]
        assert threshold == 0.5
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 10.0, 8.0]]),
                "labels": torch.tensor([10]),
                "scores": torch.tensor([0.9]),
            }
        ]


class FakeDetector(torch.nn.Module):
    class Config:
        id2label = {"10": "title"}

    config = Config()

    def forward(self, **inputs):
        return object()


def test_detector_returns_docling_label_and_coordinates():
    detector = model.DoclingLayoutDetector(
        FakeProcessor(), FakeDetector(), {10: "Title"}, torch.device("cpu"), 0.5
    )

    detections = detector.detect(torch.zeros((10, 20, 3), dtype=torch.uint8).numpy())

    assert len(detections) == 1
    assert detections[0].label == "Title"
    assert detections[0].score == pytest.approx(0.9)
    assert detections[0].coordinates == (1.0, 2.0, 10.0, 8.0)


def test_detector_propagates_inference_errors():
    class BrokenDetector(torch.nn.Module):
        def forward(self, **inputs):
            raise RuntimeError("inference failed")

    detector = model.DoclingLayoutDetector(
        FakeProcessor(), BrokenDetector(), {}, torch.device("cpu"), 0.5
    )
    with pytest.raises(RuntimeError, match="inference failed"):
        detector.detect(torch.zeros((10, 10, 3), dtype=torch.uint8).numpy())


def test_load_model_reports_dependency_or_weight_failure():
    with patch(
        "transformers.RTDetrImageProcessor.from_pretrained",
        side_effect=OSError("weights unavailable"),
    ):
        loaded, model_type, error = model.load_model()

    assert loaded is None
    assert model_type is None
    assert error is not None
    assert "weights unavailable" in error
