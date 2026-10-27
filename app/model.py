"""Docling Heron document-layout model loading and inference adapter."""

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from app.config import get_settings

DOCLING_LAYOUT_MODEL = "docling-project/docling-layout-heron"
DOCLING_LABELS = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
    11: "Document Index",
    12: "Code",
    13: "Checkbox-Selected",
    14: "Checkbox-Unselected",
    15: "Form",
    16: "Key-Value Region",
}

_model: "DoclingLayoutDetector | None" = None
_model_status: dict[str, Any] = {
    "loaded": False,
    "model_type": None,
    "error": None,
}


@dataclass(frozen=True)
class ModelInfo:
    loaded: bool
    model_type: str | None
    error: str | None


@dataclass(frozen=True)
class LayoutDetection:
    label: str
    score: float
    coordinates: tuple[float, float, float, float]


class DoclingLayoutDetector:
    """Run Docling Heron's RT-DETR model on RGB numpy images."""

    def __init__(
        self,
        processor: Any,
        detector: Any,
        categories: dict[int, str],
        device: torch.device,
        threshold: float,
    ) -> None:
        self._processor = processor
        self._detector = detector.to(device).eval()
        self._categories = categories
        self._device = device
        self._threshold = threshold

    def detect(self, image: Any) -> list[LayoutDetection]:
        """Return document-layout detections in the supplied image coordinates."""
        from PIL import Image

        pil_image = Image.fromarray(image, mode="RGB")
        inputs = self._processor(images=[pil_image], return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._detector(**inputs)

        target_sizes = torch.tensor(
            [[pil_image.height, pil_image.width]], device=self._device
        )
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self._threshold,
        )
        result = results[0]

        detections = []
        for score, label_id, box in zip(
            result["scores"].detach().cpu().tolist(),
            result["labels"].detach().cpu().tolist(),
            result["boxes"].detach().cpu().tolist(),
        ):
            detections.append(
                LayoutDetection(
                    label=self._categories.get(label_id, f"class_{label_id}"),
                    score=float(score),
                    coordinates=tuple(float(value) for value in box),
                )
            )
        return detections


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        logger.warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported DEVICE '{requested}'. Expected 'cpu' or 'cuda'.")


def load_model() -> tuple[DoclingLayoutDetector | None, str | None, str | None]:
    """Download and load Docling Heron on first use, reporting failures."""
    settings = get_settings()
    device = _resolve_device(settings.device)

    try:
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        processor = RTDetrImageProcessor.from_pretrained(DOCLING_LAYOUT_MODEL)
        detector = RTDetrV2ForObjectDetection.from_pretrained(DOCLING_LAYOUT_MODEL)
        categories = DOCLING_LABELS.copy()
        wrapped = DoclingLayoutDetector(
            processor=processor,
            detector=detector,
            categories=categories,
            device=device,
            threshold=settings.model_score_threshold,
        )
        logger.info(
            "Loaded Docling Heron layout model on {} with threshold {}",
            device,
            settings.model_score_threshold,
        )
        return wrapped, "docling-heron", None
    except Exception as exc:
        message = f"Model loading failed: {type(exc).__name__}: {exc}"
        logger.exception(message)
        return None, None, message


def get_model() -> DoclingLayoutDetector:
    if _model is None:
        raise RuntimeError(
            "Model is not loaded. Check /api/v1/health for the startup error."
        )
    return _model


def get_model_info() -> ModelInfo:
    return ModelInfo(
        loaded=bool(_model_status["loaded"]),
        model_type=_model_status["model_type"],
        error=_model_status["error"],
    )


def init_model() -> None:
    """Initialize the singleton model used by the API process."""
    global _model, _model_status

    model, model_type, error = load_model()
    _model = model
    _model_status = {
        "loaded": model is not None,
        "model_type": model_type,
        "error": error,
    }
