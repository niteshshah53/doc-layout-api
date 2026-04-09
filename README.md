# Document Layout Detection API

A production-ready REST API for document layout segmentation.
Detects **text blocks, titles, tables, figures, and lists** in document images
using a [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)-trained Detectron2 model.

## Tech Stack
- **FastAPI** — REST API framework
- **layoutparser + Detectron2** — document layout detection
- **PyTorch** — inference backend (CUDA GPU acceleration)
- **Docker** — containerised deployment
- **Pydantic v2** — request/response validation

---

## Project Structure

```
doc-layout-api/
├── app/
│   ├── main.py        # FastAPI app factory + lifespan (startup/shutdown)
│   ├── routes.py      # HTTP endpoints
│   ├── inference.py   # Pipeline: decode → preprocess → infer → postprocess
│   ├── model.py       # Singleton model loader
│   ├── schemas.py     # Pydantic request/response schemas
│   └── config.py      # Settings via environment variables
├── tests/
│   ├── test_inference.py   # Unit tests (no model needed)
│   └── test_routes.py      # Integration tests (mocked model)
├── models/            # Place model weights here (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start (Local)

### 1. Clone and set up environment
```bash
git clone <your-repo>
cd doc-layout-api
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `install.sh` is used instead of the raw requirements file, it will try the
prebuilt Detectron2 wheel first and fall back to a source build when no wheel
matches your Python version. If both paths fail, use Python 3.10 with a CUDA
toolkit-enabled environment.

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set DEVICE=cpu if no GPU
```

### 3. Run the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open Swagger UI
Navigate to: http://localhost:8000/docs

---

## Quick Start (Docker)

### With GPU (NVIDIA):
```bash
docker compose up --build
```

### Without GPU (CPU only):
```bash
DEVICE=cpu docker compose up --build
```

---

## API Endpoints

### `POST /api/v1/predict`
Upload a document image, receive layout predictions.

**Request:** `multipart/form-data`
- `file`: image file (PNG, JPEG, TIFF, BMP, WEBP) — max 10MB

**Response:**
```json
{
  "num_blocks": 3,
  "image_size": {"width": 2480, "height": 3508},
  "inference_time_ms": 87.4,
  "blocks": [
    {
      "label": "Title",
      "score": 0.9821,
      "bbox": {"x1": 120.0, "y1": 98.0, "x2": 1400.0, "y2": 160.0}
    },
    {
      "label": "Text",
      "score": 0.9543,
      "bbox": {"x1": 120.0, "y1": 200.0, "x2": 1400.0, "y2": 800.0}
    }
  ]
}
```

### `GET /api/v1/health`
Health check endpoint for Docker/load balancer monitoring.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run without a GPU or real model (mocked).

---

## Configuration

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODEL_SCORE_THRESHOLD` | `0.5` | Minimum confidence to return a block |
| `MAX_IMAGE_SIZE` | `1024` | Resize longest edge to this (px) |
| `MAX_FILE_SIZE_MB` | `10` | Reject uploads larger than this |
| `DEBUG` | `false` | Enable debug logging |

---

## Performance Notes

- Model loads **once at startup** — not per request
- `torch.no_grad()` applied during inference — ~50% less GPU memory
- Images are resized to `MAX_IMAGE_SIZE` before inference
- Typical GPU inference time: **50–150ms** per image
