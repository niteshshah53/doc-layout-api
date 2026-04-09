# Document Layout Detection API

A production-ready REST API that detects objects and regions in images using a Faster R-CNN model trained on the COCO dataset.

**What does it do?**  
Upload an image and it will automatically detect objects, identify their locations with bounding boxes, and provide confidence scores. Works with general images containing people, animals, vehicles, furniture, and other common objects. Perfect for image analysis, object detection pipelines, and automated content understanding.

---

## Prerequisites

Before getting started, make sure you have:
- **Python 3.9+** installed ([download here](https://www.python.org/downloads/))
- **Git** installed ([download here](https://git-scm.com/))
- (Optional) **Docker** for containerized deployment ([download here](https://www.docker.com/))
- (Optional) **NVIDIA GPU** with CUDA toolkit for faster inference (CPU-only is also supported)

**Not sure what these are?** Don't worry! The quickstart guide below will walk you through everything step-by-step.

---

## Tech Stack
- **FastAPI** — Web framework for building the API
- **Detectron2 + layoutparser** — COCO-Detection Faster R-CNN model for general object detection
- **PyTorch 2.5.1** — Deep learning library
- **Docker** — Package the app to run anywhere
- **Pydantic** — Validation for inputs/outputs

---

## Project Structure

```
doc-layout-api/
├── app/                       # Main API application package
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI app factory + lifespan (startup/shutdown)
│   ├── routes.py             # HTTP endpoints (/api/v1/predict, /api/v1/health)
│   ├── inference.py          # Pipeline: decode → preprocess → infer → postprocess
│   ├── model.py              # Singleton model loader (loads on startup)
│   ├── schemas.py            # Pydantic request/response schemas
│   └── config.py             # Settings via environment variables
├── tests/                     # Test suite
│   ├── conftest.py           # Pytest configuration and fixtures
│   ├── test_inference.py      # Unit tests (no model needed)
│   └── test_routes.py        # Integration tests (mocked model)
├── .env.example              # Example environment variables (copy to .env)
├── .gitignore                # Git ignore file
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── install.sh                # Installation script (handles detectron2 setup)
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── test_api.py               # Manual API testing file
└── README.md                 # This file
```

---

## Installation & Quick Start (For Beginners)

### Step 1: Clone the Repository
```bash
# Copy the repository to your computer
git clone https://github.com/niteshshah53/doc-layout-api.git
cd doc-layout-api
```

### Step 2: Create a Virtual Environment
A virtual environment isolates this project's dependencies from others on your computer.

```bash
# Create the environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# You should see (.venv) at the start of your terminal prompt
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**⚠️ Important:** This project requires PyTorch 2.5.1+ and torchvision 0.20.1+ for compatibility. If you encounter errors:
```bash
bash install.sh
```
This script automatically installs compatible versions and configures Detectron2.

### Step 4: Setup Configuration
```bash
# Copy the example environment file
cp .env.example .env

# Optional: Edit .env if you want to use CPU instead of GPU
# Change: DEVICE=cuda to DEVICE=cpu
```

### Step 5: Run the API
```bash
# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Press CTRL+C to quit
```

### Step 6: Test the API
Open your browser and go to: **http://localhost:8000/docs**

You'll see an interactive interface where you can test the API by uploading a document image!

---

## Quick Start (Docker - Simpler Alternative)

If you prefer not to install Python dependencies locally, use Docker:

```bash
# Build and run with Docker (uses GPU if available)
docker compose up --build
```

**Without GPU (CPU only):**
```bash
DEVICE=cpu docker compose up --build
```

The API will be available at: http://localhost:8000/docs

---

## Troubleshooting for Beginners

**Q: I get "command not found: python"**  
A: Python isn't installed or not in your PATH. [Install Python](https://www.python.org/downloads/) and retry.

**Q: Virtual environment activation didn't work**  
A: Make sure you're in the project directory (`doc-layout-api/`) and try again.

**Q: "pip install" fails**  
A: Try upgrading pip first:
```bash
python -m pip install --upgrade pip
```

**Q: "detectron2" installation fails**  
A: Run `bash install.sh` instead — it handles special cases.

**Q: GPU not detected**  
A: Use CPU instead by setting `DEVICE=cpu` in `.env`

**Q: Still stuck?**  
A: Check you're using Python 3.9+ (`python --version`)

---

## How to Use the API

### Upload a Document Image

**Easiest Way: Use the Web UI**
1. Go to http://localhost:8000/docs
2. Click "Try it out"
3. Upload a document image (PNG, JPEG, TIFF, etc.)
4. Click "Execute"

**Using cURL (Command Line):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "accept: application/json" \
  -F "file=@path/to/your/document.png"
```

**Using Python:**
```python
import requests

files = {"file": open("document.png", "rb")}
response = requests.post("http://localhost:8000/api/v1/predict", files=files)
print(response.json())
```

### Example Response
```json
{
  "num_blocks": 2,
  "image_size": {"width": 1344, "height": 2016},
  "inference_time_ms": 418.39,
  "blocks": [
    {
      "label": "Object",
      "score": 0.9542,
      "bbox": {"x1": 120.0, "y1": 98.0, "x2": 1240.0, "y2": 800.0}
    },
    {
      "label": "Object",
      "score": 0.8913,
      "bbox": {"x1": 150.0, "y1": 850.0, "x2": 1200.0, "y2": 1500.0}
    }
  ]
}
```

---

## Running Tests (Optional)

To verify everything works correctly:
```bash
pytest
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
  "num_blocks": 2,
  "image_size": {"width": 1344, "height": 2016},
  "inference_time_ms": 418.39,
  "blocks": [
    {
      "label": "Object",
      "score": 0.9542,
      "bbox": {"x1": 120.0, "y1": 98.0, "x2": 1240.0, "y2": 800.0}
    },
    {
      "label": "Object",
      "score": 0.8913,
      "bbox": {"x1": 150.0, "y1": 850.0, "x2": 1200.0, "y2": 1500.0}
    }
  ]
}
```

### `GET /api/v1/health`
Health check endpoint for Docker/load balancer monitoring. Returns model status including which variant is loaded (primary/fallback).

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_info": {
    "type": "primary",
    "error": null
  },
  "version": "1.0.0"
}
```

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

## Model & Performance

### Current Model
- **Type:** Faster R-CNN with ResNet-50 backbone trained on COCO dataset
- **Detects:** General objects (people, vehicles, animals, furniture, etc.)
- **Does NOT detect:** Document-specific elements (titles, paragraphs, tables)

### Fallback Mechanism
If the primary model fails to load, the API automatically cascades through:
1. **Primary:** COCO-Detection Faster R-CNN (GPU)
2. **Fallback-1:** Mask R-CNN variant (GPU)
3. **Fallback-2:** Same model on CPU (slower but always available)

The `/api/v1/health` endpoint shows which model is loaded.

### Performance Notes
- Model loads **once at startup** — not per request (typically 1-2 seconds)
- `torch.no_grad()` applied during inference — ~50% less GPU memory
- Images are resized to `MAX_IMAGE_SIZE` before inference
- Typical GPU inference time: **50–500ms** per image (depends on image complexity)
- GPU memory used: ~0.17GB with default settings
