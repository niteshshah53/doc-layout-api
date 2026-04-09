# Document Layout Detection API

A simple REST API that analyzes document images and automatically detects different parts like titles, text, tables, and figures.

**What does it do?**  
Upload an image of a document (PDF page, scanned document, etc.), and it will identify and locate all the content blocks within it. Perfect for document processing, OCR pipelines, or automated document understanding.

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
- **Detectron2 + layoutparser** — AI model that detects document layouts
- **PyTorch** — Deep learning library
- **Docker** — Package the app to run anywhere
- **Pydantic** — Validation for inputs/outputs

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

**⚠️ Tip:** If you see errors about `detectron2`, it's normal — the `install.sh` script handles this:
```bash
bash install.sh
```

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
  "num_blocks": 3,
  "image_size": {"width": 2480, "height": 3508},
  "inference_time_ms": 87.4,
  "blocks": [
    {
      "label": "Title",
      "confidence": 0.95,
      "bbox": [100, 50, 500, 150]
    },
    {
      "label": "Text",
      "confidence": 0.92,
      "bbox": [100, 160, 500, 400]
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
