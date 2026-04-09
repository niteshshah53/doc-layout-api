# ─────────────────────────────────────────────────────────────────────────────
# Document Layout Detection API — Production Dockerfile
#
# MULTI-STAGE BUILD EXPLAINED:
#   Stage 1 (builder): installs all build tools and compiles dependencies.
#   Stage 2 (runtime): copies only the compiled packages, not build tools.
#   Result: ~40-60% smaller final image — important for faster deploys.
#
# WHY nvidia/cuda BASE IMAGE:
#   CUDA libraries must be present in the container for torch.cuda to work.
#   The "runtime" variant has libcudnn but not the compiler tools — correct
#   for inference (we don't train inside the container).
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Prevents interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    ninja-build \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment (isolates deps from system Python)
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first (old pip often resolves deps incorrectly)
RUN pip install --upgrade pip

# Copy and install dependencies
# COPY requirements.txt FIRST (before app code) so Docker caches this layer.
# If requirements.txt hasn't changed, Docker reuses the cache even if app code changed.
COPY requirements.txt .

# Step 1: PyTorch first — detectron2's setup.py imports torch at build time
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: detectron2 wheel first, then source build if no wheel exists
RUN bash -lc 'set -euo pipefail; \
    wheel_url="https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.3/index.html"; \
    if pip install --no-cache-dir detectron2 -f "$wheel_url"; then \
      exit 0; \
    fi; \
    echo "Falling back to a source build of Detectron2"; \
        FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"'

# Step 3: Everything else
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime-only system packages (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Application setup ─────────────────────────────────────────────────────────

WORKDIR /app

# Create a non-root user (security best practice — don't run as root in prod)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home appuser

# Copy application code
COPY app/ ./app/

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# ── Environment variables ─────────────────────────────────────────────────────
# These are DEFAULTS — override at runtime via docker run -e or docker-compose
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cuda \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_FILE_SIZE_MB=10

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will call this every 30s. If it fails 3 times, container is "unhealthy".
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --workers 1  : One uvicorn worker. For multiple GPUs, use multiple containers
#                (horizontal scaling) rather than multiple workers in one container.
# --host 0.0.0.0: Accept connections from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
