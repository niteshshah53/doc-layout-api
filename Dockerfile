FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Use the official CPU wheels for a reproducible CPU container. GPU execution
# remains an optional host-native deployment and is not claimed as verified.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0 torchvision==0.18.0 && \
    pip install -r requirements.txt

COPY app/ ./app/

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --home-dir /home/appuser appuser && \
    chown -R appuser:appuser /app /home/appuser
USER appuser

ENV DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_FILE_SIZE_MB=10

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
