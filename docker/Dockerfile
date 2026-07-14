# ---------------------------------------------------------------------------
# sem_search — семантическая поисковая система (ВКР)
# Dockerfile: production-ready FastAPI service with sentence-transformers.
#
# Build:
#   docker build -t sem_search -f docker/Dockerfile .
#
# Run standalone (embedded Qdrant, no separate qdrant service):
#   docker run -p 8000:8000 -v ./data:/app/data sem_search
#
# Or use docker-compose.yml (app + qdrant service, remote Qdrant mode):
#   docker compose -f docker/docker-compose.yml up
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Avoid .pyc writes and stdout/stderr buffering in Docker logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face cache directory — mount as a volume so models survive rebuilds.
ENV HF_HOME=/app/.hf_cache

WORKDIR /app

# --- system dependencies ---------------------------------------------------
# libgomp1, libgcc-s1: required by sentence-transformers / torch runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---------------------------------------------------
COPY requirements.txt .

# Install torch CPU-only first to avoid pulling the full CUDA bundle (~2.5 GB).
# --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- application source ----------------------------------------------------
COPY app/ ./app/
COPY config/ ./config/
# Sample corpus (optional — the image works without it; mount your own data).
COPY data/sample_corpus/ ./data/sample_corpus/

# Create directories that the app writes to at runtime (so they exist even
# if no volume is mounted).
RUN mkdir -p /app/data/logs /app/data/lexical /app/data/qdrant

EXPOSE 8000

# The app loads config/config.yaml by default (embedded Qdrant — fine for
# standalone `docker run`). In docker-compose, vector_store Qdrant settings
# are overridden via SEM_SEARCH_VECTOR_STORE__QDRANT__MODE=remote /
# SEM_SEARCH_VECTOR_STORE__QDRANT__URL=http://qdrant:6333 environment vars.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
