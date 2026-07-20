# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

# libgomp1: OpenMP runtime needed by torch / BGE-M3. curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first so FlagEmbedding does NOT pull the multi-GB CUDA build.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1" && \
    pip install -r requirements.txt

# App code + the prebuilt index (data/index, data/processed). data/raw is .dockerignore'd.
COPY . .

RUN mkdir -p /models
EXPOSE 8000

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Single worker keeps memory bounded and the idle-unload singleton coherent.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
