# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SYNORA_EXTRAS=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SYNORA_HOME=/data/synora

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md ./
COPY synora ./synora

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision torchaudio && \
    if [ -n "${SYNORA_EXTRAS}" ]; then \
        python -m pip install --editable ".[${SYNORA_EXTRAS}]"; \
    else \
        python -m pip install --editable .; \
    fi && \
    synora version && \
    mkdir -p "${SYNORA_HOME}"

VOLUME ["/data/synora"]

ENTRYPOINT ["synora"]
CMD ["--help"]
