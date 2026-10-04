# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TARGETARCH

RUN --mount=type=cache,id=typingo-kokoro-apt-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=typingo-kokoro-apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       espeak-ng \
       ffmpeg \
       libsndfile1 \
       curl

WORKDIR /app

# Install CPU-only PyTorch first so Kokoro does not pull CUDA/NVIDIA wheels
# from the default PyPI dependency resolution path.
RUN --mount=type=cache,id=typingo-kokoro-pip-${TARGETARCH},target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX_URL} torch

COPY requirements.txt .
COPY scripts/install_dependencies.py /app/install_dependencies.py
RUN --mount=type=cache,id=typingo-kokoro-pip-${TARGETARCH},target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=data/wheels/common,target=/wheels,readonly \
    python /app/install_dependencies.py --wheelhouse /wheels --requirements /app/requirements.txt

COPY app ./app
COPY web ./web

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
