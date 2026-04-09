FROM docker.1ms.run/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml setup.py README.md ./
COPY ctc_forced_aligner ./ctc_forced_aligner
COPY scripts ./scripts

RUN pip install --upgrade pip setuptools wheel pybind11 \
    && pip install .

ARG ALIGN_MODEL=MahmoudAshraf/mms-300m-1130-forced-aligner
ENV ALIGN_MODEL=${ALIGN_MODEL} \
    ALIGN_DEVICE=cpu \
    ALIGN_COMPUTE_DTYPE=float32 \
    ALIGN_MAX_CONCURRENCY=2 \
    ALIGN_BATCH_SIZE=4 \
    ALIGN_WINDOW_SIZE=30 \
    ALIGN_CONTEXT_SIZE=2

# Pre-download model to avoid cold-start latency in runtime containers.
RUN python scripts/preload_model.py

# Avoid import shadowing: runtime should load the installed package
# (which contains the compiled pybind extension) instead of /app source tree.
RUN rm -rf /app/ctc_forced_aligner /app/scripts

EXPOSE 8000

CMD ["uvicorn", "ctc_forced_aligner.api:app", "--host", "0.0.0.0", "--port", "8000"]
