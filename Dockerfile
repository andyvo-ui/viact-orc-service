# Gateway + fast lane in one image — the gateway imports fast_lane/ocr_engine.py
# in-process, so they cannot be split into separate containers.
#
# CPU build. For the GPU fast lane see SETUP.md ("Fast lane on GPU") — that path
# needs a CUDA base image and is unverified on Blackwell/sm_120.
FROM python:3.11-slim

# opencv comes in via paddleocr and needs these shared libs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY fast_lane/requirements.txt fast_lane/requirements.txt
COPY gateway/requirements.txt gateway/requirements.txt
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir \
        -r fast_lane/requirements.txt \
        -r gateway/requirements.txt

COPY fast_lane/ fast_lane/
COPY gateway/ gateway/

# Bake the PP-OCRv6_small ONNX weights (~31MB) into the image: the container then
# starts with no network, and the first request pays no download. A build failure
# here means the weights could not be fetched — which is what you want to find out
# at build time, not in production.
RUN python fast_lane/prefetch_models.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "gateway", "--host", "0.0.0.0", "--port", "8000"]
