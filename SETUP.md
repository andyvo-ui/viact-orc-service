# Setup — Linux server

**Version numbering.** Two unrelated schemes, easy to confuse:

- `paddleocr` **3.7.0** — the Python *library* version. PP-OCRv6 first shipped here
  (2026-06-11); 3.5/3.6 predate it and cannot resolve `PP-OCRv6_*` model names.
- **PP-OCRv6**, **PaddleOCR-VL-1.6** — *model* versions, downloaded by the library.

Two lanes, installed independently:

| Lane | Endpoint | Model | Runs on | Weights |
|---|---|---|---|---|
| fast | `POST /ocr` | PP-OCRv6_small (det+rec, ONNX) | CPU by default | ~31 MB, auto-downloaded |
| doc | `POST /parse` | PaddleOCR-VL-1.6-0.9B | GPU via vLLM | ~2 GB, inside docker image |

Nothing needs to be downloaded by hand. Everything below is a command to run on
the server.

---

## 0. Prerequisites

```bash
python3 --version          # need 3.9+
nvidia-smi                 # confirm GPU + driver (doc lane only)
docker --version && docker compose version
```

## 1. Fast lane

```bash
cd ocr-service/fast_lane
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` installs the **CPU** ORT build by default. For GPU, edit it to
use `onnxruntime-gpu` instead — never both, they conflict.

Pull the weights now rather than on the first request:

```bash
python prefetch_models.py     # caches PP-OCRv6_small det+rec ONNX under ~/.paddlex/
```

Check what landed:

```bash
du -sh ~/.paddlex/
find ~/.paddlex/ -name "*.onnx"
```

### 1b. Only if you want the fast lane on GPU

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
python check_device.py        # MUST print "OK: GPU is live"
export OCR_DEVICE=gpu
```

If `check_device.py` says ORT fell back to CPU, keep `OCR_DEVICE=cpu`. ORT does
not raise on a Blackwell/sm_120 kernel mismatch — it silently runs on CPU, so
setting the env var alone proves nothing.

## 2. Doc lane

```bash
cd ocr-service
docker compose pull doc-lane      # ~15 GB
docker compose up -d doc-lane
docker compose logs -f doc-lane   # wait for the vLLM server to report ready
curl http://localhost:8118/health
```

If the Baidu registry is slow from your network, pull it on a machine that can
reach it and move the image over:

```bash
# on a machine with good connectivity
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu
docker save <image> | gzip > paddleocr-vl.tar.gz
# on the server
gunzip -c paddleocr-vl.tar.gz | docker load
```

## 3. Gateway

```bash
cd ocr-service/gateway
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The gateway imports `fast_lane/ocr_engine.py` directly, so it needs the fast-lane
deps too. Simplest is one shared venv for both — install both requirements files
into it and skip the second venv.

## 4. Verify

```bash
curl http://localhost:8000/health
curl -F "file=@sample.jpg" http://localhost:8000/ocr
curl -F "file=@sample.pdf" http://localhost:8000/parse
```

---

## Air-gapped server

If the server cannot reach the internet:

```bash
# on a connected machine
pip download -r fast_lane/requirements.txt -r gateway/requirements.txt -d wheels/
python fast_lane/prefetch_models.py
tar czf paddlex-cache.tar.gz -C ~ .paddlex

# on the server
pip install --no-index --find-links wheels/ -r fast_lane/requirements.txt -r gateway/requirements.txt
tar xzf paddlex-cache.tar.gz -C ~
```

Plus the `docker save` / `docker load` step above for the doc lane.

---

## Not verified yet — check these before calling it done

1. **`ocr_engine.py` constructor kwargs** against the installed paddleocr>=3.7 API
   (`engine` / `engine_config` were introduced in 3.5; confirm the exact names).
2. **Doc-lane request/response shape.** The gateway's `/parse` assumes multipart
   `file` in, JSON out. Confirm against the real endpoint.
3. **`GPU_MEMORY_UTILIZATION` in docker-compose.yml.** vLLM defaults to 90% of
   *total* VRAM, which would starve the LLM on the same GPU. Verify that
   `paddleocr genai_server` actually forwards this env var to vLLM — if not, pass
   it as a CLI flag in the compose `command:` instead. Check with `nvidia-smi`
   after the container is up.
4. **Accuracy on real HK documents.** Benchmark 30–50 real Traditional Chinese +
   English documents through the fast lane. Published numbers for the small tier
   are TC 77.0% / printed EN 93.3%; if that is not enough, switch `DET_MODEL` /
   `REC_MODEL` in `ocr_engine.py` to the `medium` tier (TC 78.6% / EN 94.1%) —
   that is the only change needed.
