# Setup — Docker

**Version numbering.** Two unrelated schemes, easy to confuse:

- `paddleocr` **3.7.0** — the Python *library* version. PP-OCRv6 first shipped here
  (2026-06-11); 3.5/3.6 predate it and cannot resolve `PP-OCRv6_*` model names.
- **PP-OCRv6**, **PaddleOCR-VL-1.6** — *model* versions, downloaded by the library.

Two containers:

| Service | Port | Endpoint | Model | Runs on | Size |
|---|---|---|---|---|---|
| `gateway` | 8000 | `POST /ocr` | PP-OCRv6_small det+rec (ONNX), ×2 orientation pipelines | CPU | ~31 MB + ~7 MB orientation, baked in |
| `doc-lane` | 8118 | `POST /parse` | PaddleOCR-VL-1.6-0.9B | GPU (vLLM) | 5.8 GB pull, ~12–15 GB on disk |

The fast lane is **not** a separate container — it is a Python library imported
in-process by the gateway. Only the doc lane is a standalone server.

---

## 0. Prerequisites

```bash
docker --version && docker compose version
nvidia-smi                              # doc lane only
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

That last command verifies the NVIDIA Container Toolkit is wired up. If it fails,
the doc lane cannot see the GPU no matter what compose says.

**What the host CUDA version does and does not affect.** The number in
`nvidia-smi`'s header is the highest CUDA version the *driver* supports, and that
is the only thing containers care about — each image ships its own CUDA runtime.
So on a CUDA 13.2 host:

| component | affected by host CUDA? |
|---|---|
| `gateway` (fast lane) | **No.** `python:3.11-slim`, `onnxruntime` CPU build. No CUDA at all. |
| `doc-lane` (vLLM) | Only via the driver. The image carries its own runtime; a newer driver runs it fine. |
| `check_device.py` / GPU fast lane | **Yes** — see below. |

The one real trap: official `onnxruntime-gpu` wheels are built against **CUDA 12**,
not 13. On a host with only CUDA 13 runtime libraries, `CUDAExecutionProvider` will
fail to load its dependencies. That is not silent — paddlex validates the provider
list and raises, and `check_device.py` reports it — but it does mean "install
onnxruntime-gpu" is not a one-liner on a CUDA-13-only box. Keep `OCR_DEVICE=cpu`,
which is the default and the recorded decision (DECISIONS.md #2) anyway.

## 1. Run everything

```bash
docker compose up -d --build
docker compose ps
```

`--build` builds the gateway image. That build downloads the PP-OCRv6 weights and
bakes them in, so it needs network access — but only once, at build time.

Watch the doc lane come up (vLLM takes a while to load the model):

```bash
docker compose logs -f doc-lane
```

## 2. Verify

```bash
curl http://localhost:8000/health          # device + the fast-lane limits + default orientation
curl -F "file=@sample.jpg" http://localhost:8000/ocr                      # fast lane
curl -F "file=@photo.jpg" "http://localhost:8000/ocr?orientation=auto"    # site photo, angled text
curl -F "file=@scan.pdf"  "http://localhost:8000/ocr?orientation=upright" # flat scan, faster
curl -F "file=@sample.pdf" http://localhost:8000/parse                    # doc lane
curl http://localhost:8118/health                                         # doc lane direct
```

Then look at real output with your own documents:

```bash
python scripts/report_live.py /path/to/real/docs/ --orientation auto
```

## Choosing a lane, and the fast-lane bounds

The split is **flat text vs preserved structure**, not image vs PDF:

| input | endpoint |
|---|---|
| photo of signage / nameplate / equipment ID | `/ocr?orientation=auto` |
| short **scanned** PDF, text only | `/ocr` (a scan is a photo in a PDF wrapper) |
| PDF with **tables**, long contract | `/parse` — `/ocr` returns 200 and silently flattens it |

`/ocr` bounds, both env-tunable and both reported on `/health`:

| env | default | over the limit |
|---|---|---|
| `OCR_MAX_PAGES` | `10` | `413`, message names `/parse` |
| `OCR_MAX_UPLOAD_BYTES` | `52428800` (50 MB) | `413`, aborts mid-stream |
| `OCR_DETECT_ORIENTATION` | `1` (= `auto`) | — |

The page cap is set by the blocking issue, not by the model: `run_ocr` occupies the
event loop, so a long PDF stops `/health` answering and compose eventually restarts a
working container. Raise it after benchmarking with `test_c1` / `test_c2`.

Check the doc lane is not eating the whole GPU:

```bash
nvidia-smi
```

## Fast lane only (skip the GPU container)

The gateway has no `depends_on` for the doc lane, so it runs alone:

```bash
docker compose up -d --build gateway
```

`/ocr` works; `/parse` returns 502. Saves ~15 GB of disk and all the VRAM.

## Other services calling this

From the host or another compose project on the same network:

```
POST http://<server>:8000/ocr      multipart form field "file"
POST http://<server>:8000/parse    multipart form field "file"
```

To put another container on the same network, add to its compose file:

```yaml
networks:
  default:
    external: true
    name: ocr-service_default
```

then call `http://gateway:8000/ocr`.

## Rebuild after code changes

```bash
docker compose up -d --build gateway
```

Changing only `gateway/` or `fast_lane/` source reuses the cached pip layer — the
rebuild is fast. Changing a `requirements.txt` reinstalls everything.

---

## Fast lane on GPU

Not the default, and unverified on this hardware. To try it:

1. `Dockerfile`: swap the base image to a CUDA one (e.g.
   `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04` plus a python install).
2. `fast_lane/requirements.txt`: `onnxruntime` → `onnxruntime-gpu`.
3. `docker-compose.yml`: set `OCR_DEVICE=gpu` and give the gateway a
   `deploy.resources.reservations.devices` block like the doc lane has.
4. Prove it actually works:
   ```bash
   docker compose exec gateway python fast_lane/check_device.py
   ```
   It must print `OK: GPU is live`. ONNX Runtime does **not** raise on a
   Blackwell/sm_120 kernel mismatch — it silently runs on CPU, so `OCR_DEVICE=gpu`
   alone proves nothing.

Read the note at the top of `fast_lane/ocr_engine.py` on why CPU is the default:
sharing the GPU with the LLM makes fast-lane p95 latency unpredictable.

## Air-gapped server

```bash
# on a connected machine
docker compose build gateway
docker save ocr-service-gateway:latest | gzip > ocr-gateway.tar.gz
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu
docker save ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  | gzip > ocr-doc-lane.tar.gz

# on the server
gunzip -c ocr-gateway.tar.gz | docker load
gunzip -c ocr-doc-lane.tar.gz | docker load
docker compose up -d              # no --build, images already present
```

The gateway image carries its weights, so it needs no network at run time.

## Running without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r fast_lane/requirements.txt -r gateway/requirements.txt
python fast_lane/prefetch_models.py
DOC_LANE_URL=http://localhost:8118 uvicorn main:app --app-dir gateway --host 0.0.0.0 --port 8000
```

One venv for both — the gateway imports `fast_lane/ocr_engine.py` directly.

---

## Verified by reading the shipped wheels

Settled against `paddleocr==3.7.0` + `paddlex==3.7.0` source, so these no longer
need a running container:

- **`ocr_engine.py` constructor kwargs are correct.** `engine` / `engine_config`
  reach `parse_common_args` through `PaddleOCR.__init__(**kwargs)`; `"onnxruntime"`
  is in `SUPPORTED_INFERENCE_ENGINE_LIST`; `providers` is a declared field on
  `ONNXRuntimeRunnerConfig`, which is `extra="forbid"` — so a misspelled key would
  raise rather than be ignored. `warmup()` will not die on kwargs.
- **`PP-OCRv6_small_det` / `_rec` exist and have official ONNX builds** (both are in
  `ONNX_SUPPORTED_MODELS`), so the build-time bake resolves `..._onnx` and decisions
  #1 / #4 / #6 hold.
- **PDF input to `/ocr` works.** `paddleocr` pulls `paddlex[ocr-core]`, which
  includes `pypdfium2>=4`. `ocr-core` also pulls `opencv-contrib-python`, so the
  `libgl1` / `libglib2.0-0` apt packages in the Dockerfile are genuinely needed.
- **A missing CUDA EP is loud, not silent.** `_validate_providers` raises if a
  requested provider is absent. The silent-CPU-fallback risk applies only with
  `onnxruntime-gpu` installed and an sm_120 kernel mismatch — that is the case
  `check_device.py` exists for.

## Not verified yet — check these before calling it done

1. **Doc-lane request/response shape.** The gateway's `/parse` assumes multipart
   `file` in, JSON out. Confirm against the real endpoint. A non-JSON 200 now
   surfaces as a 502 naming the problem rather than a gateway traceback, but the
   assumed request shape is still an assumption.
2. **`/ocr` under concurrency.** `async def ocr` calls the fully blocking
   `run_ocr`, so concurrent requests serialise on the event loop and `/health` can
   miss the 5s compose healthcheck under load. `tests/test_concurrency.py::test_c1`
   and `::test_c2` measure it. Deliberately not "fixed" — the remedy is a design
   choice (`run_in_threadpool`, a plain `def` endpoint, or accepting serialisation).
3. **`GPU_MEMORY_UTILIZATION` in docker-compose.yml.** vLLM defaults to 90% of
   *total* VRAM, which would starve the LLM on the same GPU. Verify that
   `paddleocr genai_server` actually forwards this env var — if not, pass it as a
   CLI flag in the compose `command:`. Check with `nvidia-smi` after startup.
4. **Accuracy on real HK documents.** Benchmark 30–50 real Traditional Chinese +
   English documents through the fast lane. Published numbers for the small tier
   are TC 77.0% / printed EN 93.3%; if that is not enough, switch `DET_MODEL` /
   `REC_MODEL` in `ocr_engine.py` to the `medium` tier (TC 78.6% / EN 94.1%) —
   that is the only change needed.
5. **Text orientation.** `use_textline_orientation` is off. Fine for scanned
   documents shot straight on; wrong for site photos where signage is at an angle.
