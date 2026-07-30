"""OCR gateway: fast lane (PP-OCRv6 small, ONNX) and doc lane (PaddleOCR-VL, vLLM).

Routing is the caller's choice, not a guess:
  POST /ocr    -> fast lane. Single image, plain text out.
  POST /parse  -> doc lane. PDF / structured document, Markdown+JSON out.

NOTE: not smoke-tested end to end. The /parse proxy assumes the doc-lane server
takes multipart 'file' and returns JSON — verify against the running container.
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent / "fast_lane"))
from ocr_engine import DEVICE, run_ocr, warmup  # noqa: E402

DOC_LANE_URL = os.getenv("DOC_LANE_URL", "http://localhost:8118")
DOC_LANE_TIMEOUT = float(os.getenv("DOC_LANE_TIMEOUT", "300"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download + session init here, so the first real request does not pay for it.
    warmup()
    yield


app = FastAPI(title="ocr-service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "fast_lane_device": DEVICE}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """Fast lane: one image, plain text extraction."""
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        return {"device": DEVICE, "pages": run_ocr(str(tmp_path))}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    """Doc lane: structured document parsing, proxied to the vLLM container."""
    async with httpx.AsyncClient(timeout=DOC_LANE_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{DOC_LANE_URL}/parse",
                files={"file": (file.filename, await file.read())},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"doc lane unreachable: {exc}") from exc
    return resp.json()
