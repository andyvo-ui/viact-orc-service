"""OCR gateway: fast lane (PP-OCRv6 small, ONNX) and doc lane (PaddleOCR-VL, vLLM).

Routing is the caller's choice, not a guess. The split is FLAT TEXT vs PRESERVED
STRUCTURE — not image vs PDF:

  POST /ocr    -> fast lane. Image OR short scanned PDF. text + boxes + scores,
                  one entry per page. No reading order, no tables, no markdown.
                  CPU, no GPU contention. Bounded: see fast_lane/input_limits.py.
  POST /parse  -> doc lane. Anything where the STRUCTURE matters (tables, a long
                  contract). Markdown+JSON out. GPU.

A scanned PDF is a photo in a PDF wrapper, so it belongs on the fast lane. A
born-digital contract with tables belongs on the doc lane — sending it to /ocr
returns 200 and silently discards the structure you wanted. Page count is the cheap
proxy for that line, which is why /ocr caps it rather than refusing PDFs outright.

NOTE: not smoke-tested end to end. The /parse proxy assumes the doc-lane server
takes multipart 'file' and returns JSON — verify against the running container.
/ocr's response shape IS now verified against paddleocr 3.7.0 by source reading;
see run_ocr in fast_lane/ocr_engine.py.
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent / "fast_lane"))
# Imported as a MODULE, not by value: the limits are module globals read at call
# time, so `from input_limits import MAX_PAGES` would snapshot them at import and
# /health would report numbers the request path no longer uses.
import input_limits  # noqa: E402
from input_limits import (  # noqa: E402
    TooManyPages,
    UploadTooLarge,
    enforce_page_limit,
    stream_upload_to,
)
from ocr_engine import (  # noqa: E402
    DETECT_ORIENTATION_DEFAULT,
    DEVICE,
    EngineRejectedInput,
    run_ocr,
    warmup,
)

DOC_LANE_URL = os.getenv("DOC_LANE_URL", "http://localhost:8118")
DOC_LANE_TIMEOUT = float(os.getenv("DOC_LANE_TIMEOUT", "300"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download + session init here, so the first real request does not pay for it.
    warmup()
    yield


app = FastAPI(title="ocr-service", lifespan=lifespan)


class Orientation(str, Enum):
    """Which fast-lane pipeline handles the request.

    Two pipelines exist because the real input is ~50/50 site photos and flat scans,
    and one global setting is wrong half the time. See fast_lane/ocr_engine.py for
    why this cannot be a per-predict flag.
    """

    auto = "auto"       # detect rotated textlines — signage, site photos
    upright = "upright"  # assume upright — flat scans, slightly faster


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fast_lane_device": DEVICE,
        "limits": {
            "max_pages": input_limits.MAX_PAGES,
            "max_upload_bytes": input_limits.MAX_UPLOAD_BYTES,
        },
        "default_orientation": "auto" if DETECT_ORIENTATION_DEFAULT else "upright",
    }


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    orientation: Orientation | None = Query(
        None,
        description="auto = handle rotated textlines (site photos). "
        "upright = assume upright, slightly faster (flat scans). "
        "Defaults to OCR_DETECT_ORIENTATION.",
    ),
):
    """Fast lane: flat text from one image, or from each page of a short PDF."""
    suffix = Path(file.filename or "upload").suffix
    # mkstemp rather than NamedTemporaryFile: the body is streamed in below with a
    # size limit, so nothing is written at creation time.
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    tmp_path = Path(name)

    detect = None if orientation is None else (orientation is Orientation.auto)

    try:
        try:
            await stream_upload_to(file, tmp_path)
        except UploadTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {exc.limit_bytes} bytes "
                f"(OCR_MAX_UPLOAD_BYTES)",
            ) from exc

        try:
            enforce_page_limit(tmp_path)
        except TooManyPages as exc:
            raise HTTPException(
                status_code=413,
                detail=f"{exc.pages}-page PDF exceeds the {exc.limit}-page fast-lane "
                f"limit (OCR_MAX_PAGES). Long documents belong on POST /parse, which "
                f"also preserves tables and reading order.",
            ) from exc

        try:
            pages = run_ocr(str(tmp_path), detect_orientation=detect)
        except EngineRejectedInput as exc:
            # The engine refused the request rather than the file — a 502-shaped
            # problem on our side, not the caller's. Never report it as a 4xx.
            raise HTTPException(
                status_code=500, detail=f"OCR engine rejected the request: {exc}"
            ) from exc

        return {"device": DEVICE, "pages": pages}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    """Doc lane: structured document parsing, proxied to the vLLM container.

    Error mapping is deliberately three-way. The previous version had
    raise_for_status() inside the `except httpx.HTTPError` block, and
    HTTPStatusError subclasses HTTPError — so a doc-lane 400 ("your file is
    garbage") reached the caller as 502 ("our GPU box is down"). Callers cannot
    tell those apart, so they retry a request that can never succeed.
    """
    async with httpx.AsyncClient(timeout=DOC_LANE_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{DOC_LANE_URL}/parse",
                files={"file": (file.filename or "upload", await file.read())},
            )
        except httpx.HTTPError as exc:
            # Transport level only: DNS, connection refused, timeout. This is the
            # sole case that is genuinely "the doc lane is unreachable".
            raise HTTPException(status_code=502, detail=f"doc lane unreachable: {exc}") from exc

    # Caller's fault — pass the status through so a retry loop can give up.
    if 400 <= resp.status_code < 500:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"doc lane rejected the request: {resp.text[:500]}",
        )
    # Doc lane's fault — that IS an infrastructure error from the caller's side.
    if resp.status_code >= 500:
        raise HTTPException(
            status_code=502,
            detail=f"doc lane failed with {resp.status_code}: {resp.text[:500]}",
        )

    try:
        return resp.json()
    except ValueError as exc:
        # SETUP.md #2 is still open: "JSON out" is an assumption about an endpoint
        # nobody here has called. Surface it as a bad upstream, not a gateway crash.
        raise HTTPException(
            status_code=502, detail=f"doc lane returned non-JSON: {resp.text[:200]}"
        ) from exc
