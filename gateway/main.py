"""OCR gateway: fast lane (PP-OCRv6 small, ONNX) and doc lane (PaddleOCR-VL, Ollama).

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

The doc lane is NOT part of this compose stack — it is an Ollama server already
running on the GPU host, serving PaddleOCR-VL over its OpenAI-compatible API.
Point DOC_LANE_URL at it (default: localhost:11434).

NOTE: /parse is not smoke-tested end to end against that server yet.
/ocr's response shape IS now verified against paddleocr 3.7.0 by source reading;
see run_ocr in fast_lane/ocr_engine.py.
"""

import base64
import mimetypes
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
    UnreadablePDF,
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

DOC_LANE_URL = os.getenv("DOC_LANE_URL", "http://localhost:11434")
DOC_LANE_TIMEOUT = float(os.getenv("DOC_LANE_TIMEOUT", "300"))

# Ollama's library naming (namespace/model:tag), not a vLLM --model path. Used as
# the fallback when the live /v1/models lookup fails; the fetched name wins.
DOC_LANE_MODEL = os.getenv("MODEL", "AuditAid/PaddleOCR-VL-1.6-0.9B:latest")


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
            "parse_max_pages": input_limits.PARSE_MAX_PAGES,
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


DOC_LANE_MAX_TOKENS = 4096


async def _doc_lane_model_name(client: httpx.AsyncClient) -> str:
    """Best-effort, mirroring scripts/test_doclane.sh: ask the doc lane what it's
    actually serving, falling back to DOC_LANE_MODEL on any failure.

    Not worth its own error path — a wrong model name surfaces on the real
    completions call below as an upstream 4xx, which already has a mapping.
    """
    try:
        resp = await client.get(f"{DOC_LANE_URL}/v1/models")
        return resp.json()["data"][0]["id"]
    except Exception:
        return DOC_LANE_MODEL


async def _doc_lane_complete(
    client: httpx.AsyncClient, model: str, image_bytes: bytes, mime: str
) -> str:
    """POST one page to the doc lane's OpenAI-compatible chat-completions endpoint
    and return the extracted markdown.

    Error mapping is deliberately three-way (unchanged from the original /parse
    proxy). The previous version had raise_for_status() inside the
    `except httpx.HTTPError` block, and HTTPStatusError subclasses HTTPError — so a
    doc-lane 400 ("your file is garbage") reached the caller as 502 ("our GPU box is
    down"). Callers cannot tell those apart, so they retry a request that can never
    succeed.
    """
    b64 = base64.b64encode(image_bytes).decode()
    try:
        resp = await client.post(
            f"{DOC_LANE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                            {"type": "text", "text": "OCR:"},
                        ],
                    }
                ],
                "max_tokens": DOC_LANE_MAX_TOKENS,
                "temperature": 0.0,
            },
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
        data = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502, detail=f"doc lane returned non-JSON: {resp.text[:200]}"
        ) from exc

    # Valid JSON, but not the OpenAI chat-completion shape we asked for — a
    # DIFFERENT failure than non-JSON, so it gets its own message.
    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content is not a string")
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"doc lane returned a malformed completion: {resp.text[:200]}",
        ) from exc

    return content


@app.post(
    "/parse",
    deprecated=True,
    summary="NOT READY — do not test this endpoint",
    description=(
        "Structured document parsing. **Not yet verified against the real doc "
        "lane.** This now speaks the doc lane's actual protocol (vLLM's "
        "OpenAI-compatible `/v1/chat/completions`, one image per page), but nobody "
        "has run it against the live container yet — see SETUP.md. Use `POST /ocr` "
        "instead until this notice is removed."
    ),
)
async def parse(file: UploadFile = File(...)):
    """Doc lane: structured document parsing via PaddleOCR-VL, served by Ollama.

    The doc lane is a vision-language model behind an OpenAI-compatible chat API —
    it has no /parse route and does not accept multipart file uploads or raw PDF
    bytes. It takes one image per request (`image_url` content block, base64
    data URI). So a PDF is rasterised here, page by page (`render_pdf_pages_to_png`,
    reusing the same pypdfium2 dependency /ocr's page-counting already uses), and
    each page is sent as a separate completion; a plain image is sent as-is. Pages
    are joined with an HTML-comment marker rather than silently concatenated, so the
    boundary survives as a signal without being a hard cut — same convention as the
    WhatsApp/knowledge-base ingestion side of this project.

    Deliberately kept `deprecated=True` / "NOT READY" above even though this is a
    real implementation now: it has not been run against the live doc-lane
    container. Flip that only after a manual check (see SETUP.md / DECISIONS.md).
    """
    suffix = Path(file.filename or "upload").suffix
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    tmp_path = Path(name)

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
            pdf_pages = enforce_page_limit(tmp_path, max_pages=input_limits.PARSE_MAX_PAGES)
        except TooManyPages as exc:
            raise HTTPException(
                status_code=413,
                detail=f"{exc.pages}-page PDF exceeds the {exc.limit}-page doc-lane "
                f"limit (OCR_PARSE_MAX_PAGES).",
            ) from exc

        if pdf_pages is not None:
            try:
                page_images = [
                    (png, "image/png")
                    for png in input_limits.render_pdf_pages_to_png(tmp_path)
                ]
            except UnreadablePDF as exc:
                raise HTTPException(
                    status_code=400, detail=f"could not read PDF: {exc}"
                ) from exc
        else:
            mime = mimetypes.guess_type(file.filename or "")[0] or "image/jpeg"
            page_images = [(tmp_path.read_bytes(), mime)]

        async with httpx.AsyncClient(timeout=DOC_LANE_TIMEOUT) as client:
            model = await _doc_lane_model_name(client)
            page_markdowns = [
                await _doc_lane_complete(client, model, image_bytes, mime)
                for image_bytes, mime in page_images
            ]

        if len(page_markdowns) <= 1:
            markdown = page_markdowns[0] if page_markdowns else ""
        else:
            markdown = "\n\n".join(
                f"<!-- page {i} -->\n\n{text}"
                for i, text in enumerate(page_markdowns, start=1)
            )
        return {"markdown": markdown, "pages": len(page_markdowns)}
    finally:
        tmp_path.unlink(missing_ok=True)
