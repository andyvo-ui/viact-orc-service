"""Bounds on what /ocr will accept. It previously had none.

WHY BOUNDS AT ALL — this is not defensive boilerplate, it is load-bearing:

`run_ocr` is fully blocking and is called from an `async def` endpoint, so it
occupies the event loop for its whole duration and /health cannot answer while it
runs. The compose healthcheck is `timeout: 5s, retries: 3, interval: 30s`, so a
steady stream of long requests marks a perfectly working container unhealthy and
gets it restarted mid-request. Until that blocking is resolved
(tests/test_concurrency.py::test_c1), refusing work that takes too long is the only
defence available.

Two separate limits, because they fail differently:

  bytes  — the old code did `await file.read()`, putting the entire upload in RAM
           before writing it out. A 2GB PDF was a 2GB allocation. We now stream to
           disk and abort as soon as the limit is crossed, so an oversized upload
           costs one chunk of memory, not all of it.
  pages  — cost is per page, so a 5MB 200-page PDF is cheap to upload and very
           expensive to OCR. Bytes alone do not bound the work.

Both are env-tunable rather than baked, because the right numbers depend on CPU
speed and on how many pages your real permits and contracts actually have. Measure
before raising them.
"""

import io
import os
from pathlib import Path

CHUNK_BYTES = 1024 * 1024

MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# 10 pages at the closest published CPU figure (DECISIONS.md §7: 0.59s/page on
# Xeon+OpenVINO) is ~6s of blocking — inside one healthcheck interval, outside the
# 5s timeout for that single probe but not enough to trip `retries: 3`. Raise it
# only after test_c1/test_c2 have run on the real box.
MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "10"))

# Doc lane cost is a GPU round-trip per page (vLLM), not local CPU inference, so it
# gets its own cap rather than sharing MAX_PAGES. 20 is a starting point, not a
# measured number — raise or lower once real per-page latency is known (see
# DECISIONS.md / SETUP.md).
PARSE_MAX_PAGES = int(os.getenv("OCR_PARSE_MAX_PAGES", "20"))


class UploadTooLarge(Exception):
    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        super().__init__(f"upload exceeds {limit_bytes} bytes")


class TooManyPages(Exception):
    def __init__(self, pages: int, limit: int):
        self.pages = pages
        self.limit = limit
        super().__init__(f"{pages} pages exceeds the {limit}-page limit")


class UnreadablePDF(Exception):
    """A file sniffs as a PDF (`%PDF-` magic bytes) but pypdfium2 cannot open or
    render it — e.g. truncated or corrupted past the header."""


async def stream_upload_to(upload, dest: Path, *, max_bytes: int | None = None) -> int:
    """Copy an UploadFile to `dest` in chunks. Raises UploadTooLarge past the limit.

    Aborts on the chunk that crosses the limit rather than after reading everything,
    so the memory cost of a hostile upload is one chunk. `dest` may hold a partial
    file when this raises — the caller's `finally` unlink handles that.

    The limit is read from the module global at CALL time, not bound as a default
    argument. A default would be evaluated once at import and then be impossible to
    change or to override in a test — which is a bug this module already had.
    """
    limit = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await upload.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise UploadTooLarge(limit)
            out.write(chunk)
    return total


def count_pdf_pages(path: Path) -> int | None:
    """Page count for a PDF, or None when the file is not a PDF.

    Sniffs the magic bytes rather than trusting the filename — the extension comes
    from the client and /ocr does not validate it. Uses pypdfium2, which is already
    installed: paddleocr pulls paddlex[ocr-core], which requires pypdfium2>=4. It is
    the same library paddlex itself uses to rasterise PDFs, so this count is the
    count paddlex will iterate.

    Returns None (rather than raising) for a corrupt PDF: rejecting it here would
    turn a "your file is broken" into a "too many pages", which is a worse message.
    Let the engine produce the real error.
    """
    try:
        with path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                return None
    except OSError:
        return None

    try:
        import pypdfium2
    except ImportError:
        return None

    try:
        doc = pypdfium2.PdfDocument(str(path))
    except Exception:
        return None
    try:
        return len(doc)
    finally:
        close = getattr(doc, "close", None)
        if close is not None:
            close()


def enforce_page_limit(path: Path, *, max_pages: int | None = None) -> int | None:
    """Raise TooManyPages if `path` is a PDF with more pages than allowed.

    Returns the page count, or None when the input is not a PDF (a single image is
    always one page and never hits this). Limit read at call time — see the note in
    stream_upload_to.
    """
    limit = MAX_PAGES if max_pages is None else max_pages
    pages = count_pdf_pages(path)
    if pages is not None and pages > limit:
        raise TooManyPages(pages, limit)
    return pages


def render_pdf_pages_to_png(path: Path, *, scale: float = 2.0) -> list[bytes]:
    """Rasterise each page of a PDF to a PNG, in reading order.

    The doc lane is a vision-language model behind an OpenAI-compatible chat API —
    it takes images (`image_url`), not PDF bytes. This renders pages so /parse can
    hand it one image per page. `scale=2.0` matches paddlex's own default PDF-to-
    image zoom (`PDFReaderBackend`), so fast lane and doc lane see visually
    equivalent input for the same file.

    Uses `.to_pil()`, not `.to_numpy()` + cv2: cv2 is only guaranteed in the
    production container (arrives transitively via paddleocr, which the offline
    test suite deliberately does not install — see tests/offline/conftest.py).
    Pillow is guaranteed in BOTH: it is an unconditional paddlex dependency (not
    behind an extras marker, unlike opencv-contrib-python) and is already an
    explicit offline-test dependency (tests/requirements.txt, for synthesising
    fixtures). Verified directly: `PdfBitmap.to_pil()` returns proper RGB, no
    channel-order bugs to route around.
    """
    import pypdfium2
    from PIL import Image  # noqa: F401 — import error should surface, not just PIL usage below

    try:
        doc = pypdfium2.PdfDocument(str(path))
    except Exception as exc:
        raise UnreadablePDF(str(exc)) from exc

    try:
        pages = []
        for page in doc:
            try:
                image = page.render(scale=scale).to_pil()
            finally:
                page.close()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            pages.append(buf.getvalue())
        return pages
    except pypdfium2.PdfiumError as exc:
        raise UnreadablePDF(str(exc)) from exc
    finally:
        doc.close()
