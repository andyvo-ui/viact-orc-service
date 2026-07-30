"""Offline — the bounds /ocr now enforces, and the error it must NOT return.

Context: /ocr accepts PDFs on purpose (a scanned PDF is a photo in a wrapper), but
`run_ocr` blocks the event loop, so unbounded page counts turn one upload into a
health-check outage. These tests pin the bound and, just as importantly, pin that
the rejection is a 413 naming /parse — not a 500, and not a silent 200.

Real PDFs are built here with pypdfium2 rather than mocked, because the whole point
is that the page count agrees with the library paddlex itself uses to rasterise.
"""

import pytest

from shape import assert_ocr_envelope

pytestmark = pytest.mark.offline

pypdfium2 = pytest.importorskip(
    "pypdfium2", reason="pypdfium2 ships with paddlex[ocr-core]; install test deps"
)

ONE_PAGE = [{"rec_texts": ["X"], "rec_scores": [0.9], "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}]


def make_pdf_bytes(pages: int) -> bytes:
    """A structurally valid PDF with `pages` blank pages."""
    doc = pypdfium2.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 200)
    import io

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# page limit                                                                   #
# --------------------------------------------------------------------------- #


def test_pdf_within_the_limit_is_accepted(post_ocr, fake_pages):
    from input_limits import MAX_PAGES

    fake_pages(ONE_PAGE)
    resp = post_ocr(make_pdf_bytes(min(3, MAX_PAGES)), filename="short.pdf")

    assert resp.status_code == 200, resp.text[:500]
    assert_ocr_envelope(resp.json())


def test_pdf_over_the_limit_is_413_and_names_the_alternative(post_ocr, fake_pages, monkeypatch):
    """A caller who gets a 413 with no route forward will just retry. The message
    has to say where a long document actually goes."""
    import input_limits

    monkeypatch.setattr(input_limits, "MAX_PAGES", 3)
    fake_pages(ONE_PAGE)

    resp = post_ocr(make_pdf_bytes(8), filename="long.pdf")

    assert resp.status_code == 413, f"{resp.status_code}: {resp.text[:500]}"
    body = resp.text
    assert "8" in body and "3" in body, f"neither count is in the message: {body[:300]}"
    assert "/parse" in body, f"the 413 does not point anywhere: {body[:300]}"


def test_page_limit_is_checked_before_any_ocr_runs(post_ocr, fake_pages, monkeypatch):
    """The limit exists to avoid spending the time, so it must reject BEFORE
    inference. If it ran after, the 413 would be honest and useless."""
    import input_limits
    import main

    monkeypatch.setattr(input_limits, "MAX_PAGES", 2)
    fake_pages(ONE_PAGE)

    called = []
    monkeypatch.setattr(main, "run_ocr", lambda *a, **k: called.append(1) or [])

    resp = post_ocr(make_pdf_bytes(9), filename="long.pdf")

    assert resp.status_code == 413, resp.text[:300]
    assert not called, "run_ocr ran despite the page limit rejecting the request"


def test_a_plain_image_never_trips_the_page_limit(post_ocr, fake_pages, monkeypatch):
    """count_pdf_pages sniffs magic bytes; an image must come back None, not 0 or 1
    from a failed parse, or images would be judged against a PDF rule."""
    import input_limits

    monkeypatch.setattr(input_limits, "MAX_PAGES", 1)
    fake_pages(ONE_PAGE)

    resp = post_ocr(b"\x89PNG\r\n\x1a\n" + b"junk" * 100, filename="photo.png")

    assert resp.status_code == 200, resp.text[:300]


def test_pdf_extension_on_a_non_pdf_is_not_a_page_error(post_ocr, fake_pages):
    """The extension comes from the client and is not validated. A .pdf name on
    non-PDF bytes must not be reported as a page-count problem."""
    fake_pages(ONE_PAGE)
    resp = post_ocr(b"this is plainly not a pdf", filename="liar.pdf")
    assert resp.status_code != 413, (
        f"non-PDF bytes rejected as too many pages: {resp.text[:300]}"
    )


def test_corrupt_pdf_is_not_reported_as_too_many_pages(post_ocr, fake_pages):
    """A truncated PDF has the right magic bytes and an unreadable page tree.
    count_pdf_pages returns None so the engine produces the real error, instead of
    the caller being told to split a file that is simply broken."""
    fake_pages(ONE_PAGE)
    truncated = make_pdf_bytes(4)[: 40]
    resp = post_ocr(truncated, filename="truncated.pdf")
    assert resp.status_code != 413, resp.text[:300]


# --------------------------------------------------------------------------- #
# size limit                                                                   #
# --------------------------------------------------------------------------- #


def test_upload_over_the_byte_limit_is_413(post_ocr, fake_pages, monkeypatch):
    import input_limits

    monkeypatch.setattr(input_limits, "MAX_UPLOAD_BYTES", 4096)
    fake_pages(ONE_PAGE)

    resp = post_ocr(b"\x89PNG\r\n\x1a\n" + b"A" * 20_000, filename="big.png")

    assert resp.status_code == 413, f"{resp.status_code}: {resp.text[:300]}"
    assert "4096" in resp.text, resp.text[:300]


def test_upload_at_the_limit_is_accepted(post_ocr, fake_pages, monkeypatch):
    """Off-by-one guard: the limit is inclusive. A file of exactly N bytes passes."""
    import input_limits

    monkeypatch.setattr(input_limits, "MAX_UPLOAD_BYTES", 4096)
    fake_pages(ONE_PAGE)

    resp = post_ocr(b"A" * 4096, filename="exact.png")

    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"


def test_oversized_upload_leaves_no_temp_file(post_ocr, fake_pages, monkeypatch, tmp_path):
    """The size check aborts mid-stream, so a partial file exists when it raises.
    The finally block must still unlink it, or every rejected upload leaks."""
    import tempfile

    import input_limits

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(input_limits, "MAX_UPLOAD_BYTES", 2048)
    fake_pages(ONE_PAGE)

    assert post_ocr(b"A" * 50_000, filename="big.png").status_code == 413
    assert list(tmp_path.iterdir()) == [], f"leaked: {list(tmp_path.iterdir())}"


def test_limits_are_advertised_on_health(app_client):
    """A caller cannot respect a limit it cannot discover. If these are only in env
    vars, every client hardcodes a guess."""
    body = app_client.get("/health").json()
    assert body["limits"]["max_pages"] >= 1, body
    assert body["limits"]["max_upload_bytes"] > 0, body
