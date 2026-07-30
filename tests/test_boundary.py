"""Group B — input space. Where the gateway currently validates nothing.

/ocr takes any UploadFile, writes it to a temp file keeping only the filename's
suffix (gateway/main.py:45-48), and hands the path to PaddleOCR. There is no size
limit, no content-type check, no page limit. These tests map what that means.

Several of these have no obviously-correct answer — a 500 on a .txt upload may be
acceptable for an internal service. They are written to make the behaviour VISIBLE
so the decision is yours, and the assert is on "does not hang / does not leak an
unhandled traceback", not on a status code I invented.
"""

import pytest

from shape import assert_ocr_envelope, assert_page_shape, joined


def test_b1_blank_image_returns_empty_not_error(post_file, make_image):
    img = make_image("blank.png", text=None)
    resp = post_file("/ocr", img)
    assert resp.status_code == 200, resp.text[:2000]
    pages = assert_ocr_envelope(resp.json())
    # Without this, pages == [] makes the loop body never run and the test passes
    # while asserting nothing. One image in must be one page entry out.
    assert len(pages) == 1, f"one blank image produced {len(pages)} page entries"
    for page in pages:
        assert_page_shape(page)
        assert page["texts"] == [], f"invented text on a blank image: {page['texts']}"


def test_b2_filename_without_extension(post_file, make_image):
    """main.py:45 -> suffix='' -> the temp file has no extension. Whether PaddleOCR
    can still sniff the format is unverified."""
    img = make_image("noext_src.png", text="NO EXTENSION")
    resp = post_file("/ocr", img, filename="upload")
    assert resp.status_code in (200, 400, 415, 422, 500), resp.status_code
    if resp.status_code == 200:
        assert_ocr_envelope(resp.json())
    else:
        pytest.xfail(f"extension-less upload rejected with {resp.status_code} — decide if that is acceptable")


def test_b3_unicode_and_spaces_in_filename(post_file, make_image):
    img = make_image("unicode_src.png", text="REPORT V2")
    resp = post_file("/ocr", img, filename="報告 v2.png")
    assert resp.status_code == 200, resp.text[:2000]
    assert_ocr_envelope(resp.json())


def test_b4_text_file_disguised_as_image(post_file):
    """No content-type validation exists. This documents what a caller mistake
    looks like from the outside."""
    resp = post_file("/ocr", b"this is plainly not an image", filename="fake.jpg")
    assert resp.status_code != 200 or resp.json()["pages"] == [], (
        "a text file was accepted and produced OCR output — that is a lie"
    )
    if resp.status_code >= 500:
        pytest.xfail("bad input surfaces as 500, not 4xx — caller cannot tell it is their fault")


def test_b5_single_page_pdf(post_file, make_pdf):
    """PDF on the fast lane is supported ON PURPOSE, not incidentally: a scanned PDF
    is a photo in a PDF wrapper, which is the same job. What /ocr will not do is
    preserve structure — that is the doc lane. See gateway/main.py's module docstring."""
    pdf = make_pdf("one_page.pdf", pages=1)
    resp = post_file("/ocr", pdf)
    assert resp.status_code == 200, resp.text[:2000]
    pages = assert_ocr_envelope(resp.json())
    assert len(pages) == 1, f"1-page PDF produced {len(pages)} page entries"
    assert "PAGE 1" in joined(resp.json())


def test_b5b_multi_page_pdf_maps_one_entry_per_page(post_file, make_pdf):
    """run_ocr loops predict() and appends per page (ocr_engine.py:73-82). Confirm
    the page count survives, and that page order is preserved."""
    pdf = make_pdf("three_page.pdf", pages=3)
    resp = post_file("/ocr", pdf)
    assert resp.status_code == 200, resp.text[:2000]
    pages = assert_ocr_envelope(resp.json())
    assert len(pages) == 3, f"3-page PDF produced {len(pages)} page entries"
    for i, page in enumerate(pages, start=1):
        assert f"PAGE {i}" in " ".join(page["texts"]).upper(), (
            f"page {i} entry does not contain page {i} text — order is scrambled"
        )


def test_b6_pdf_over_the_page_limit_is_refused_not_attempted(post_file, make_pdf, client):
    """This used to accept any of (200, 413, 500, 504), i.e. it asserted nothing.

    /ocr now caps PDF pages, because run_ocr blocks the event loop and a long
    document takes /health down with it. The cap is discoverable on /health, so the
    test reads it instead of hardcoding a number.
    """
    limit = client.get("/health").json()["limits"]["max_pages"]
    pdf = make_pdf(f"over_limit_{limit + 5}.pdf", pages=limit + 5)

    resp = post_file("/ocr", pdf)

    assert resp.status_code == 413, (
        f"a {limit + 5}-page PDF returned {resp.status_code}, expected 413. "
        f"Unbounded page counts are what makes one upload an outage.\n{resp.text[:500]}"
    )
    assert "/parse" in resp.text, f"the 413 does not route the caller: {resp.text[:300]}"


@pytest.mark.slow
def test_b6b_pdf_at_the_page_limit_still_completes(post_file, make_pdf, client):
    """The other side of the cap: exactly at the limit must work, or the limit is
    effectively one lower than advertised."""
    limit = client.get("/health").json()["limits"]["max_pages"]
    pdf = make_pdf(f"at_limit_{limit}.pdf", pages=limit)

    resp = post_file("/ocr", pdf)

    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:500]}"
    assert len(resp.json()["pages"]) == limit


def test_b7_one_pixel_image(post_file, make_image):
    img = make_image("tiny.png", text=None, size=(1, 1))
    resp = post_file("/ocr", img)
    assert resp.status_code in (200, 400, 500), resp.status_code
    if resp.status_code == 200:
        assert_ocr_envelope(resp.json())


@pytest.mark.slow
def test_b8_very_large_image(post_file, make_image):
    """8000x8000 is ~190MB decoded. main.py:47 does `await file.read()` — the whole
    body lands in RAM, then PaddleOCR decodes it again."""
    img = make_image("huge.png", text="LARGE CANVAS", size=(8000, 8000))
    resp = post_file("/ocr", img)
    assert resp.status_code in (200, 413, 500), resp.status_code


def test_b9_missing_file_field(client):
    resp = client.post("/ocr", files={"wrong_field": ("a.png", b"x")})
    assert resp.status_code == 422, f"expected FastAPI validation error, got {resp.status_code}"


def test_b9b_empty_file(post_file):
    resp = post_file("/ocr", b"", filename="empty.png")
    assert resp.status_code in (200, 400, 422, 500), resp.status_code
    assert resp.status_code != 200 or resp.json()["pages"] in ([], [{"texts": [], "scores": [], "boxes": []}])


def test_b10_rotated_90_degrees_is_read_with_orientation_auto(post_file, make_image):
    """Input is ~50/50 site photos, so rotated text is a first-class case, not an
    edge case — this asserts rather than xfails.

    ?orientation=auto routes to the pipeline built with use_textline_orientation=True.
    If this fails, the orientation pipeline is not doing its job and half the real
    traffic is silently losing text.
    """
    upright = make_image("rot_base.png", text="SITE ENTRANCE")
    rotated = make_image("rot_90.png", text="SITE ENTRANCE", rotate=90)

    up = post_file("/ocr?orientation=auto", upright)
    rot = post_file("/ocr?orientation=auto", rotated)
    assert up.status_code == 200 and rot.status_code == 200

    up_text, rot_text = joined(up.json()), joined(rot.json())
    print(f"\n  upright -> {up_text!r}\n  rotated -> {rot_text!r}")
    assert "ENTRANCE" in up_text, f"baseline itself failed: {up_text!r}"
    assert "ENTRANCE" in rot_text, (
        f"90-degree text not read WITH orientation=auto: {rot_text!r}. Either the "
        f"orientation model is not loaded or textline orientation does not cover a "
        f"full 90 degrees — check whether the doc lane is needed for rotated input."
    )


@pytest.mark.accuracy
def test_b10b_upright_mode_is_the_one_allowed_to_miss_rotated_text(post_file, make_image):
    """Documents the trade-off the ?orientation param exists for. ?orientation=upright
    is the fast path for flat scans; missing rotated text there is the expected cost,
    not a bug. This is why `auto` is the default."""
    rotated = make_image("rot_90.png", text="SITE ENTRANCE", rotate=90)
    resp = post_file("/ocr?orientation=upright", rotated)
    assert resp.status_code == 200, resp.text[:2000]
    text = joined(resp.json())
    print(f"\n  rotated + upright mode -> {text!r}")
    if "ENTRANCE" not in text:
        pytest.xfail("rotated text missed in upright mode — expected, hence auto default")


@pytest.mark.accuracy
def test_b11_skewed_15_degrees(post_file, make_image):
    """15 degrees is the harder case: too little to be a 90-degree rotation, enough
    to distort the cropped textline strip. Neither mode is guaranteed to get it."""
    skewed = make_image("skew_15.png", text="SITE ENTRANCE", rotate=15)
    resp = post_file("/ocr?orientation=auto", skewed)
    assert resp.status_code == 200, resp.text[:2000]
    text = joined(resp.json())
    print(f"\n  skewed 15deg (auto) -> {text!r}")
    if "ENTRANCE" not in text:
        pytest.xfail("15-degree skew not read even with orientation=auto — if your "
                     "site photos look like this, the doc lane may be the answer")
