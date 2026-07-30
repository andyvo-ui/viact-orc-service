"""Offline — which pipeline serves a request, and that the wrong one is never silent.

Real input is ~50/50 site photos (signage at an angle) and flat scans, so /ocr takes
`?orientation=auto|upright` and keeps one pipeline per setting.

Why two pipelines rather than a predict-time flag — verified against paddlex 3.7.0:
OCRPipeline.__init__ creates the textline-orientation model ONLY when the flag is
true at construction (inference/pipelines/ocr/pipeline.py:89-96). On a pipeline built
without it, predict(use_textline_orientation=True) does not enable anything —
check_model_settings_valid() logs an error and predict() YIELDS {"error": ...} and
then carries on, with no `return`. These tests pin that we build the right pipeline,
and that if that error record ever appears it becomes a loud failure rather than an
extra empty page in the caller's results.
"""

import pytest

pytestmark = pytest.mark.offline

ONE_PAGE = [{
    "rec_texts": ["SITE ENTRANCE"],
    "rec_scores": [0.97],
    "rec_polys": [[[0, 0], [9, 0], [9, 3], [0, 3]]],
}]


def orientation_flags_used(stub) -> list:
    return [k.get("use_textline_orientation") for k in stub.constructions]


# --------------------------------------------------------------------------- #
# the query param selects the pipeline                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "param,expected_flag",
    [("auto", True), ("upright", False)],
)
def test_orientation_param_selects_the_matching_pipeline(
    post_ocr, fake_pages, stub_paddleocr, param, expected_flag
):
    import ocr_engine

    fake_pages(ONE_PAGE)
    ocr_engine._pipelines.clear()
    stub_paddleocr.constructions = []

    resp = post_ocr(orientation=param)

    assert resp.status_code == 200, resp.text[:400]
    assert expected_flag in orientation_flags_used(stub_paddleocr), (
        f"?orientation={param} did not build a pipeline with "
        f"use_textline_orientation={expected_flag}; built "
        f"{orientation_flags_used(stub_paddleocr)}"
    )
    assert ocr_engine._pipelines[expected_flag] is not None


def test_the_two_settings_use_different_pipeline_objects(post_ocr, fake_pages):
    """If both settings resolved to one cached pipeline, the param would be a no-op
    that still returns 200 — the worst kind of wrong."""
    import ocr_engine

    fake_pages(ONE_PAGE)
    ocr_engine._pipelines.clear()

    assert post_ocr(orientation="auto").status_code == 200
    assert post_ocr(orientation="upright").status_code == 200

    assert set(ocr_engine._pipelines) == {True, False}, ocr_engine._pipelines
    assert ocr_engine._pipelines[True] is not ocr_engine._pipelines[False]


def test_omitting_the_param_uses_the_configured_default(post_ocr, fake_pages, stub_paddleocr):
    import ocr_engine

    fake_pages(ONE_PAGE)
    ocr_engine._pipelines.clear()
    stub_paddleocr.constructions = []

    assert post_ocr().status_code == 200

    assert ocr_engine.DETECT_ORIENTATION_DEFAULT in orientation_flags_used(stub_paddleocr)


def test_default_is_orientation_detection(app_client):
    """A deliberate choice recorded as a test, because the failure modes are not
    symmetric: assuming upright on a rotated photo loses text silently and looks
    like success, while detecting orientation on a flat scan only costs latency.
    If you flip OCR_DETECT_ORIENTATION=0 after benchmarking, change this test and
    say why in DECISIONS.md — do not just delete it."""
    body = app_client.get("/health").json()
    assert body["default_orientation"] == "auto", (
        f"default is {body['default_orientation']!r}; if that is intentional, update "
        f"this test and DECISIONS.md together"
    )


def test_an_unknown_orientation_value_is_422_not_a_silent_fallback(post_ocr, fake_pages):
    """A typo like ?orientation=rotated must not quietly get the default — the caller
    would believe it asked for something it did not get."""
    fake_pages(ONE_PAGE)
    resp = post_ocr(orientation="rotated")
    assert resp.status_code == 422, f"{resp.status_code}: {resp.text[:300]}"


def test_orientation_choice_does_not_change_the_response_shape(post_ocr, fake_pages):
    """Whatever pipeline serves it, the contract out is identical — callers must not
    have to branch on which one ran."""
    from shape import assert_ocr_envelope, assert_page_shape

    fake_pages(ONE_PAGE)
    for param in ("auto", "upright"):
        payload = post_ocr(orientation=param).json()
        for page in assert_ocr_envelope(payload):
            assert_page_shape(page)


# --------------------------------------------------------------------------- #
# the error record must never become a phantom page                            #
# --------------------------------------------------------------------------- #


def test_engine_error_record_is_raised_not_returned_as_an_empty_page(post_ocr, fake_pages):
    """paddlex yields {"error": ...} and CONTINUES when model settings are invalid.
    With a plain .get() that record became a page with no text — a page count that is
    silently one too high, indistinguishable from a document with a blank page."""
    fake_pages([{"error": "the input params for model settings are invalid!"}])

    resp = post_ocr()

    assert resp.status_code == 500, (
        f"expected a loud 500, got {resp.status_code}. If this is 200, check the "
        f"page count: {resp.text[:300]}"
    )
    assert "rejected" in resp.text.lower(), resp.text[:300]


def test_error_record_mixed_with_real_pages_still_fails(post_ocr, fake_pages):
    """The realistic shape: the error is yielded FIRST and real pages follow, so a
    naive reader sees N+1 pages and no exception."""
    fake_pages([
        {"error": "the input params for model settings are invalid!"},
        ONE_PAGE[0],
    ])

    resp = post_ocr()

    assert resp.status_code == 500, (
        f"got {resp.status_code} — an error record was folded into the page list "
        f"instead of failing: {resp.text[:400]}"
    )


def test_run_ocr_raises_a_typed_error(fake_pages):
    """Callers other than the gateway (batch scripts) need something catchable, not
    a bare RuntimeError message they have to string-match."""
    import ocr_engine

    fake_pages([{"error": "boom"}])
    with pytest.raises(ocr_engine.EngineRejectedInput):
        ocr_engine.run_ocr("whatever.jpg")
