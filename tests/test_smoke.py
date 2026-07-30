"""Group A — smoke. If anything here fails, stop; the rest is noise.

A3 was written as a prediction: `rec_polys` came straight out of PaddleOCR as numpy
arrays, and FastAPI's encoder cannot handle those. That has since been CONFIRMED —
paddlex appends raw ndarrays (inference/pipelines/ocr/pipeline.py) and
jsonable_encoder raises ValueError("'numpy.int64' object is not iterable") on that
exact shape — and fixed in run_ocr, which now coerces to builtins.

So A3 is no longer a probe, it is the regression test. Note the old failure was
input-dependent: a BLANK image returned 200 because the list was empty, so /health
and B1 looked fine while every real document 500'd.
"""

import pytest

from shape import assert_ocr_envelope, assert_page_shape


def test_a2_health_reports_device(client):
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["fast_lane_device"] in ("cpu", "gpu"), body


def test_a3_ocr_returns_200_on_a_real_image(post_file, real_sample):
    resp = post_file("/ocr", real_sample)
    assert resp.status_code == 200, (
        f"HTTP {resp.status_code}. A 500 mentioning ndarray / numpy / not JSON "
        f"serializable means run_ocr's builtin coercion (fast_lane/ocr_engine.py) "
        f"has regressed or a new numpy-typed field was added to the response.\n"
        f"{resp.text[:2000]}"
    )


def test_a3b_ocr_response_matches_the_declared_shape(post_file, real_sample):
    resp = post_file("/ocr", real_sample)
    assert resp.status_code == 200, resp.text[:2000]
    pages = assert_ocr_envelope(resp.json())
    assert pages, "no pages returned for a document that has text"
    for page in pages:
        assert_page_shape(page)


def test_a3c_ocr_actually_read_something(post_file, real_sample):
    """Separate from shape on purpose: a 200 with zero text is a silent failure,
    and it looks identical to success in a status-code-only check."""
    resp = post_file("/ocr", real_sample)
    assert resp.status_code == 200, resp.text[:2000]
    texts = [t for p in resp.json()["pages"] for t in p["texts"]]
    assert texts, "200 OK but zero text extracted from a real document"


def test_a3d_synthetic_image_roundtrip(post_file, make_image):
    """Runs without any fixture file. Weaker evidence than A3 — a clean 32px
    render is far easier than a real scan — but it proves the pipeline is wired."""
    img = make_image("smoke_en.png", text="INVOICE 2026")
    resp = post_file("/ocr", img)
    assert resp.status_code == 200, resp.text[:2000]
    joined = " ".join(t for p in resp.json()["pages"] for t in p["texts"]).upper()
    assert "INVOICE" in joined or "2026" in joined, f"read nothing usable: {joined!r}"


@pytest.mark.docker
def test_a4_weights_are_baked_into_the_image(compose_exec):
    """DECISIONS.md #6: weights bake in at build time so the container needs no
    network. If ~/.paddlex is empty, that decision is not actually in effect."""
    proc = compose_exec("python", "-c", "import pathlib,sys;"
                        "p=pathlib.Path('/root/.paddlex');"
                        "sys.stdout.write(str(sum(1 for _ in p.rglob('*.onnx'))) if p.exists() else 'NO_DIR')")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    assert out != "NO_DIR", "/root/.paddlex does not exist — prefetch did not run"
    assert int(out) > 0, "no .onnx files cached — weights were not baked in"
