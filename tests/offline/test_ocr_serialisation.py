"""Offline — the /ocr response contract. Input on the left, output on the right.

The bug this guards: paddlex appends raw numpy arrays to rec_polys
(inference/pipelines/ocr/pipeline.py), so `boxes` reached FastAPI as
list[np.ndarray] and jsonable_encoder raised
    ValueError("'numpy.int64' object is not iterable")
i.e. HTTP 500 for every image containing text. A BLANK image returned 200 because
the list was empty — which is why /health and the blank-image test stayed green
while every real document failed.

Each case below states exactly what predict() yields and exactly what /ocr must
return, so the contract is readable without running anything.
"""

import numpy as np
import pytest

from shape import assert_ocr_envelope, assert_page_shape

pytestmark = pytest.mark.offline


def poly(*points, dtype="int64"):
    """A detection polygon as paddlex produces it: an ndarray, not a list."""
    return np.array(points, dtype=dtype)


# --------------------------------------------------------------------------- #
# INPUT (what the stubbed predict() yields)  ->  OUTPUT (what /ocr must return) #
# --------------------------------------------------------------------------- #

CASES = [
    pytest.param(
        # numpy ints + numpy float32 scores: the exact shape that used to 500
        [{
            "rec_texts": ["INVOICE 2026"],
            "rec_scores": [np.float32(0.9912)],
            "rec_polys": [poly([10, 10], [190, 12], [190, 46], [10, 44])],
        }],
        [{
            "texts": ["INVOICE 2026"],
            "scores": [pytest.approx(0.9912, abs=1e-6)],
            "boxes": [[[10, 10], [190, 12], [190, 46], [10, 44]]],
        }],
        id="numpy_int_polys_and_float32_scores",
    ),
    pytest.param(
        # float32 polys — the other dtype paddlex emits, depending on the det head
        [{
            "rec_texts": ["總金額"],
            "rec_scores": [0.8734],
            "rec_polys": [poly([1.5, 2.5], [9.0, 2.5], [9.0, 8.0], [1.5, 8.0], dtype="float32")],
        }],
        [{
            "texts": ["總金額"],
            "scores": [pytest.approx(0.8734, abs=1e-6)],
            "boxes": [[[1.5, 2.5], [9.0, 2.5], [9.0, 8.0], [1.5, 8.0]]],
        }],
        id="float32_polys_traditional_chinese",
    ),
    pytest.param(
        # blank image: the case that USED to pass and hid the bug
        [{"rec_texts": [], "rec_scores": [], "rec_polys": []}],
        [{"texts": [], "scores": [], "boxes": []}],
        id="blank_image_stays_empty",
    ),
    pytest.param(
        # multi-page PDF: order must survive, one entry per page
        [
            {"rec_texts": ["PAGE 1"], "rec_scores": [0.99],
             "rec_polys": [poly([0, 0], [10, 0], [10, 5], [0, 5])]},
            {"rec_texts": ["PAGE 2"], "rec_scores": [0.98],
             "rec_polys": [poly([0, 0], [10, 0], [10, 5], [0, 5])]},
        ],
        [
            {"texts": ["PAGE 1"], "scores": [pytest.approx(0.99)],
             "boxes": [[[0, 0], [10, 0], [10, 5], [0, 5]]]},
            {"texts": ["PAGE 2"], "scores": [pytest.approx(0.98)],
             "boxes": [[[0, 0], [10, 0], [10, 5], [0, 5]]]},
        ],
        id="two_pages_keep_their_order",
    ),
    pytest.param(
        # a whole-page ndarray instead of a list of ndarrays — np.asarray handles both
        [{
            "rec_texts": ["A", "B"],
            "rec_scores": [0.9, 0.8],
            "rec_polys": np.array([
                [[0, 0], [4, 0], [4, 2], [0, 2]],
                [[0, 5], [4, 5], [4, 7], [0, 7]],
            ], dtype="int64"),
        }],
        [{
            "texts": ["A", "B"],
            "scores": [pytest.approx(0.9), pytest.approx(0.8)],
            "boxes": [[[0, 0], [4, 0], [4, 2], [0, 2]], [[0, 5], [4, 5], [4, 7], [0, 7]]],
        }],
        id="rec_polys_as_one_big_ndarray",
    ),
]


@pytest.mark.parametrize("predict_yields,expected_pages", CASES)
def test_ocr_returns_exactly_this_json(post_ocr, fake_pages, predict_yields, expected_pages):
    fake_pages(predict_yields)

    resp = post_ocr()

    assert resp.status_code == 200, (
        f"HTTP {resp.status_code}. A 500 naming ndarray/numpy means run_ocr's "
        f"coercion regressed.\n{resp.text[:1500]}"
    )
    assert resp.json() == {"device": "cpu", "pages": expected_pages}


@pytest.mark.parametrize("predict_yields,expected_pages", CASES)
def test_every_case_satisfies_the_shared_shape_contract(
    post_ocr, fake_pages, predict_yields, expected_pages
):
    """Same inputs, but asserted through tests/shape.py — the file the live suite
    uses. Keeps the two suites from drifting into different contracts."""
    fake_pages(predict_yields)
    payload = post_ocr().json()
    for page in assert_ocr_envelope(payload):
        assert_page_shape(page)


# --------------------------------------------------------------------------- #
# types, not just values                                                       #
# --------------------------------------------------------------------------- #


def test_no_numpy_type_survives_anywhere_in_the_response(post_ocr, fake_pages):
    """Value equality can pass while types are still numpy (np.float32(0.5) == 0.5).
    A JSON encoder further down the chain would still choke, so assert the types."""
    fake_pages([{
        "rec_texts": [np.str_("NUMPY STR")],
        "rec_scores": [np.float64(0.5)],
        "rec_polys": [poly([1, 2], [3, 4], [5, 6], [7, 8])],
    }])

    page = post_ocr().json()["pages"][0]

    assert type(page["texts"][0]) is str
    assert type(page["scores"][0]) is float
    for point in page["boxes"][0]:
        for coord in point:
            assert type(coord) in (int, float), f"coord is {type(coord).__name__}"


def test_response_is_json_round_trippable(post_ocr, fake_pages):
    """The end state that actually matters: json.dumps must not raise."""
    import json

    fake_pages([{
        "rec_texts": ["ROUND TRIP"],
        "rec_scores": [np.float32(0.77)],
        "rec_polys": [poly([0, 0], [1, 0], [1, 1], [0, 1])],
    }])
    payload = post_ocr().json()
    assert json.loads(json.dumps(payload)) == payload


def test_pipeline_is_built_once_and_reused(post_ocr, fake_pages, stub_paddleocr):
    """get_pipeline() caches one pipeline per orientation setting. If that cache
    breaks, every request pays model init — invisible in correctness tests, fatal
    for latency."""
    import ocr_engine

    fake_pages([{"rec_texts": [], "rec_scores": [], "rec_polys": []}])
    post_ocr()
    built_after_first = len(stub_paddleocr.constructions)
    snapshot = dict(ocr_engine._pipelines)
    post_ocr()
    post_ocr()

    assert built_after_first > 0, "no pipeline was ever constructed"
    assert len(stub_paddleocr.constructions) == built_after_first, (
        "a pipeline was rebuilt on a later request"
    )
    assert ocr_engine._pipelines == snapshot, "the pipeline cache changed identity"


# --------------------------------------------------------------------------- #
# the constructor kwargs, checked without installing paddleocr                  #
# --------------------------------------------------------------------------- #


def test_engine_kwargs_match_what_paddlex_accepts(post_ocr, fake_pages, stub_paddleocr):
    """Guards the names verified against paddleocr 3.7.0 / paddlex 3.7.0 source.
    The stub asserts engine_config keys against ONNXRuntimeRunnerConfig's real
    field set (extra='forbid'), so a typo fails here instead of at deploy time."""
    fake_pages([{"rec_texts": [], "rec_scores": [], "rec_polys": []}])
    post_ocr()

    built = stub_paddleocr.constructions
    assert built, "PaddleOCR was never constructed — warmup()/get_pipeline() did not run"
    for kwargs in built:
        assert kwargs["engine"] == "onnxruntime", kwargs
        assert kwargs["text_detection_model_name"].startswith("PP-OCRv6_"), kwargs
        assert kwargs["text_recognition_model_name"].startswith("PP-OCRv6_"), kwargs
        assert "CPUExecutionProvider" in kwargs["engine_config"]["providers"], kwargs
        # Whole-page classify/unwarp stay off in BOTH pipelines — that is the doc
        # lane's job. Only use_textline_orientation differs between them.
        assert kwargs["use_doc_orientation_classify"] is False
        assert kwargs["use_doc_unwarping"] is False


def test_warmup_builds_both_orientation_pipelines(app_client, stub_paddleocr):
    """DECISIONS.md #6: the container runs with no network, so BOTH pipelines must
    be built at boot. If only the default one is, the first ?orientation=auto request
    in production tries to download a model from an offline container."""
    from starlette.testclient import TestClient

    import main
    import ocr_engine

    ocr_engine._pipelines.clear()
    stub_paddleocr.constructions = []
    with TestClient(main.app):
        pass  # entering the context runs lifespan -> warmup()

    flags = {k.get("use_textline_orientation") for k in stub_paddleocr.constructions}
    assert flags == {True, False}, (
        f"warmup built pipelines for {flags}, expected both True and False"
    )


def test_health_reports_the_configured_device(app_client):
    body = app_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["fast_lane_device"] in ("cpu", "gpu"), body


# --------------------------------------------------------------------------- #
# temp-file hygiene, without a container                                       #
# --------------------------------------------------------------------------- #


def test_temp_file_is_removed_after_a_successful_request(post_ocr, fake_pages, tmp_path, monkeypatch):
    """main.py writes the upload to NamedTemporaryFile(delete=False) and unlinks in
    a finally. Point tempfile at an empty dir and assert it is empty afterwards."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    fake_pages([{"rec_texts": ["X"], "rec_scores": [0.9],
                 "rec_polys": [poly([0, 0], [1, 0], [1, 1], [0, 1])]}])

    assert post_ocr().status_code == 200
    assert list(tmp_path.iterdir()) == [], f"temp file leaked: {list(tmp_path.iterdir())}"


def test_temp_file_is_removed_even_when_ocr_raises(post_ocr, fake_pages, tmp_path, monkeypatch):
    """The leak that matters is the one on the error path — that is the one that
    fills the container's disk over days of bad uploads."""
    import tempfile

    import ocr_engine

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    fake_pages([])

    def boom(_path, **_kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr("main.run_ocr", boom)
    monkeypatch.setattr(ocr_engine, "run_ocr", boom, raising=False)

    with pytest.raises(RuntimeError):
        post_ocr()

    assert list(tmp_path.iterdir()) == [], f"temp file leaked on the error path: {list(tmp_path.iterdir())}"
