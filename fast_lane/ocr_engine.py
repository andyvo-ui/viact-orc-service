"""Fast-lane OCR: PP-OCRv6 small, ONNX Runtime, CPU or GPU.

Weights are resolved by model_name — PaddleOCR fetches the ONNX build itself on
first use (because engine="onnxruntime") and caches it under ~/.paddlex/.
Nothing is vendored in this repo. Call warmup() at service start so that download
happens at deploy time instead of inside the first user request.

Device is selected by OCR_DEVICE ("cpu" | "gpu"), defaulting to cpu.

WHY cpu is the default: this GPU is shared with an LLM and the doc-lane vLLM
server. CUDA time-slicing makes fast-lane latency depend on LLM load — good p50,
unpredictable p95. CPU keeps this lane's latency independent. Flip to gpu only
after measuring that CPU throughput is genuinely insufficient.

WARNING (Blackwell / RTX 50-series), with the failure modes now separated:
  - onnxruntime (CPU build) + OCR_DEVICE=gpu -> LOUD. paddlex validates the
    provider list against ort.get_available_providers() and raises RuntimeError
    (inference/models/runners/onnxruntime_runner.py::_validate_providers).
  - onnxruntime-gpu + a kernel/arch mismatch (sm_120) -> SILENT. The CUDA EP is
    "available", passes validation, and ORT then runs ops on CPU anyway.
The second case is the one OCR_DEVICE=gpu cannot prove. Run `check_device.py`.

Version note: "PP-OCRv6" is the MODEL version; the paddleocr LIBRARY that ships it
is 3.7.0+. The two numbering schemes are unrelated.

Kwargs below are verified against paddleocr 3.7.0 / paddlex 3.7.0: `engine` and
`engine_config` reach parse_common_args via PaddleOCR's **kwargs, "onnxruntime" is
in SUPPORTED_INFERENCE_ENGINE_LIST, and `providers` is a declared field on
ONNXRuntimeRunnerConfig (which is extra="forbid", so a typo would raise).
"""

import os

import numpy as np
from paddleocr import PaddleOCR

DET_MODEL = "PP-OCRv6_small_det"
REC_MODEL = "PP-OCRv6_small_rec"

DEVICE = os.getenv("OCR_DEVICE", "cpu").lower()

# Real input is ~50/50 site photos (signage at an angle) and flat scans, so one
# global orientation setting is wrong half the time. Callers pick per request.
#
# WHY TWO PIPELINES AND NOT A PER-PREDICT FLAG — verified against paddlex 3.7.0:
# OCRPipeline.__init__ creates the textline-orientation model ONLY when the flag is
# true at construction (inference/pipelines/ocr/pipeline.py:89-96). Passing
# use_textline_orientation=True to predict() on a pipeline built without it does
# not enable anything: check_model_settings_valid() logs an error and predict()
# yields {"error": ...} and then CARRIES ON — which run_ocr would have reported as
# an extra empty page. Hence one pipeline per setting, built lazily.
#
# Default is "detect orientation" because the failure modes are not symmetric:
# assuming upright on a rotated photo loses the text silently and looks exactly
# like success, while detecting orientation on a flat scan only costs a little
# latency. Flip it with OCR_DETECT_ORIENTATION=0 once you have measured.
DETECT_ORIENTATION_DEFAULT = os.getenv("OCR_DETECT_ORIENTATION", "1") not in (
    "0", "false", "False", "no", "",
)

# The orientation classifier is a SEPARATE model download (~7MB). It has an
# official ONNX build (PP-LCNet_x1_0_textline_ori is in paddlex's
# ONNX_SUPPORTED_MODELS), so this stays off paddlepaddle — DECISIONS.md #4 holds.
_pipelines: dict[bool, PaddleOCR] = {}


def _engine_config() -> dict:
    """ONNX Runtime provider list for the selected device.

    CPUExecutionProvider stays as the tail entry either way — ORT needs a fallback
    for any op the CUDA EP does not implement.
    """
    if DEVICE == "gpu":
        return {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    return {"providers": ["CPUExecutionProvider"]}


def get_pipeline(detect_orientation: bool | None = None) -> PaddleOCR:
    """One cached pipeline per orientation setting. See the note above on why this
    cannot be a per-predict flag."""
    if detect_orientation is None:
        detect_orientation = DETECT_ORIENTATION_DEFAULT
    if detect_orientation not in _pipelines:
        _pipelines[detect_orientation] = PaddleOCR(
            text_detection_model_name=DET_MODEL,
            text_recognition_model_name=REC_MODEL,
            # Still off in both pipelines: these classify/unwarp the WHOLE PAGE,
            # which is the doc lane's job. use_textline_orientation is per-textline
            # and is the one that matters for signage photographed at an angle.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=detect_orientation,
            engine="onnxruntime",
            engine_config=_engine_config(),
        )
    return _pipelines[detect_orientation]


def warmup() -> None:
    """Force model download + session init before serving traffic.

    Builds BOTH pipelines. That is deliberate: the container is meant to run with
    no network (DECISIONS.md #6), so the orientation classifier has to be fetched at
    build time too — otherwise the first `?orientation=auto` request in production
    tries to download a model from an offline container and fails.
    """
    get_pipeline(detect_orientation=False)
    get_pipeline(detect_orientation=True)


class EngineRejectedInput(RuntimeError):
    """predict() yielded an error record instead of a result."""


def run_ocr(image_path: str, *, detect_orientation: bool | None = None) -> list[dict]:
    """Read text from one image, or from every page of one PDF.

    Returns one entry per page — a single image gives a one-element list. PDF input
    is supported on purpose (a scanned PDF is a photo in a PDF wrapper, which is the
    same job); the gateway caps how many pages it will accept. Structure-preserving
    parsing is the doc lane's job, not this one.

    Coercion to builtins happens HERE, not in the gateway. paddlex appends raw
    numpy arrays to rec_polys (inference/pipelines/ocr/pipeline.py), and FastAPI's
    jsonable_encoder cannot encode numpy scalars — it raises
    ValueError("'numpy.int64' object is not iterable") on that exact shape, i.e. a
    500 for any image that contains text. A blank image returned 200 because the
    list was empty, which is why this hid so well.

    Doing it here rather than at the JSON boundary keeps every future caller of
    run_ocr (batch scripts, other services) from repeating the same conversion.
    """
    pages = []
    for page in get_pipeline(detect_orientation).predict(image_path):
        # paddlex signals invalid model settings by YIELDING {"error": ...} and then
        # continuing (pipelines/ocr/pipeline.py:331-332) — there is no `return`. With
        # a plain .get() that record became a page with no text: a silent off-by-one
        # in the page count that looks like a document with a blank page.
        if isinstance(page, dict) and page.get("error"):
            raise EngineRejectedInput(str(page["error"]))
        pages.append(
            {
                "texts": [str(t) for t in page.get("rec_texts", [])],
                "scores": [float(s) for s in page.get("rec_scores", [])],
                "boxes": [np.asarray(p).tolist() for p in page.get("rec_polys", [])],
            }
        )
    return pages
