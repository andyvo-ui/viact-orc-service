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

WARNING (Blackwell / RTX 50-series): ONNX Runtime silently falls back to CPU when
its CUDA kernels do not match the GPU arch (sm_120). OCR_DEVICE=gpu is NOT proof
the GPU is being used — run `python check_device.py` to confirm.

Version note: "PP-OCRv6" is the MODEL version; the paddleocr LIBRARY that ships it
is 3.7.0+. The two numbering schemes are unrelated.

NOTE: not smoke-tested against an installed paddleocr>=3.7. Verify kwargs with
    python -c "from paddleocr import PaddleOCR; help(PaddleOCR)"
"""

import os

from paddleocr import PaddleOCR

DET_MODEL = "PP-OCRv6_small_det"
REC_MODEL = "PP-OCRv6_small_rec"

DEVICE = os.getenv("OCR_DEVICE", "cpu").lower()

_pipeline: PaddleOCR | None = None


def _engine_config() -> dict:
    """ONNX Runtime provider list for the selected device.

    CPUExecutionProvider stays as the tail entry either way — ORT needs a fallback
    for any op the CUDA EP does not implement.
    """
    if DEVICE == "gpu":
        return {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    return {"providers": ["CPUExecutionProvider"]}


def get_pipeline() -> PaddleOCR:
    global _pipeline
    if _pipeline is None:
        _pipeline = PaddleOCR(
            text_detection_model_name=DET_MODEL,
            text_recognition_model_name=REC_MODEL,
            # These are separate models with their own weights to download. The
            # fast lane handles already-upright single images — leave them off.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="onnxruntime",
            engine_config=_engine_config(),
        )
    return _pipeline


def warmup() -> None:
    """Force model download + session init before serving traffic."""
    get_pipeline()


def run_ocr(image_path: str) -> list[dict]:
    """Detect + recognise text in one image. Returns one entry per page."""
    pages = []
    for page in get_pipeline().predict(image_path):
        pages.append(
            {
                "texts": page.get("rec_texts", []),
                "scores": page.get("rec_scores", []),
                "boxes": page.get("rec_polys", []),
            }
        )
    return pages
