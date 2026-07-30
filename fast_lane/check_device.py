"""Confirm whether ONNX Runtime actually runs on the GPU.

Run this BEFORE trusting OCR_DEVICE=gpu. ORT does not raise when its CUDA kernels
are unusable on the installed GPU (notably Blackwell / sm_120) — it silently drops
to CPU, so a "gpu" config can be a lie.

    python check_device.py

Reading the output:
  actually using: ['CUDAExecutionProvider', ...]  -> GPU is live
  actually using: ['CPUExecutionProvider']        -> silent fallback, GPU is NOT used

A very slow first run (tens of seconds) suggests the driver is JIT-compiling PTX
for this GPU arch. That is expected once, not per request.
"""

import os
import sys
from pathlib import Path

import onnxruntime as ort

# paddlex caches downloaded weights under $PADDLE_PDX_CACHE_HOME (default ~/.paddlex)
# in official_models/<model_name>_onnx/inference.onnx — verified against paddlex
# 3.7.0 (utils/cache.py DEFAULT_CACHE_DIR, official_models.py _save_dir,
# constants.py MODEL_FILE_PREFIX). Nothing is ever written under fast_lane/models/.
CACHE_DIR = Path(os.getenv("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex"))
MODEL_ROOT = CACHE_DIR / "official_models"


def _find_det_onnx() -> Path | None:
    """Any det .onnx will do — this probes the runtime, not a specific model."""
    for candidate in sorted(MODEL_ROOT.glob("*_det_onnx/inference.onnx")):
        return candidate
    for candidate in sorted(MODEL_ROOT.rglob("inference.onnx")):
        return candidate
    return None


def main() -> int:
    print("build providers:", ort.get_available_providers())

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("\nFAIL: this onnxruntime build has no CUDA EP.")
        print("      Installed 'onnxruntime' instead of 'onnxruntime-gpu'? "
              "The two conflict — uninstall both, then install only onnxruntime-gpu.")
        return 1

    det_onnx = _find_det_onnx()
    if det_onnx is None:
        print(f"\nFAIL: no cached ONNX model under {MODEL_ROOT}")
        print("      Run `python fast_lane/prefetch_models.py` first (the Docker")
        print("      build does this at image-build time).")
        return 1
    print("probing with:", det_onnx)

    session = ort.InferenceSession(
        str(det_onnx),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = session.get_providers()
    print("actually using:", active)

    if active and active[0] == "CUDAExecutionProvider":
        print("\nOK: GPU is live. OCR_DEVICE=gpu is safe to use.")
        return 0

    print("\nFAIL: ORT fell back to CPU — the CUDA EP could not load kernels for this GPU.")
    print("      Keep OCR_DEVICE=cpu, or investigate a TensorRT EP build.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
