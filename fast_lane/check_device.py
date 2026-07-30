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

import sys
from pathlib import Path

import onnxruntime as ort

DET_ONNX = Path(__file__).parent / "models" / "PP-OCRv6_small_det_onnx" / "inference.onnx"


def main() -> int:
    print("build providers:", ort.get_available_providers())

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("\nFAIL: this onnxruntime build has no CUDA EP.")
        print("      Installed 'onnxruntime' instead of 'onnxruntime-gpu'? "
              "The two conflict — uninstall both, then install only onnxruntime-gpu.")
        return 1

    if not DET_ONNX.exists():
        print(f"\nFAIL: model not found at {DET_ONNX}")
        print("      Run ./download_models.sh first.")
        return 1

    session = ort.InferenceSession(
        str(DET_ONNX),
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
