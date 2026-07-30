"""Trigger the PP-OCRv6 ONNX download ahead of serving traffic.

PaddleOCR downloads weights lazily on first use. Run this once after install so
the first real request is not paying for a download.

    python prefetch_models.py
"""

import sys

from ocr_engine import DET_MODEL, DEVICE, REC_MODEL, warmup


def main() -> int:
    print(f"device={DEVICE}  det={DET_MODEL}  rec={REC_MODEL}")
    print("initialising (downloads ONNX weights on first run)...")
    warmup()
    print("done — weights cached under ~/.paddlex/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
