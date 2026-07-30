# tests/samples — synthetic, committed, safe to share

Regenerate with `python scripts/make_samples.py`. Run them all at once:

```bash
./scripts/test_samples.sh                        # default http://localhost:8000
./scripts/test_samples.sh http://localhost:8001
```

**Two sample directories, do not mix them up:**

| dir | contents | in git? |
|---|---|---|
| `tests/samples/` | synthetic renders, this directory | **yes** |
| `tests/fixtures/` | your real HK customer documents | **no** — gitignored |

A clean font render is far easier than a phone photo of a Traditional Chinese
contract. A green run here proves the pipeline is wired and every boundary behaves.
It proves **nothing** about accuracy on your customer's documents — that still needs
30–50 real files through `scripts/report_live.py`.

## What each file is for

| file | endpoint | expect | what a failure means |
|---|---|---|---|
| `01_english.png` | `/ocr` | 200, reads `INVOICE 2026` / `NO ENTRY` / `PANEL P-04B` | pipeline is broken; nothing else matters |
| `02_chinese_traditional.png` | `/ocr` | 200, reads `停車場` / `危險 請勿進入` / `總金額 HK$1,240` | TC recognition is not working — the primary language |
| `03_mixed_tc_en.png` | `/ocr` | 200, reads TC and EN **on the same line** | mixed-script lines get split or dropped |
| `04_rotated_90.png` | `/ocr?orientation=auto` | 200, reads `SITE ENTRANCE` | the orientation pipeline is not doing its job — half your input is site photos |
| `04_rotated_90.png` | `/ocr?orientation=upright` | 200, may read **nothing** | this is the expected trade-off, not a bug. It is why `auto` is the default |
| `05_skewed_15.png` | `/ocr?orientation=auto` | 200, may read nothing | 15° is the hard case: too little for a 90° rotation, enough to distort the cropped strip |
| `06_blank.png` | `/ocr` | 200, **0 lines** | any text here is invented |
| `07_small_text.png` | `/ocr` | 200, reads `SERIAL 8842-XK-01` | 22px is near the floor; a miss tells you the minimum text size you can rely on |
| `08_two_page.pdf` | `/ocr` | 200, **2 page entries in order** | PDF paging or ordering is broken |
| `09_over_page_limit.pdf` | `/ocr` | **413**, message names `/parse` | the page cap is not enforced — one upload can take `/health` down |
| `10_not_an_image.jpg` | `/ocr` | non-200, **or** 200 with 0 lines | 200 **with text** means it invented output from a text file |
| `11_unsupported.docx` | `/ocr` | non-200 | see below |
| `12_unsupported.xlsx` | `/ocr` | non-200 | see below |
| `13_empty.png` | `/ocr` | non-200, or 200 with 0 lines | a 0-byte upload should not crash the worker |

## Word and Excel are not supported by either lane

Not an oversight — neither lane can read them:

- **fast lane** — `paddleocr` pulls `paddlex[ocr-core]`, which handles images and PDF
  (via `pypdfium2`). No Office formats.
- **doc lane** — PaddleOCR-VL is a vision model. It takes pixels.
- The `python-docx` / `openpyxl` / `python-pptx` dependencies live in paddleocr's
  **`doc2md` extra**, which this image does not install.

`11_unsupported.docx` and `12_unsupported.xlsx` are structurally valid Office files
(written with `zipfile`, not renamed `.txt`) so the test exercises a real rejection.

If you need Office input, that is a separate decision with two shapes — convert to
PDF before upload (LibreOffice headless in front of the gateway), or install the
`doc2md` extra and add an endpoint for it. Neither is built.

## A note on generating these

`scripts/make_samples.py` checks glyph coverage per character before picking a CJK
font. PIL silently draws **nothing** for a missing glyph and raises no error, so the
first version of `03_mixed_tc_en.png` rendered `緊急出口` as `急出口` — a fixture that
did not contain the text it claimed. `require_glyphs()` is why the file is now
trustworthy; do not remove it when adding samples.
