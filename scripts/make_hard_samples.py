"""Generate HARD samples in tests/samples/hard/ plus their ground truth.

    python scripts/make_hard_samples.py

The point of writing the generator rather than downloading files: we know exactly
what text was drawn, so scripts/score_samples.py can compute a real accuracy number
(character accuracy, missed lines, invented lines) with no human transcription.

That number is a REGRESSION baseline, not a business answer. These are rendered
documents with simulated degradation; a phone photo of a real HK contract is harder
in ways this cannot fake (paper texture, motion blur, ink bleed, stamps over text,
handwriting). Real documents still go in tests/fixtures/ and get scored separately.

What each sample deliberately attacks:
    20 permit form      labelled fields, boxes, dense TC+EN, codes and dates
    21 invoice table    ruled table - the case flat OCR cannot reconstruct
    22 site photo       perspective + uneven lighting + JPEG artefacts
    23 nameplate        small dense text, low contrast on metal grey
    24 two column       reading order across columns
    25 low contrast     grey on grey, faded print
    26 noisy scan       speckle, skew, heavy JPEG - a bad photocopy
    27 complex.pdf      3 pages combining the above
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "tests" / "samples" / "hard"

LATIN_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
CJK_FONTS = [
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 1),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
]

TRUTH: dict[str, dict] = {}


def missing_glyphs(f, text):
    out = []
    for ch in set(text):
        if ch.isspace():
            continue
        try:
            if f.getmask(ch).getbbox() is None:
                out.append(ch)
        except Exception:
            out.append(ch)
    return out


def font(size, *, must_render=""):
    """Any font that covers `must_render`. CJK candidates first when needed.

    PIL draws NOTHING for a missing glyph and raises nothing, so without this check
    a sample silently fails to contain the text its ground truth claims.
    """
    needs_cjk = any(ord(c) > 0x2E80 for c in must_render)
    cands = CJK_FONTS if needs_cjk else [(p, 0) for p in LATIN_FONTS]
    for path, idx in cands:
        if not Path(path).exists():
            continue
        try:
            f = ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
        if must_render and missing_glyphs(f, must_render):
            continue
        return f
    return None


def record(name, lines, note):
    TRUTH[name] = {"lines": [ln for ln in lines if ln.strip()], "note": note}


# --------------------------------------------------------------------------- #
# degradation helpers — what turns a clean render into something photo-like     #
# --------------------------------------------------------------------------- #


def add_lighting(img, strength=0.55):
    """Diagonal light falloff, like a phone photo under one lamp."""
    w, h = img.size
    grad = Image.new("L", (w, h))
    px = grad.load()
    for y in range(h):
        for x in range(0, w, 4):
            v = 255 - int(strength * 255 * ((x / w) * 0.6 + (y / h) * 0.4))
            for dx in range(4):
                if x + dx < w:
                    px[x + dx, y] = max(0, v)
    return Image.composite(img, Image.new("RGB", (w, h), "black"), grad.point(lambda v: v))


def add_speckle(img, amount=0.04):
    import random

    random.seed(7)  # deterministic: a fixture that changes per run is not a fixture
    px = img.load()
    w, h = img.size
    for _ in range(int(w * h * amount)):
        x, y = random.randrange(w), random.randrange(h)
        v = random.choice((0, 0, 255))
        px[x, y] = (v, v, v)
    return img


def perspective(img, tilt=0.10):
    """Shoot the sign from an angle rather than straight on."""
    w, h = img.size
    dx = int(w * tilt)
    coeffs = _perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(dx, int(h * 0.04)), (w, 0), (w - dx, h), (0, int(h * 0.96))],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor="white")


def _perspective_coeffs(src, dst):
    import numpy as np

    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(m, dtype=float)
    B = np.array(src, dtype=float).reshape(8)
    return np.linalg.solve(A, B).tolist()


def save_jpeg(img, path, quality=45):
    img.convert("RGB").save(path, "JPEG", quality=quality)


# --------------------------------------------------------------------------- #
# samples                                                                      #
# --------------------------------------------------------------------------- #


def permit_form(path):
    lines = [
        "WORK PERMIT / 工作許可證",
        "PERMIT NO: HK-WP-2026-04817",
        "SITE: KOWLOON BAY DEPOT / 九龍灣車廠",
        "CONTRACTOR: VIACT ENGINEERING LTD",
        "ISSUE DATE: 2026-03-14",
        "EXPIRY DATE: 2026-03-21",
        "AREA: BLOCK C LEVEL B2",
        "HOT WORK: YES",
        "CONFINED SPACE: NO",
        "SUPERVISOR: CHAN TAI MAN",
        "CONTACT: +852 9123 4567",
    ]
    f_title = font(44, must_render=lines[0])
    f_body = font(30, must_render="".join(lines))
    if not f_title or not f_body:
        return print("  SKIP 20_permit_form.png (font)")
    img = Image.new("RGB", (1240, 900), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 1210, 870], outline="black", width=3)
    d.text((60, 55), lines[0], fill="black", font=f_title)
    d.line([30, 120, 1210, 120], fill="black", width=2)
    y = 150
    for ln in lines[1:]:
        d.rectangle([55, y - 6, 1185, y + 42], outline=(140, 140, 140), width=1)
        d.text((70, y), ln, fill="black", font=f_body)
        y += 62
    img.save(path)
    record(path.name, lines, "form: boxes, labels, codes, dates, TC+EN")
    print(f"  {path.name}")


def invoice_table(path):
    header = ["ITEM", "DESC", "QTY", "UNIT PRICE", "AMOUNT"]
    rows = [
        ["1", "ANCHOR BOLT M16", "120", "8.50", "1,020.00"],
        ["2", "SAFETY HELMET", "45", "96.00", "4,320.00"],
        ["3", "SCAFFOLD CLAMP", "300", "12.75", "3,825.00"],
        ["4", "WARNING TAPE 50M", "18", "34.00", "612.00"],
    ]
    title = "INVOICE 發票  NO. INV-2026-0917"
    total = ["", "TOTAL 總金額", "", "", "9,777.00"]
    f_t = font(38, must_render=title)
    f_b = font(26, must_render="".join(header + sum(rows, []) + total))
    if not f_t or not f_b:
        return print("  SKIP 21_invoice_table.png (font)")

    img = Image.new("RGB", (1300, 620), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, fill="black", font=f_t)
    cols = [40, 140, 620, 760, 990, 1260]
    top, rh = 110, 70
    nrows = len(rows) + 2
    for i in range(nrows + 1):
        d.line([cols[0], top + i * rh, cols[-1], top + i * rh], fill="black", width=2)
    for c in cols:
        d.line([c, top, c, top + nrows * rh], fill="black", width=2)
    for i, cell in enumerate(header):
        d.text((cols[i] + 12, top + 20), cell, fill="black", font=f_b)
    for r, row in enumerate(rows, start=1):
        for i, cell in enumerate(row):
            d.text((cols[i] + 12, top + r * rh + 20), cell, fill="black", font=f_b)
    for i, cell in enumerate(total):
        if cell:
            d.text((cols[i] + 12, top + (len(rows) + 1) * rh + 20), cell, fill="black", font=f_b)
    img.save(path)
    record(path.name, [title] + [" ".join(header)]
           + [" ".join(r) for r in rows] + [" ".join(c for c in total if c)],
           "ruled table: flat OCR cannot rebuild rows/columns")
    print(f"  {path.name}")


def site_photo(path):
    lines = ["緊急出口", "EMERGENCY EXIT", "GATE 3 / 三號閘", "NO ENTRY 禁止進入"]
    f = font(72, must_render="".join(lines))
    if not f:
        return print("  SKIP 22_site_photo.jpg (font)")
    img = Image.new("RGB", (1400, 800), (245, 243, 238))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 1340, 740], outline=(30, 30, 30), width=8)
    y = 140
    for ln in lines:
        d.text((130, y), ln, fill=(20, 20, 20), font=f)
        y += 140
    img = perspective(img, tilt=0.09)
    img = add_lighting(img, 0.45)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    save_jpeg(img, path, quality=50)
    record(path.name, lines, "photo: perspective + uneven light + JPEG + blur")
    print(f"  {path.name}")


def nameplate(path):
    lines = [
        "MODEL: VG-2200XT",
        "SERIAL NO: 8842-XK-01193",
        "VOLTAGE: 380V 50Hz 3PH",
        "RATED LOAD: 2,200 kg",
        "MFG DATE: 2025-11",
        "CAL DUE: 2026-11-30",
        "ASSET TAG: VIACT-EQ-00417",
    ]
    f = font(24, must_render="".join(lines))
    if not f:
        return print("  SKIP 23_nameplate.jpg (font)")
    img = Image.new("RGB", (760, 420), (168, 170, 172))
    d = ImageDraw.Draw(img)
    d.rectangle([16, 16, 744, 404], outline=(110, 112, 114), width=4)
    y = 50
    for ln in lines:
        d.text((45, y), ln, fill=(58, 58, 60), font=f)
        y += 48
    img = ImageEnhance.Contrast(img).enhance(0.72)
    img = add_lighting(img, 0.3)
    save_jpeg(img, path, quality=55)
    record(path.name, lines, "small dense text, low contrast on metal grey")
    print(f"  {path.name}")


def two_column(path):
    left = [
        "SECTION 1 - SCOPE",
        "The contractor shall provide all",
        "labour, plant and materials for",
        "the works described herein and",
        "shall comply with the site rules",
        "issued by the project manager.",
    ]
    right = [
        "第一節 - 工程範圍",
        "承包商須提供所有勞工",
        "機械及材料以完成本文",
        "所述之工程並須遵守",
        "項目經理發出之工地規則",
    ]
    f = font(28, must_render="".join(left + right))
    if not f:
        return print("  SKIP 24_two_column.png (font)")
    img = Image.new("RGB", (1240, 500), "white")
    d = ImageDraw.Draw(img)
    d.line([620, 40, 620, 460], fill=(180, 180, 180), width=2)
    y = 60
    for ln in left:
        d.text((60, y), ln, fill="black", font=f)
        y += 52
    y = 60
    for ln in right:
        d.text((660, y), ln, fill="black", font=f)
        y += 52
    img.save(path)
    record(path.name, left + right, "two columns: reading order across the gutter")
    print(f"  {path.name}")


def low_contrast(path):
    lines = ["REF: HK-2026-0331-B", "FADED CARBON COPY", "施工日誌 SITE DIARY"]
    f = font(40, must_render="".join(lines))
    if not f:
        return print("  SKIP 25_low_contrast.png (font)")
    img = Image.new("RGB", (1000, 320), (206, 206, 202))
    d = ImageDraw.Draw(img)
    y = 50
    for ln in lines:
        d.text((60, y), ln, fill=(150, 150, 148), font=f)
        y += 80
    img.save(path)
    record(path.name, lines, "grey on grey, faded print")
    print(f"  {path.name}")


def noisy_scan(path):
    lines = [
        "DAILY SITE REPORT 每日工地報告",
        "DATE: 2026-03-14  WEATHER: FINE",
        "MANPOWER: 34   PLANT: 6",
        "INCIDENT: NIL 無事故",
    ]
    f = font(38, must_render="".join(lines))
    if not f:
        return print("  SKIP 26_noisy_scan.jpg (font)")
    img = Image.new("RGB", (1200, 460), "white")
    d = ImageDraw.Draw(img)
    y = 60
    for ln in lines:
        d.text((70, y), ln, fill=(35, 35, 35), font=f)
        y += 90
    img = img.rotate(-2.2, expand=True, fillcolor="white", resample=Image.BICUBIC)
    img = add_speckle(img, 0.05)
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    save_jpeg(img, path, quality=35)
    record(path.name, lines, "bad photocopy: skew + speckle + heavy JPEG")
    print(f"  {path.name}")


def complex_pdf(path, srcs):
    pages = []
    for s in srcs:
        if not s.exists():
            continue
        im = Image.open(s).convert("RGB")
        canvas = Image.new("RGB", (1240, 1754), "white")
        im.thumbnail((1120, 1500))
        canvas.paste(im, ((1240 - im.width) // 2, 120))
        pages.append(canvas)
    if not pages:
        return print("  SKIP 27_complex.pdf")
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:])
    merged = []
    for s in srcs:
        merged += TRUTH.get(s.name, {}).get("lines", [])
    record(path.name, merged, f"{len(pages)}-page PDF combining the samples above")
    print(f"  {path.name} ({len(pages)} pages)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUT}")
    permit_form(OUT / "20_permit_form.png")
    invoice_table(OUT / "21_invoice_table.png")
    site_photo(OUT / "22_site_photo.jpg")
    nameplate(OUT / "23_nameplate.jpg")
    two_column(OUT / "24_two_column.png")
    low_contrast(OUT / "25_low_contrast.png")
    noisy_scan(OUT / "26_noisy_scan.jpg")
    complex_pdf(OUT / "27_complex.pdf",
                [OUT / "20_permit_form.png", OUT / "21_invoice_table.png",
                 OUT / "24_two_column.png"])

    gt = OUT / "ground_truth.json"
    gt.write_text(json.dumps(TRUTH, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nground truth -> {gt}")
    print("score them with:  python scripts/score_samples.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
