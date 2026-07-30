"""Generate the synthetic test files in tests/samples/.

Run it to regenerate; the outputs are committed so nobody has to.

    python scripts/make_samples.py

WHY THESE ARE SYNTHETIC AND WHY THAT IS A LIMIT
A clean font render is far easier than a phone photo of a Traditional Chinese
contract. These files prove the PIPELINE is wired and that each boundary behaves —
they prove nothing about accuracy on your customer's documents. Real documents go in
tests/fixtures/, which is gitignored precisely so they never reach git history.

Two directories, deliberately separate:
    tests/samples/   synthetic, committed, safe to share  <- this script
    tests/fixtures/  real customer documents, gitignored
"""

import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "tests" / "samples"

# Arial has no CJK coverage, so a Traditional Chinese sample rendered with it comes
# out as empty boxes and the test becomes a false negative. Fall back in order.
LATIN_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
# Note the (path, index) pairs. Songti.ttc face 0 is a SUBSET that silently drops
# most of these glyphs — PIL renders nothing and raises nothing, so the fixture ends
# up not containing the text it claims to. Face 1 is fine. Never trust a CJK font
# without checking coverage: see require_glyphs().
CJK_FONTS = [
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 1),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/arphic/uming.ttc", 0),
]


def require_glyphs(f, text: str) -> list[str]:
    """Characters `f` cannot draw. PIL substitutes nothing and warns nothing, so a
    missing glyph becomes a silently wrong fixture rather than an error."""
    missing = []
    for ch in set(text):
        if ch.isspace():
            continue
        try:
            if f.getmask(ch).getbbox() is None:
                missing.append(ch)
        except Exception:
            missing.append(ch)
    return missing


def font(size: int, *, cjk: bool = False, must_render: str = ""):
    candidates = CJK_FONTS if cjk else [(p, 0) for p in LATIN_FONTS]
    for path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            f = ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
        missing = require_glyphs(f, must_render) if must_render else []
        if missing:
            print(f"    skip {Path(path).name}#{index}: no glyph for {''.join(missing)}")
            continue
        return f
    return None


def text_image(path: Path, lines, *, size=(1000, 400), rotate=0, cjk=False, fsize=56):
    f = font(fsize, cjk=cjk, must_render="".join(lines))
    if f is None:
        print(f"  SKIP {path.name} — no font covers {'CJK ' if cjk else ''}{lines}")
        return False
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        draw.text((50, y), line, fill="black", font=f)
        y += int(fsize * 1.6)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor="white")
    img.save(path)
    print(f"  {path.name}")
    return True


def pdf(path: Path, pages: int, *, size=(1240, 1754)):
    f = font(72)
    if f is None:
        print(f"  SKIP {path.name} — no latin font")
        return False
    imgs = []
    for i in range(pages):
        img = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(img)
        d.text((90, 140), f"PAGE {i + 1} OF {pages}", fill="black", font=f)
        d.text((90, 260), f"DOC REF HK-{2026}-{i + 1:03d}", fill="black", font=f)
        imgs.append(img)
    imgs[0].save(path, "PDF", save_all=True, append_images=imgs[1:])
    print(f"  {path.name} ({pages} pages)")
    return True


def minimal_docx(path: Path, text: str):
    """A valid .docx written with zipfile only — no python-docx needed.

    Deliberately real rather than a renamed .txt: the point is that /ocr rejects a
    genuine Office file, not that it rejects garbage bytes.
    """
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", body)
    print(f"  {path.name}")


def minimal_xlsx(path: Path, rows):
    """A valid .xlsx written with zipfile only — no openpyxl needed."""
    def cell(col, row, value):
        return f'<c r="{col}{row}" t="inlineStr"><is><t>{value}</t></is></c>'

    xml_rows = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(cell(chr(ord("A") + c), r, v) for c, v in enumerate(row))
        xml_rows.append(f'<row r="{r}">{cells}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    print(f"  {path.name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUT}")

    # --- supported: images -------------------------------------------------- #
    text_image(OUT / "01_english.png", ["INVOICE 2026", "NO ENTRY", "PANEL P-04B"])
    text_image(OUT / "02_chinese_traditional.png",
               ["停車場", "危險 請勿進入", "總金額 HK$1,240"], cjk=True)
    text_image(OUT / "03_mixed_tc_en.png",
               ["緊急出口 EMERGENCY EXIT", "電錶房 METER ROOM B2", "SITE REF HK-2026-017"],
               cjk=True, size=(1200, 400))
    text_image(OUT / "04_rotated_90.png", ["SITE ENTRANCE", "GATE 3"], rotate=90)
    text_image(OUT / "05_skewed_15.png", ["SITE ENTRANCE", "GATE 3"], rotate=15)
    text_image(OUT / "06_blank.png", [], size=(800, 300))
    text_image(OUT / "07_small_text.png",
               ["SERIAL 8842-XK-01", "CAL DUE 2026-11-30"], fsize=22, size=(700, 200))

    # --- supported: PDF ------------------------------------------------------ #
    pdf(OUT / "08_two_page.pdf", 2)
    pdf(OUT / "09_over_page_limit.pdf", 12)  # > OCR_MAX_PAGES default of 10

    # --- rejected on purpose ------------------------------------------------- #
    (OUT / "10_not_an_image.jpg").write_bytes(b"This is plainly not an image.\n" * 8)
    minimal_docx(OUT / "11_unsupported.docx", "Neither lane can read a .docx.")
    minimal_xlsx(OUT / "12_unsupported.xlsx",
                 [["item", "qty", "unit"], ["anchor bolt", "120", "pcs"]])
    (OUT / "13_empty.png").write_bytes(b"")

    print("\ndone. Expected results per file: tests/samples/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
