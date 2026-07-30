# OCR Service — Testing Guide

Written so anyone can test it without needing to read code.

---

## Server

| | |
|---|---|
| Address | `http://172.16.1.22:8000` |
| Requires | Company **VPN** connection |
| Status | Under test, not yet released |

Check your connection first — open this link in a browser:

```
http://172.16.1.22:8000/health
```

If you see text containing `"status": "ok"`, you are connected. If nothing loads,
check your VPN.

---

## Option 1 — Test in a browser (easiest, nothing to install)

**Step 1.** Open your browser and go to:

```
http://172.16.1.22:8000/docs
```

**Step 2.** Click on the line **`POST /ocr`** to expand it.

**Step 3.** Click the **`Try it out`** button on the right.

**Step 4.** Next to `file`, click **Choose File** and pick the image or PDF you want
to read.

**Step 5.** In the `orientation` box, choose:
- **`auto`** — site photos, signage shot at an angle, tilted text
- **`upright`** — flat scans shot straight on (slightly faster)
- Leaving it blank is fine; the system uses `auto` by default

**Step 6.** Click the blue **`Execute`** button.

The result appears just below, under **Response body**.

---

## Option 2 — Send a file from the command line

```bash
curl -F "file=@path/to/image.jpg" http://172.16.1.22:8000/ocr
```

For a photo taken at an angle:

```bash
curl -F "file=@image.jpg" "http://172.16.1.22:8000/ocr?orientation=auto"
```

---

## Reading the result

The response looks like this:

```json
{
  "pages": [
    {
      "texts":  ["緊急出口", "EMERGENCY EXIT", "GATE 3"],
      "scores": [0.991, 0.964, 0.887],
      "boxes":  [ ... ]
    }
  ]
}
```

- **`texts`** — the text that was read, one entry per line
- **`scores`** — **confidence** for each line, from 0 to 1
- **`boxes`** — where that line sits on the image (four corner coordinates)
- **`pages`** — one entry per PDF page. A normal image has just one

About `scores`:

| Value | Meaning |
|---|---|
| above 0.90 | Read with confidence |
| 0.80 – 0.90 | Probably right, worth a glance |
| below 0.80 | Doubtful, needs a human to confirm |

⚠️ **A high score does not guarantee the reading is correct.** The model can be
confidently wrong — especially with Chinese characters that differ by small strokes.
Always compare against the original image.

---

## What to test

To judge this properly, please try all of the following:

1. **Site signage and nameplates** — phone photos, including ones shot at an angle
2. **Equipment labels** — serial numbers, asset tags, small print
3. **Scanned permits** — ruled boxes, reference codes, dates
4. **Contracts or quotations containing tables**
5. **Traditional Chinese documents** — this is the primary language to evaluate
6. **Documents mixing Chinese and English on the same line**
7. **Faded photocopies, poorly lit photos** — the worst realistic case

The closer these are to what you handle day to day, the more useful the result.

---

## What is expected vs what to report

**Expected — not a defect:**
- A few wrong characters on blurry images or very small print
- Handwriting is not read
- Output comes back as separate lines and **does not preserve table shape** — this is
  a known limitation of the current version
- A blank image returns an empty result

**Please report these:**
- Text returned that **does not exist** in the image
- A **whole clear, easily readable line is missed**
- A red `500` error
- A request that never returns

---

## Current limitations

| File type | Supported |
|---|---|
| Images — JPG, PNG | ✅ |
| Scanned PDF, **up to 10 pages** | ✅ |
| PDF over 10 pages | ❌ returns `413` |
| Files over 50 MB | ❌ returns `413` |
| **Word (.docx), Excel (.xlsx)** | ❌ **not supported** |

**Why Word and Excel are not supported:** OCR reads **images**. A Word or Excel file
is not an image — the text inside is already digital data. Extracting it is done with
a different tool, which is 100% accurate and needs no OCR at all.

If a Word or Excel file contains a **scanned image inside it**, convert the file to
PDF first, then submit that.

**Tables:** the current version returns individual cells as separate lines and does
not reassemble them into a table. Structure-preserving table extraction is being
evaluated separately.

---

## When reporting a problem, please include

1. **The file you tested** (or a description of the document type if it cannot be shared)
2. **The result you got** — copy the Response body
3. **What the correct text should have been**

Item 3 matters most. Without it there is no way to measure how far off the result was.
