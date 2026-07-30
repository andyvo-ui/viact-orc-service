"""Send real files to a RUNNING gateway and print input -> output for eyeballing.

This is not a test. It asserts almost nothing, on purpose: the point is to put the
input and the output side by side so a human can judge whether the OCR is good
enough. Only a human with the source documents can do that, and it is the one
question the whole test suite cannot answer (tests/README.md, "Group H").

    # on the server, gateway already up
    python scripts/report_live.py tests/fixtures/                  # a directory
    python scripts/report_live.py a.jpg b.png contract.pdf         # explicit files
    python scripts/report_live.py --parse docs/                    # doc lane too
    python scripts/report_live.py --url http://localhost:8001 x.jpg

Writes reports/ocr-report-<label>.md and .json next to the repo root, so the
markdown can be pasted into a ticket and the JSON diffed run over run.

Exit code is 0 unless a request failed outright. A bad *reading* is not an error
here — it is a finding for you to read.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DOC_SUFFIXES = {".pdf"}


def collect(paths: list[str], *, want_docs: bool) -> list[Path]:
    wanted = IMAGE_SUFFIXES | (DOC_SUFFIXES if want_docs else set())
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files += sorted(c for c in p.rglob("*") if c.suffix.lower() in wanted)
        elif p.is_file():
            files.append(p)
        else:
            print(f"  ! skipping {p} (not found)", file=sys.stderr)
    return files


def probe_health(client: httpx.Client) -> dict:
    resp = client.get("/health", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if "fast_lane_device" not in body:
        raise SystemExit(
            f"/health returned {body} — that is not ocr-service. Wrong --url or port?"
        )
    return body


def call(client: httpx.Client, endpoint: str, path: Path, *, params=None) -> dict:
    started = time.perf_counter()
    try:
        resp = client.post(
            endpoint,
            files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
            params=params or {},
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "status": None, "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}"}
    out = {"ok": resp.status_code == 200, "status": resp.status_code,
           "seconds": round(time.perf_counter() - started, 2)}
    try:
        out["body"] = resp.json()
    except ValueError:
        out["body"] = None
        out["error"] = resp.text[:500]
    else:
        # A 4xx from this service carries its reason in `detail` and IS valid JSON,
        # so the except branch never runs for it. Without this, a 413's explanation
        # was dropped and the report printed "None" where the reason should be.
        if not out["ok"]:
            body = out["body"]
            detail = body.get("detail") if isinstance(body, dict) else None
            out["error"] = detail if detail else resp.text[:500]
    return out


def summarise_ocr(body) -> dict:
    """Flatten an /ocr response into the few numbers worth reading at a glance."""
    pages = (body or {}).get("pages") or []
    texts, scores = [], []
    for page in pages:
        texts += page.get("texts") or []
        scores += page.get("scores") or []
    return {
        "pages": len(pages),
        "lines": len(texts),
        "chars": sum(len(t) for t in texts),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "min_score": round(min(scores), 4) if scores else None,
        "texts": texts,
    }


def image_dims(path: Path):
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            return f"{img.width}x{img.height}"
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="files and/or directories")
    ap.add_argument("--url", default="http://localhost:8000", help="gateway base URL")
    ap.add_argument("--parse", action="store_true",
                    help="ALSO send PDFs to /parse, so you can compare flat text "
                         "against the doc lane's structured output on the same file")
    ap.add_argument("--skip-pdf", action="store_true",
                    help="do not send PDFs to /ocr at all (images only)")
    ap.add_argument("--orientation", choices=("auto", "upright"), default=None,
                    help="which fast-lane pipeline to use; omit for the server default. "
                         "auto = site photos with rotated text, upright = flat scans")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--label", default="run", help="suffix for the report filenames")
    ap.add_argument("--low-score", type=float, default=0.80,
                    help="flag lines below this confidence in the markdown")
    args = ap.parse_args()

    files = collect(args.paths, want_docs=not args.skip_pdf)
    if not files:
        return print("no input files found") or 1

    ocr_params = {} if args.orientation is None else {"orientation": args.orientation}

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        health = probe_health(client)
        limits = health.get("limits") or {}
        print(f"gateway {args.url}  device={health['fast_lane_device']}  files={len(files)}")
        print(f"  fast-lane limits: max_pages={limits.get('max_pages', '?')} "
              f"max_upload_bytes={limits.get('max_upload_bytes', '?')}")
        print(f"  orientation: {args.orientation or health.get('default_orientation', '?') + ' (server default)'}")
        # PDFs go to /ocr on purpose — a scanned PDF is a photo in a wrapper. But a
        # PDF where the STRUCTURE matters belongs on /parse, and /ocr will return 200
        # while silently flattening it. Say so once rather than letting the report
        # imply the fast lane is the right tool for every PDF.
        pdfs = [p for p in files if p.suffix.lower() in DOC_SUFFIXES]
        if pdfs and not args.parse:
            print(f"  note: {len(pdfs)} PDF(s) going to /ocr only. /ocr returns FLAT "
                  f"text — no tables, no reading order.")
            print(f"        add --parse to compare against the doc lane, or --skip-pdf "
                  f"to exclude them.")
        print()

        records = []
        for i, path in enumerate(files, 1):
            rec = {
                "input": {
                    "name": path.name,
                    "path": str(path),
                    "kb": round(path.stat().st_size / 1024, 1),
                    "dims": image_dims(path),
                    "suffix": path.suffix.lower(),
                }
            }
            ocr = call(client, "/ocr", path, params=ocr_params)
            rec["ocr"] = ocr
            rec["input"]["orientation"] = args.orientation or "(server default)"
            rec["ocr_summary"] = summarise_ocr(ocr.get("body")) if ocr["ok"] else None

            if args.parse and path.suffix.lower() in DOC_SUFFIXES:
                rec["parse"] = call(client, "/parse", path)

            s = rec["ocr_summary"]
            if s:
                head = f"{s['lines']:>3} lines  {s['chars']:>5} chars  score~{s['mean_score']}"
            elif ocr.get("status") == 413:
                # Not a failure of the model — a bound doing its job. Say which one.
                head = f"REFUSED 413 (over a fast-lane limit) {str(ocr.get('error'))[:80]}"
            else:
                head = f"ERROR {ocr.get('status')} {str(ocr.get('error'))[:60]}"
            print(f"[{i}/{len(files)}] {path.name:<40} {ocr['seconds']:>6.2f}s  {head}")
            if s and s["texts"]:
                for line in s["texts"][:5]:
                    print(f"          | {line}")
                if len(s["texts"]) > 5:
                    print(f"          | ... {len(s['texts']) - 5} more lines")
            records.append(rec)

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f"ocr-report-{args.label}"

    payload = {"gateway": args.url, "health": health, "files": len(files), "records": records}
    stem.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(
        render_markdown(payload, low_score=args.low_score), encoding="utf-8"
    )

    # A 4xx is the service refusing input on purpose (too many pages, too large) —
    # that is a working limit, not a broken service. Only 5xx and transport errors
    # are failures, and only those set a non-zero exit code.
    ok = [r for r in records if r["ocr"]["ok"]]
    refused = [r for r in records
               if not r["ocr"]["ok"] and (r["ocr"]["status"] or 500) < 500]
    failed = [r for r in records
              if not r["ocr"]["ok"] and (r["ocr"]["status"] or 500) >= 500]
    empty = [r for r in records if r["ocr_summary"] and r["ocr_summary"]["lines"] == 0]

    print(f"\nreport: {stem.with_suffix('.md')}")
    print(f"        {stem.with_suffix('.json')}")
    print(f"\n{len(ok)}/{len(records)} returned 200"
          f"{f', {len(refused)} refused (4xx)' if refused else ''}"
          f"{f', {len(failed)} FAILED (5xx/transport)' if failed else ''}"
          f"{f', {len(empty)} returned zero text' if empty else ''}")
    if empty:
        print("  zero-text results are the ones to look at first — a 200 with no text")
        print("  is a silent failure and looks identical to success in a status check.")
    if refused:
        print("  refused files hit a fast-lane bound. Route long/structured documents")
        print("  to /parse, or raise OCR_MAX_PAGES / OCR_MAX_UPLOAD_BYTES if the limit")
        print("  is genuinely too tight for your real inputs.")
    return 1 if failed else 0


def render_markdown(payload, *, low_score: float) -> str:
    lines = [
        "# OCR live report",
        "",
        f"- gateway: `{payload['gateway']}`",
        f"- fast lane device: `{payload['health'].get('fast_lane_device')}`",
        f"- files: {payload['files']}",
        "",
        "Read the **Extracted** column against the source document. A row with a 200",
        "and no text is a silent failure. Scores are the model's own confidence and",
        "are not accuracy — a confidently wrong character scores high.",
        "",
        "| # | file | size | dims | status | secs | pages | lines | mean score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, rec in enumerate(payload["records"], 1):
        inp, ocr, s = rec["input"], rec["ocr"], rec["ocr_summary"]
        lines.append(
            f"| {i} | `{inp['name']}` | {inp['kb']} KB | {inp['dims'] or '—'} "
            f"| {ocr['status'] or 'ERROR'} | {ocr['seconds']} "
            f"| {s['pages'] if s else '—'} | {s['lines'] if s else '—'} "
            f"| {s['mean_score'] if s else '—'} |"
        )

    lines += ["", "## Extracted text", ""]
    for i, rec in enumerate(payload["records"], 1):
        inp, ocr, s = rec["input"], rec["ocr"], rec["ocr_summary"]
        lines.append(f"### {i}. {inp['name']}")
        lines.append("")
        if not s:
            status = ocr["status"] or 0
            label = "REFUSED" if 400 <= status < 500 else "FAILED"
            lines += [f"**{label}** — HTTP {ocr['status']}", "", "```",
                      str(ocr.get("error"))[:1000], "```", ""]
            continue
        if not s["texts"]:
            lines += ["**200 OK but zero text extracted.** Silent failure — check "
                      "whether the document really has machine-readable text, then "
                      "whether orientation or resolution is the cause.", ""]
            continue
        body = (rec["ocr"]["body"] or {}).get("pages") or []
        for pno, page in enumerate(body, 1):
            if len(body) > 1:
                lines.append(f"**page {pno}**")
                lines.append("")
            lines.append("| conf | text |")
            lines.append("|---|---|")
            for text, score in zip(page.get("texts") or [], page.get("scores") or []):
                flag = " ⚠️" if score < low_score else ""
                safe = text.replace("|", "\\|")
                lines.append(f"| {score:.3f}{flag} | {safe} |")
            lines.append("")
        if rec.get("parse"):
            p = rec["parse"]
            lines += [f"**/parse** → HTTP {p['status']}", "", "```json",
                      json.dumps(p.get("body"), ensure_ascii=False, indent=2)[:2000],
                      "```", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
