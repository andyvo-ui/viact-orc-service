"""Score the gateway against known ground truth. Prints real accuracy numbers.

    python scripts/score_samples.py                        # localhost:8000
    python scripts/score_samples.py --url http://172.16.1.22:8000
    python scripts/score_samples.py --orientation auto

Reads tests/samples/hard/ground_truth.json — written by make_hard_samples.py, which
knows exactly what text it drew — so no human transcription is needed.

THREE NUMBERS, because "accuracy" alone hides the failure that matters:

  char accuracy  how many characters are right, on the lines it DID find.
                 High here with low recall means "reads well, but misses things".
  recall         how much of the real text it found at all. A line never detected
                 is invisible in char accuracy - this is the silent failure.
  spurious       lines returned that match nothing real. Invented text.

A run is only good if all three are good. Optimising one alone is easy and useless.

LIMIT: these are rendered documents with simulated degradation. They cannot fake
paper texture, motion blur, ink bleed, stamps over text, or handwriting. Treat the
number as a REGRESSION BASELINE, not as the business answer - that still needs real
documents scored the same way (--dir tests/fixtures with your own ground_truth.json).
"""

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "tests" / "samples" / "hard"


def normalise(s: str) -> str:
    """Fold away differences nobody would call an OCR error.

    NFKC collapses full-width forms (２０２６ vs 2026) and CJK compatibility
    variants; without it the score punishes correct readings. Whitespace is
    collapsed because line-wrapping is a layout choice, not a reading error.
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_one(truth_lines, ocr_lines, *, match_threshold=0.45):
    """Greedy best-match between expected and returned lines.

    Greedy rather than optimal assignment: with a handful of lines the difference
    is negligible, and a wrong pairing shows up as a low char score either way.
    """
    truth = [normalise(t) for t in truth_lines if normalise(t)]
    got = [normalise(t) for t in ocr_lines if normalise(t)]
    unused = list(range(len(got)))

    total_chars = sum(len(t) for t in truth)
    total_dist = 0
    matched, missed = [], []

    for t in truth:
        best_i, best_r = None, 0.0
        for i in unused:
            r = similarity(t, got[i])
            if r > best_r:
                best_i, best_r = i, r
        if best_i is not None and best_r >= match_threshold:
            unused.remove(best_i)
            total_dist += edit_distance(t, got[best_i])
            matched.append((t, got[best_i], best_r))
        else:
            total_dist += len(t)
            missed.append(t)

    spurious = [got[i] for i in unused]
    char_acc = 1 - (total_dist / total_chars) if total_chars else 1.0
    recall = len(matched) / len(truth) if truth else 1.0
    return {
        "char_accuracy": max(0.0, char_acc),
        "recall": recall,
        "matched": matched,
        "missed": missed,
        "spurious": spurious,
        "n_truth": len(truth),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--orientation", choices=("auto", "upright"), default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every matched line, not just problems")
    args = ap.parse_args()

    d = Path(args.dir)
    gt_path = d / "ground_truth.json"
    if not gt_path.exists():
        sys.exit(f"no ground truth at {gt_path}\nrun: python scripts/make_hard_samples.py")
    truth = json.loads(gt_path.read_text(encoding="utf-8"))
    params = {} if args.orientation is None else {"orientation": args.orientation}

    with httpx.Client(base_url=args.url, timeout=args.timeout) as c:
        try:
            h = c.get("/health", timeout=10).json()
        except Exception as e:
            sys.exit(f"gateway unreachable at {args.url}: {e}")
        if "fast_lane_device" not in h:
            sys.exit(f"{args.url} is not ocr-service: {h}")
        print(f"gateway {args.url}  device={h['fast_lane_device']}  "
              f"orientation={args.orientation or h.get('default_orientation','?')+' (default)'}")
        print("=" * 78)

        rows, agg_dist, agg_chars, agg_found, agg_truth, agg_spur = [], 0, 0, 0, 0, 0
        for name in sorted(truth):
            f = d / name
            if not f.exists():
                print(f"  {name:<26} MISSING FILE"); continue
            try:
                r = c.post("/ocr", files={"file": (name, f.read_bytes(),
                                                   "application/octet-stream")},
                           params=params)
            except Exception as e:
                print(f"  {name:<26} REQUEST FAILED: {e}"); continue
            if r.status_code != 200:
                print(f"  {name:<26} HTTP {r.status_code}  {r.text[:120]}")
                continue
            lines = [t for p in r.json().get("pages", []) for t in p.get("texts", [])]
            s = score_one(truth[name]["lines"], lines)

            tl = [normalise(x) for x in truth[name]["lines"] if normalise(x)]
            agg_chars += sum(len(x) for x in tl)
            agg_dist += round((1 - s["char_accuracy"]) * sum(len(x) for x in tl))
            agg_found += len(s["matched"]); agg_truth += s["n_truth"]
            agg_spur += len(s["spurious"])
            rows.append((name, s, truth[name].get("note", "")))

        for name, s, note in rows:
            flag = "  " if s["char_accuracy"] >= 0.9 and s["recall"] >= 0.9 else "！ "
            print(f"{flag}{name:<26} char {s['char_accuracy']*100:5.1f}%  "
                  f"found {len(s['matched'])}/{s['n_truth']}  "
                  f"invented {len(s['spurious'])}   {note}")
            for t in s["missed"]:
                print(f"      MISSED   {t[:70]}")
            for t in s["spurious"]:
                print(f"      INVENTED {t[:70]}")
            for t, g, r in s["matched"]:
                if args.verbose or r < 0.97:
                    print(f"      expect   {t[:70]}")
                    print(f"      got      {g[:70]}")

        print("=" * 78)
        if agg_chars:
            print(f"OVERALL   char accuracy {100*(1-agg_dist/agg_chars):5.1f}%   "
                  f"recall {100*agg_found/max(1,agg_truth):5.1f}%   "
                  f"invented lines {agg_spur}")
        print("\nRegression baseline only. Real HK documents, scored the same way, are")
        print("what decides whether the model is good enough. See the docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
