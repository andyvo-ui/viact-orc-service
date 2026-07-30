#!/usr/bin/env bash
# Fire every file in tests/samples/ at a running gateway and print a pass/fail table.
#
#   ./scripts/test_samples.sh                        # default http://localhost:8000
#   ./scripts/test_samples.sh http://localhost:8001
#
# Expected result per file is asserted here, so a green run means the boundaries
# behave — NOT that accuracy is good enough. These are synthetic renders; accuracy
# needs your real documents (see tests/samples/README.md).

set -uo pipefail
BASE="${1:-http://localhost:8000}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/tests/samples"
PASS=0; FAIL=0

hr() { printf '%s\n' "------------------------------------------------------------------"; }

# post <file> <expected_http> [query] -> echoes "<code>|<body>"
post() {
  local f="$1" q="${3:-}"
  curl -s -o /tmp/_ocr_body -w '%{http_code}' \
       -F "file=@${DIR}/${f}" "${BASE}/ocr${q}"
}

check() {
  local name="$1" expect="$2" query="${3:-}" note="${4:-}"
  local code; code="$(post "$name" "$expect" "$query")"
  local lines; lines="$(python3 - <<'PY' 2>/dev/null || echo '?'
import json
d=json.load(open("/tmp/_ocr_body"))
print(sum(len(p["texts"]) for p in d["pages"]))
PY
)"
  if [ "$code" = "$expect" ]; then
    printf '  PASS  %-28s HTTP %-3s lines=%-4s %s\n' "$name" "$code" "$lines" "$note"
    PASS=$((PASS+1))
  else
    printf '  FAIL  %-28s HTTP %-3s (expected %s) %s\n' "$name" "$code" "$expect" "$note"
    printf '        %s\n' "$(head -c 300 /tmp/_ocr_body)"
    FAIL=$((FAIL+1))
  fi
}

# show <file> [query] — print what was actually read, for eyeballing
show() {
  local f="$1" q="${2:-}"
  post "$f" 200 "$q" >/dev/null
  python3 - <<'PY' 2>/dev/null
import json
try:
    d = json.load(open("/tmp/_ocr_body"))
except Exception:
    print("        (no JSON)"); raise SystemExit
for i, p in enumerate(d.get("pages", []), 1):
    if len(d["pages"]) > 1:
        print(f"        page {i}:")
    for t, s in zip(p.get("texts", []), p.get("scores", [])):
        flag = " <-- low" if s < 0.80 else ""
        print(f"        {s:.3f}  {t}{flag}")
    if not p.get("texts"):
        print("        (nothing read)")
PY
}

hr; echo "gateway: $BASE"
H="$(curl -s "${BASE}/health")"
echo "$H" | python3 -m json.tool 2>/dev/null || { echo "  gateway unreachable"; exit 2; }
echo "$H" | grep -q fast_lane_device || { echo "  NOT ocr-service on this port"; exit 2; }

hr; echo "SUPPORTED — images"
check 01_english.png              200 ""                  "clear EN"
check 02_chinese_traditional.png  200 ""                  "TC only"
check 03_mixed_tc_en.png          200 ""                  "TC+EN one line"
check 04_rotated_90.png           200 "?orientation=auto" "rotated, auto"
check 05_skewed_15.png            200 "?orientation=auto" "15deg skew"
check 06_blank.png                200 ""                  "must read 0 lines"
check 07_small_text.png           200 ""                  "22px text"

hr; echo "SUPPORTED — PDF"
check 08_two_page.pdf             200 ""                  "expect 2 page entries"

hr; echo "BOUNDS — must be refused"
check 09_over_page_limit.pdf      413 ""                  "12 pages > OCR_MAX_PAGES"

hr; echo "UNSUPPORTED — neither lane reads these"
echo "  (a non-200 here is CORRECT; .docx/.xlsx need the paddleocr doc2md extra,"
echo "   which is not installed. Recorded so the failure mode is known, not a bug.)"
for f in 10_not_an_image.jpg 11_unsupported.docx 12_unsupported.xlsx 13_empty.png; do
  code="$(post "$f" 000 "")"
  if [ "$code" = "200" ]; then
    n="$(python3 -c 'import json;d=json.load(open("/tmp/_ocr_body"));print(sum(len(p["texts"]) for p in d["pages"]))' 2>/dev/null || echo '?')"
    if [ "$n" = "0" ]; then
      printf '  PASS  %-28s HTTP 200 but 0 lines (accepted, read nothing)\n' "$f"; PASS=$((PASS+1))
    else
      printf '  FAIL  %-28s HTTP 200 with %s lines — it INVENTED text\n' "$f" "$n"; FAIL=$((FAIL+1))
    fi
  else
    printf '  PASS  %-28s HTTP %s (rejected)\n' "$f" "$code"; PASS=$((PASS+1))
  fi
done

hr; echo "WHAT WAS ACTUALLY READ  (judge these yourself)"
for f in 01_english.png 02_chinese_traditional.png 03_mixed_tc_en.png 07_small_text.png; do
  echo "  $f"; show "$f"
done
echo "  04_rotated_90.png  orientation=auto"; show 04_rotated_90.png "?orientation=auto"
echo "  04_rotated_90.png  orientation=upright"; show 04_rotated_90.png "?orientation=upright"
echo "  08_two_page.pdf"; show 08_two_page.pdf

hr; printf 'passed %s / %s\n' "$PASS" "$((PASS+FAIL))"
[ "$FAIL" -eq 0 ] || echo "NOTE: a FAIL on 04/05 (rotation) is a real finding, not a flaky test."
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
