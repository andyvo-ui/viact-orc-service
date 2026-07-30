#!/usr/bin/env bash
# Test the doc lane (PaddleOCR-VL on vLLM) DIRECTLY on :8118.
#
#   ./scripts/test_doclane.sh <image-or-pdf-page.jpg> [host:port]
#   ./scripts/test_doclane.sh tests/samples/03_mixed_tc_en.png 172.16.1.22:8118
#
# Why not through the gateway: `paddleocr genai_server --backend vllm` runs vLLM's
# OpenAI-compatible server, so the doc lane speaks /v1/chat/completions. It has no
# /parse endpoint, which means the gateway's /parse proxy 404s against it. Until
# that is resolved, this is how you exercise the model.

set -uo pipefail
IMG="${1:-}"
ADDR="${2:-localhost:8118}"
[ -f "$IMG" ] || { echo "usage: $0 <image> [host:port]"; exit 2; }

BASE="http://${ADDR}"
REQ=/tmp/_doclane_req.json

echo "== 1. is it alive =="
code=$(curl -s -o /tmp/_dl_health -w '%{http_code}' --max-time 10 "${BASE}/health")
echo "  GET /health -> HTTP ${code}"
if [ "$code" != "200" ]; then
  echo "  doc lane not answering. Checks:"
  echo "    docker compose ps"
  echo "    docker compose logs --tail=50 doc-lane"
  echo "  If it is 'up' but unreachable, confirm --host 0.0.0.0 is in the compose"
  echo "  command: genai_server defaults to localhost and binds inside the container."
  exit 1
fi

echo "== 2. what model name does it serve =="
MODEL=$(curl -s --max-time 10 "${BASE}/v1/models" \
  | python3 -c 'import sys,json
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("")' 2>/dev/null)
if [ -z "$MODEL" ]; then
  echo "  could not read /v1/models; falling back to PaddleOCR-VL-1.6-0.9B"
  MODEL="PaddleOCR-VL-1.6-0.9B"
fi
echo "  served-model-name = ${MODEL}"

echo "== 3. VRAM before =="
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || echo "  (no nvidia-smi here)"

echo "== 4. OCR ${IMG} =="
# Build the request with python so base64 never goes through argv (a large image
# would blow past ARG_MAX as a shell argument).
python3 - "$IMG" "$MODEL" "$REQ" <<'PY'
import base64, json, mimetypes, sys
img, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
mime = mimetypes.guess_type(img)[0] or "image/jpeg"
b64 = base64.b64encode(open(img, "rb").read()).decode()
json.dump({
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": "OCR:"},
    ]}],
    "max_tokens": 4096,
    "temperature": 0.0,
}, open(out, "w"))
PY

start=$(date +%s)
code=$(curl -s -o /tmp/_dl_resp -w '%{http_code}' --max-time 300 \
  -H 'Content-Type: application/json' -d @"$REQ" \
  "${BASE}/v1/chat/completions")
elapsed=$(( $(date +%s) - start ))
echo "  HTTP ${code} in ${elapsed}s"

python3 - <<'PY'
import json
try:
    d = json.load(open("/tmp/_dl_resp"))
except Exception:
    print("  --- raw response ---")
    print(open("/tmp/_dl_resp").read()[:800]); raise SystemExit
if "choices" in d:
    print("  --- text returned ---")
    print(d["choices"][0]["message"]["content"][:4000])
    u = d.get("usage") or {}
    if u:
        print(f"\n  tokens: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
else:
    print("  --- error ---")
    print(json.dumps(d, ensure_ascii=False, indent=2)[:1200])
PY

echo "== 5. VRAM after =="
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
echo
echo "Doc lane should sit around 5GB. If total used jumped by ~16GB, gpu-memory-utilization"
echo "is not being applied - stop it NOW (docker compose stop doc-lane) before it starves Qwen."
