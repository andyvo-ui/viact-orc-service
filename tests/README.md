# tests — gateway

Black-box, over HTTP, against a **running** gateway. Nothing is mocked: the parts
that are unverified (paddleocr kwargs, numpy→JSON, docker DNS to the doc lane) are
exactly the parts a mock would replace.

Run these **on the VM**, not on a laptop — they need the container up.

`tests/offline/` is the exception: no Docker, no GPU, no paddleocr, no network. See
[its conftest](offline/conftest.py) for why that directory is allowed to stub things
when this one is not.

## Run

```bash
pip install -r tests/requirements.txt

# --- anywhere, including a laptop with nothing running -----------------------
pytest tests/offline -v            # ~33 tests, under 10s

# --- on the VM, gateway up ---------------------------------------------------
docker compose up -d --build gateway

pytest -m "not slow"               # fast pass — SEE THE WARNING BELOW
OCR_TEST_DOCKER=1 pytest           # everything
pytest tests/test_boundary.py -v   # one group
```

Then look at real output with your own documents:

```bash
python scripts/report_live.py tests/fixtures/ --parse
```

That is not a test — it prints input beside output and writes
`reports/ocr-report-*.md` so you can judge accuracy yourself. It flags the two
things a status-code check cannot see: a 200 with zero text, and lines below 0.80
confidence.

Config:

| env | default | what it does |
|---|---|---|
| `OCR_BASE_URL` | `http://localhost:8000` | where the gateway is |
| `OCR_DOC_LANE_URL` | `http://localhost:8118` | probed directly to decide if the doc lane is up |
| `OCR_TEST_TIMEOUT` | `120` | per-request timeout |
| `OCR_TEST_DOCKER` | unset | enables tests that `docker compose exec` into the container |
| `OCR_TEST_CONCURRENCY` | `8` | parallel requests in C1 |
| `OCR_TEST_MIN_SPEEDUP` | `1.5` | C1 threshold; `1.0` means fully serialised |
| `OCR_TEST_HEALTH_BUDGET` | `5.0` | C2 threshold; matches the compose healthcheck timeout |

## Put a real document in `tests/fixtures/`

Several tests skip without one. Name it `sample.jpg` / `sample.png`:

```bash
cp /path/to/a/real/hk_document.jpg tests/fixtures/sample.jpg
```

**Synthetic images are not a substitute.** A clean 64px Arial render is far easier
than a phone photo of a Traditional Chinese contract. Passing on synthetic input
proves the pipeline is wired, not that the model is good enough.

## Coverage

| file | group | covers |
|---|---|---|
| `offline/test_ocr_serialisation.py` | — | the /ocr JSON contract, input→output table, no deps |
| `offline/test_parse_error_mapping.py` | — | /parse transport vs 4xx vs 5xx vs non-JSON, no deps |
| `offline/test_input_limits.py` | — | page cap + byte cap, 413 wording, no leak on rejection |
| `offline/test_orientation_routing.py` | — | `?orientation=` picks the right pipeline; engine error records raise |
| `test_smoke.py` | A | health, 200 on a real image, response shape, weights baked in |
| `test_boundary.py` | B | blank / no-extension / unicode name / fake image / PDF pages / 1px / 8000px / missing field / rotation |
| `test_concurrency.py` | C | serialisation, `/health` under load, thread-safety of the shared pipeline, RSS + temp-file growth |
| `test_failure.py` | D | doc lane down → 502 while `/ocr` stays 200, fail-fast, error mapping, disconnect leak |
| `test_idempotency.py` | G | repeat determinism, warmup effectiveness, golden snapshot |

Three tests were written to **fail against the then-current code**. Two of those
bugs have since been confirmed and fixed, so those tests are now regression guards
rather than probes:

- `test_a3*` — **FIXED.** `rec_polys` reached FastAPI as `list[np.ndarray]`, which
  `jsonable_encoder` cannot encode → 500 on any image containing text (a *blank*
  image returned 200, which is why it hid). `run_ocr` now coerces to builtins.
- `test_d3` — **FIXED.** `raise_for_status()` sat inside the `except httpx.HTTPError`
  block and `HTTPStatusError` subclasses `HTTPError`, so a doc-lane 400 became a
  gateway 502. `/parse` now maps transport / upstream-4xx / upstream-5xx separately.
- `test_c1` / `test_c2` — **STILL OPEN.** Sync `run_ocr` inside `async def ocr` blocks
  the event loop. Not fixed here; it is a design change (`run_in_threadpool`, or
  `def` instead of `async def`), not a bug fix. Do not relax the thresholds.

⚠️ `pytest -m "not slow"` **skips `c1` and `c2`**, the only two predicted bugs still
live. A green fast pass is not evidence about concurrency — run the full suite.

## Groups D and G that need a manual step

```bash
# D1/D2 — the whole point of having no depends_on
docker compose stop doc-lane
pytest tests/test_failure.py -v -m doclane_down
docker compose start doc-lane

# D4 — timeout behaviour
DOC_LANE_TIMEOUT=5 docker compose up -d gateway
#   then send a large PDF to /parse; expect 502 at ~5s, not a hang
docker compose up -d gateway            # restore

# D5 — hard kill mid-request
docker kill -s SIGKILL ocr-gateway && docker compose up -d gateway

# D7 — restart must not need network (weights are baked in)
docker network disconnect ocr-service_default ocr-gateway
docker compose restart gateway && curl -F "file=@tests/fixtures/sample.jpg" localhost:8000/ocr
docker network connect ocr-service_default ocr-gateway

# G3 — output stable across a restart
pytest tests/test_idempotency.py::test_g4_golden_snapshot
docker compose restart gateway
pytest tests/test_idempotency.py::test_g4_golden_snapshot
```

## Not automated here — deliberately

**Group E (resources).** Needs `nvidia-smi` on the host and the LLM running
alongside. Asserting VRAM numbers from inside pytest would encode a machine state
that changes daily.

```bash
docker compose up -d doc-lane && sleep 120
nvidia-smi                       # doc lane should be ~5GB, NOT ~28GB
                                 # ~28GB => GPU_MEMORY_UTILIZATION is not forwarded (SETUP.md #3)
docker stats --no-stream
docker image ls ocr-service-gateway
```

**Group F (exposure).** A decision, not a bug. Port 8000 binds `0.0.0.0` with no
auth, and 8118 is published too.

```bash
curl http://<VM-IP>:8000/health          # from another machine — succeeds today
```

Fix, if you decide it needs one: bind `127.0.0.1:8000:8000` and put a reverse
proxy in front, or drop `doc-lane`'s `ports:` entirely since only the gateway
calls it.

**Group H (accuracy).** This is the one that decides whether the service is usable,
and it is the one I cannot write. It needs 30–50 real HK documents with ground
truth, scored per category (printed contract / handwriting / signage / tables) and
split TC vs EN. `test_g4_golden_snapshot` only detects *change*, never *correctness*.

`scripts/report_live.py` is the closest thing to a harness for it: point it at a
directory of real documents and it produces a markdown table of input → extracted
text → confidence. It supplies the *output* side; you still supply the ground truth.

## Writing more tests

[`docs/TEST_AUTHORING_PROMPT.md`](../docs/TEST_AUTHORING_PROMPT.md) is a brief you
can hand to any model. It enumerates cases by failure mode rather than by
imagination, forbids inventing status codes for undecided behaviour, and requires a
"cases I did NOT cover" section — that section is where your domain knowledge enters,
and it is the part worth reading first.

## Blind spots in this suite

I derived these cases from the same reading of the code that produced the bug
predictions above — so my testing blind spots correlate with my reading blind spots.
Known gaps:

- **No ground truth.** Every accuracy-shaped test is either `xfail` or a print.
- **`doc_lane_up` assumes port 8118 is reachable from wherever pytest runs.** It is
  probed directly rather than through the gateway, because the gateway's error
  space cannot distinguish "doc lane down" from "doc lane rejected this file" —
  the old in-band probe made every `doclane_down` test pass while the doc lane was
  UP. If pytest and the gateway are on different hosts, set `OCR_DOC_LANE_URL`.
- **No real input profile.** B10/B11 (rotation) assert nothing hard because I do not
  know whether your input is flat scans or site photos. That single fact flips
  SETUP.md #5 and changes which failures matter.
- **QPS is invented.** `OCR_TEST_CONCURRENCY=8` is a guess.
- **`/parse` has no schema assertion** — SETUP.md #2 is still open, so `test_d3b`
  only prints the response keys. Write the real contract once you have seen one.
- **No PDF-with-text-layer case.** A digitally generated PDF and a scanned PDF are
  different inputs; I do not know which you get.
- **No multi-language-in-one-line case** automated — needs real fixtures.
- **Nothing tests the doc lane directly** on `:8118`. This suite is gateway-only,
  as asked.
