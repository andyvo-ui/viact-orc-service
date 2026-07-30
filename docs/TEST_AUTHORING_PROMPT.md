# Prompt: write a test file for ocr-service

A reusable brief for handing to Claude / Gemini / any model when you want new tests
for this repo. Copy **Part 2** verbatim as the prompt, fill in the `<<>>` slots, and
attach the files listed in Part 1.

Why a document and not just an ad-hoc ask: the value of an LLM-written test suite
lives almost entirely in *which cases get enumerated*. Left to itself a model writes
the happy path plus two obvious edge cases, and it is pulled toward tests that pass.
This brief forces enumeration by failure mode, forbids the shortcuts, and requires
the model to declare what it did not cover.

---

## Part 1 — what to attach

Always:

- `gateway/main.py`, `fast_lane/ocr_engine.py` — the code under test
- `tests/shape.py` — the response contract, so a new test reuses it instead of
  inventing a second one
- `tests/conftest.py` **or** `tests/offline/conftest.py` — whichever suite the new
  file belongs to; fixtures must be reused, not redefined
- `DECISIONS.md` — the claims that are supposed to be true
- This file

Add when relevant: `docker-compose.yml` and `Dockerfile` (anything about container
behaviour, ports, healthchecks, baked weights), `SETUP.md` (anything about a
still-unverified assumption).

Do **not** attach a diff of the code you want tested if you can avoid it. A model
that reads the implementation writes tests shaped like the implementation, and
inherits its blind spots. Attach the *contract* and the *surrounding* code instead.

---

## Part 2 — the prompt

> You are writing tests for `ocr-service`, a two-lane OCR gateway. Read the attached
> files first. Do not write any code until you have.
>
> **Target:** `<<which file, e.g. tests/offline/test_upload_limits.py>>`
> **Scope:** `<<what behaviour, e.g. request-size and content-type handling on /ocr>>`
>
> ### Which suite you are writing for
>
> This repo has two suites with deliberately different rules. Pick the right one and
> follow its rule; do not blend them.
>
> - `tests/` — **black-box over HTTP against a running gateway. Nothing may be
>   mocked.** The whole point is to exercise what is actually deployed: real
>   paddleocr kwargs, real JSON serialisation, real docker DNS. A mock here replaces
>   exactly the part nobody has verified.
> - `tests/offline/` — **no Docker, no GPU, no paddleocr, no network.** `PaddleOCR`
>   is stubbed and the doc lane is a throwaway localhost HTTP server; the gateway
>   app, FastAPI routing, its JSON encoder and httpx all run for real. This suite
>   may only assert things about **code in this repo**. The moment a test needs to
>   know what PaddleOCR actually returns, it belongs in `tests/`.
>
> ### Hard rules
>
> 1. **Reuse the existing fixtures.** `post_file`, `make_image`, `make_pdf`,
>    `real_sample`, `compose_exec`, `client` (live suite); `post_ocr`, `post_parse`,
>    `fake_pages`, `doc_lane`, `dead_doc_lane_url`, `app_client` (offline suite).
>    Adding a near-duplicate fixture is a defect, not a style choice.
> 2. **Assert the response shape through `tests/shape.py`.** Do not re-implement it.
>    If the contract needs a new assertion, extend `shape.py` — one file, one truth.
> 3. **Never invent a status code and assert it.** If the correct behaviour is a
>    judgement call the maintainer has not made (is a `.txt` upload a 400 or a 500?),
>    assert only what must be true regardless — "does not hang", "does not leak an
>    unhandled traceback", "does not return 200 with fabricated output" — then
>    `pytest.xfail` with a message saying which decision is outstanding.
> 4. **No test may pass vacuously.** Any `for x in collection: assert ...` must be
>    preceded by an assertion that the collection is non-empty. State in a comment
>    what makes each test impossible to pass by accident.
> 5. **Every test's docstring names the failure it would catch**, and where —
>    `file:line` or function name. Not "tests that /ocr works".
> 6. Mark `slow` (tens of seconds+), `docker` (needs `docker compose exec`),
>    `offline`, `accuracy` (verdict is a human's) as declared in `pytest.ini`. Use
>    `--strict-markers`-safe names only; register any new marker in `pytest.ini`.
> 7. Keep the file under ~250 lines. If the scope does not fit, say so and propose a
>    split rather than writing 600 lines.
>
> ### Enumerate by failure mode, not by imagination
>
> Group the cases under these headings and say explicitly when a heading does not
> apply. Enumeration survives blind spots; brainstorming does not.
>
> - **boundary** — empty, 1 byte, 1 pixel, 1 page, 50 pages, max int, missing field,
>   None, unicode and spaces in filenames, no file extension
> - **failure / timeout** — upstream down, upstream slow, upstream returns the wrong
>   content type, client disconnects mid-request, malformed body
> - **concurrency** — shared mutable state (there is exactly one global PaddleOCR
>   pipeline), request interleaving, does output stay correct rather than merely
>   not crashing
> - **scale** — N+1 work per page, memory that grows per request, unbounded input
> - **idempotency** — same input twice, same input after a restart, same input under
>   a different filename
> - **auth / exposure** — port 8000 binds `0.0.0.0` with no auth today; this is a
>   recorded decision, not an oversight. Test it only if asked.
>
> ### Then, in the same reply, write two sections
>
> **"Cases I did NOT cover"** — every case you thought of and dropped, with the
> reason (needs real documents / needs the GPU box / correct behaviour undecided /
> would be flaky). This section is not optional and "none" is not an acceptable
> answer.
>
> **"Where my test design is likely to be wrong"** — you derived these cases from
> the same reading of the code that a previous model used to write the code, so your
> testing blind spots correlate with its coding blind spots. Name the specific
> assumptions you made about the domain, the input profile, or the deployment that
> could be false. Be concrete: "I assumed input documents are flat scans, not phone
> photos taken at an angle — if it is the latter, `use_textline_orientation=False`
> matters far more than these tests suggest."
>
> ### Frame
>
> Do not ask "what tests should exist for this code". Ask **"how would I break this
> in production on a shared-GPU single box serving Hong Kong documents?"** — then
> write the test that proves it. A green suite is not evidence of correctness; it is
> evidence that the cases you thought of pass.

---

## Part 3 — reviewing what comes back

Do not read the model's summary first. Read the test bodies, then the two
self-declared sections, then decide.

The four checks that catch most of it:

1. **Pick two tests and try to make them pass with broken code.** If a test would
   still pass with the behaviour it claims to check removed, it is decoration. This
   is worth doing by hand — revert the fix, run the test, confirm red. Both fixes in
   `tests/offline/` were validated this way (11 and 9 tests go red respectively).
2. **Grep for invented status codes.** `assert resp.status_code == 400` on an
   endpoint that has never been specified to return 400 is the model deciding
   policy on your behalf.
3. **Check every loop has a non-empty guard.** `for page in pages: assert ...` with
   `pages == []` passes while asserting nothing. This exact bug was in `test_b1`.
4. **Read "Cases I did NOT cover" as the actual deliverable.** Your additions to
   that list are where the domain knowledge enters — no model has your customer's
   document profile, and that single fact decides which failures matter.

Then ask the question no model can answer for you: *is the input profile flat
scanned contracts, or photos of signage on a site?* That flips `SETUP.md` #5 and
changes which half of this suite is worth running.
