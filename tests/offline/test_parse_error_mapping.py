"""Offline — /parse error mapping. The doc lane here is a real localhost server.

Only the doc-lane *server* is a stand-in; the gateway's httpx client, its exception
types and its status mapping all execute for real. Real sample files (tests/samples)
exercise the real pypdfium2/PIL rasterisation path too — nothing about *that* is
stubbed here, unlike paddleocr.

The bug this guards: raise_for_status() used to sit INSIDE the
`except httpx.HTTPError` block, and httpx.HTTPStatusError subclasses HTTPError. So
a doc-lane 400 ("your file is garbage") reached the caller as 502 ("our GPU box is
down"), and a retry loop would hammer a request that can never succeed.

The mapping the gateway must implement:

    upstream situation              gateway responds
    -----------------------------  -----------------------------------------
    connection refused / DNS       502  "doc lane unreachable"
    upstream 4xx                   the SAME 4xx, "doc lane rejected"
    upstream 5xx                   502  "doc lane failed with <code>"
    upstream 200, not JSON         502  "doc lane returned non-JSON"
    upstream 200, valid JSON,      502  "doc lane returned a malformed completion"
      but no choices[0].message.content
    upstream 200, a real           200  {"markdown": <content>, "pages": N}
      OpenAI chat-completion

Every test below also triggers a GET to /v1/models first (the gateway asks the doc
lane what it's serving before the real completion call) — the stub's do_GET replies
with whatever .responds() last set, same as do_POST, but only do_POST is recorded
into .requests. _doc_lane_model_name() swallows any failure from that GET and falls
back to a hardcoded model name, so it never changes what these tests assert.
"""

import json

import pytest

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# transport failure — the only genuine "unreachable"                           #
# --------------------------------------------------------------------------- #


def test_nothing_listening_is_502_unreachable(post_parse, dead_doc_lane_url):
    resp = post_parse()
    assert resp.status_code == 502, f"{resp.status_code}: {resp.text[:300]}"
    assert "doc lane unreachable" in resp.text, resp.text[:300]


def test_transport_failure_returns_immediately(post_parse, dead_doc_lane_url):
    """DOC_LANE_TIMEOUT is 300s by default. A refused connection must not wait for
    it — otherwise callers hang five minutes on a container that is simply down."""
    import time

    started = time.perf_counter()
    resp = post_parse()
    elapsed = time.perf_counter() - started

    assert resp.status_code == 502
    assert elapsed < 10, f"a refused connection took {elapsed:.1f}s to report 502"


# --------------------------------------------------------------------------- #
# upstream 4xx — the caller's fault, must stay distinguishable                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("upstream_status", [400, 413, 415, 422])
def test_upstream_4xx_passes_through(post_parse, doc_lane, upstream_status):
    doc_lane.responds(upstream_status, b"that file is not a document", "text/plain")

    resp = post_parse()

    assert resp.status_code == upstream_status, (
        f"upstream {upstream_status} came back as {resp.status_code}. If this is 502, "
        f"client errors are being masked as infrastructure errors again.\n{resp.text[:300]}"
    )
    assert "rejected" in resp.text, resp.text[:300]
    assert "unreachable" not in resp.text, "a rejection must not be called unreachable"


def test_upstream_4xx_body_is_surfaced_to_the_caller(post_parse, doc_lane):
    """A caller who gets a 4xx needs to know WHY, or the status alone is useless."""
    doc_lane.responds(422, b"page 3 is an unsupported colour space", "text/plain")
    resp = post_parse()
    assert "unsupported colour space" in resp.text, resp.text[:400]


# --------------------------------------------------------------------------- #
# upstream 5xx — infrastructure, but not "unreachable"                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("upstream_status", [500, 503])
def test_upstream_5xx_becomes_502_naming_the_code(post_parse, doc_lane, upstream_status):
    doc_lane.responds(upstream_status, b"vllm out of memory", "text/plain")

    resp = post_parse()

    assert resp.status_code == 502, f"{resp.status_code}: {resp.text[:300]}"
    assert f"doc lane failed with {upstream_status}" in resp.text, resp.text[:400]
    assert "unreachable" not in resp.text, (
        "a 5xx means the doc lane answered — calling it unreachable loses the fact "
        "that it is up but broken, which is a different fix"
    )


# --------------------------------------------------------------------------- #
# happy path                                                                    #
# --------------------------------------------------------------------------- #


def _completion(content: str) -> bytes:
    """Build a real OpenAI-shaped chat-completion 200 body, the shape verified
    against the live doc lane via scripts/test_doclane.sh."""
    return json.dumps(
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45},
        }
    ).encode()


def test_upstream_completion_is_normalised_to_markdown(post_parse, doc_lane):
    """The gateway no longer passes the doc lane's raw OpenAI-shaped JSON through —
    it extracts choices[0].message.content and returns it as {"markdown": ...},
    so every caller of /parse gets the same shape regardless of what serves the
    doc lane underneath."""
    content = "# Invoice\n\n| item | qty |\n|---|---|\n| bolt | 12 |\n\n總金額"
    doc_lane.responds(200, _completion(content), "application/json")

    resp = post_parse()

    assert resp.status_code == 200, resp.text[:300]
    assert resp.json() == {"markdown": content, "pages": 1}


def test_gateway_sends_one_base64_image_per_page(post_parse, doc_lane):
    """The doc lane is vLLM's OpenAI-compatible server — it has no /parse route and
    wants JSON chat-completions with a base64 image, not multipart 'file'. This
    pins the request shape actually sent, matching scripts/test_doclane.sh."""
    doc_lane.responds(200, _completion("stub"))

    post_parse()

    assert len(doc_lane.requests) == 1, doc_lane.requests
    sent = doc_lane.requests[0]
    assert sent["path"] == "/v1/chat/completions", sent["path"]
    assert sent["content_type"].startswith("application/json"), sent["content_type"]

    body = json.loads(sent["raw"])
    assert body["model"], "model field must be set (from /v1/models or the fallback)"
    content_blocks = body["messages"][0]["content"]
    image_block = next(b for b in content_blocks if b["type"] == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,"), (
        image_block["image_url"]["url"][:60]
    )


def test_missing_filename_does_not_break_the_proxy(app_client, doc_lane):
    """UploadFile.filename can be None. The gateway must still fall back to a
    usable suffix/mime-type instead of crashing on a None filename.

    422 is an accepted outcome, not just 200: httpx's multipart encoder treats a
    `(None, bytes)` file tuple as a plain form field rather than a file, so
    Starlette's own `UploadFile` validation may reject it before the handler body
    ever runs. That is a pre-existing Starlette/httpx quirk, unrelated to /parse's
    own logic — the thing under test is "does not crash" (no 500), not which of
    200/422 Starlette's request validation happens to pick.
    """
    doc_lane.responds(200, _completion("stub"))

    resp = app_client.post("/parse", files={"file": (None, b"bytes-with-no-name")})

    assert resp.status_code in (200, 422), f"{resp.status_code}: {resp.text[:300]}"


def test_malformed_completion_is_502_not_a_gateway_traceback(post_parse, doc_lane):
    """Valid JSON, upstream 200 — but not the OpenAI chat-completion shape asked
    for. Distinct from the non-JSON case below: `.json()` succeeds here, so this
    must be caught by field access, not by the JSON parse."""
    doc_lane.responds(200, b'{"ok": true}', "application/json")

    resp = post_parse()

    assert resp.status_code == 502, f"{resp.status_code}: {resp.text[:300]}"
    assert "malformed completion" in resp.text, resp.text[:400]


def test_two_page_pdf_is_stitched_with_page_markers(post_parse, doc_lane, tmp_path):
    """A multi-page PDF is rasterised page-by-page (pypdfium2) and sent as separate
    completions — the doc lane takes one image per request, not a whole PDF."""
    from pathlib import Path

    pdf_bytes = (Path("tests/samples/08_two_page.pdf")).read_bytes()
    doc_lane.responds(200, _completion("PAGE TEXT"))

    resp = post_parse(payload=pdf_bytes, filename="doc.pdf", content_type="application/pdf")

    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert body["pages"] == 2, body
    assert len(doc_lane.requests) == 2, doc_lane.requests
    assert "<!-- page 1 -->" in body["markdown"]
    assert "<!-- page 2 -->" in body["markdown"]
    assert body["markdown"].count("PAGE TEXT") == 2


# --------------------------------------------------------------------------- #
# upstream 200 that is not JSON                                                 #
# --------------------------------------------------------------------------- #


def test_non_json_200_is_502_not_a_gateway_traceback(post_parse, doc_lane):
    """`return resp.json()` on an HTML error page raises inside the handler, which
    surfaces as a bare 500 with a traceback and no hint about whose fault it is."""
    doc_lane.responds(200, b"<html><body>Gateway Timeout</body></html>", "text/html")

    resp = post_parse()

    assert resp.status_code == 502, f"{resp.status_code}: {resp.text[:300]}"
    assert "non-JSON" in resp.text, resp.text[:400]


def test_empty_200_body_is_502(post_parse, doc_lane):
    doc_lane.responds(200, b"", "application/json")
    resp = post_parse()
    assert resp.status_code == 502, f"{resp.status_code}: {resp.text[:300]}"


# --------------------------------------------------------------------------- #
# isolation — DECISIONS.md #7, the part that CAN be tested offline              #
# --------------------------------------------------------------------------- #


def test_ocr_still_works_while_the_doc_lane_is_dead(post_ocr, fake_pages, dead_doc_lane_url):
    """The offline half of DECISIONS.md #7: no import-time or module-level coupling
    makes /ocr depend on the doc lane. The container-level half (no depends_on, DNS
    failure inside compose) still needs the real stack — see tests/test_failure.py."""
    fake_pages([{"rec_texts": ["FAST LANE ALIVE"], "rec_scores": [0.99],
                 "rec_polys": [[[0, 0], [9, 0], [9, 3], [0, 3]]]}])

    resp = post_ocr()

    assert resp.status_code == 200, resp.text[:300]
    assert resp.json()["pages"][0]["texts"] == ["FAST LANE ALIVE"]


def test_health_is_green_while_the_doc_lane_is_dead(app_client, dead_doc_lane_url):
    """If /health probed the doc lane, compose would restart a healthy gateway."""
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
