"""Offline — /parse error mapping. The doc lane here is a real localhost server.

Only the doc-lane *server* is a stand-in; the gateway's httpx client, its exception
types and its status mapping all execute for real.

The bug this guards: raise_for_status() used to sit INSIDE the
`except httpx.HTTPError` block, and httpx.HTTPStatusError subclasses HTTPError. So
a doc-lane 400 ("your file is garbage") reached the caller as 502 ("our GPU box is
down"), and a retry loop would hammer a request that can never succeed.

The mapping the gateway must now implement:

    upstream situation              gateway responds
    -----------------------------  -----------------------------------------
    connection refused / DNS       502  "doc lane unreachable"
    upstream 4xx                   the SAME 4xx, "doc lane rejected"
    upstream 5xx                   502  "doc lane failed with <code>"
    upstream 200, valid JSON       200  body passed through unchanged
    upstream 200, not JSON         502  "doc lane returned non-JSON"
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


def test_upstream_json_passes_through_unchanged(post_parse, doc_lane):
    """SETUP.md #2 is still open — the real doc-lane schema is unknown. So this
    asserts pass-through fidelity, NOT a schema: whatever it sends, the caller gets
    byte-for-byte the same object."""
    payload = {
        "markdown": "# Invoice\n\n| item | qty |\n|---|---|\n| bolt | 12 |",
        "pages": 2,
        "nested": {"tables": [{"rows": 3}], "unicode": "總金額"},
    }
    doc_lane.responds(200, json.dumps(payload).encode(), "application/json")

    resp = post_parse()

    assert resp.status_code == 200, resp.text[:300]
    assert resp.json() == payload


def test_gateway_forwards_multipart_file_field(post_parse, doc_lane):
    """main.py assumes the doc lane wants multipart 'file'. That assumption is still
    unverified against the real server, but this pins what we actually send, so the
    day someone reads the real API there is one place to compare against."""
    doc_lane.responds(200, b'{"ok": true}')

    post_parse(payload=b"%PDF-1.4 sentinel-bytes", filename="contract.pdf")

    assert len(doc_lane.requests) == 1, doc_lane.requests
    sent = doc_lane.requests[0]
    assert sent["path"] == "/parse", sent["path"]
    assert sent["content_type"].startswith("multipart/form-data"), sent["content_type"]
    assert b'name="file"' in sent["raw"], sent["raw"][:200]
    assert b"contract.pdf" in sent["raw"], sent["raw"][:200]
    assert b"sentinel-bytes" in sent["raw"], "the upload body did not reach the doc lane"


def test_missing_filename_does_not_break_the_proxy(app_client, doc_lane):
    """UploadFile.filename can be None. main.py falls back to 'upload' rather than
    handing httpx a None filename."""
    doc_lane.responds(200, b'{"ok": true}')

    resp = app_client.post("/parse", files={"file": (None, b"bytes-with-no-name")})

    assert resp.status_code in (200, 422), f"{resp.status_code}: {resp.text[:300]}"


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
