"""Group D — failure isolation. The core architectural claim of this repo.

DECISIONS.md #7: no depends_on, so /ocr must stay alive when the GPU lane is dead.
That claim is only worth anything if it is actually tested with the doc lane down.

    docker compose stop doc-lane
    pytest tests/test_failure.py
    docker compose start doc-lane

Tests marked doclane_down self-skip when the doc lane IS reachable, so the file is
safe to run in either state — it just covers less.
"""

import pytest

from shape import assert_ocr_envelope


@pytest.fixture
def require_doc_lane_down(doc_lane_up):
    if doc_lane_up:
        pytest.skip("doc lane is up — run `docker compose stop doc-lane` to cover this")


@pytest.fixture
def require_doc_lane_up(doc_lane_up):
    if not doc_lane_up:
        pytest.skip("doc lane is down — run `docker compose start doc-lane` to cover this")


# --------------------------------------------------------------------------- #
# D1 / D2 — the isolation claim                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.doclane_down
def test_d1_parse_returns_502_when_doc_lane_is_down(post_file, make_pdf, require_doc_lane_down):
    pdf = make_pdf("d1.pdf", pages=1)
    resp = post_file("/parse", pdf)
    assert resp.status_code == 502, f"expected 502, got {resp.status_code}: {resp.text[:500]}"
    assert "doc lane unreachable" in resp.text, resp.text[:500]


@pytest.mark.doclane_down
def test_d1b_ocr_still_works_while_doc_lane_is_down(post_file, make_image, require_doc_lane_down):
    """THE test for DECISIONS.md #7. If this fails, the no-depends_on decision
    bought nothing."""
    img = make_image("d1b.png", text="FAST LANE ALIVE")
    resp = post_file("/ocr", img)
    assert resp.status_code == 200, resp.text[:2000]
    assert_ocr_envelope(resp.json())


@pytest.mark.doclane_down
def test_d1c_health_is_green_while_doc_lane_is_down(client, require_doc_lane_down):
    """/health must not depend on the doc lane, or compose will restart a gateway
    that is perfectly healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.doclane_down
def test_d2_unresolvable_doc_lane_fails_fast(post_file, make_pdf, require_doc_lane_down):
    """DOC_LANE_TIMEOUT is 300s. A DNS failure must NOT wait that long — it should
    come back immediately. A slow 502 here means callers hang for 5 minutes."""
    import time

    pdf = make_pdf("d2.pdf", pages=1)
    started = time.perf_counter()
    resp = post_file("/parse", pdf)
    elapsed = time.perf_counter() - started

    assert resp.status_code == 502, resp.status_code
    assert elapsed < 30, (
        f"a dead doc lane took {elapsed:.0f}s to report 502 — callers will hang"
    )


# --------------------------------------------------------------------------- #
# D3 — error mapping when the doc lane IS up                                   #
# --------------------------------------------------------------------------- #


def test_d3_client_error_is_indistinguishable_from_infrastructure_error(
    post_file, require_doc_lane_up
):
    """Regression test for the raise_for_status()-inside-except bug.

    It used to be that raise_for_status() sat inside the `except httpx.HTTPError`
    block, and HTTPStatusError subclasses HTTPError — so a 400 from the doc lane
    ("your file is garbage") reached the caller as 502 ("our GPU box is down") and
    a retry loop would hammer a request that can never succeed. main.py now maps
    transport errors, upstream 4xx and upstream 5xx separately.

    Renamed intent: this no longer documents a defect, it guards the fix."""
    resp = post_file("/parse", b"definitely not a document", filename="junk.pdf")

    if resp.status_code == 502 and "doc lane unreachable" in resp.text:
        pytest.fail(
            "REGRESSION: a doc-lane rejection is being reported as 'unreachable' "
            f"again — client errors are masked as infrastructure errors.\n{resp.text[:500]}"
        )
    if resp.status_code == 502 and "doc lane failed with" in resp.text:
        pytest.xfail(
            "the doc lane itself answers 5xx on a malformed upload rather than 4xx. "
            "The gateway's mapping is correct; the upstream's is questionable."
        )
    assert resp.status_code in (200, 400, 415, 422), f"{resp.status_code}: {resp.text[:500]}"


def test_d3b_parse_passes_the_doc_lane_response_through_unchanged(
    post_file, make_pdf, require_doc_lane_up
):
    """SETUP.md #2: the /parse contract is UNVERIFIED — main.py assumes multipart
    'file' in, JSON out. This test does not assert a schema (I do not know it); it
    asserts the round trip produces JSON at all, and prints the keys so you can
    write the real contract afterwards."""
    pdf = make_pdf("d3b.pdf", pages=1)
    resp = post_file("/parse", pdf)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:1000]}"
    body = resp.json()
    print(f"\n  /parse top-level keys: {list(body) if isinstance(body, dict) else type(body).__name__}")
    assert body, "doc lane returned an empty body"


# --------------------------------------------------------------------------- #
# D6 — client disconnect                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.docker
def test_d6_client_disconnect_does_not_leak_a_temp_file(base_url, make_image, compose_exec):
    """Abort mid-request. main.py's finally should still unlink. If uvicorn kills
    the task before finally runs, the temp file survives forever.

    The timeout is split on purpose. A single small timeout aborts during UPLOAD,
    before main.py has written any temp file at all — so "no leak" comes back true
    without the finally block ever being reached. A generous write timeout plus a
    tiny read timeout uploads the whole body, lets the server enter run_ocr, and
    only THEN disconnects. That is the case worth testing.
    """
    import httpx

    img = make_image("d6.png", text="DISCONNECT PROBE", size=(2400, 600))
    abort_timeout = httpx.Timeout(connect=10.0, write=60.0, pool=10.0, read=0.5)

    def tmp_entries() -> set[str]:
        proc = compose_exec("python", "-c",
                            "import pathlib;print('\\n'.join(p.name for p in pathlib.Path('/tmp').iterdir()))")
        assert proc.returncode == 0, proc.stderr
        return set(proc.stdout.split())

    before = tmp_entries()
    with pytest.raises(httpx.ReadTimeout):
        with httpx.Client(base_url=base_url, timeout=abort_timeout) as c:
            c.post("/ocr", files={"file": (img.name, img.read_bytes())})

    # Give the server time to finish the aborted request and run its finally.
    import time
    time.sleep(20)

    leaked = tmp_entries() - before
    assert not leaked, f"temp files left behind after a client disconnect: {sorted(leaked)}"
