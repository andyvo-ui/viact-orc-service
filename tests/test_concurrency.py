"""Group C — concurrency.

The claim under test: gateway/main.py:43 declares `async def ocr` but calls
`run_ocr`, which is fully blocking (ONNX inference in C). A blocking call inside an
async endpoint occupies the event loop, so concurrent /ocr requests serialise AND
/health cannot answer while OCR is running.

If that claim is right, test_c1 fails and test_c2 fails. Those failures are the
deliverable — do not "fix" them by loosening the thresholds.

Thresholds are env-tunable because the right numbers depend on the VM:
    OCR_TEST_CONCURRENCY    default 8
    OCR_TEST_MIN_SPEEDUP    default 1.5   (1.0 == perfectly serialised)
    OCR_TEST_HEALTH_BUDGET  default 5.0   (matches the compose healthcheck timeout)
"""

import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

N = int(os.getenv("OCR_TEST_CONCURRENCY", "8"))
MIN_SPEEDUP = float(os.getenv("OCR_TEST_MIN_SPEEDUP", "1.5"))
HEALTH_BUDGET = float(os.getenv("OCR_TEST_HEALTH_BUDGET", "5.0"))


def _post_once(base_url: str, path, timeout: float) -> tuple[int, float]:
    data = path.read_bytes()
    started = time.perf_counter()
    with httpx.Client(base_url=base_url, timeout=timeout) as c:
        resp = c.post("/ocr", files={"file": (path.name, data)})
    return resp.status_code, time.perf_counter() - started


@pytest.fixture(scope="module")
def load_image(make_image):
    # Big enough that inference dominates HTTP overhead, small enough to repeat.
    return make_image("load.png", text="CONCURRENCY PROBE 12345", size=(1600, 400))


@pytest.mark.slow
def test_c1_concurrent_requests_are_not_serialised(base_url, load_image):
    status, t_single = _post_once(base_url, load_image, 120)
    assert status == 200, f"baseline request failed with {status}"

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as pool:
        results = list(pool.map(lambda _: _post_once(base_url, load_image, 300), range(N)))
    wall = time.perf_counter() - started

    codes = [c for c, _ in results]
    assert all(c == 200 for c in codes), f"non-200 under load: {codes}"

    speedup = (N * t_single) / wall
    print(
        f"\n  single={t_single:.2f}s  n={N}  wall={wall:.2f}s  "
        f"speedup={speedup:.2f}x (1.0 == fully serialised)"
    )
    assert speedup >= MIN_SPEEDUP, (
        f"{N} concurrent requests took {wall:.1f}s vs {t_single:.1f}s for one "
        f"(speedup {speedup:.2f}x). Consistent with the event loop being blocked by "
        f"the sync run_ocr call in `async def ocr` (gateway/main.py:43)."
    )


@pytest.mark.slow
def test_c2_health_stays_responsive_under_load(base_url, load_image, client):
    """The compose healthcheck has timeout: 5s. If /health blocks past that while
    OCR runs, the container gets marked unhealthy under normal traffic."""
    stop = threading.Event()
    errors: list[str] = []

    def hammer():
        while not stop.is_set():
            try:
                code, _ = _post_once(base_url, load_image, 300)
                if code != 200:
                    errors.append(f"load request -> {code}")
            except httpx.HTTPError as exc:
                errors.append(f"load request raised {exc}")

    workers = [threading.Thread(target=hammer, daemon=True) for _ in range(max(2, N // 2))]
    for w in workers:
        w.start()

    samples: list[float] = []
    try:
        time.sleep(1.0)  # let the load actually land before sampling
        for _ in range(8):
            t0 = time.perf_counter()
            resp = client.get("/health", timeout=30)
            samples.append(time.perf_counter() - t0)
            assert resp.status_code == 200, resp.status_code
            time.sleep(0.25)
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=300)

    worst = max(samples)
    print(f"\n  /health under load: p50={statistics.median(samples):.2f}s worst={worst:.2f}s")
    assert not errors, errors[:5]
    assert worst < HEALTH_BUDGET, (
        f"/health took {worst:.1f}s under load, over the {HEALTH_BUDGET}s compose "
        f"healthcheck timeout — the container would flip to unhealthy under traffic"
    )


def test_c3_concurrent_results_match_the_serial_result(base_url, load_image, post_file):
    """One global PaddleOCR pipeline is shared by every request (ocr_engine.py:35).
    If predict() is not thread-safe, concurrency corrupts output rather than
    raising — which is the failure mode nobody notices in production."""
    serial = post_file("/ocr", load_image)
    assert serial.status_code == 200, serial.text[:2000]
    expected = [t for p in serial.json()["pages"] for t in p["texts"]]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_concurrent_texts, base_url, load_image) for _ in range(4)]
        got = [f.result() for f in futures]

    for i, texts in enumerate(got):
        assert texts == expected, (
            f"concurrent request {i} produced different text than the serial run.\n"
            f"  serial:     {expected}\n  concurrent: {texts}"
        )


def _concurrent_texts(base_url, path) -> list[str]:
    data = path.read_bytes()
    with httpx.Client(base_url=base_url, timeout=300) as c:
        resp = c.post("/ocr", files={"file": (path.name, data)})
    resp.raise_for_status()
    return [t for p in resp.json()["pages"] for t in p["texts"]]


@pytest.mark.slow
@pytest.mark.docker
def test_c4_memory_does_not_climb_over_a_burst(base_url, load_image, compose_exec):
    """Temp files are unlinked in a finally (main.py:52), and the ONNX session is
    reused — so RSS should plateau. A steady climb means something retains."""
    def rss_kb() -> int:
        # PID 1 is uvicorn in this image (single worker). If --workers is ever added,
        # this measures the parent only and the test silently stops being meaningful.
        proc = compose_exec("python", "-c", "print(open('/proc/1/status').read())")
        # Without this assert a broken probe reads as "skipped", not "broken".
        assert proc.returncode == 0, f"could not exec into the container: {proc.stderr}"
        for line in proc.stdout.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        pytest.skip("could not read VmRSS from the container")

    baseline = rss_kb()
    for _ in range(20):
        code, _ = _post_once(base_url, load_image, 300)
        assert code == 200
    after = rss_kb()

    growth_mb = (after - baseline) / 1024
    print(f"\n  RSS {baseline / 1024:.0f}MB -> {after / 1024:.0f}MB (+{growth_mb:.0f}MB over 20 requests)")
    assert growth_mb < 300, f"RSS grew {growth_mb:.0f}MB over 20 requests — suspect a leak"


@pytest.mark.docker
def test_c4b_no_temp_files_left_behind(base_url, load_image, compose_exec):
    """main.py:46 writes to NamedTemporaryFile(delete=False) and relies on the
    finally block. Any leak here fills the container's disk over days."""
    def tmp_count() -> int:
        proc = compose_exec("python", "-c",
                            "import pathlib;print(len(list(pathlib.Path('/tmp').iterdir())))")
        assert proc.returncode == 0, proc.stderr
        return int(proc.stdout.strip())

    before = tmp_count()
    for _ in range(5):
        code, _ = _post_once(base_url, load_image, 300)
        assert code == 200
    after = tmp_count()

    assert after <= before, f"/tmp grew from {before} to {after} entries over 5 requests"
