"""Shared fixtures. Everything here talks to a RUNNING gateway over HTTP.

Nothing is mocked on purpose: the point of this suite is to verify the thing that
is actually deployed (paddleocr kwargs, JSON serialisation, docker networking),
and a mock would replace exactly the parts that are unverified.

    OCR_BASE_URL      default http://localhost:8000
    OCR_DOC_LANE_URL  default http://localhost:8118  (probed directly, see doc_lane_up)
    OCR_TEST_TIMEOUT  default 120  (seconds, per request)
    OCR_TEST_DOCKER   set to 1 to enable tests that shell out to docker compose
"""

import os
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont

BASE_URL = os.getenv("OCR_BASE_URL", "http://localhost:8000")
DOC_LANE_PROBE_URL = os.getenv("OCR_DOC_LANE_URL", "http://localhost:8118")
TIMEOUT = float(os.getenv("OCR_TEST_TIMEOUT", "120"))
FIXTURES = Path(__file__).parent / "fixtures"

# Candidate paths for a real TTF. PIL's built-in bitmap font renders ~11px text
# that no OCR model can read, which would make every synthetic test a false
# negative. If none of these exist, text-bearing synthetic images are skipped.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return None


# --------------------------------------------------------------------------- #
# service reachability                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def client():
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=8)
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT, limits=limits) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def _service_is_up(client):
    """Fail the whole run once, loudly, instead of 30 confusing connection errors.

    Checking for 200 alone is not enough: port 8000 is a popular default, and a
    DIFFERENT local service answering /health with 200 would let the whole suite
    run against the wrong process. `fast_lane_device` is this gateway's fingerprint.
    """
    try:
        resp = client.get("/health", timeout=10)
    except httpx.HTTPError as exc:
        pytest.exit(f"gateway unreachable at {BASE_URL}: {exc}", returncode=2)
    if resp.status_code != 200:
        pytest.exit(f"gateway /health returned {resp.status_code}", returncode=2)
    try:
        body = resp.json()
    except ValueError:
        pytest.exit(f"{BASE_URL}/health did not return JSON — wrong service?", returncode=2)
    if "fast_lane_device" not in body:
        pytest.exit(
            f"something IS listening on {BASE_URL} but it is not ocr-service: "
            f"/health returned {body}. Set OCR_BASE_URL to the right port.",
            returncode=2,
        )


@pytest.fixture(scope="session")
def doc_lane_up() -> bool:
    """True when the doc lane is actually running. Probed OUT OF BAND.

    Deliberately NOT inferred from a /parse status code. The gateway maps both a
    transport failure and a doc-lane rejection into its own error space, so "junk
    request came back 502" cannot distinguish a dead doc lane from a live one that
    refused a junk file. With the old probe every `doclane_down` test passed while
    the doc lane was UP — meaning the DECISIONS.md #7 isolation claim, which those
    tests exist solely to prove, was never actually exercised.

    Assumes the doc lane's port is published to this host (compose publishes 8118).
    Set OCR_DOC_LANE_URL if it is not, or if the gateway is on another machine.
    """
    for path in ("/health", "/"):
        try:
            resp = httpx.get(f"{DOC_LANE_PROBE_URL}{path}", timeout=5)
        except httpx.HTTPError:
            continue
        # Any answer below 500 proves a server is listening — even a 404.
        if resp.status_code < 500:
            return True
    return False


# --------------------------------------------------------------------------- #
# request helper                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def post_file(client):
    """post_file('/ocr', path) -> Response. Override filename to test parsing."""

    def _post(endpoint: str, path, *, filename=None, field="file", content_type=None):
        data = Path(path).read_bytes() if not isinstance(path, bytes) else path
        name = filename if filename is not None else Path(str(path)).name
        return client.post(
            endpoint, files={field: (name, data, content_type or "application/octet-stream")}
        )

    return _post


# --------------------------------------------------------------------------- #
# synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("ocr-fixtures")


@pytest.fixture(scope="session")
def make_image(workdir):
    """Factory: make_image('name.jpg', text='ABC 123', size=(600,200), rotate=0)."""

    def _make(name, *, text="HELLO 123", size=(600, 200), rotate=0, bg="white", fg="black"):
        path = workdir / name
        if path.exists():
            return path
        img = Image.new("RGB", size, bg)
        if text:
            font = _font(max(32, size[1] // 5))
            if font is None:
                pytest.skip("no TTF font on this host; synthetic text images unusable")
            draw = ImageDraw.Draw(img)
            draw.text((size[0] // 20, size[1] // 3), text, fill=fg, font=font)
        if rotate:
            img = img.rotate(rotate, expand=True, fillcolor=bg)
        img.save(path)
        return path

    return _make


@pytest.fixture(scope="session")
def make_pdf(workdir):
    """Factory: make_pdf('doc.pdf', pages=3). Each page carries readable text."""

    def _make(name, *, pages=1, size=(1240, 1754)):
        path = workdir / name
        if path.exists():
            return path
        font = _font(64)
        if font is None:
            pytest.skip("no TTF font on this host; synthetic PDFs unusable")
        imgs = []
        for i in range(pages):
            img = Image.new("RGB", size, "white")
            ImageDraw.Draw(img).text((80, 120), f"PAGE {i + 1} OF {pages}", fill="black", font=font)
            imgs.append(img)
        imgs[0].save(path, "PDF", save_all=True, append_images=imgs[1:])
        return path

    return _make


@pytest.fixture(scope="session")
def real_sample() -> Path:
    """A REAL document you drop in tests/fixtures/. Synthetic images cannot stand in
    for this — see tests/README.md."""
    for ext in ("jpg", "jpeg", "png"):
        for candidate in sorted(FIXTURES.glob(f"sample*.{ext}")):
            return candidate
    pytest.skip(f"no real sample image in {FIXTURES} — see tests/README.md")


# --------------------------------------------------------------------------- #
# docker escape hatch                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def compose_exec():
    """Run a command inside the gateway container. Skips unless OCR_TEST_DOCKER=1."""
    if os.getenv("OCR_TEST_DOCKER") != "1":
        pytest.skip("set OCR_TEST_DOCKER=1 to run docker-level checks")
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")

    root = Path(__file__).parent.parent

    def _exec(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "compose", "exec", "-T", "gateway", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )

    return _exec
