"""Harness for the offline suite. No Docker, no GPU, no paddleocr, no network.

WHY THIS EXISTS ALONGSIDE A DELIBERATELY MOCK-FREE SUITE
--------------------------------------------------------
tests/ (the parent directory) is mock-free on purpose: it verifies the thing that
is actually deployed, and a mock would replace exactly the parts nobody has run.
That rule is right, and this directory does not break it — it tests a *different*
thing.

The parent suite asks "does the deployed service work?".
This directory asks "is OUR glue code correct?" — the numpy→JSON coercion in
run_ocr and the error mapping in /parse. Neither of those is a claim about
paddleocr; both are claims about code in this repo, and both were wrong before.

The line to hold: **nothing here may assert anything about OCR quality, model
behaviour, or paddleocr's API.** The moment a test in this directory needs to know
what PaddleOCR returns, it belongs in the parent suite against a real container.

What IS faked, and what is real:
  faked  — paddleocr.PaddleOCR (stubbed; returns whatever a test feeds it)
  faked  — the doc lane (a real throwaway HTTP server on localhost, so the actual
           httpx client code path runs; only the *server* is a stand-in)
  REAL   — gateway/main.py, fast_lane/ocr_engine.py, FastAPI routing and its JSON
           encoder, python-multipart parsing, httpx error types

Run it anywhere, including a laptop with nothing installed but pip deps:

    pytest tests/offline -v
"""

import socket
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------- #
# stub paddleocr BEFORE gateway/main.py imports ocr_engine                     #
# --------------------------------------------------------------------------- #

# Recorded from paddlex 3.7.0 ONNXRuntimeRunnerConfig, which is pydantic with
# extra="forbid" — so an engine_config key outside this set would raise at runtime.
# Asserting it here means a typo in ocr_engine.py fails offline instead of at deploy.
_ORT_CONFIG_FIELDS = {
    "device_type", "device_id", "providers", "provider_options",
    "graph_optimization_level", "intra_op_num_threads", "inter_op_num_threads",
    "execution_mode", "log_severity_level", "enable_mem_pattern",
    "enable_cpu_mem_arena", "session_options",
}

# What the current test wants predict() to return. Set via the `fake_pages` fixture.
_FAKE_PAGES: list[dict] = []


class _StubPaddleOCR:
    """Stands in for paddleocr.PaddleOCR. Must be a real class: ocr_engine annotates
    `dict[bool, PaddleOCR]`, which is evaluated at import time.

    Records EVERY construction, not just the last: ocr_engine now builds one pipeline
    per orientation setting, and a test that only saw the last one could not tell
    which settings each was built with.
    """

    constructions: list[dict] = []

    def __init__(self, **kwargs):
        type(self).constructions.append(kwargs)
        bad = set(kwargs.get("engine_config") or {}) - _ORT_CONFIG_FIELDS
        assert not bad, (
            f"engine_config carries keys paddlex would reject (extra='forbid'): {bad}"
        )
        self.use_textline_orientation = kwargs.get("use_textline_orientation")

    def predict(self, image_path, **kwargs):
        return list(_FAKE_PAGES)


if "paddleocr" not in sys.modules:
    sys.modules["paddleocr"] = types.SimpleNamespace(PaddleOCR=_StubPaddleOCR)

for path in (ROOT / "fast_lane", ROOT / "gateway"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import main as gateway_main  # noqa: E402
import ocr_engine  # noqa: E402


# --------------------------------------------------------------------------- #
# neutralise the parent conftest's live-gateway guard                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def _service_is_up():
    """Overrides tests/conftest.py's fixture of the same name.

    The parent calls pytest.exit() when no gateway answers on :8000. That is the
    right behaviour there and fatal here — this directory has no gateway to reach.
    Overriding by name in a subdirectory conftest is how pytest scopes that off.
    """
    return None


# --------------------------------------------------------------------------- #
# /ocr fixtures                                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_pages():
    """Set what the stubbed PaddleOCR.predict() yields.

        fake_pages([{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}])

    Also resets ocr_engine's module-level pipeline cache so the next call rebuilds.
    """

    def _set(pages):
        _FAKE_PAGES[:] = pages
        ocr_engine._pipelines.clear()
        return pages

    yield _set
    _FAKE_PAGES.clear()
    ocr_engine._pipelines.clear()


@pytest.fixture
def stub_paddleocr():
    """The stub class, so tests can inspect the kwargs ocr_engine passed it.

    Exposed as a fixture rather than via `import conftest` in the test: both
    tests/ and tests/offline/ contain a conftest.py, so that import resolves by
    sys.path ordering, which in turn depends on collection order. Fragile.
    """
    _StubPaddleOCR.constructions = []
    return _StubPaddleOCR


@pytest.fixture
def app_client():
    """TestClient over the REAL FastAPI app from gateway/main.py."""
    from starlette.testclient import TestClient

    with TestClient(gateway_main.app) as client:
        yield client


@pytest.fixture
def post_ocr(app_client):
    """POST an arbitrary byte payload to /ocr. The bytes are never decoded —
    predict() is stubbed — so content does not matter, only the request shape."""

    def _post(payload=b"pretend-this-is-an-image", filename="doc.jpg", **params):
        return app_client.post(
            "/ocr", files={"file": (filename, payload, "image/jpeg")}, params=params
        )

    return _post


# --------------------------------------------------------------------------- #
# doc-lane stand-in (a real server, so the real httpx path runs)               #
# --------------------------------------------------------------------------- #


class _DocLaneStub:
    """A throwaway HTTP server whose next response each test sets explicitly."""

    def __init__(self):
        self.status = 200
        self.body = b'{"markdown": "# stub"}'
        self.content_type = "application/json"
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                stub.requests.append({
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type", ""),
                    "length": length,
                    "raw": raw,
                })
                self._reply()

            def do_GET(self):
                self._reply()

            def _reply(self):
                self.send_response(stub.status)
                self.send_header("Content-Type", stub.content_type)
                self.send_header("Content-Length", str(len(stub.body)))
                self.end_headers()
                self.wfile.write(stub.body)

            def log_message(self, *args):
                pass  # keep pytest output clean

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def responds(self, status, body=b"", content_type="application/json"):
        self.status = status
        self.body = body if isinstance(body, bytes) else body.encode()
        self.content_type = content_type

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def doc_lane(monkeypatch):
    """A live stand-in doc lane, wired into the gateway via DOC_LANE_URL."""
    stub = _DocLaneStub()
    monkeypatch.setattr(gateway_main, "DOC_LANE_URL", stub.url)
    yield stub
    stub.shutdown()


@pytest.fixture
def dead_doc_lane_url(monkeypatch):
    """A URL with nothing listening: bind a port, release it, hand back the address."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(gateway_main, "DOC_LANE_URL", url)
    return url


@pytest.fixture
def post_parse(app_client):
    def _post(payload=b"%PDF-1.4 pretend", filename="doc.pdf"):
        return app_client.post(
            "/parse", files={"file": (filename, payload, "application/pdf")}
        )

    return _post
