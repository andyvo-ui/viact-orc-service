"""Group G — same input, same output.

Cheap, and it is the only thing standing between you and a silent model swap: if
someone edits DET_MODEL/REC_MODEL in ocr_engine.py (SETUP.md #4 explicitly invites
this — small -> medium), G4 is what tells you the output moved.
"""

import time

import pytest

from shape import all_texts


def test_g1_same_image_three_times_gives_identical_output(post_file, real_sample):
    runs = []
    for _ in range(3):
        resp = post_file("/ocr", real_sample)
        assert resp.status_code == 200, resp.text[:2000]
        runs.append(resp.json())

    first_texts = all_texts(runs[0])
    for i, run in enumerate(runs[1:], start=2):
        assert all_texts(run) == first_texts, f"run {i} text differs from run 1"

    first_scores = [s for p in runs[0]["pages"] for s in p["scores"]]
    for i, run in enumerate(runs[1:], start=2):
        got = [s for p in run["pages"] for s in p["scores"]]
        assert got == pytest.approx(first_scores, abs=1e-6), f"run {i} scores drifted"


def test_g2_warmup_cost_is_paid_before_serving(post_file, make_image):
    """lifespan calls warmup() at boot (main.py:30), so request #1 should not be
    dramatically slower than request #5. A big gap means warmup is not doing its
    job and the first caller after every restart eats the cost."""
    img = make_image("g2.png", text="WARMUP PROBE")

    timings = []
    for _ in range(5):
        started = time.perf_counter()
        resp = post_file("/ocr", img)
        timings.append(time.perf_counter() - started)
        assert resp.status_code == 200, resp.text[:2000]

    steady = sum(timings[1:]) / len(timings[1:])
    print(f"\n  first={timings[0]:.2f}s  steady={steady:.2f}s  all={[round(t, 2) for t in timings]}")
    assert timings[0] < steady * 5 + 2.0, (
        f"first request {timings[0]:.1f}s vs steady {steady:.1f}s — warmup() is not "
        f"actually pre-initialising the pipeline"
    )


def test_g3_reencoded_same_content_is_stable(post_file, make_image):
    """Same pixels, two filenames. Guards against anything keying off the filename
    or the temp-file suffix path in main.py:45."""
    a = make_image("g3_a.png", text="STABLE CONTENT 42")
    r1 = post_file("/ocr", a)
    r2 = post_file("/ocr", a, filename="renamed.png")
    assert r1.status_code == 200 and r2.status_code == 200
    assert all_texts(r1.json()) == all_texts(r2.json()), "output depends on the filename"


def test_g4_golden_snapshot(post_file, real_sample, request):
    """Regression tripwire. First run writes tests/fixtures/golden.json and skips;
    every later run compares against it.

    Delete the file deliberately when you change the model — never edit it to make
    a red test go green.
    """
    import json
    from pathlib import Path

    golden = Path(__file__).parent / "fixtures" / "golden.json"
    resp = post_file("/ocr", real_sample)
    assert resp.status_code == 200, resp.text[:2000]
    current = {"sample": real_sample.name, "texts": all_texts(resp.json())}

    if not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        pytest.skip(f"wrote baseline {golden} — review it by hand, then re-run")

    expected = json.loads(golden.read_text(encoding="utf-8"))
    assert expected["sample"] == current["sample"], (
        f"baseline was recorded for {expected['sample']}, not {current['sample']}"
    )
    assert current["texts"] == expected["texts"], (
        "OCR output changed against the recorded baseline — model, version, or "
        "preprocessing moved. Confirm it is intentional before updating golden.json."
    )
