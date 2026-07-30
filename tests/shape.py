"""Response-shape assertions for /ocr.

Kept separate from the tests so the contract lives in one place: if the gateway's
output shape changes, exactly one file needs editing.

The contract asserted here is the one the CODE claims (gateway/main.py:50), not one
I invented: {"device": str, "pages": [{"texts": [...], "scores": [...], "boxes": [...]}]}
"""


def assert_ocr_envelope(payload) -> list[dict]:
    """Validate the top-level /ocr response. Returns the pages list."""
    assert isinstance(payload, dict), f"expected object, got {type(payload).__name__}"
    assert "device" in payload, f"missing 'device': {list(payload)}"
    assert payload["device"] in ("cpu", "gpu"), payload["device"]
    assert "pages" in payload, f"missing 'pages': {list(payload)}"
    assert isinstance(payload["pages"], list), type(payload["pages"]).__name__
    return payload["pages"]


def assert_page_shape(page) -> None:
    """Validate one page entry, including the parallel-array invariant.

    texts/scores/boxes are three parallel arrays. Nothing in the code enforces that
    they stay the same length — if they diverge, a caller zipping them silently
    mislabels text.
    """
    assert isinstance(page, dict), type(page).__name__
    for key in ("texts", "scores", "boxes"):
        assert key in page, f"missing '{key}': {list(page)}"
        assert isinstance(page[key], list), f"{key} is {type(page[key]).__name__}, not list"

    n = len(page["texts"])
    assert len(page["scores"]) == n, f"scores {len(page['scores'])} != texts {n}"
    assert len(page["boxes"]) == n, f"boxes {len(page['boxes'])} != texts {n}"

    for t in page["texts"]:
        assert isinstance(t, str), f"text is {type(t).__name__}, not str"
    for s in page["scores"]:
        assert isinstance(s, (int, float)) and not isinstance(s, bool), type(s).__name__
        assert 0.0 <= s <= 1.0, f"score out of range: {s}"

    # boxes was the field that actually broke (numpy polys -> 500), and it was the
    # only one with no content assertion. Without this, "fixing" it with str(polys)
    # or the wrong nesting depth still passes every shape test.
    for i, box in enumerate(page["boxes"]):
        assert isinstance(box, list), f"box {i} is {type(box).__name__}, not list"
        assert len(box) >= 3, f"box {i} has {len(box)} points, need >=3 for a polygon"
        for point in box:
            assert isinstance(point, list), f"box {i} point is {type(point).__name__}, not list"
            assert len(point) == 2, f"box {i} point has {len(point)} coords, expected 2"
            for coord in point:
                assert isinstance(coord, (int, float)) and not isinstance(coord, bool), (
                    f"box {i} coord is {type(coord).__name__} — not a JSON number "
                    f"(a numpy scalar leaking through would land here)"
                )


def all_texts(payload) -> list[str]:
    return [t for page in payload.get("pages", []) for t in page.get("texts", [])]


def joined(payload) -> str:
    return " ".join(all_texts(payload)).upper()
