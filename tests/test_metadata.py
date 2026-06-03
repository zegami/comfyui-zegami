import json

from comfyui_zegami.capture import build_metadata


def test_full_capture():
    meta = build_metadata(
        prompt={"6": {"class_type": "KSampler", "inputs": {"seed": 42}}},
        extra_pnginfo={"workflow": {"nodes": []}},
        tags="campaign-x, hero",
        notes="client brief v2",
        batch_index=2,
        batch_size=5,
        prompt_id="pid-123",
    )
    assert meta["prompt"]["6"]["inputs"]["seed"] == 42
    assert meta["workflow"] == {"nodes": []}
    assert meta["tags"] == ["campaign-x", "hero"]
    assert meta["notes"] == "client brief v2"
    assert meta["batch_index"] == 2
    assert meta["exec"]["prompt_id"] == "pid-123"
    assert "python_version" in meta["exec"]
    # It becomes a single CSV cell — must be JSON-serialisable.
    json.dumps(meta)


def test_partial_capture_never_raises():
    meta = build_metadata()  # nothing supplied
    assert "exec" in meta
    assert "python_version" in meta["exec"]
    json.dumps(meta)


def test_no_credentials_leak_into_metadata():
    # The capture path never receives or serialises the API key.
    meta = build_metadata(prompt={"x": 1}, extra_pnginfo={"workflow": {}})
    assert "zeg_" not in json.dumps(meta)
