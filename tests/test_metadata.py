import json

from comfyui_zegami.capture import build_metadata, split_tags


def test_split_tags_separates_plain_tags_from_key_value_columns():
    plain, columns = split_tags("hero, style=anime, weight = 0.8 , favourite")
    assert plain == ["hero", "favourite"]
    # `key=value` entries become columns; keys + values are stripped.
    assert columns == {"style": "anime", "weight": "0.8"}


def test_split_tags_handls_edges():
    # Blank entries, blank keys, the reserved `name` join key, and value-side
    # `=` are all handled; on a duplicate key the last value wins.
    plain, columns = split_tags(" , =orphan, name=oops, a=1, a=2, url=http://x?q=1")
    assert plain == []
    assert "name" not in columns  # never shadow the row join key
    assert columns == {"a": "2", "url": "http://x?q=1"}


def test_split_tags_empty():
    assert split_tags("") == ([], {})


def test_build_metadata_keeps_only_plain_tags_in_comfy_json():
    # `key=value` pairs are promoted to columns, so they must NOT linger in the
    # opaque `_comfy_json.tags` list as literal "style=anime" strings.
    meta = build_metadata(tags="hero, style=anime")
    assert meta["tags"] == ["hero"]
    assert "style=anime" not in json.dumps(meta)


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
