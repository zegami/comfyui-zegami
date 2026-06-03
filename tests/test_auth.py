import json

from zegami_client.auth import (
    DEFAULT_ENDPOINT,
    key_fingerprint,
    resolve_api_key,
    resolve_api_key_source,
    resolve_endpoint,
)


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_node_override_beats_env_and_config(monkeypatch, tmp_path):
    # The explicit per-node override wins over both ambient sources — an input
    # named `api_key_override` must override (regression for the 403 where a
    # leftover ZEGAMI_API_KEY shadowed the key the user set on the node).
    cfg = _write(tmp_path / "config.json", {"api_key": "from_cfg"})
    monkeypatch.setenv("ZEGAMI_API_KEY", "from_env")
    assert resolve_api_key("from_node", config_path=cfg) == "from_node"


def test_env_beats_config_when_no_override(monkeypatch, tmp_path):
    cfg = _write(tmp_path / "config.json", {"api_key": "from_cfg"})
    monkeypatch.setenv("ZEGAMI_API_KEY", "from_env")
    assert resolve_api_key("", config_path=cfg) == "from_env"


def test_config_used_when_only_config(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    cfg = _write(tmp_path / "config.json", {"api_key": "from_cfg"})
    assert resolve_api_key("", config_path=cfg) == "from_cfg"


def test_none_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    assert resolve_api_key("", config_path=tmp_path / "missing.json") is None


def test_malformed_config_is_skipped(monkeypatch, tmp_path):
    # No override, no env, malformed config → tolerated as {} → None (not a crash).
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert resolve_api_key("", config_path=bad) is None


def test_source_names_where_the_key_came_from(monkeypatch, tmp_path):
    # The source label is the diagnostic that lets a user spot a stale key
    # winning from a forgotten source.
    cfg = _write(tmp_path / "config.json", {"api_key": "from_cfg"})
    monkeypatch.setenv("ZEGAMI_API_KEY", "from_env")
    assert resolve_api_key_source("from_node", config_path=cfg) == ("from_node", "node input")
    assert resolve_api_key_source("", config_path=cfg) == ("from_env", "ZEGAMI_API_KEY env")
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    key, source = resolve_api_key_source("", config_path=cfg)
    assert key == "from_cfg"
    assert source == str(cfg)


def test_source_none_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    assert resolve_api_key_source("", config_path=tmp_path / "missing.json") == (None, "none")


def test_key_fingerprint_is_non_secret(monkeypatch):
    # The fingerprint must identify a key (its `zeg_<prefix>` head, which the
    # platform stores and shows) without revealing enough to use it.
    token = "zeg_F-z19rfSIFcqd1yeBFnUPUjfgu"
    fp = key_fingerprint(token)
    assert fp.startswith("zeg_F-z19rfS")
    assert "IFcqd1yeBFnUPUjfgu" not in fp  # the secret tail is elided
    assert fp.endswith("…")
    assert key_fingerprint(None) == "<none>"
    assert key_fingerprint("") == "<none>"


def test_endpoint_override_wins(monkeypatch, tmp_path):
    # Endpoint resolution mirrors the key: explicit node input wins.
    cfg = _write(tmp_path / "c.json", {"endpoint": "https://cfg.host/"})
    monkeypatch.setenv("ZEGAMI_ENDPOINT", "https://env.host/")
    assert resolve_endpoint("https://node.host/", config_path=cfg) == "https://node.host"


def test_endpoint_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_ENDPOINT", raising=False)
    assert resolve_endpoint(config_path=tmp_path / "x.json") == DEFAULT_ENDPOINT
    cfg = _write(tmp_path / "c.json", {"endpoint": "https://self.host/"})
    assert resolve_endpoint(config_path=cfg) == "https://self.host"
    monkeypatch.setenv("ZEGAMI_ENDPOINT", "https://env.host/")
    assert resolve_endpoint(config_path=cfg) == "https://env.host"
