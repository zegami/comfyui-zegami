import json

from zegami_client.auth import DEFAULT_ENDPOINT, resolve_api_key, resolve_endpoint


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_env_beats_config_and_node(monkeypatch, tmp_path):
    cfg = _write(tmp_path / "config.json", {"api_key": "from_cfg"})
    monkeypatch.setenv("ZEGAMI_API_KEY", "from_env")
    assert resolve_api_key("from_node", config_path=cfg) == "from_env"


def test_config_beats_node(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    cfg = _write(tmp_path / "config.json", {"api_key": "from_cfg"})
    assert resolve_api_key("from_node", config_path=cfg) == "from_cfg"


def test_node_override_is_last_resort(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    assert resolve_api_key("from_node", config_path=tmp_path / "missing.json") == "from_node"


def test_none_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    assert resolve_api_key("", config_path=tmp_path / "missing.json") is None


def test_malformed_config_is_skipped(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_API_KEY", raising=False)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert resolve_api_key("nd", config_path=bad) == "nd"


def test_endpoint_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("ZEGAMI_ENDPOINT", raising=False)
    assert resolve_endpoint(config_path=tmp_path / "x.json") == DEFAULT_ENDPOINT
    cfg = _write(tmp_path / "c.json", {"endpoint": "https://self.host/"})
    assert resolve_endpoint(config_path=cfg) == "https://self.host"
    monkeypatch.setenv("ZEGAMI_ENDPOINT", "https://env.host/")
    assert resolve_endpoint(config_path=cfg) == "https://env.host"
