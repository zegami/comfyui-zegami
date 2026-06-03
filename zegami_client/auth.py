"""API-key + endpoint resolution.

Order: explicit node input → environment variable → ~/.zegami/config.json.
The per-node `api_key_override` input WINS — an input named "override" should
override ambient sources (the alternative silently shadows whatever the user
typed into the node, which is surprising and was a real source of "I set the
key but it's using the wrong one" 403s).

For *shareable* workflows, prefer env / config: a value typed into the node
widget is saved into the workflow JSON, so a published workflow could leak it;
env and config are never written into the workflow. The node never reads the
key from the workflow JSON itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_ENDPOINT = "https://app.zegami.com"
DEFAULT_CONFIG_PATH = Path.home() / ".zegami" / "config.json"


def _read_config(config_path: Path) -> dict:
    """Tolerant read — a missing or malformed config yields {}."""
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def resolve_api_key(
    node_override: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str | None:
    """Return the first key found in node input → env → config, else None.

    The explicit `api_key_override` node input takes precedence over the
    ambient `ZEGAMI_API_KEY` env var and `~/.zegami/config.json`.
    """
    if node_override and node_override.strip():
        return node_override.strip()
    env = os.environ.get("ZEGAMI_API_KEY", "").strip()
    if env:
        return env
    cfg_key = str(_read_config(config_path).get("api_key", "")).strip()
    if cfg_key:
        return cfg_key
    return None


def resolve_endpoint(
    node_override: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    """Resolve the base URL: node input → env → config → default."""
    if node_override and node_override.strip():
        return node_override.strip().rstrip("/")
    env = os.environ.get("ZEGAMI_ENDPOINT", "").strip()
    if env:
        return env.rstrip("/")
    cfg_ep = str(_read_config(config_path).get("endpoint", "")).strip()
    if cfg_ep:
        return cfg_ep.rstrip("/")
    return DEFAULT_ENDPOINT
