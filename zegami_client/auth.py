"""API-key + endpoint resolution.

Order (per the node spec): environment variable → ~/.zegami/config.json →
node input. The key is NEVER read from or written back into a workflow JSON,
so published workflows stay credential-free — a user downloads one, sets their
own key, and runs.
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
    """Return the first key found in env → config → node input, else None."""
    env = os.environ.get("ZEGAMI_API_KEY", "").strip()
    if env:
        return env
    cfg_key = str(_read_config(config_path).get("api_key", "")).strip()
    if cfg_key:
        return cfg_key
    if node_override and node_override.strip():
        return node_override.strip()
    return None


def resolve_endpoint(
    node_override: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    """Resolve the base URL: env → config → node input → default."""
    env = os.environ.get("ZEGAMI_ENDPOINT", "").strip()
    if env:
        return env.rstrip("/")
    cfg_ep = str(_read_config(config_path).get("endpoint", "")).strip()
    if cfg_ep:
        return cfg_ep.rstrip("/")
    if node_override and node_override.strip():
        return node_override.strip().rstrip("/")
    return DEFAULT_ENDPOINT
