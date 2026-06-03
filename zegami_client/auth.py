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


def resolve_api_key_source(
    node_override: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str | None, str]:
    """Like :func:`resolve_api_key`, but also returns WHERE the key came from
    (``"node input"`` / ``"ZEGAMI_API_KEY env"`` / ``"~/.zegami/config.json"`` /
    ``"none"``).

    The source is the single most useful thing to surface on an auth/scope
    failure: the classic "I minted a new key but still get 403" is almost
    always a STALE key winning from a source the user forgot about (an old
    env var, a leftover config file). Naming the source tells them which one
    to fix.
    """
    if node_override and node_override.strip():
        return node_override.strip(), "node input"
    env = os.environ.get("ZEGAMI_API_KEY", "").strip()
    if env:
        return env, "ZEGAMI_API_KEY env"
    cfg_key = str(_read_config(config_path).get("api_key", "")).strip()
    if cfg_key:
        return cfg_key, str(config_path)
    return None, "none"


def resolve_api_key(
    node_override: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str | None:
    """Return the first key found in node input → env → config, else None.

    The explicit `api_key_override` node input takes precedence over the
    ambient `ZEGAMI_API_KEY` env var and `~/.zegami/config.json`.
    """
    return resolve_api_key_source(node_override, config_path)[0]


def key_fingerprint(token: str | None) -> str:
    """A non-secret identifier for a key — the `zeg_<prefix>` head Zegami
    itself stores and shows, safe to print in logs. Never reveals enough of
    the token to use it."""
    if not token:
        return "<none>"
    t = token.strip()
    # Tokens look like `zeg_<prefix><secret>`; the platform's stored prefix is
    # the first 8 chars after `zeg_`. Show that head and elide the rest.
    head = t[:12] if t.startswith("zeg_") else t[:8]
    return f"{head}…"


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
