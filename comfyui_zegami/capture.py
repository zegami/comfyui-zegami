"""Capture everything, surface nothing.

Builds the opaque metadata dict attached to every output: the full prompt
graph (resolved backend inputs — seeds, CFG, checkpoints, LoRAs), the UI
workflow JSON (node titles), and execution metadata. Every optional field is
guarded so a partial capture never fails the upload (spec: "never fail the
upload due to metadata issues"). The result is serialised into the
`_comfy_json` column the viewer maps fields out of.
"""

from __future__ import annotations

import platform
from typing import Any


def _torch_version() -> str | None:
    try:
        import torch  # noqa: PLC0415

        return str(torch.__version__)
    except Exception:
        return None


def _comfy_version() -> str | None:
    try:
        import comfy  # type: ignore  # noqa: PLC0415

        return str(getattr(comfy, "__version__", "")) or None
    except Exception:
        return None


def build_metadata(
    prompt: Any = None,
    extra_pnginfo: Any = None,
    *,
    tags: str = "",
    notes: str = "",
    batch_index: int = 0,
    batch_size: int = 1,
    prompt_id: str | None = None,
    timestamp: str | None = None,
    gen_time_s: float | None = None,
) -> dict:
    meta: dict[str, Any] = {}
    # The prompt graph as the backend received it (the interpretable payload).
    try:
        if prompt is not None:
            meta["prompt"] = prompt
    except Exception:
        pass
    # The UI workflow (node positions + titles) — often holds user-meaningful
    # labels like "main prompt" vs "negative — quality tags".
    try:
        if isinstance(extra_pnginfo, dict) and "workflow" in extra_pnginfo:
            meta["workflow"] = extra_pnginfo["workflow"]
    except Exception:
        pass

    exec_meta: dict[str, Any] = {
        "python_version": platform.python_version(),
    }
    for key, val in (
        ("timestamp", timestamp),
        ("prompt_id", prompt_id),
        ("gen_time_s", gen_time_s),
        ("comfy_version", _comfy_version()),
        ("torch_version", _torch_version()),
    ):
        if val is not None:
            exec_meta[key] = val
    meta["exec"] = exec_meta

    meta["batch_index"] = batch_index
    meta["batch_size"] = batch_size
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    if parsed_tags:
        meta["tags"] = parsed_tags
    if notes:
        meta["notes"] = notes
    return meta
