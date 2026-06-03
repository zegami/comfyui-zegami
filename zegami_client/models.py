"""Adapter data shapes (the spec's UploadItem / UploadResult interface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MediaType = Literal["image", "video"]


@dataclass
class UploadItem:
    """One generation to push.

    `name` is the zero-padded join key (matches the zip-ingest convention);
    `media_path` is the PNG (image) or MP4 (video); for video, `thumbnail_path`
    is the poster JPEG that becomes the grid tile. `metadata` is the opaque
    capture dict (prompt graph + exec info) serialised into the `_comfy_json`
    column. `duration_s` is surfaced as its own column for the grid badge.
    """

    name: str
    media_path: Path
    media_type: MediaType
    metadata: dict = field(default_factory=dict)
    thumbnail_path: Path | None = None
    duration_s: float = 0.0


@dataclass
class UploadResult:
    success: bool
    item_id: str | None = None
    error: str | None = None


@dataclass
class BatchUploadResult:
    success: bool
    collection_id: str
    item_ids: list[str] = field(default_factory=list)
    error: str | None = None
