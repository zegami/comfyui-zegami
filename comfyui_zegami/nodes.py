"""The ZegamiBatchExport node.

Drop it inline anywhere in a workflow (it passes `images` through, so it can
sit before a Save node). It captures every output — image OR video — plus the
full workflow graph, and pushes them to a Zegami collection on a background
queue. Fail-soft: media is always encoded locally first, and a Zegami failure
never breaks the generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from zegami_client import UploadItem, ZegamiClient, resolve_api_key, resolve_endpoint

from .capture import build_metadata
from .encoding import (
    _vhs_path,
    classify_output,
    encode_image_tensor,
    encode_video_frames,
    image_count,
    make_poster,
)
from .queue import UploadJob, get_queue


class ZegamiBatchExport:
    CATEGORY = "Zegami"
    FUNCTION = "export"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "upload_status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "video": ("VIDEO",),
                "collection_id": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": ""}),
                "notes": ("STRING", {"multiline": True, "default": ""}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "api_key_override": ("STRING", {"default": ""}),
                "endpoint_override": ("STRING", {"default": ""}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):  # noqa: ANN003
        # Uploads are never cacheable — always re-run.
        return float("nan")

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _output_dir() -> Path:
        try:
            import folder_paths  # type: ignore  # ComfyUI runtime

            return Path(folder_paths.get_output_directory())
        except Exception:
            return Path(tempfile.mkdtemp(prefix="zegami_"))

    def _build_items(
        self,
        images: Any,
        video: Any,
        fps: int,
        out_dir: Path,
        base_meta: dict,
        run_id: str,
    ) -> list[UploadItem]:
        kind = classify_output(images=images, video=video)
        items: list[UploadItem] = []

        if images is not None and kind == "image_batch":
            n = max(1, image_count(images))
            for i in range(n):
                name = f"{run_id}_{i:06d}"
                png = encode_image_tensor(images, i, out_dir / f"{name}.png")
                meta = {**base_meta, "batch_index": i, "batch_size": n}
                items.append(
                    UploadItem(name=name, media_path=png, media_type="image", metadata=meta)
                )
            return items

        name = f"{run_id}_000000"
        if kind == "video_frames":
            try:
                mp4 = encode_video_frames(video, out_dir / f"{name}.mp4", fps=fps)
                dur = image_count(video) / float(fps or 16)
                poster = self._safe_poster(mp4, video, out_dir / f"{name}.jpg", dur)
                items.append(
                    UploadItem(
                        name=name,
                        media_path=mp4,
                        media_type="video",
                        metadata=base_meta,
                        thumbnail_path=poster,
                        duration_s=dur,
                    )
                )
            except Exception:
                # Video encode failed — upload the first frame as a still.
                png = encode_image_tensor(video, 0, out_dir / f"{name}.png")
                items.append(
                    UploadItem(name=name, media_path=png, media_type="image", metadata=base_meta)
                )
        elif kind == "single_frame":
            png = encode_image_tensor(video, 0, out_dir / f"{name}.png")
            items.append(
                UploadItem(name=name, media_path=png, media_type="image", metadata=base_meta)
            )
        elif kind == "vhs_video":
            path = _vhs_path(video)
            if path is not None:
                poster = self._safe_poster(path, None, out_dir / f"{name}.jpg", 0.0)
                items.append(
                    UploadItem(
                        name=name,
                        media_path=path,
                        media_type="video",
                        metadata=base_meta,
                        thumbnail_path=poster,
                    )
                )
        return items

    @staticmethod
    def _safe_poster(media: Path, frames: Any, out: Path, dur: float) -> Path | None:
        try:
            return make_poster(media, out, duration_s=dur)
        except Exception:
            if frames is not None:
                try:
                    from PIL import Image

                    from .encoding import _as_array, _frame_to_uint8

                    arr = _as_array(frames)
                    if arr is not None:
                        Image.fromarray(_frame_to_uint8(arr[0])).save(out, format="JPEG")
                        return out
                except Exception:
                    return None
            return None

    # ── entry point ─────────────────────────────────────────────────────
    def export(
        self,
        images: Any = None,
        video: Any = None,
        collection_id: str = "",
        tags: str = "",
        notes: str = "",
        fps: int = 16,
        api_key_override: str = "",
        endpoint_override: str = "",
        enabled: bool = True,
        prompt: Any = None,
        extra_pnginfo: Any = None,
        unique_id: Any = None,
    ) -> tuple[Any, str]:
        if not enabled:
            return (images, json.dumps({"success": False, "error": "disabled"}))

        out_dir = self._output_dir()
        run_id = str(unique_id or "run").replace("/", "_")
        base_meta = build_metadata(
            prompt,
            extra_pnginfo,
            tags=tags,
            notes=notes,
            prompt_id=str(unique_id) if unique_id is not None else None,
        )

        # Encode locally FIRST — the fail-soft anchor. A later upload error
        # never costs the user their generation.
        try:
            items = self._build_items(images, video, fps, out_dir, base_meta, run_id)
        except Exception as e:
            return (images, json.dumps({"success": False, "error": f"encode failed: {e}"}))

        if not items:
            return (images, json.dumps({"success": False, "error": "no image/video input"}))

        api_key = resolve_api_key(api_key_override)
        if not api_key:
            return (
                images,
                json.dumps(
                    {
                        "success": False,
                        "error": "no API key — set ZEGAMI_API_KEY, ~/.zegami/config.json, "
                        "or the api_key_override input",
                        "saved_local": [str(it.media_path) for it in items],
                    }
                ),
            )
        if not collection_id:
            return (images, json.dumps({"success": False, "error": "collection_id is required"}))

        client = ZegamiClient(resolve_endpoint(endpoint_override), api_key)
        get_queue().submit(UploadJob(client=client, collection_id=collection_id, items=items))
        return (
            images,
            json.dumps(
                {
                    "success": True,
                    "status": "queued",
                    "collection_id": collection_id,
                    "items": len(items),
                }
            ),
        )
