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
import time
import uuid
from pathlib import Path
from typing import Any

from zegami_client import (
    UploadItem,
    ZegamiClient,
    key_fingerprint,
    resolve_api_key,
    resolve_api_key_source,
    resolve_endpoint,
)

from .capture import build_metadata
from .encoding import (
    _vhs_path,
    classify_output,
    encode_image_tensor,
    encode_video_frames,
    image_count,
    make_poster,
    save_native_video,
    video_fps,
)
from .queue import UploadJob, get_queue


class ZegamiBatchExport:
    CATEGORY = "Zegami"
    FUNCTION = "export"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "upload_status")
    OUTPUT_NODE = True

    @staticmethod
    def _collection_choices() -> list[str]:
        """Existing collection names for the picker dropdown.

        INPUT_TYPES runs at node-load and can't see the per-node
        api_key_override widget, so this can only use an AMBIENT key
        (ZEGAMI_API_KEY env / ~/.zegami/config.json). Fails soft to `[""]`
        (blank = no pick) on any error — missing/slow/scoped Zegami must never
        block ComfyUI startup. The list refreshes when the node graph reloads.
        NB: a useful picker wants an account/workspace key; a collection-scoped
        key returns a trimmed list. To create a NEW collection, type into
        `collection_name` instead of picking here."""
        try:
            key = resolve_api_key("")
            if not key:
                return [""]
            client = ZegamiClient(resolve_endpoint(""), key, timeout=4)
            names = sorted({c["name"] for c in client.list_collections() if c.get("name")})
            return [""] + names
        except Exception:
            return [""]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "video": ("VIDEO",),
                # Dropdown of existing collections when an ambient key is set
                # (else a lone blank option). Selecting one targets it; leave
                # blank to use collection_id / collection_name.
                "collection_picker": (cls._collection_choices(),),
                "collection_id": ("STRING", {"default": ""}),
                # Create-or-reuse a collection by NAME when no id is given —
                # so a workflow doesn't need a pre-existing collection. id wins
                # if both are set. Needs a key that can create (account- or
                # workspace-scoped; a collection-scoped key can't create).
                "collection_name": ("STRING", {"default": ""}),
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

    @classmethod
    def VALIDATE_INPUTS(cls, fps=None):  # noqa: ANN001, ANN206
        """Tolerate a blank/odd `fps` widget.

        ComfyUI type-validates EVERY declared input before the node runs, and
        an empty `fps` widget (serialised as "") makes its `int("")` coercion
        raise — so ComfyUI rejects the node and silently drops the export
        ('Output will be ignored'). That blocks even IMAGE-only exports, which
        never use fps at all. Declaring `fps` here hands its validation to us;
        we accept anything and coerce defensively in `export()`. Every other
        input keeps ComfyUI's normal validation.
        """
        return True

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _coerce_fps(value: Any, default: int = 16) -> int:
        """Best-effort int in [1, 120]; blank/garbage → default. Mirrors the
        INPUT_TYPES bounds. Because VALIDATE_INPUTS defers fps validation, the
        raw widget value (possibly "" or "16.0") reaches export() unconverted."""
        try:
            v = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default
        return min(120, max(1, v))

    @staticmethod
    def _output_dir() -> Path:
        try:
            import folder_paths  # type: ignore  # ComfyUI runtime

            return Path(folder_paths.get_output_directory())
        except Exception:
            return Path(tempfile.mkdtemp(prefix="zegami_"))

    @staticmethod
    def _make_run_id(unique_id: Any) -> str:
        """A per-execution name prefix for this batch's items.

        `unique_id` is the node's GRAPH id — stable across every queue run of
        the same workflow, so on its own it only disambiguates multiple export
        nodes within one workflow. It must NOT be the sole name key: otherwise
        every generation emits the same item names (`<nodeid>_000000`), which
        extract to the same `raw/<name>` blob path and overwrite the previous
        generation instead of accumulating as a new tile (the "my new image
        isn't showing up" bug).

        Salt it with a per-execution timestamp + short uuid so separate
        generations are distinct. The timestamp keeps names lexicographically
        time-sortable; the uuid suffix closes the gap when two prompts are
        queued in the same wall-clock second (batch / API submit).
        """
        node_id = str(unique_id or "run").replace("/", "_")
        return f"{node_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    @staticmethod
    def _resolve_target_collection(
        client: Any, picker: str, cid: str, cname: str
    ) -> tuple[str | None, str | None]:
        """Resolve the target collection id. Explicit id wins; else a picked
        existing collection; else a typed name (create-or-reuse via
        ensure_collection, which returns the existing one on a name match).
        Returns ``(collection_id, None)`` or ``(None, error_message)``."""
        target = (cid or "").strip()
        if target:
            return target, None
        name = (picker or "").strip() or (cname or "").strip()
        if not name:
            return None, (
                "pick a collection, or set collection_id / collection_name "
                "(name creates/reuses one)"
            )
        try:
            rid = client.ensure_collection(name)
        except Exception as e:  # noqa: BLE001
            return None, (
                f"could not create/find collection {name!r}: {e} "
                "(a collection-scoped key can't create — use an account/workspace key)"
            )
        if not rid:
            return None, f"ensure_collection returned no id for {name!r}"
        return rid, None

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
                # A native VIDEO carries its own frame rate — honour it (encoding
                # at the widget default would play a real clip at the wrong
                # speed); an IMAGE-batch-as-video has none, so the widget wins.
                eff_fps = video_fps(video, fps)
                mp4_path = out_dir / f"{name}.mp4"
                # Prefer the native VIDEO's own serializer: it muxes the AUDIO
                # track (matching ComfyUI's SaveVideo). The frame re-encode only
                # sees the RGB frames and drops audio — fall back to it only for
                # a raw IMAGE-batch-as-video, which has no audio to lose.
                mp4 = save_native_video(video, mp4_path) or encode_video_frames(
                    video, mp4_path, fps=eff_fps
                )
                dur = image_count(video) / float(eff_fps or 16)
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
        collection_picker: str = "",
        collection_id: str = "",
        collection_name: str = "",
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

        # VALIDATE_INPUTS defers fps validation to us, so it may arrive raw
        # (e.g. "" from a blank widget). Coerce before any use.
        fps = self._coerce_fps(fps)

        out_dir = self._output_dir()
        run_id = self._make_run_id(unique_id)
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

        api_key, key_source = resolve_api_key_source(api_key_override)
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
        endpoint = resolve_endpoint(endpoint_override)
        client = ZegamiClient(endpoint, api_key, key_source=key_source)
        # One console line per submit naming the key + source: makes "which key
        # is this node actually using?" answerable at a glance (the upload runs
        # async on the queue, so a later scope-403 is otherwise hard to trace).
        print(
            f"[zegami] export → {endpoint} "
            f"(key {key_fingerprint(api_key)} from {key_source})",
            flush=True,
        )
        # Resolve the target collection (id > picked > typed name). The name
        # branch create-or-reuses via ensure_collection — a quick bounded call;
        # on failure we fail soft like the no-key path so a Zegami hiccup never
        # costs the user their generation.
        target_id, col_err = self._resolve_target_collection(
            client, collection_picker, collection_id, collection_name
        )
        if col_err:
            return (
                images,
                json.dumps(
                    {
                        "success": False,
                        "error": col_err,
                        "saved_local": [str(it.media_path) for it in items],
                    }
                ),
            )

        get_queue().submit(UploadJob(client=client, collection_id=target_id, items=items))
        return (
            images,
            json.dumps(
                {
                    "success": True,
                    "status": "queued",
                    "collection_id": target_id,
                    "items": len(items),
                    "key": key_fingerprint(api_key),
                    "key_source": key_source,
                }
            ),
        )
