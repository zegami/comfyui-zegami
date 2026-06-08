"""Tensor → media encoding.

IMAGE is the primary path: a ComfyUI IMAGE tensor `[N, H, W, C]` is N separate
still items (one PNG each). VIDEO is the secondary path: an explicit `video`
input of frames `[frames, H, W, C]` (frames > 1) encodes to one MP4 via ffmpeg,
streaming frames through a temp raw file so a long clip is never held in RAM
twice; a VHS_VIDEO dict is passed through by file path.

numpy is used for shape handling (ComfyUI ships it); PIL for PNG; ffmpeg (a
system binary) for video + poster frames.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

Runner = Callable[..., Any]


def _as_array(x: Any) -> np.ndarray | None:
    """Best-effort convert a torch tensor / numpy array / list to numpy.
    Returns None if it isn't array-like."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # torch tensor (duck-typed so we don't import torch)
    detach = getattr(x, "detach", None)
    if callable(detach):
        try:
            return detach().cpu().numpy()
        except Exception:
            return None
    try:
        return np.asarray(x)
    except Exception:
        return None


def _video_components(video: Any) -> tuple[np.ndarray | None, int | None]:
    """Frames + fps from a ComfyUI NATIVE VIDEO object (`VideoFromComponents`
    and any other `comfy_api` VideoInput — duck-typed on `get_components`).
    ComfyUI's newer video nodes emit these instead of an IMAGE batch tensor;
    `get_components()` yields `.images` (`[frames, H, W, C]`) plus a
    `.frame_rate` (a `Fraction`). Returns (None, None) when `video` isn't a
    native VIDEO, so callers fall through to the tensor / VHS paths.

    Without this, `_as_array(video)` wraps the object in a 0-d numpy *object*
    array (not None), the clip is misclassified as a single frame, and
    `_frame_to_uint8(...).astype(np.uint8)` calls `int()` on the video object —
    `int() argument must be ... not 'VideoFromComponents'`.
    """
    get_components = getattr(video, "get_components", None)
    if not callable(get_components):
        return None, None
    try:
        comps = get_components()
        frames = _as_array(getattr(comps, "images", None))
    except Exception:
        return None, None
    fps: int | None = None
    rate = getattr(comps, "frame_rate", None)
    if rate is not None:
        try:
            fps = max(1, round(float(rate)))
        except (TypeError, ValueError):
            fps = None
    return frames, fps


def _frames_array(x: Any) -> np.ndarray | None:
    """Frames `[N, H, W, C]` from either a native VIDEO object or an array-like
    tensor. Native VIDEO is tried first so it never reaches `np.asarray` (which
    would 0-d-wrap the object and break the downstream encode)."""
    frames, _ = _video_components(x)
    if frames is not None:
        return frames
    return _as_array(x)


def video_fps(video: Any, default: int) -> int:
    """A native VIDEO's own frame rate (authoritative — re-timing a real clip to
    a guessed default would play it at the wrong speed), else the caller's
    default (the node's `fps` widget)."""
    _, fps = _video_components(video)
    return fps if fps else default


def _vhs_path(video: Any) -> Path | None:
    """Extract a file path from a VideoHelperSuite-style VHS_VIDEO object."""
    if isinstance(video, dict):
        for key in ("filename", "file", "path", "video_path"):
            val = video.get(key)
            if isinstance(val, str) and val:
                return Path(val)
    fn = getattr(video, "filename", None) or getattr(video, "path", None)
    if isinstance(fn, str) and fn:
        return Path(fn)
    return None


def classify_output(images: Any = None, video: Any = None) -> str:
    """Decide how to handle the node's output.

    Returns one of: 'image_batch' (N stills from `images`), 'video_frames'
    (encode an MP4 from a frames tensor), 'vhs_video' (pass through a file),
    'single_frame' (a 1-frame video → treat as one image).
    """
    if video is not None:
        if _vhs_path(video) is not None:
            return "vhs_video"
        # Native VIDEO (VideoFromComponents) OR a raw frames tensor — both
        # normalise to `[frames, H, W, C]` via `_frames_array`.
        arr = _frames_array(video)
        if arr is not None and arr.ndim == 4 and arr.shape[0] > 1:
            return "video_frames"
        return "single_frame"
    # `images` is the primary path — always a batch of stills.
    return "image_batch"


def image_count(images: Any) -> int:
    arr = _frames_array(images)
    if arr is None:
        return 0
    return int(arr.shape[0]) if arr.ndim == 4 else 1


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    """HWC float[0,1] or uint8 → uint8 RGB (drops alpha if present)."""
    if frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    if np.issubdtype(frame.dtype, np.floating):
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame.astype(np.uint8)


def encode_image_tensor(images: Any, index: int, out_path: Path) -> Path:
    """Write the `index`-th still of an IMAGE batch as a PNG."""
    from PIL import Image  # lazy: ComfyUI provides PIL at runtime

    arr = _frames_array(images)
    if arr is None:
        raise ValueError("images is not array-like")
    frame = arr[index] if arr.ndim == 4 else arr
    Image.fromarray(_frame_to_uint8(frame)).save(out_path, format="PNG")
    return out_path


def encode_video_frames(
    frames: Any, out_path: Path, fps: int = 16, *, runner: Runner = subprocess.run
) -> Path:
    """Encode `[frames, H, W, C]` to an MP4. Frames are streamed one at a time
    to a temp rawvideo file (never a second full-clip copy in RAM), then ffmpeg
    transcodes from disk."""
    arr = _frames_array(frames)
    if arr is None or arr.ndim != 4:
        raise ValueError("video frames must be a [frames, H, W, C] tensor or a native VIDEO")
    n, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    raw = Path(out_path).with_suffix(".rgb24.raw")
    with open(raw, "wb") as f:
        for i in range(n):
            f.write(_frame_to_uint8(arr[i]).tobytes())
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", str(raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    try:
        runner(cmd, check=True, capture_output=True)
    finally:
        raw.unlink(missing_ok=True)
    return Path(out_path)


def save_native_video(video: Any, out_path: Path) -> Path | None:
    """Serialize a ComfyUI NATIVE VIDEO object to `out_path`, preserving its
    AUDIO track. Duck-typed on `save_to` — ComfyUI's `VideoInput.save_to`
    muxes video + audio (it's exactly what the built-in SaveVideo uses), so the
    clip Zegami receives matches the `/output/video` copy that has sound.

    `encode_video_frames` only ever sees `get_components().images` (the RGB
    frames) and so DROPS the audio; prefer this whenever the VIDEO object can
    serialize itself.

    Returns None when `video` has no usable `save_to` — a raw IMAGE-batch-as-
    video frames tensor, which carries no audio to lose — so the caller falls
    back to the frame re-encode. Also returns None (→ fallback) if `save_to`
    raises or writes nothing, so a serializer quirk can't lose the clip
    entirely.
    """
    save_to = getattr(video, "save_to", None)
    if not callable(save_to):
        return None
    try:
        # AUTO container/codec: ComfyUI infers mp4 + h264 from the `.mp4`
        # suffix and muxes the audio stream when the VIDEO has one.
        save_to(str(out_path))
    except Exception:
        return None
    p = Path(out_path)
    return p if p.exists() and p.stat().st_size > 0 else None


def make_poster(
    video_path: Path, out_path: Path, *, duration_s: float = 0.0, runner: Runner = subprocess.run
) -> Path:
    """Extract a 512-wide midpoint poster JPEG from a video (matches the
    server-side AARO convention)."""
    ss = max(0.0, duration_s / 2.0)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{ss:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", "scale=512:-1",
        str(out_path),
    ]
    runner(cmd, check=True, capture_output=True)
    return Path(out_path)
