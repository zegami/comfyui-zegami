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
        arr = _as_array(video)
        if arr is not None and arr.ndim == 4 and arr.shape[0] > 1:
            return "video_frames"
        return "single_frame"
    # `images` is the primary path — always a batch of stills.
    return "image_batch"


def image_count(images: Any) -> int:
    arr = _as_array(images)
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

    arr = _as_array(images)
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
    arr = _as_array(frames)
    if arr is None or arr.ndim != 4:
        raise ValueError("video frames must be a [frames, H, W, C] tensor")
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
