from fractions import Fraction

import numpy as np

from comfyui_zegami.encoding import (
    classify_output,
    encode_image_tensor,
    encode_video_frames,
    image_count,
    make_poster,
    save_native_video,
    video_fps,
)


class _FakeComponents:
    def __init__(self, images, frame_rate):
        self.images = images
        self.frame_rate = frame_rate


class FakeNativeVideo:
    """Duck-types ComfyUI's native VIDEO (VideoFromComponents): the only thing
    the node relies on is `get_components()` → an object with `.images`
    ([frames, H, W, C]) and a `.frame_rate` (a Fraction)."""

    def __init__(self, frames, frame_rate=Fraction(24, 1)):
        self._frames = frames
        self._frame_rate = frame_rate

    def get_components(self):
        return _FakeComponents(self._frames, self._frame_rate)


def test_classify_image_batch():
    # The `images` input is always N stills — the primary path.
    assert classify_output(images=np.zeros((4, 8, 8, 3))) == "image_batch"
    assert image_count(np.zeros((4, 8, 8, 3))) == 4
    assert classify_output(images=np.zeros((1, 8, 8, 3))) == "image_batch"


def test_classify_video_frames():
    assert classify_output(video=np.zeros((8, 8, 8, 3))) == "video_frames"


def test_classify_single_frame():
    assert classify_output(video=np.zeros((1, 8, 8, 3))) == "single_frame"


def test_classify_vhs_video():
    assert classify_output(video={"filename": "/tmp/clip.mp4"}) == "vhs_video"


def test_encode_image_tensor_writes_png(tmp_path):
    out = encode_image_tensor(np.zeros((2, 4, 4, 3), dtype=np.uint8), 1, tmp_path / "a.png")
    assert out.exists() and out.stat().st_size > 0


def test_encode_video_frames_builds_streaming_ffmpeg_cmd(tmp_path):
    cmds = []

    def runner(cmd, **kw):
        cmds.append(cmd)

    encode_video_frames(
        np.zeros((3, 4, 6, 3), dtype=np.uint8), tmp_path / "v.mp4", fps=24, runner=runner
    )
    assert len(cmds) == 1
    cmd = cmds[0]
    assert "rawvideo" in cmd
    assert "6x4" in cmd  # WxH
    assert "24" in cmd  # fps
    assert "libx264" in cmd
    # The temp rawvideo scratch file is cleaned up.
    assert not (tmp_path / "v.rgb24.raw").exists()


def test_classify_native_video_frames():
    # ComfyUI's native VIDEO (VideoFromComponents) — regression for the
    # "int() argument must be ... not 'VideoFromComponents'" crash, where a
    # native VIDEO was 0-d-wrapped by np.asarray and mis-encoded as a still.
    vid = FakeNativeVideo(np.zeros((8, 8, 8, 3)))
    assert classify_output(video=vid) == "video_frames"
    assert image_count(vid) == 8


def test_classify_native_single_frame_video():
    vid = FakeNativeVideo(np.zeros((1, 8, 8, 3)))
    assert classify_output(video=vid) == "single_frame"


def test_encode_native_video_frames(tmp_path):
    cmds = []
    vid = FakeNativeVideo(np.zeros((3, 4, 6, 3), dtype=np.uint8), frame_rate=Fraction(30, 1))
    encode_video_frames(vid, tmp_path / "v.mp4", fps=16, runner=lambda cmd, **kw: cmds.append(cmd))
    assert len(cmds) == 1 and "6x4" in cmds[0] and "libx264" in cmds[0]
    assert not (tmp_path / "v.rgb24.raw").exists()


def test_encode_native_single_frame_writes_png(tmp_path):
    # The exact failing path: a 1-frame native VIDEO routed to encode_image_tensor.
    vid = FakeNativeVideo(np.zeros((1, 4, 4, 3), dtype=np.uint8))
    out = encode_image_tensor(vid, 0, tmp_path / "a.png")
    assert out.exists() and out.stat().st_size > 0


def test_video_fps_prefers_native_rate():
    assert video_fps(FakeNativeVideo(np.zeros((2, 4, 4, 3)), Fraction(24, 1)), default=16) == 24
    # A raw frames tensor has no inherent rate → the widget default wins.
    assert video_fps(np.zeros((2, 4, 4, 3)), default=16) == 16


class FakeSavableVideo(FakeNativeVideo):
    """A native VIDEO that can serialize itself (`save_to`) — ComfyUI's real
    VideoFromComponents does this and muxes the audio track. The sentinel bytes
    stand in for "a real mp4 with audio"."""

    def __init__(self, frames, frame_rate=Fraction(24, 1)):
        super().__init__(frames, frame_rate)
        self.saved_to: str | None = None

    def save_to(self, path, *args, **kwargs):
        self.saved_to = str(path)
        with open(path, "wb") as f:
            f.write(b"mp4-with-audio")


def test_save_native_video_uses_save_to(tmp_path):
    vid = FakeSavableVideo(np.zeros((3, 4, 4, 3)))
    out = save_native_video(vid, tmp_path / "v.mp4")
    assert out == tmp_path / "v.mp4"
    assert out.exists() and out.read_bytes() == b"mp4-with-audio"
    assert vid.saved_to == str(tmp_path / "v.mp4")


def test_save_native_video_none_without_save_to(tmp_path):
    # A frames-only native VIDEO (no audio, no serializer) → caller re-encodes.
    assert save_native_video(FakeNativeVideo(np.zeros((3, 4, 4, 3))), tmp_path / "v.mp4") is None
    assert save_native_video(np.zeros((3, 4, 4, 3)), tmp_path / "v.mp4") is None


def test_save_native_video_none_on_failure(tmp_path):
    class Boom:
        def save_to(self, path, *a, **k):
            raise RuntimeError("encoder blew up")

    # A serializer that throws must not strand the upload — return None so the
    # node falls back to the frame re-encode rather than crashing.
    assert save_native_video(Boom(), tmp_path / "v.mp4") is None


def test_save_native_video_none_when_nothing_written(tmp_path):
    class NoOp:
        def save_to(self, path, *a, **k):
            pass  # claims success but writes no file

    assert save_native_video(NoOp(), tmp_path / "v.mp4") is None


def test_make_poster_uses_midpoint(tmp_path):
    cmds = []
    make_poster(
        tmp_path / "v.mp4",
        tmp_path / "p.jpg",
        duration_s=4.0,
        runner=lambda cmd, **kw: cmds.append(cmd),
    )
    joined = " ".join(cmds[0])
    assert "scale=512:-1" in joined
    assert "2.000" in joined  # midpoint of a 4s clip
