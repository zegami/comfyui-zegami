import numpy as np

from comfyui_zegami.encoding import (
    classify_output,
    encode_image_tensor,
    encode_video_frames,
    image_count,
    make_poster,
)


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
