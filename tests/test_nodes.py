"""Naming regression tests for ZegamiBatchExport.

The node's item names are the join key AND the blob path stem (zip entries are
written as `<name>.<ext>` and extracted to `raw/<name>`). If two separate
generations of the same workflow emit the same names, the second overwrites the
first's blobs instead of accumulating as a new tile — the "my new ComfyUI image
isn't showing up" bug. `unique_id` is the node's stable graph id, so it cannot
be the sole name key.
"""

import time

from comfyui_zegami.nodes import ZegamiBatchExport


def test_run_id_distinct_across_executions_with_same_node_id(monkeypatch):
    # Freeze the clock so the timestamp is identical between calls — the ONLY
    # remaining disambiguator is the uuid salt. This is the real regression:
    # ComfyUI passes the SAME unique_id (the graph id) on every queue run, and
    # before the fix run_id == str(unique_id) made both runs identical.
    monkeypatch.setattr(time, "time", lambda: 1_780_000_000)

    a = ZegamiBatchExport._make_run_id("9")
    b = ZegamiBatchExport._make_run_id("9")

    assert a != b, "separate generations of the same node must get distinct run_ids"
    # The node id is preserved as a prefix so multiple export nodes in one
    # workflow stay disambiguated too.
    assert a.startswith("9_") and b.startswith("9_")


def test_run_id_unique_over_many_executions():
    # Stress the uniqueness guarantee across a burst of same-node executions
    # (e.g. a queued batch of prompts) — every one must be distinct.
    ids = [ZegamiBatchExport._make_run_id("7") for _ in range(200)]
    assert len(set(ids)) == 200


def test_run_id_falls_back_when_unique_id_missing():
    # A missing unique_id must still produce a usable, distinct prefix.
    rid = ZegamiBatchExport._make_run_id(None)
    assert rid.startswith("run_")
    assert rid != ZegamiBatchExport._make_run_id(None)


# ── collection resolution (id > picker > name → ensure_collection) ──────────

class _FakeClient:
    def __init__(self, ensure_id="col-ensured", raise_ensure=False):
        self._id = ensure_id
        self._raise = raise_ensure
        self.ensure_calls = []

    def ensure_collection(self, name):
        self.ensure_calls.append(name)
        if self._raise:
            raise RuntimeError("boom")
        return self._id


def test_resolve_target_explicit_id_wins():
    c = _FakeClient()
    tid, err = ZegamiBatchExport._resolve_target_collection(c, "picked", "explicit-id", "typed")
    assert (tid, err) == ("explicit-id", None)
    assert c.ensure_calls == []  # id given → no ensure round-trip


def test_resolve_target_picker_beats_name():
    c = _FakeClient(ensure_id="col-1")
    tid, err = ZegamiBatchExport._resolve_target_collection(c, "MyColl", "", "Other")
    assert (tid, err) == ("col-1", None)
    assert c.ensure_calls == ["MyColl"]


def test_resolve_target_typed_name_is_stripped_and_ensured():
    c = _FakeClient(ensure_id="col-2")
    tid, err = ZegamiBatchExport._resolve_target_collection(c, "", "", "  New Coll  ")
    assert (tid, err) == ("col-2", None)
    assert c.ensure_calls == ["New Coll"]


def test_resolve_target_errors_when_nothing_set():
    tid, err = ZegamiBatchExport._resolve_target_collection(_FakeClient(), "", "", "")
    assert tid is None and "collection" in err


def test_resolve_target_ensure_failure_is_soft():
    c = _FakeClient(raise_ensure=True)
    tid, err = ZegamiBatchExport._resolve_target_collection(c, "", "", "X")
    assert tid is None and "could not create/find" in err


def test_resolve_target_ensure_returns_no_id():
    c = _FakeClient(ensure_id=None)
    tid, err = ZegamiBatchExport._resolve_target_collection(c, "", "", "X")
    assert tid is None and "no id" in err


# ── fps robustness (empty widget must not drop the export) ──────────────────

def test_validate_inputs_accepts_blank_fps():
    # The real-world failure: ComfyUI validates inputs before running, and an
    # empty `fps` widget ("") makes its int("") coercion raise → 'Output will
    # be ignored' → export() never runs (no upload, no status). VALIDATE_INPUTS
    # must accept any fps so ComfyUI doesn't reject the node.
    assert ZegamiBatchExport.VALIDATE_INPUTS(fps="") is True
    assert ZegamiBatchExport.VALIDATE_INPUTS(fps=None) is True
    assert ZegamiBatchExport.VALIDATE_INPUTS(fps="30") is True


def test_coerce_fps_handles_blank_and_garbage():
    coerce = ZegamiBatchExport._coerce_fps
    assert coerce("") == 16          # blank widget → default
    assert coerce(None) == 16
    assert coerce("not-a-number") == 16
    assert coerce("24") == 24        # string int
    assert coerce("16.0") == 16      # string float
    assert coerce(30) == 30          # already int
    assert coerce("0") == 1          # clamped to min
    assert coerce("9999") == 120     # clamped to max


# ── video audio preservation (native VIDEO must keep its audio track) ───────

def test_build_items_video_prefers_save_to_for_audio(tmp_path):
    # A native VIDEO with audio must reach Zegami via its own serializer
    # (save_to → muxes audio), NOT the frame re-encode (RGB frames only →
    # silent). Regression: the LTX clip arrived silent because it was rebuilt
    # from get_components().images.
    from fractions import Fraction

    import numpy as np

    class _SavableVideo:
        def __init__(self, frames):
            self._frames = frames
            self.encoded_frames = False

        def get_components(self):
            comps = type("C", (), {})()
            comps.images = self._frames
            comps.frame_rate = Fraction(24, 1)
            return comps

        def save_to(self, path, *args, **kwargs):
            with open(path, "wb") as f:
                f.write(b"mp4-with-audio")

    items = ZegamiBatchExport()._build_items(
        images=None,
        video=_SavableVideo(np.zeros((4, 8, 8, 3))),
        fps=16,
        out_dir=tmp_path,
        base_meta={},
        run_id="r",
    )
    assert len(items) == 1
    it = items[0]
    assert it.media_type == "video"
    # The sentinel bytes prove the clip came from save_to (audio path), not the
    # frame-only ffmpeg re-encode.
    assert it.media_path.read_bytes() == b"mp4-with-audio"


def test_collection_choices_failsoft_without_ambient_key(monkeypatch):
    # INPUT_TYPES must never raise / block at node-load; no ambient key → [""].
    from comfyui_zegami import nodes
    monkeypatch.setattr(nodes, "resolve_api_key", lambda *_a, **_k: "")
    assert ZegamiBatchExport._collection_choices() == [""]


def test_collection_choices_failsoft_on_list_error(monkeypatch):
    from comfyui_zegami import nodes

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def list_collections(self):
            raise RuntimeError("network down")

    monkeypatch.setattr(nodes, "resolve_api_key", lambda *_a, **_k: "zeg_x")
    monkeypatch.setattr(nodes, "resolve_endpoint", lambda *_a, **_k: "https://z.test")
    monkeypatch.setattr(nodes, "ZegamiClient", _Boom)
    assert ZegamiBatchExport._collection_choices() == [""]
