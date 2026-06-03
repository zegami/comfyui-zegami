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
