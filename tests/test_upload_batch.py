import io
import json
import zipfile

from comfyui_zegami.queue import UploadJob, process_job
from zegami_client import UploadItem, ZegamiClient


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text or json.dumps(self._json)
        self.content = self.text.encode()

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Routes every request through a user-supplied handler(method, url, kwargs)."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def put(self, url, **kw):
        return self._do("PUT", url, kw)

    def post(self, url, **kw):
        return self._do("POST", url, kw)

    def get(self, url, **kw):
        return self._do("GET", url, kw)

    def _do(self, method, url, kw):
        self.calls.append((method, url, kw))
        return self.handler(method, url, kw)


def _client(handler):
    return ZegamiClient(
        "https://z.test", "zeg_secret", session=FakeSession(handler), sleep=lambda *_: None
    )


def _image_items(tmp_path, n=2):
    items = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(b"\x89PNG")
        items.append(
            UploadItem(
                name=f"{i:06d}",
                media_path=p,
                media_type="image",
                metadata={"prompt": {"6": {"class_type": "KSampler", "inputs": {"seed": i}}}},
            )
        )
    return items


def test_upload_batch_promotes_tag_columns_to_csv(tmp_path):
    # `key=value` tags ride their own CSV columns so the ingest promotes them to
    # real, filterable dataset columns. Keys are unioned across items (first-seen
    # order) and DictWriter blank-fills a row that lacks one.
    import csv as _csv
    import io as _io

    captured = {}

    def handler(method, url, kw):
        if "upload-stream" in url:
            if "kind=csv" in url:
                captured["csv"] = kw["data"].read().decode()
            return FakeResponse(200, {"blobPath": "b"})
        if url.endswith("/manage/import-zip"):
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404, text="unexpected")

    items = _image_items(tmp_path, n=2)
    items[0].columns = {"style": "anime"}
    items[1].columns = {"campaign": "spring"}

    _client(handler).upload_batch("col1", items)

    rows = list(_csv.DictReader(_io.StringIO(captured["csv"])))
    assert "style" in rows[0] and "campaign" in rows[0]
    assert rows[0]["style"] == "anime"
    assert rows[0]["campaign"] == ""  # blank-filled — item 0 had no campaign
    assert rows[1]["style"] == ""
    assert rows[1]["campaign"] == "spring"
    # Base columns untouched.
    assert rows[0]["name"] == "000000"
    assert "_comfy_json" in rows[0]


def test_upload_batch_image_happy_path(tmp_path):
    captured = {}

    captured["stage_content_types"] = {}

    captured["upload_ids"] = {}

    def handler(method, url, kw):
        if "upload-stream" in url:
            kind = "csv" if "kind=csv" in url else "zip"
            # The staging PUTs MUST send a Content-Type — without it the
            # server's request adapter drops the body (→ empty_body 400).
            captured["stage_content_types"][kind] = kw["headers"].get("Content-Type")
            # Capture the per-run uploadId so we can assert zip + csv share it.
            from urllib.parse import parse_qs, urlparse
            captured["upload_ids"][kind] = parse_qs(urlparse(url).query).get("uploadId", [None])[0]
            if kind == "csv":
                captured["csv"] = kw["data"].read().decode()
            return FakeResponse(200, {"blobPath": f"col1/_staging/{kind}"})
        if url.endswith("/manage/import-zip"):
            captured["body"] = kw["json"]
            captured["auth"] = kw["headers"]["Authorization"]
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404, text="unexpected")

    res = _client(handler).upload_batch("col1", _image_items(tmp_path))
    assert res.success
    assert res.item_ids == ["000000", "000001"]
    assert captured["body"]["csvJoinCol"] == "name"
    assert captured["body"]["zipBlobs"] == ["col1/_staging/zip"]
    assert captured["body"]["csvBlob"] == "col1/_staging/csv"
    # Default is append=True so successive generations accumulate as
    # distinct tiles rather than overwriting the previous batch.
    assert captured["body"]["append"] is True
    assert captured["auth"] == "Bearer zeg_secret"
    # Both staging PUTs carry a Content-Type (regression — see commit msg).
    assert captured["stage_content_types"] == {"zip": "application/zip", "csv": "text/csv"}
    # The zip + csv of one batch share a single non-empty uploadId, so the
    # run's staged blobs are namespaced together and can't be clobbered by a
    # concurrent upload to the same collection.
    assert captured["upload_ids"]["zip"]
    assert captured["upload_ids"]["zip"] == captured["upload_ids"]["csv"]
    # The opaque prompt graph rides the `_comfy_json` CSV column.
    assert "_comfy_json" in captured["csv"]
    assert "KSampler" in captured["csv"]
    assert "media_kind" in captured["csv"]


def test_upload_batch_append_false_is_forwarded(tmp_path):
    # A caller can opt out of accumulation (one-shot dataset push) and the
    # flag must reach the server body verbatim.
    captured = {}

    def handler(method, url, kw):
        if "upload-stream" in url:
            kind = "csv" if "kind=csv" in url else "zip"
            return FakeResponse(200, {"blobPath": f"col1/_staging/{kind}"})
        if url.endswith("/manage/import-zip"):
            captured["body"] = kw["json"]
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404, text="unexpected")

    res = _client(handler).upload_batch("col1", _image_items(tmp_path), append=False)
    assert res.success
    assert captured["body"]["append"] is False


def test_upload_batch_video_sets_media_kind_and_raw_ext(tmp_path):
    captured = {}
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"MP4")
    poster = tmp_path / "clip.jpg"
    poster.write_bytes(b"JPG")
    items = [
        UploadItem(
            name="000000",
            media_path=mp4,
            media_type="video",
            metadata={},
            thumbnail_path=poster,
            duration_s=3.25,
        )
    ]

    def handler(method, url, kw):
        if "upload-stream" in url:
            body = kw["data"].read()
            if "kind=csv" in url:
                captured["csv"] = body.decode()
            else:
                captured["zip"] = body
            return FakeResponse(200, {"blobPath": "b"})
        if url.endswith("/manage/import-zip"):
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404)

    res = _client(handler).upload_batch("col1", items)
    assert res.success
    assert "video" in captured["csv"]
    assert "mp4" in captured["csv"]  # raw_ext
    assert "3.250" in captured["csv"]  # duration_s
    # The zip carries BOTH the poster (grid tile) and the clip itself (so the
    # server preserves it at raw_assets/<slot>.mp4 for playback) under one stem.
    names = set(zipfile.ZipFile(io.BytesIO(captured["zip"])).namelist())
    assert names == {"000000.jpg", "000000.mp4"}


def test_upload_batch_image_zip_has_no_video(tmp_path):
    captured = {}

    def handler(method, url, kw):
        if "upload-stream" in url:
            if "kind=zip" in url:
                captured["zip"] = kw["data"].read()
            return FakeResponse(200, {"blobPath": "b"})
        if url.endswith("/manage/import-zip"):
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404)

    _client(handler).upload_batch("col1", _image_items(tmp_path, n=1))
    names = set(zipfile.ZipFile(io.BytesIO(captured["zip"])).namelist())
    assert names == {"000000.png"}  # image-only: no extra video entry


def test_upload_batch_retries_transient_then_succeeds(tmp_path):
    state = {"enqueue": 0}

    def handler(method, url, kw):
        if "upload-stream" in url:
            return FakeResponse(200, {"blobPath": "b"})
        if url.endswith("/manage/import-zip"):
            state["enqueue"] += 1
            if state["enqueue"] <= 2:
                return FakeResponse(503, text="busy")
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404)

    res = _client(handler).upload_batch("col1", _image_items(tmp_path, 1))
    assert res.success
    assert state["enqueue"] == 3  # 503, 503, 202


def test_permanent_failure_writes_pending_sidecar(tmp_path):
    def handler(method, url, kw):
        return FakeResponse(401, text="unauthorized")

    items = _image_items(tmp_path, 1)
    job = UploadJob(client=_client(handler), collection_id="col1", items=items)
    res = process_job(job)
    assert not res.success
    sidecar = items[0].media_path.with_name(items[0].media_path.name + ".zegami-pending")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["collection_id"] == "col1"


def test_permanent_failure_is_logged_loudly(tmp_path, capsys):
    # Fail-soft must not be fail-silent: a permanent (e.g. scope-403) failure
    # is printed to the console naming the key fingerprint + source so the
    # user can spot a stale key, and the actionable server body survives.
    server_msg = (
        '{"error":"API key \'zeg_F-z19rfS…\' is not scoped to collection col1 '
        "— mint a key for this collection (or its workspace) under Settings → "
        'API access and point your integration at that key."}'
    )

    def handler(method, url, kw):
        return FakeResponse(403, text=server_msg)

    items = _image_items(tmp_path, 1)
    client = ZegamiClient(
        "https://z.test",
        "zeg_F-z19rfSIFcqd1yeBFnUPUjfgu",
        session=FakeSession(handler),
        sleep=lambda *_: None,
        key_source="ZEGAMI_API_KEY env",
    )
    res = process_job(UploadJob(client=client, collection_id="col1", items=items))
    assert not res.success
    # The full actionable server message survived (not clipped at 200 chars).
    assert "Settings → API access" in res.error
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "zeg_F-z19rfS…" in out  # the key fingerprint
    assert "ZEGAMI_API_KEY env" in out  # the source
    assert "secret" not in out.lower()  # the secret tail never printed


def test_ensure_collection(tmp_path):
    def handler(method, url, kw):
        if url.endswith("/collections"):
            assert kw["json"]["ensure"] is True
            assert kw["json"]["name"] == "My Gallery"
            return FakeResponse(200, {"id": "col_abc"})
        return FakeResponse(404)

    assert _client(handler).ensure_collection("My Gallery") == "col_abc"


def test_list_collections_parses_id_and_name():
    def handler(method, url, kw):
        if method == "GET" and url.endswith("/collections"):
            return FakeResponse(200, [
                {"id": "a", "name": "Alpha"},
                {"id": "b", "name": "Beta"},
                {"name": "no-id"},   # skipped — no id
                {"id": "c"},          # name falls back to id
            ])
        return FakeResponse(404, text="unexpected")

    cols = _client(handler).list_collections()
    assert cols == [
        {"id": "a", "name": "Alpha"},
        {"id": "b", "name": "Beta"},
        {"id": "c", "name": "c"},
    ]
