import json

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
            if "kind=csv" in url:
                captured["csv"] = kw["data"].read().decode()
            return FakeResponse(200, {"blobPath": "b"})
        if url.endswith("/manage/import-zip"):
            return FakeResponse(202, {"status": "queued"})
        return FakeResponse(404)

    res = _client(handler).upload_batch("col1", items)
    assert res.success
    assert "video" in captured["csv"]
    assert "mp4" in captured["csv"]  # raw_ext
    assert "3.250" in captured["csv"]  # duration_s


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
