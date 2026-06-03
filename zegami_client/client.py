"""ZegamiClient — the thin HTTP adapter over Zegami's frozen ingest contract
(see docs/zegami-ingest-contract-v1.md in the platform repo, or
GET /api/v1/ingest/schema.json on any instance).

Flow per batch:
  1. build one zip of media (PNG stills / video poster JPEGs) on disk
  2. build a metadata.csv (RFC-4180 quoted) with the `name` join key,
     `media_kind`, `raw_ext`, `duration_s`, and the opaque `_comfy_json`
     prompt-graph column
  3. stage both via PUT .../import-zip/upload-stream?kind=zip|csv
  4. enqueue via POST .../import-zip {csvJoinCol: "name", ...} — auto-triggers
     the pipeline

This is the productised form of the platform's job-side upload-and-finalise
logic; it depends only on `requests` so the ComfyUI node (and, later, the
Zegami CLI) can share it.
"""

from __future__ import annotations

import csv
import json
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import requests

from .errors import PermanentError, RetryableError
from .models import BatchUploadResult, UploadItem, UploadResult


class ZegamiClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self._sleep = sleep

    # ── HTTP helpers ────────────────────────────────────────────────────
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    def _with_retry(self, fn: Callable[[], requests.Response]) -> requests.Response:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = fn()
                self._raise_for_status(resp)
                return resp
            except RetryableError as e:
                last = e
                if attempt < self.max_retries:
                    self._sleep(2**attempt)  # 1s, 2s, 4s …
                else:
                    raise
        assert last is not None  # unreachable
        raise last

    # ── Public API ──────────────────────────────────────────────────────
    def ensure_collection(self, name: str) -> str | None:
        """Idempotent create-by-name. Returns the collection id.

        NB: relies on the platform's `ensure` flag (POST /collections).
        """

        def call() -> requests.Response:
            return self.session.post(
                f"{self.endpoint}/collections",
                json={"name": name, "ensure": True},
                headers=self._headers(),
                timeout=self.timeout,
            )

        data = self._with_retry(call).json()
        return data.get("id")

    def upload_item(
        self,
        collection_id: str,
        media_path: Path,
        media_type: str,
        metadata: dict,
        thumbnail_path: Path | None = None,
        *,
        name: str | None = None,
        duration_s: float = 0.0,
    ) -> UploadResult:
        item = UploadItem(
            name=name or Path(media_path).stem,
            media_path=Path(media_path),
            media_type=media_type,  # type: ignore[arg-type]
            metadata=metadata,
            thumbnail_path=Path(thumbnail_path) if thumbnail_path else None,
            duration_s=duration_s,
        )
        res = self.upload_batch(collection_id, [item])
        return UploadResult(
            success=res.success, item_id=item.name if res.success else None, error=res.error
        )

    def upload_batch(
        self, collection_id: str, items: Sequence[UploadItem]
    ) -> BatchUploadResult:
        if not items:
            return BatchUploadResult(success=True, collection_id=collection_id, item_ids=[])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            zip_path = tmp_dir / "batch.zip"
            csv_path = tmp_dir / "metadata.csv"
            self._build_zip(zip_path, items)
            self._build_csv(csv_path, items)

            zip_blob = self._stage(collection_id, zip_path, "zip")
            csv_blob = self._stage(collection_id, csv_path, "csv")

            def enqueue() -> requests.Response:
                return self.session.post(
                    f"{self.endpoint}/collection/{collection_id}/manage/import-zip",
                    json={
                        "zipBlobs": [zip_blob],
                        "csvBlob": csv_blob,
                        "csvJoinCol": "name",
                        "thumbSize": 512,
                    },
                    headers=self._headers(),
                    timeout=self.timeout,
                )

            self._with_retry(enqueue)

        return BatchUploadResult(
            success=True, collection_id=collection_id, item_ids=[it.name for it in items]
        )

    def poll_status(self, collection_id: str) -> dict:
        def call() -> requests.Response:
            return self.session.get(
                f"{self.endpoint}/collection/{collection_id}/manage/pipeline",
                headers=self._headers(),
                timeout=self.timeout,
            )

        return self._with_retry(call).json()

    # ── Build helpers ───────────────────────────────────────────────────
    @staticmethod
    def _grid_image(item: UploadItem) -> Path:
        """The file that becomes the grid tile: the poster for a video,
        else the image itself."""
        if item.media_type == "video" and item.thumbnail_path:
            return item.thumbnail_path
        return item.media_path

    def _build_zip(self, zip_path: Path, items: Sequence[UploadItem]) -> None:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for it in items:
                img = self._grid_image(it)
                ext = (img.suffix.lstrip(".") or "png").lower()
                zf.write(img, f"{it.name}.{ext}")

    def _build_csv(self, csv_path: Path, items: Sequence[UploadItem]) -> None:
        # csv.writer quoting keeps the minified-JSON `_comfy_json` cell intact
        # through the server's pyarrow CSV reader → Parquet VARCHAR.
        fields = ["name", "media_kind", "raw_ext", "duration_s", "_comfy_json"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for it in items:
                is_video = it.media_type == "video"
                w.writerow(
                    {
                        "name": it.name,
                        "media_kind": it.media_type,
                        "raw_ext": it.media_path.suffix.lstrip(".").lower() if is_video else "",
                        "duration_s": f"{it.duration_s:.3f}" if is_video and it.duration_s else "",
                        "_comfy_json": json.dumps(it.metadata, separators=(",", ":")),
                    }
                )

    def _stage(self, collection_id: str, path: Path, kind: str) -> str:
        url = (
            f"{self.endpoint}/collection/{collection_id}"
            f"/manage/import-zip/upload-stream?kind={kind}"
        )

        def call() -> requests.Response:
            with open(path, "rb") as body:
                return self.session.put(
                    url, data=body, headers=self._headers(), timeout=self.timeout
                )

        data = self._with_retry(call).json()
        return data.get("blobPath") or data.get("blob") or ""
