"""Background upload queue + fail-soft sidecar.

A single global queue with a small daemon worker pool drains upload jobs off
the generation thread so the node returns immediately and never slows a
generation. The client already retries transient failures; a PermanentError
(or exhausted retries) writes a `.zegami-pending` sidecar next to the media so
a future retry CLI (v1.1) can pick it up — the generation always completes.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from zegami_client import BatchUploadResult, UploadItem, ZegamiClient
from zegami_client.errors import PermanentError


@dataclass
class UploadJob:
    client: ZegamiClient
    collection_id: str
    items: list[UploadItem] = field(default_factory=list)


def _write_pending(job: UploadJob, error: str) -> None:
    for it in job.items:
        media = Path(it.media_path)
        sidecar = media.with_name(media.name + ".zegami-pending")
        payload = {
            "collection_id": job.collection_id,
            "endpoint": job.client.endpoint,
            "name": it.name,
            "media": str(media),
            "media_type": it.media_type,
            "error": error,
        }
        try:
            sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass  # best-effort — never raise out of the worker


def _log_failure(job: UploadJob, error: str) -> None:
    """Surface an upload failure on the ComfyUI console. Fail-SOFT must not
    mean fail-SILENT: the generation is safe, but the user needs to know the
    push didn't land — and which key (fingerprint + source) was used, since
    the #1 cause is a stale key winning from a forgotten env var / config."""
    label = getattr(job.client, "key_label", "?")
    source = getattr(job.client, "key_source", "unknown")
    print(
        f"[zegami] upload to '{job.collection_id}' FAILED "
        f"(key {label} from {source}): {error}\n"
        f"[zegami] {len(job.items)} item(s) saved locally with a "
        f".zegami-pending sidecar — generation was not affected.",
        flush=True,
    )


def process_job(job: UploadJob) -> BatchUploadResult:
    """Run one job to completion. Never raises — on failure it writes the
    fail-soft sidecar, logs to the console, and returns an unsuccessful
    result."""
    try:
        return job.client.upload_batch(job.collection_id, job.items)
    except PermanentError as e:
        _write_pending(job, str(e))
        _log_failure(job, str(e))
        return BatchUploadResult(
            success=False, collection_id=job.collection_id, error=str(e)
        )
    except Exception as e:  # exhausted retries / unexpected
        _write_pending(job, str(e))
        _log_failure(job, str(e))
        return BatchUploadResult(
            success=False, collection_id=job.collection_id, error=str(e)
        )


class UploadQueue:
    """Lazily-started daemon-thread pool. Bounded so a 1000-image XYZ plot
    back-pressures rather than OOMing."""

    def __init__(self, workers: int = 2) -> None:
        self._q: queue.Queue[UploadJob] = queue.Queue(maxsize=256)
        self._workers = max(1, workers)
        self._threads: list[threading.Thread] = []
        self._started = False
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            for _ in range(self._workers):
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
                self._threads.append(t)

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            try:
                process_job(job)
            finally:
                self._q.task_done()

    def submit(self, job: UploadJob) -> None:
        self._ensure_started()
        self._q.put(job)

    def join(self) -> None:
        self._q.join()


_GLOBAL_QUEUE: UploadQueue | None = None


def get_queue() -> UploadQueue:
    global _GLOBAL_QUEUE
    if _GLOBAL_QUEUE is None:
        _GLOBAL_QUEUE = UploadQueue()
    return _GLOBAL_QUEUE
