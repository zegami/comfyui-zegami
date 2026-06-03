"""zegami_client — a tiny, dependency-light (requests-only) HTTP client for
Zegami's ingest contract. Reusable by the ComfyUI node and the Zegami CLI."""

from .auth import (
    key_fingerprint,
    resolve_api_key,
    resolve_api_key_source,
    resolve_endpoint,
)
from .client import ZegamiClient
from .errors import PermanentError, RetryableError, ZegamiClientError
from .models import BatchUploadResult, UploadItem, UploadResult

__all__ = [
    "ZegamiClient",
    "UploadItem",
    "UploadResult",
    "BatchUploadResult",
    "resolve_api_key",
    "resolve_api_key_source",
    "key_fingerprint",
    "resolve_endpoint",
    "ZegamiClientError",
    "RetryableError",
    "PermanentError",
]
