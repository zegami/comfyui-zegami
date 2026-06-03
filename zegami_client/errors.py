"""Error taxonomy so the upload queue knows what to retry vs. fail-soft."""


class ZegamiClientError(Exception):
    """Base for all client errors."""


class RetryableError(ZegamiClientError):
    """Transient — network blip, 5xx, or 429. Worth retrying with backoff."""


class PermanentError(ZegamiClientError):
    """Won't fix itself — 4xx auth / permission / bad request. Fail soft."""
