"""HTTP retrieval with a content-addressed on-disk cache and SEC-safe pacing.

Why this module exists rather than calling ``httpx`` directly:

1. **Cost and politeness.** Every retrieval is cached by URL. Re-running the
   pipeline is free and does not re-hit SEC.
2. **Reproducibility.** Each cached artifact carries a sha256 and a retrieval
   timestamp, so a report can state exactly which bytes it was built from.
3. **Rate compliance.** SEC's fair-access policy caps automated access; we pace
   requests rather than risk an IP block.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .config import Settings


class RetrievalError(RuntimeError):
    """Raised when an artifact cannot be retrieved after exhausting retries.

    Callers at the pipeline level catch this and record a *failed source* rather
    than crashing the run - a partially-supported company is still useful, but it
    must be reported as partial.
    """


@dataclass(frozen=True)
class CachedArtifact:
    """A retrieved document plus the metadata needed to cite it."""

    url: str
    content: bytes
    retrieved_at: datetime
    content_hash: str
    cache_path: Path
    from_cache: bool
    content_type: str | None = None
    status_code: int = 200

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    @property
    def size_kb(self) -> float:
        return round(len(self.content) / 1024, 1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Cache:
    """Content-addressed store: ``cache/http/<sha256(url)>/``.

    Layout per entry::

        cache/http/<key>/body.bin     raw bytes exactly as received
        cache/http/<key>/meta.json    url, retrieval time, hash, content type

    Caching the raw bytes (not a parsed form) means an extractor change never
    invalidates the cache, and the cache is trivially auditable.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]

    def entry_dir(self, url: str) -> Path:
        return self.root / self.key_for(url)

    def get(self, url: str, max_age: timedelta | None = None) -> CachedArtifact | None:
        d = self.entry_dir(url)
        body_path, meta_path = d / "body.bin", d / "meta.json"
        if not (body_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            retrieved_at = datetime.fromisoformat(meta["retrieved_at"])
        except (json.JSONDecodeError, KeyError, ValueError):
            # A corrupt cache entry is a cache miss, not a crash.
            return None

        if max_age is not None:
            if datetime.now(timezone.utc) - retrieved_at > max_age:
                return None

        content = body_path.read_bytes()
        return CachedArtifact(
            url=url,
            content=content,
            retrieved_at=retrieved_at,
            content_hash=meta.get("content_hash") or _sha256(content),
            cache_path=body_path,
            from_cache=True,
            content_type=meta.get("content_type"),
            status_code=meta.get("status_code", 200),
        )

    def put(
        self, url: str, content: bytes, content_type: str | None, status_code: int = 200
    ) -> CachedArtifact:
        d = self.entry_dir(url)
        d.mkdir(parents=True, exist_ok=True)
        body_path, meta_path = d / "body.bin", d / "meta.json"
        now = datetime.now(timezone.utc)
        digest = _sha256(content)

        body_path.write_bytes(content)
        meta_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "retrieved_at": now.isoformat(),
                    "content_hash": digest,
                    "content_type": content_type,
                    "status_code": status_code,
                    "bytes": len(content),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return CachedArtifact(
            url=url,
            content=content,
            retrieved_at=now,
            content_hash=digest,
            cache_path=body_path,
            from_cache=False,
            content_type=content_type,
            status_code=status_code,
        )

    def stats(self) -> dict[str, int]:
        entries = [p for p in self.root.iterdir() if p.is_dir()]
        return {
            "entries": len(entries),
            "bytes": sum(
                f.stat().st_size for e in entries for f in e.glob("*") if f.is_file()
            ),
        }


class HttpClient:
    """Paced, retrying, caching HTTP client.

    Pacing is applied between outbound requests only; cache hits cost nothing.
    """

    def __init__(self, settings: Settings, cache: Cache | None = None):
        self.settings = settings
        self.cache = cache or Cache(settings.cache_root / "http")
        self._last_request_at: float = 0.0
        self._client = httpx.Client(
            timeout=settings.http.timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": settings.user_agent,
                "Accept-Encoding": "gzip, deflate",
                # SEC returns JSON/HTML; being explicit avoids content negotiation surprises.
                "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            },
        )
        self.network_requests = 0
        self.cache_hits = 0

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------- #

    def _pace(self) -> None:
        """Enforce the minimum spacing between outbound requests."""
        gap = self.settings.http.min_seconds_between_requests
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _decompress(resp: httpx.Response) -> bytes:
        """Return decoded bytes.

        httpx transparently decodes gzip when reading ``.content``, but only if it
        handled the encoding itself. SEC serves gzip and responses are large
        (companyfacts can exceed 10 MB), so we normalise defensively.
        """
        raw = resp.content
        if resp.headers.get("content-encoding", "").lower() == "gzip":
            try:
                return gzip.decompress(raw)
            except (OSError, EOFError):
                return raw
        return raw

    # -- public API --------------------------------------------------------- #

    def get(
        self,
        url: str,
        *,
        max_age: timedelta | None = None,
        force_refresh: bool = False,
    ) -> CachedArtifact:
        """Fetch a URL, preferring the cache.

        Retries on 429 and 5xx with exponential backoff. A 403 is not retried:
        it means bot-blocking or a missing User-Agent, which retrying cannot fix.
        """
        if not force_refresh:
            cached = self.cache.get(url, max_age=max_age)
            if cached is not None:
                self.cache_hits += 1
                return cached

        last_error: str = "unknown error"
        for attempt in range(self.settings.http.max_retries):
            self._pace()
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
                continue

            self.network_requests += 1

            if resp.status_code == 200:
                body = self._decompress(resp)
                return self.cache.put(
                    url, body, resp.headers.get("content-type"), resp.status_code
                )

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}"
                time.sleep(2**attempt)
                continue

            if resp.status_code == 403:
                raise RetrievalError(
                    f"HTTP 403 Forbidden for {url}. SEC requires a descriptive "
                    f"User-Agent with contact info; bot-protected company sites "
                    f"need a headless browser instead. Not retrying - this is a "
                    f"policy response, not a transient failure."
                )

            last_error = f"HTTP {resp.status_code}"

        raise RetrievalError(f"Failed to retrieve {url} after retries: {last_error}")

    def get_json(self, url: str, **kwargs: object) -> object:
        import json as _json

        artifact = self.get(url, **kwargs)  # type: ignore[arg-type]
        try:
            return _json.loads(artifact.text)
        except _json.JSONDecodeError as exc:
            raise RetrievalError(f"Expected JSON from {url}: {exc}") from exc
