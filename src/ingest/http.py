"""A polite async JSON client shared by every Parliament API ingest."""

import asyncio
import random
import time
from typing import Any

import httpx

MAX_RETRY_AFTER = 60.0


class FetchError(Exception):
    """A request failed for good: retries exhausted, or a non-retryable status."""


class _Gate:
    """Spaces request starts at least `interval` seconds apart, across all tasks."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            await asyncio.sleep(start - now)

    def penalise(self, seconds: float) -> None:
        """Push every waiter back, e.g. after a 429."""
        self._next = max(self._next, time.monotonic() + seconds)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return min(float(response.headers["Retry-After"]), MAX_RETRY_AFTER)
    except (KeyError, ValueError):
        return None  # absent, or an HTTP-date; fall back to exponential backoff


class PoliteClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        concurrency: int = 6,
        min_interval: float = 0.1,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
    ) -> None:
        self._http = http
        self._sem = asyncio.Semaphore(concurrency)
        self._gate = _Gate(min_interval)
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any | None:
        """GET `url` as JSON. Returns None on 404; raises FetchError once retries are exhausted."""
        return await self._request_json("GET", url, params=params)

    async def post_json(self, url: str, data: dict[str, Any], headers: dict[str, str] | None = None) -> Any | None:
        """POST a form and read JSON back (SPARQL queries are too long for a GET URL). Same retry rules."""
        return await self._request_json("POST", url, data=data, headers=headers)

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> Any | None:
        for attempt in range(1, self._max_attempts + 1):
            retry_after: float | None = None
            async with self._sem:
                await self._gate.wait()
                try:
                    response = await self._http.request(method, url, **kwargs)
                    if response.status_code == 404:
                        return None
                    if response.status_code < 400:
                        return response.json()
                    if response.status_code != 429 and response.status_code < 500:
                        raise FetchError(f"{url} -> HTTP {response.status_code}")
                    reason = f"HTTP {response.status_code}"
                    if response.status_code == 429:
                        retry_after = _retry_after(response)
                        self._gate.penalise(retry_after if retry_after is not None else self._backoff(attempt))
                except (httpx.TransportError, ValueError) as exc:  # ValueError: body was not JSON
                    reason = repr(exc)
            if attempt == self._max_attempts:
                raise FetchError(f"{url} failed after {attempt} attempts ({reason})")
            await asyncio.sleep(retry_after if retry_after is not None else self._backoff(attempt))
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> float:
        return self._backoff_base * 2 ** (attempt - 1) * (1 + random.random() * 0.25)
