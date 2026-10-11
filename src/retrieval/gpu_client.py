"""Async client for the local GPU service (gpu/api.py, 127.0.0.1:8001): query embedding and reranking."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
from pgvector import Vector

from app.config import settings

EMBEDDING_DIM = 384  # the service's /embed answers with 384-dimensional unit vectors (BAAI/bge-small-en-v1.5)
_UNIT_TOLERANCE = 1e-3
DEFAULT_TIMEOUT = httpx.Timeout(connect=2.0, read=30.0, write=5.0, pool=5.0)  # read is generous: a busy GPU queues


class GpuError(Exception):
    """Base of everything this client raises."""


class GpuUnavailable(GpuError):
    """Nothing usable answered: refused, unreachable, dropped, or the circuit breaker is open."""


class GpuTimeout(GpuUnavailable):
    """The service accepted the request and did not answer in time."""


class GpuBusy(GpuError):
    """503 after every retry: the service is alive but its queue is full or its GPU is out of memory."""


class GpuBadRequest(GpuError):
    """4xx: the request was refused (a bug on our side, not a service problem)."""


class GpuServerError(GpuError):
    """5xx other than 503: the service failed on this request."""


class GpuBadResponse(GpuError):
    """A 200 whose body cannot be trusted (wrong shape, wrong size, not unit length, not finite)."""


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; after `cooldown` seconds it lets calls through again, and a
    success closes it while a failure re-opens it. Only availability failures count: a busy or a rejecting service
    is alive.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown, self._clock = threshold, cooldown, clock
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        return self._opened_at is None or self._clock() - self._opened_at >= self.cooldown

    def success(self) -> None:
        self._failures, self._opened_at = 0, None

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = self._clock()


class GpuClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        retries: int = 2,
        max_backoff: float = 2.0,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.timeout, self.retries, self.max_backoff, self._sleep = timeout, retries, max_backoff, sleep
        self.breaker = CircuitBreaker(breaker_threshold, breaker_cooldown, clock)
        self._http = httpx.AsyncClient(base_url=(base_url or settings.gpu_url).rstrip("/"), timeout=timeout, transport=transport)

    @property
    def base_url(self) -> httpx.URL:
        return self._http.base_url

    async def __aenter__(self) -> "GpuClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- the calls --------------------------------------------------------------------------------------------------

    async def embed_query(self, text: str) -> Vector:
        """The unit vector for a search query. `is_query` is always true: BGE embeds a query with its retrieval
        instruction and a passage without, and a query embedded as a passage retrieves noticeably worse.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("cannot embed an empty query")
        body = await self._post("/embed", {"texts": [text], "is_query": True})
        vectors = body.get("embeddings") if isinstance(body, dict) else None
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise GpuBadResponse("the embedding answer does not hold exactly one vector")
        vector = vectors[0]
        if vector is None:
            raise GpuBadResponse("the service could not embed this query")
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
            raise GpuBadResponse(f"expected a vector of {EMBEDDING_DIM} numbers")
        try:
            floats = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else math.nan for x in vector]
        except (TypeError, ValueError):
            raise GpuBadResponse("the vector holds a non-number") from None
        if not all(math.isfinite(x) for x in floats):
            raise GpuBadResponse("the vector holds a non-finite number")
        if abs(math.sqrt(sum(x * x for x in floats)) - 1.0) > _UNIT_TOLERANCE:
            raise GpuBadResponse("the vector is not unit length")
        return Vector(floats)

    async def verify(self, pairs: Sequence[tuple[str, str]],
                     threshold: float | None = None) -> list[float | None]:  # fmt: skip
        """One support probability per (evidence, claim) pair, in the order given."""
        if not pairs:
            return []
        payload: dict[str, object] = {
            "pairs": [{"evidence": evidence, "claim": claim} for evidence, claim in pairs]
        }
        if threshold is not None:
            payload["threshold"] = threshold
        body = await self._post("/verify", payload)
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise GpuBadResponse("the verify answer holds no results list")
        scores: list[float | None] = [None] * len(pairs)
        for item in results:
            if not isinstance(item, dict):
                raise GpuBadResponse("a verify result is not an object")
            index, score = item.get("index"), item.get("score")
            if not isinstance(index, int) or not 0 <= index < len(pairs):
                raise GpuBadResponse(f"a verify result names index {index!r}, which was not requested")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
                raise GpuBadResponse("a verify result holds a non-finite score")
            scores[index] = float(score)
        return scores

    async def rerank(self, query: str, candidates: Sequence[str]) -> list[float | None]:
        """One relevance score per candidate, in the order given; None for a candidate the service could not score.
        """
        if not candidates:
            return []
        body = await self._post("/rerank", {"query": query, "candidates": list(candidates)})
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise GpuBadResponse("the rerank answer has no results")
        scores: list[float | None] = [None] * len(candidates)
        seen: set[int] = set()
        for item in results:
            index = item.get("index") if isinstance(item, dict) else None
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(candidates) or index in seen:
                raise GpuBadResponse("the rerank answer has a bad candidate index")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise GpuBadResponse("the rerank answer has a bad score")
            seen.add(index)
            scores[index] = float(score)
        return scores

    async def health(self) -> dict[str, Any]:
        response = await self._send("GET", "/health", None)
        return response

    # -- transport --------------------------------------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        return await self._send("POST", path, payload)

    async def _send(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        if not self.breaker.allow():
            raise GpuUnavailable("the GPU service is failing repeatedly: circuit open, not trying again yet")
        for attempt in range(self.retries + 1):
            try:
                response = await self._http.request(method, path, json=payload)
            except httpx.TimeoutException as exc:
                if isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout)):
                    self.breaker.failure()
                    raise GpuUnavailable("could not connect to the GPU service in time") from None
                self.breaker.failure()
                raise GpuTimeout("the GPU service did not answer in time") from None
            except httpx.TransportError as exc:
                self.breaker.failure()
                raise GpuUnavailable(f"the GPU service is unreachable ({type(exc).__name__})") from None
            if response.status_code == 503:
                if attempt < self.retries:
                    await self._sleep(self._backoff(response, attempt))
                    continue
                raise GpuBusy("the GPU service is busy (503) and stayed busy through the retries")
            if response.status_code >= 500:
                self.breaker.failure()
                raise GpuServerError(f"the GPU service failed ({response.status_code})")
            if response.status_code >= 400:
                raise GpuBadRequest(f"the GPU service refused the request ({response.status_code})")
            self.breaker.success()
            try:
                return response.json()
            except ValueError:
                raise GpuBadResponse("the GPU service answered with something that is not JSON") from None
        raise GpuBusy("the GPU service is busy")  # unreachable: the loop always returns or raises

    def _backoff(self, response: httpx.Response, attempt: int) -> float:
        """The service's Retry-After when it is a sensible number, else 0.25 s doubling each attempt; never above
        max_backoff.
        """
        header = response.headers.get("Retry-After")
        try:
            wanted = float(header) if header is not None else 0.25 * 2**attempt
        except ValueError:
            wanted = 0.25 * 2**attempt
        if not math.isfinite(wanted) or wanted < 0:
            wanted = 0.25 * 2**attempt
        return min(wanted, self.max_backoff)
