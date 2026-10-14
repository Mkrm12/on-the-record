"""Local GPU service: query embedding and reranking on 127.0.0.1:8001."""

from __future__ import annotations

import argparse
import asyncio
import gc
import ipaddress
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from embedder import DIMENSIONS, MODEL_NAME, MODEL_REVISION, clean_for_embedding, validate_vectors
import verifier as verifier_module
from reranker import RERANKER_NAME, RERANKER_REVISION, GpuOutOfMemory

log = logging.getLogger("gpu.api")

HOST = "127.0.0.1"
PORT = 8001
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]  # Host headers accepted; Starlette cannot parse an IPv6 literal, so IPv4 only

MAX_EMBED_TEXTS = 512  # per /embed call
MAX_RERANK_CANDIDATES = 512  # per /rerank call (hybrid retrieval sends a few dozen)
MAX_VERIFY_PAIRS = 128  # per /verify call; an answer has a handful of claims, not hundreds
MAX_TEXT_CHARS = 10_000  # longer text is cut; 512 tokens is about 2,000 characters, so nothing useful is lost
MAX_BODY_BYTES = 32 * 2**20  # refused before it is read.
# with every character sent as a \uXXXX escape (6 bytes): 30.7 MB
MAX_QUERY_CHARS = 2_000  # a rerank query is paired with every candidate, so it is held to a tighter cap
MAX_QUEUED = 8  # requests running or waiting for the GPU; beyond this the service answers 503

EMBED_BATCH_SIZE = 128  # worst case (512 texts of 512 tokens) measured at +579 MiB
MIN_FREE_VRAM_MIB = 4096  # the headroom the maintainer wants left for everything else; /health reports whether it holds


# --------------------------------------------------------------------------- input hygiene (pure)
# ---------------------------------------------------------------------------


def sanitize(text: str | None, limit: int = MAX_TEXT_CHARS) -> str | None:
    """A client string made safe for a model, or None when nothing embeddable is left. Over-long text is cut at `limit`
    characters, and the cut is checked again (a cut can leave only punctuation).
    """
    cleaned = clean_for_embedding(text)
    if cleaned is not None and len(cleaned) > limit:
        cleaned = clean_for_embedding(cleaned[:limit])
    return cleaned


def dedupe(texts: Sequence[str | None]) -> tuple[list[str], list[int | None]]:
    """(unique strings in first-seen order, and for each input the index of its string in that list, None for None).
    Identical inputs are scored once and get identical answers.
    """
    unique: list[str] = []
    position: dict[str, int] = {}
    slots: list[int | None] = []
    for text in texts:
        if text is None:
            slots.append(None)
            continue
        if text not in position:
            position[text] = len(unique)
            unique.append(text)
        slots.append(position[text])
    return unique, slots


def vector_ok(vector: np.ndarray) -> bool:
    """True for a finite, unit-length vector of the right size. Anything else is a model fault and is never sent out.
    """
    try:
        validate_vectors(np.asarray(vector)[None, :])
    except (ValueError, IndexError):
        return False
    return True


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.version == 4 and address.is_loopback


# --------------------------------------------------------------------------- the GPU: one thread, one job at a time,
# bounded queue ---------------------------------------------------------------------------


class GpuBusy(Exception):
    """More than MAX_QUEUED requests are already running or waiting."""


class GpuQueue:
    """Runs every model call on a single worker thread. That gives exclusive use of the GPU without a lock (so VRAM is
    bounded by one micro-batch), keeps the CUDA handles warm in one thread, and leaves the event loop free.
    """

    def __init__(self, max_queued: int = MAX_QUEUED):
        self.max_queued = max_queued
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._lock = threading.Lock()
        self._queued = 0

    @property
    def queued(self) -> int:
        return self._queued

    def _release(self, _future) -> None:  # also runs when a waiting job is cancelled, so the count cannot leak
        with self._lock:
            self._queued -= 1

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        with self._lock:
            if self._queued >= self.max_queued:
                raise GpuBusy(f"{self._queued} requests already waiting for the GPU")
            self._queued += 1
        future = self._pool.submit(fn, *args)
        future.add_done_callback(self._release)
        return await asyncio.wrap_future(future)

    async def run_unbounded(self, fn: Callable[..., Any], *args: Any) -> Any:
        """For startup and shutdown work, which must never be refused."""
        return await asyncio.wrap_future(self._pool.submit(fn, *args))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


# --------------------------------------------------------------------------- the engine: the two models behind one
# small interface ---------------------------------------------------------------------------


class Engine(Protocol):
    def embed(self, texts: list[str], is_query: bool) -> list[np.ndarray | None]: ...
    def score(self, query: str, texts: list[str]) -> np.ndarray: ...
    def info(self) -> dict: ...
    def close(self) -> None: ...


def release_gpu_memory(torch_module) -> None:
    """Hand cached VRAM back to the driver: collect garbage first so freed tensors are really free."""
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()


def _looks_like_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError"  # torch.cuda.OutOfMemoryError, without importing torch here


class GpuEngine:
    """The real thing: Embedder (bge-small-en-v1.5) and Reranker (bge-reranker-base), both fp16 on one GPU."""

    def __init__(self, embedder, reranker, device: str = "cuda:0", torch_module=None):
        self.embedder, self.reranker, self.device = embedder, reranker, device
        self._torch = torch_module

    @classmethod
    def load(cls, device: str = "cuda:0") -> GpuEngine:
        import torch

        from embedder import Embedder
        from reranker import Reranker

        return cls(Embedder(device=device, batch_size=EMBED_BATCH_SIZE), Reranker(device=device), device, torch)

    def embed(self, texts: list[str], is_query: bool) -> list[np.ndarray | None]:
        encode = self.embedder.encode_queries if is_query else self.embedder.encode_documents
        try:
            return encode(texts)
        except Exception as exc:
            if _looks_like_oom(exc):
                release_gpu_memory(self._torch)
                raise GpuOutOfMemory(f"out of memory embedding {len(texts)} texts") from exc
            raise

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return self.reranker.score(query, texts)

    def info(self) -> dict:
        cuda, mib = self._torch.cuda, 2**20
        free, total = cuda.mem_get_info(self.device)
        return {
            "device": cuda.get_device_name(self.device),
            "models": {
                "embedder": {"name": MODEL_NAME, "revision": MODEL_REVISION, "dim": DIMENSIONS, "dtype": self.embedder.dtype},
                "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "dtype": self.reranker.dtype},
            },
            "vram": {
                "total_mib": round(total / mib, 1),
                "used_mib": round((total - free) / mib, 1),  # the whole card: this service, the CUDA context, other programs
                "free_mib": round(free / mib, 1),
                "torch_allocated_mib": round(cuda.memory_allocated(self.device) / mib, 1),  # what the models themselves hold
                "torch_peak_mib": round(cuda.max_memory_allocated(self.device) / mib, 1),
            },
        }

    def close(self) -> None:
        self.embedder = self.reranker = None
        release_gpu_memory(self._torch)


# --------------------------------------------------------------------------- the HTTP layer
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo such as "is_querry" must not silently fall back to the default

    texts: list[str | None] = Field(max_length=MAX_EMBED_TEXTS)
    is_query: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float] | None]  # one per text, in order; null where nothing was embeddable
    dim: int
    count: int  # texts received (== len(embeddings))
    skipped: list[int]  # positions that are null


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    candidates: list[str | None] = Field(max_length=MAX_RERANK_CANDIDATES)
    top_k: int | None = Field(default=None, ge=1)


class RerankItem(BaseModel):
    index: int  # position in the request's candidates
    score: float
    text: str  # the candidate exactly as sent


class RerankResponse(BaseModel):
    results: list[RerankItem]  # best first; equal scores keep request order
    count: int
    skipped: list[int]  # candidates that could not be scored (empty, letterless, or a non-finite score)


class VerifyPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: str
    claim: str


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[VerifyPair] = Field(max_length=MAX_VERIFY_PAIRS)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class VerifyItem(BaseModel):
    index: int  # position in the request's pairs
    score: float  # MiniCheck's support probability
    verdict: str  # supported | unsupported


class VerifyResponse(BaseModel):
    results: list[VerifyItem]  # request order, never reordered: the caller matches these to claims by index
    model: str
    precision: str
    threshold: float
    skipped: list[int]  # pairs with no scorable text


class BodyLimit:
    """Refuses a POST whose declared size is over the limit (413) or that does not declare one (411), before reading
    it.
    """

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            headers = {k: v for k, v in scope["headers"]}
            declared = headers.get(b"content-length")
            problem = None
            if declared is None or b"transfer-encoding" in headers:
                problem = (411, "send a Content-Length (chunked bodies are not accepted)")
            elif not declared.isdigit() or int(declared) > self.max_bytes:
                problem = (413, f"request body over {self.max_bytes // 2**20} MiB")
            if problem:
                response = JSONResponse({"detail": problem[1]}, status_code=problem[0])
                return await response(scope, receive, send)
        await self.app(scope, receive, send)


async def ensure_verifier(state) -> Any:
    """The verifier, loaded on first use and kept afterwards."""
    existing = getattr(state, "verifier", None)
    if existing is not None:
        return existing
    lock = getattr(state, "verifier_lock", None)
    if lock is None:
        lock = state.verifier_lock = asyncio.Lock()
    async with lock:
        if getattr(state, "verifier", None) is not None:
            return state.verifier
        try:
            loaded = await state.gpu.run_unbounded(verifier_module.Verifier)
        except verifier_module.VerifierUnavailable as exc:
            log.warning("verifier unavailable: %s", exc)
            raise HTTPException(503, f"verifier unavailable: {exc}") from exc
        state.verifier = loaded
        log.info("verifier loaded: %s", loaded.info())
        return loaded


def create_app(engine_factory: Callable[[], Engine] | None = None, *, max_queued: int = MAX_QUEUED) -> FastAPI:
    """The service. `engine_factory` builds the models (default: GpuEngine.load); tests pass a fake."""
    factory = engine_factory or GpuEngine.load

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        gpu = GpuQueue(max_queued)
        app.state.gpu = gpu
        try:
            app.state.engine = await gpu.run_unbounded(factory)  # loaded on the GPU thread, which then serves every call
            app.state.started = time.monotonic()
            log.info("models loaded; ready")
            try:
                yield
            finally:
                loaded_verifier = getattr(app.state, "verifier", None)
                if loaded_verifier is not None:
                    await gpu.run_unbounded(loaded_verifier.close)
                await gpu.run_unbounded(app.state.engine.close)  # del, gc.collect, empty_cache, on the same thread
                log.info("models released")
        finally:
            gpu.shutdown()

    app = FastAPI(title="On the Record GPU service", version="1", lifespan=lifespan)
    app.add_middleware(BodyLimit)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)  # added last, so it runs first

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError):
        # FastAPI's default 422 echoes the offending input, which here could be megabytes of request text
        problems = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()[:20]]
        return JSONResponse({"detail": problems}, status_code=422)

    @app.exception_handler(GpuBusy)
    async def busy(_request: Request, exc: GpuBusy):
        return JSONResponse({"detail": "GPU busy; retry shortly"}, status_code=503, headers={"Retry-After": "1"})

    @app.exception_handler(GpuOutOfMemory)
    async def out_of_memory(_request: Request, exc: GpuOutOfMemory):
        log.warning("GPU out of memory: %s", exc)
        return JSONResponse({"detail": "GPU out of memory; retry with fewer texts"}, status_code=503, headers={"Retry-After": "2"})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.error("unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/health")
    async def health(request: Request):
        state = request.app.state
        try:
            info = await asyncio.to_thread(state.engine.info)
        except Exception:
            log.exception("GPU probe failed")
            return JSONResponse({"status": "error", "detail": "GPU probe failed"}, status_code=503)
        vram = info["vram"]
        vram["min_free_mib"] = MIN_FREE_VRAM_MIB
        vram["headroom_ok"] = vram["free_mib"] >= MIN_FREE_VRAM_MIB
        return {
            "status": "ok" if vram["headroom_ok"] else "low_vram",
            **info,
            # Reported rather than probed: /health must not load 3.1 GB of weights as a side effect of asking whether
            # the service is up.
            "verifier": (getattr(state, "verifier", None).info() if getattr(state, "verifier", None)
                         else {"loaded": False, "note": "loaded on first /verify call"}),
            "queue": {"queued": state.gpu.queued, "max_queued": state.gpu.max_queued},
            "uptime_s": round(time.monotonic() - state.started, 1),
        }

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(body: EmbedRequest, request: Request):
        state = request.app.state
        unique, slots = dedupe([sanitize(t) for t in body.texts])
        vectors: list[np.ndarray | None] = []
        if unique:
            vectors = await state.gpu.run(state.engine.embed, unique, body.is_query)
            if len(vectors) != len(unique):
                raise RuntimeError(f"engine returned {len(vectors)} vectors for {len(unique)} texts")
        embeddings: list[list[float] | None] = []
        skipped: list[int] = []
        for position, slot in enumerate(slots):
            vector = None if slot is None else vectors[slot]
            if vector is None or not vector_ok(vector):
                if vector is not None:
                    log.warning("engine returned an unusable vector for text %d; skipped", position)
                embeddings.append(None)
                skipped.append(position)
            else:
                embeddings.append(np.asarray(vector, dtype=np.float32).tolist())  # exactly the float32 values, as JSON numbers
        return EmbedResponse(embeddings=embeddings, dim=DIMENSIONS, count=len(embeddings), skipped=skipped)

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(body: RerankRequest, request: Request):
        state = request.app.state
        if not body.candidates:
            return RerankResponse(results=[], count=0, skipped=[])
        query = sanitize(body.query, MAX_QUERY_CHARS)
        if query is None:
            raise HTTPException(422, "query has no letters or digits to score against")
        unique, slots = dedupe([sanitize(c) for c in body.candidates])
        scores = np.empty(0, dtype=np.float32)
        if unique:
            scores = np.asarray(await state.gpu.run(state.engine.score, query, unique), dtype=np.float64)
            if scores.shape != (len(unique),):
                raise RuntimeError(f"engine returned scores of shape {scores.shape} for {len(unique)} candidates")
        ranked: list[tuple[float, int]] = []
        skipped: list[int] = []
        for index, slot in enumerate(slots):
            if slot is None or not math.isfinite(scores[slot]):
                if slot is not None:
                    log.warning("engine returned a non-finite score for candidate %d; skipped", index)
                skipped.append(index)
            else:
                ranked.append((float(scores[slot]), index))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        if body.top_k is not None:
            ranked = ranked[: body.top_k]
        # the candidate as sent, minus lone surrogates: they cannot be written as UTF-8 JSON, so echoing one is a 500
        results = [
            RerankItem(index=index, score=score, text=body.candidates[index].encode("utf-8", "ignore").decode("utf-8"))
            for score, index in ranked
        ]
        return RerankResponse(results=results, count=len(results), skipped=skipped)

    @app.post("/verify", response_model=VerifyResponse)
    async def verify(body: VerifyRequest, request: Request):
        """Does this evidence support this claim?"""
        state = request.app.state
        if not body.pairs:
            return VerifyResponse(results=[], model="", precision="", threshold=0.0, skipped=[])

        verifier = await ensure_verifier(state)
        threshold = body.threshold if body.threshold is not None else verifier_module.SUPPORT_THRESHOLD

        scorable: list[int] = []
        skipped: list[int] = []
        for index, pair in enumerate(body.pairs):
            if sanitize(pair.evidence) is None or sanitize(pair.claim) is None:
                skipped.append(index)
            else:
                scorable.append(index)

        scores: list[float] = []
        if scorable:
            evidence = [body.pairs[i].evidence for i in scorable]
            claims = [body.pairs[i].claim for i in scorable]
            scores = await state.gpu.run(verifier.score, evidence, claims)
            if len(scores) != len(scorable):
                raise RuntimeError(f"verifier returned {len(scores)} scores for {len(scorable)} pairs")

        results = []
        for index, score in zip(scorable, scores, strict=True):
            # A non-finite score is never reported as a verdict. fp16 Flan-T5 emits NaNs, and a NaN compared against a
            # threshold is False, which would arrive as a confident "unsupported".
            if not math.isfinite(score):
                log.warning("verifier returned a non-finite score for pair %d; skipped", index)
                skipped.append(index)
                continue
            results.append(VerifyItem(index=index, score=float(score),
                                      verdict="supported" if score >= threshold else "unsupported"))  # fmt: skip
        return VerifyResponse(results=results, model=verifier_module.VERIFIER_NAME,
                              precision=verifier.precision, threshold=threshold,
                              skipped=sorted(skipped))  # fmt: skip

    return app


app = create_app()  # models load in the lifespan, not at import, so importing this module is cheap and torch-free


# --------------------------------------------------------------------------- running it
# ---------------------------------------------------------------------------


def make_server(application: FastAPI, host: str = HOST, port: int = PORT, log_level: str = "info") -> uvicorn.Server:
    return uvicorn.Server(uvicorn.Config(application, host=host, port=port, log_level=log_level))


def run_server(server: uvicorn.Server) -> None:
    """Serve until told to stop. On Windows the loop is a SelectorEventLoop: the project rule for every async server
    here (psycopg needs it), and it also avoids the Proactor loop's noisy connection-reset tracebacks.
    """
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    try:
        with asyncio.Runner(loop_factory=factory) as runner:
            runner.run(server.serve())
    except KeyboardInterrupt:
        pass  # after a clean shutdown uvicorn re-raises the signal that stopped it; for Ctrl+C that is this


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local GPU service: query embedding and reranking.")
    parser.add_argument("--host", default=HOST, help="a loopback address only (default: %(default)s)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args(argv)

    if not is_loopback_host(args.host):
        print(f"refusing to listen on {args.host}: this service has no authentication and is for this machine only", file=sys.stderr)
        return 2

    import check_cuda

    errors, warnings = check_cuda.assess(check_cuda.inspect_environment())
    for message in warnings:
        print(f"warning: {message}", file=sys.stderr)
    if errors:
        for message in errors:
            print(f"error: {message}", file=sys.stderr)
        return 2

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = make_server(app, args.host, args.port, args.log_level)
    run_server(server)
    return 0 if server.started else 3


if __name__ == "__main__":
    sys.exit(main())
