"""BAAI/bge-reranker-base: a cross-encoder that scores (query, passage) pairs on the GPU."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

RERANKER_NAME = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"  # pinned so the same weights are used every time
RERANK_MAX_LENGTH = 512  # tokens for query + passage together; the model has 514 positions, two are reserved
DEFAULT_RERANK_BATCH = 32  # pairs per forward pass; the service's VRAM ceiling rests on this


class GpuOutOfMemory(RuntimeError):
    """The GPU ran out of memory. The service answers 503 and stays up; the caller may retry with less."""


def score_in_batches(
    query: str,
    texts: Sequence[str],
    forward: Callable[[str, list[str]], np.ndarray],
    batch_size: int = DEFAULT_RERANK_BATCH,
) -> np.ndarray:
    """One float32 score per text, in the order given, computed `batch_size` pairs at a time."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, not {batch_size}")
    out = np.empty(len(texts), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        scores = np.asarray(forward(query, [texts[i] for i in chosen]), dtype=np.float32)
        if scores.shape != (len(chosen),):
            raise ValueError(f"forward returned scores of shape {scores.shape} for {len(chosen)} texts")
        out[chosen] = scores
    return out


def _load_reranker(device: str, dtype: str):
    """(tokenizer, model) for the pinned reranker, from the local Hugging Face cache; downloaded only on a cache miss
    (the same reasoning as embedder._load_model: a cached load must not depend on the network).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def load(local_files_only: bool):
        options = {"revision": RERANKER_REVISION, "local_files_only": local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_NAME, **options)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME, **options)
        if dtype == "float16":
            model = model.half()  # cast on the CPU first, so the GPU never holds the 1.1 GB float32 copy
        return tokenizer, model.to(device)

    try:
        return load(True)
    except OSError:  # a cache miss surfaces as OSError; anything else is a real error and propagates
        return load(False)


class Reranker:
    """bge-reranker-base on the GPU, loaded once."""

    name = RERANKER_NAME
    revision = RERANKER_REVISION

    def __init__(self, device: str = "cuda:0", dtype: str = "float16", batch_size: int = DEFAULT_RERANK_BATCH):
        if dtype not in ("float16", "float32"):
            raise ValueError(f"dtype must be float16 or float32, not {dtype!r}")
        import torch

        self._torch = torch
        self.device, self.dtype, self.batch_size = device, dtype, batch_size
        self.tokenizer, self.model = _load_reranker(device, dtype)
        self.model.eval()
        self.score("warm up the kernels", ["a short passage to score"])
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Relevance of each text to the query, in the order given. Callers pass tidy, non-empty strings (the service
        cleans them first); a pair longer than 512 tokens is truncated, longest side first, never rejected.
        """
        return score_in_batches(query, texts, self._forward, self.batch_size)

    def _forward(self, query: str, texts: list[str]) -> np.ndarray:
        torch = self._torch
        try:
            encoded = self.tokenizer(
                [query] * len(texts), texts, padding=True, truncation="longest_first", max_length=RERANK_MAX_LENGTH, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise GpuOutOfMemory(f"out of memory scoring {len(texts)} pairs") from exc
        return logits[:, 0].float().cpu().numpy()

    def vram_mib(self) -> tuple[float, float]:
        """(currently allocated, peak allocated) by this process, in MiB."""
        if not self.device.startswith("cuda"):
            return 0.0, 0.0
        t = self._torch.cuda
        return t.memory_allocated(self.device) / 2**20, t.max_memory_allocated(self.device) / 2**20
