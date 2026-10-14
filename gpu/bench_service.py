"""Benchmark the GPU service as a real process: start it, measure, stop it. Run from gpu/ with the database up."""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np

import check_cuda
import embedder

HERE = Path(__file__).parent
URL = "http://127.0.0.1:8001"
# The dense search through the live index's own ORDER BY: any other spelling is a sequential scan, and this would
# benchmark the wrong thing.
SEARCH = ("SELECT parent_body FROM chunk WHERE embedding IS NOT NULL AND NOT is_boilerplate ORDER BY "
          f"{embedder._vector_index().order_expression(embedder.index_form(), alias='', param='%s')} LIMIT 20")


def smi(*fields: str) -> list[int]:
    out = subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], text=True)
    return [int(float(x)) for x in out.strip().split(",")]


def summary(ms: list[float]) -> str:
    return f"median {statistics.median(ms):6.1f} ms   p95 {sorted(ms)[int(0.95 * len(ms))]:6.1f} ms   (n={len(ms)})"


def wait_until_ready(http: httpx.Client, proc: subprocess.Popen, seconds: float = 120) -> float:
    start = time.time()
    while time.time() - start < seconds and proc.poll() is None:
        try:
            if http.get("/health").status_code == 200:
                return time.time() - start
        except httpx.TransportError:
            time.sleep(0.25)
    raise SystemExit("the service did not become ready (see its log)")


def worst_case(path: str, payload: dict) -> tuple[float, int, list[float]]:
    """Time one maximum-size request while polling /health; returns (seconds, status, health latencies in ms)."""
    box: dict = {}

    def send():
        start = time.perf_counter()
        box["status"] = httpx.post(URL + path, json=payload, timeout=180).status_code
        box["seconds"] = time.perf_counter() - start

    worker = threading.Thread(target=send)
    worker.start()
    polls: list[float] = []
    time.sleep(0.3)
    while worker.is_alive():
        start = time.perf_counter()
        httpx.get(URL + "/health", timeout=30)
        polls.append((time.perf_counter() - start) * 1000)
        time.sleep(0.05)
    worker.join()
    return box["seconds"], box["status"], polls


def main() -> int:
    battery = check_cuda.on_battery()
    print(f"power source: {'BATTERY (GPU is power-capped: expect ~4x slower than on mains)' if battery else 'mains' if battery is False else 'unknown'}")
    used_before, total = smi("memory.used", "memory.total")
    print(f"GPU before the service: {used_before} MiB used of {total} MiB")

    log_path = Path(tempfile.gettempdir()) / "gpu_service_bench.log"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "api.py"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    try:
        with httpx.Client(base_url=URL, timeout=120) as http:
            print(f"startup, both models loaded before the port opens: {wait_until_ready(http, proc):.1f} s")
            health = http.get("/health").json()
            idle_used = smi("memory.used")[0]
            print(f"VRAM, service idle: torch holds {health['vram']['torch_allocated_mib']:.0f} MiB; nvidia-smi {idle_used} MiB used "
                  f"({idle_used - used_before} MiB for the service); CUDA driver reports {health['vram']['free_mib']:.0f} MiB free")

            peak = [idle_used]
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    peak[0] = max(peak[0], smi("memory.used")[0])
                    time.sleep(0.1)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()

            gold = json.loads((HERE.parent / "tests" / "data" / "gold_eval.json").read_text(encoding="utf-8"))
            questions = [q["question"] for q in gold["queries"]]
            embed_ms, rerank_ms = [], []
            with embedder.connect(autocommit=True) as conn:
                for question in questions * 2:
                    start = time.perf_counter()
                    vector = http.post("/embed", json={"texts": [question]}).json()["embeddings"][0]
                    embed_ms.append((time.perf_counter() - start) * 1000)
                    windows = [r[0] for r in conn.execute(SEARCH, (np.asarray(vector, dtype=np.float32),)).fetchall()]
                    start = time.perf_counter()
                    response = http.post("/rerank", json={"query": question, "candidates": windows})
                    rerank_ms.append((time.perf_counter() - start) * 1000)
                    assert response.status_code == 200 and response.json()["skipped"] == []
            print(f"POST /embed  1 query:                     {summary(embed_ms)}")
            print(f"POST /rerank 20 real parent windows:      {summary(rerank_ms)}")

            texts = [(f"parliament debate number {i} " * 800)[:10_000] for i in range(512)]  # every text beyond 512 tokens
            seconds, status, polls = worst_case("/rerank", {"query": "parliament debate", "candidates": texts})
            print(f"worst /rerank 512 x 512 tokens: {seconds:5.1f} s (HTTP {status}); /health meanwhile: median {statistics.median(polls):.0f} ms, max {max(polls):.0f} ms")
            seconds, status, polls = worst_case("/embed", {"texts": texts})
            print(f"worst /embed  512 x 512 tokens: {seconds:5.1f} s (HTTP {status}); /health meanwhile: median {statistics.median(polls):.0f} ms, max {max(polls):.0f} ms")

            stop.set()
            sampler.join()
            print(f"peak VRAM over the whole run: {peak[0]} MiB used -> {total - peak[0]} MiB free of {total} MiB "
                  f"(the owner's requirement is at least 4,096 free)")
            print(f"GPU clock now: {smi('clocks.sm')[0]} MHz of {smi('clocks.max.sm')[0]} MHz")

        os.kill(proc.pid, signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)  # what Ctrl+C does
        proc.wait(timeout=30)
        time.sleep(1.5)
        log_text = log_path.read_text(encoding="utf-8")
        print(f"shutdown: 'models released' in the log: {'models released' in log_text}; tracebacks: {log_text.count('Traceback')}; "
              f"VRAM after: {smi('memory.used')[0]} MiB used (before: {used_before})")
    finally:
        if proc.poll() is None:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
