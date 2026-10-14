# GPU project: embedding worker and HTTP service

Two programs share one GPU and one environment: the **embedding worker** (`embedder.py`), which embeds every
non-boilerplate row of `chunk` with `BAAI/bge-small-en-v1.5` and rebuilds the HNSW index, and the **HTTP service**
(`api.py`), which embeds queries and reranks candidates for the search code. This is a separate uv project on
purpose: torch's CUDA wheels never enter the root environment or CI's install. Run everything from this directory.

```powershell
uv sync                                    # once; about 3 GB (torch 2.11 from PyTorch's cu128 index)
uv run python check_cuda.py                # pre-flight: CUDA usable, RTX 3070 found, VRAM free, power source
uv run pytest                              # 276 tests, 30 to 70 s depending on GPU clocks (35 of them need the GPU)

uv run python api.py                       # the HTTP service on 127.0.0.1:8001 (see "The HTTP service" below)
uv run python bench_service.py             # start it, measure latency and VRAM, stop it (plug the laptop in first)

uv run python embedder.py --status         # embedded / pending counts and the index state
uv run python embedder.py --limit 100      # smoke test: 100 chunks, the index is left alone
uv run python embedder.py                  # everything pending; ~75 s for all 90,040 chunks
uv run python embedder.py --index-only     # drop and rebuild the index, then VACUUM ANALYZE
```

The worker needs the database from `docker compose up -d` and reads `PG_DSN` (default: the local container).
It is idempotent and resumable: a re-run embeds only chunks that still have no vector, so after
`python -m ingest.chunk --rebuild` in the root project, run it again and only changed sentences are re-embedded.

## How a full run behaves

1. If the queue is at least 5,000 chunks, `DROP INDEX chunk_embedding_idx` (row-by-row HNSW inserts are slow and
   build a worse graph). Smaller runs leave the index alone.
2. Stream pending chunks in pages of 2,048 (keyset paging on `id`), embed them in fp16 batches of 256, normalise to
   unit length, and write each page with one binary COPY plus one UPDATE, committing per page. Each row gets
   `embedding`, `embedding_model = 'BAAI/bge-small-en-v1.5'` and `embedded_at`.
3. Build the index once over the finished table: `USING hnsw (embedding vector_cosine_ops) WITH (m = 16,
   ef_construction = 64)`, serially (see below), then `VACUUM ANALYZE chunk`.
4. Ctrl-C or a crash never loses more than the current page. After Ctrl-C the index is not rebuilt (the run says
   so); the next run, or `--index-only`, restores it. After an ordinary error the worker still rebuilds it.

Measured on the RTX 3070 Laptop, 2026-09-20: 90,040 chunks embedded in 24.7 s (3,644 chunks/s end to end, limited
by the database write; the GPU alone does about 7,000/s), index build 39.8 s (174 MiB), peak PyTorch VRAM
935 MiB (3.1 GB of the card's 8 GB including the CUDA context and Windows).

## Decisions worth knowing

- **fp16, batch 256.** fp16 was 2.9x faster than fp32 at batch 512 (6,120 against 2,123 chunks/s) and fastest of
  all at batch 256 (7,160), with cosine 0.999999 to the fp32 vectors and 99.7 % top-10 neighbour overlap. Vectors are normalised in float64 after the model, so they are unit
  length whatever precision the model ran in.
- **Documents are embedded as-is; queries get BGE's instruction** ("Represent this sentence for searching relevant
  passages: "). Use `Embedder.encode_queries` at search time, never `encode_documents`.
- **The model loads from the local Hugging Face cache first** and only downloads if it is missing. Without this,
  every load sends network requests that were being reset and retried for 15+ seconds.
- **The index is built with zero parallel workers.** The container's `/dev/shm` is 64 MB and a parallel HNSW build
  fails at once with `DiskFull: could not resize shared memory segment`. Serial takes about 40 s; parallel would
  need `shm_size: 1gb` in `docker-compose.yml` and a container recreate, for a one-off saving of seconds.
- **The revision is pinned** (`MODEL_REVISION`), so the same weights are used every time.
- **Reading vectors back:** pgvector 0.5.0 returns `Vector` objects; call `.to_numpy()`. Writing accepts numpy arrays.

## The HTTP service

`uv run python api.py` loads both models (about 18 s), then listens on `127.0.0.1:8001`. The port opens only after
the models are loaded, so a 200 from `/health` means ready. Stop it with Ctrl+C: it releases the models and the
VRAM first. It never touches the database.

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | | `status` (`ok`, or `low_vram` when under 4,096 MiB is free), `device`, model names and revisions, `vram` (`total_mib`, `used_mib`, `free_mib`, `torch_allocated_mib`, `torch_peak_mib`, `headroom_ok`), `queue`, `uptime_s` |
| `POST /embed` | `{"texts": [...], "is_query": true}` | `{"embeddings": [[384 floats] or null, ...], "dim": 384, "count": n, "skipped": [positions]}` |
| `POST /rerank` | `{"query": "...", "candidates": [...], "top_k": null}` | `{"results": [{"index", "score", "text"}, ...], "count": n, "skipped": [positions]}` |

- **`is_query` defaults to true**, which puts BGE's retrieval instruction in front of the text. Send `false` for
  passages. Vectors are unit length, exactly float32 values, 384 long.
- **A text with nothing to embed** (null, empty, no letter or digit) gets `null` at its position and its index in
  `skipped`, so the answer always lines up with the request. `/rerank` leaves such candidates out and lists them in
  `skipped`; a blank `query` is a 422 (nothing to score against). An empty list is answered at once with no GPU work.
- **Scores** are the reranker's raw logit: higher is more relevant, negative is normal for a poor match, and they
  are only comparable within one query. Equal scores keep request order; identical candidates are scored once.
- **`text` in a rerank result** is the candidate as sent, except that lone surrogates are dropped (they cannot be
  written as UTF-8 JSON).
- **Limits:** 512 texts or candidates per call; each text is cut at 10,000 characters (a rerank query at 2,000) and
  the models truncate at 512 tokens, so long input is shortened, never rejected; request bodies up to 32 MiB.
  Errors: 422 malformed request (the message never echoes your text), 411/413 body framing or size, 400 wrong `Host`
  header, 503 with `Retry-After` when more than 8 requests are queued or the GPU is out of memory, 500 unexpected.

Calling it from the search code. The response is a plain list, and psycopg adapts a list as `double precision[]`,
which pgvector cannot compare, so wrap the vector:

```python
vector = httpx.post("http://127.0.0.1:8001/embed", json={"texts": [question]}).json()["embeddings"][0]
rows = conn.execute("SELECT id FROM chunk ORDER BY embedding <=> %s LIMIT 20", (np.asarray(vector, dtype=np.float32),))
```

How it stays up, each rule tested:

- **One GPU thread, one job at a time.** VRAM is bounded by a single micro-batch (embed 128 texts, rerank 32 pairs),
  the event loop is never blocked (`/health` answers during a 10-second job), and the bounded queue turns overload
  into a 503 instead of a pile-up. The models are loaded and released on that same thread.
- **Dirty input cannot raise.** NULs, lone surrogates (which crash the tokenizer), control characters, whitespace
  runs, emoji-only text, a 1 MB single word, non-Latin scripts: cleaned or reported in `skipped`, never a 500.
  Non-finite vectors and scores are never sent.
- **Local only.** It refuses to bind anything but a loopback address, rejects other `Host` headers (DNS rebinding)
  and has no CORS. On Windows it runs on a `SelectorEventLoop`, the project rule for every async server here.
- **Request text is never logged.**

Measured on the RTX 3070 Laptop with `bench_service.py`, **on mains power** (2026-09-21): both models resident hold
612 MiB (nvidia-smi: 793 MiB for the process); peak over the whole benchmark including two worst-case requests 2,662 MiB
used, **5,530 MiB free** of 8,192. One query embed **16 ms**, a 20-candidate rerank **50 ms** over HTTP (medians; p95 26
and 81 ms). Worst case (512 candidates of 512 tokens) 2.7 s for a rerank and 1.1 s for an embed, with `/health` answering
throughout. Startup 12 s. **On battery the same laptop's GPU is power-capped** (about 630 MHz at a flat 30 W against a
2,100 MHz maximum) and everything is roughly 2 to 4 times slower: embed 30 ms, rerank 184 ms, worst case 10 s (owner's
run: 28 ms and 166 ms at 885 MHz). `check_cuda.py` and `bench_service.py` say which power source the numbers came from.

## Files

- `api.py`: the HTTP service (`create_app`, `GpuQueue`, `GpuEngine`, input hygiene, limits, `main`).
- `reranker.py`: `Reranker` (bge-reranker-base, revision pinned, fp16, longest-first micro-batches) and the torch-free
  batching function `score_in_batches`.
- `embedder.py`: the model wrapper (`Embedder`), the text and vector helpers, the database functions, the index
  lifecycle, `run()` and the command line. The service calls `Embedder.encode_queries` / `encode_documents`.
- `check_cuda.py`: the pre-flight above; its decisions (`assess`) are unit tested. Warns when on battery.
- `bench_service.py`: starts the service as a real process and measures it (needs the database for real windows).
- `tests/`: unit tests, database tests against a scratch database built from the repo's own `db/`, HTTP tests against
  a fake engine (no GPU, run in CI), and real-model tests (`test_gpu_*`, need the card). `conftest.py` creates and
  drops that scratch database, so the live one is never touched. Name a test `test_gpu_*` only if it needs the GPU:
  CI and the mutation suites deselect on that prefix.
