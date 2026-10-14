"""Embed the chunk table with BAAI/bge-small-en-v1.5 on the GPU, then rebuild the HNSW index."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned so the same weights are used every time

# What text was embedded, not just which model embedded it.
TEXT_RECIPE = "full"
RECIPE_VERSION = "v2"
EMBEDDING_RECIPE = f"{MODEL_NAME}|{TEXT_RECIPE}|{RECIPE_VERSION}"
DIMENSIONS = 384
MAX_SEQ_LENGTH = 512  # the model's limit; the longest chunk is about 1,500 characters, well inside it
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# `hansard_v2` is the corpus (app.safety.TARGET); `hansard` is the protected rollback and has no embedding_recipe column
# at all.
DEFAULT_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/hansard_v2")
DEFAULT_DTYPE = "float16"
DEFAULT_BATCH_SIZE = 256
DEFAULT_PAGE_SIZE = 2048  # rows per database page; the encoder length-sorts within it

# The index's name, parameters and SQL come from `retrieval.vector_index`: the dense leg's ORDER BY has to repeat the
# index's expression exactly or every search is a sequential scan, so both sides read one definition.
INDEX_NAME = "chunk_embedding_idx"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
# The HNSW graph is built in memory and then written out.
INDEX_MEM_FLOOR_MB = 512
INDEX_MEM_CEILING_MB = 4_096  # the container may use 15.5 GiB; a single serial build has no need of more
INDEX_MAINTENANCE_WORK_MEM = "512MB"  # the floor, kept for callers that build without a row count
INDEX_PARALLEL_WORKERS = 0  # measured: workers=2 fails at once with DiskFull (the container's 64 MB /dev/shm); serial takes ~80 s
INDEX_THRESHOLD = 5_000  # a run writing at least this many vectors manages the index lifecycle itself


# --------------------------------------------------------------------------- text and vector helpers (no GPU, no
# database) ---------------------------------------------------------------------------


def _src_on_path() -> None:
    """Make the root project's pure-Python modules importable from the GPU project."""
    path = str(Path(__file__).resolve().parents[1] / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


def _vector_index():
    """`retrieval.vector_index`: pure, and the one definition of the index and the query that uses it."""
    _src_on_path()
    from retrieval import vector_index

    ours = (INDEX_NAME, HNSW_M, HNSW_EF_CONSTRUCTION, DIMENSIONS)
    theirs = (vector_index.INDEX_NAME, vector_index.HNSW_M, vector_index.HNSW_EF_CONSTRUCTION, vector_index.DIM)
    if ours != theirs:
        raise RuntimeError(f"gpu/embedder.py {ours} and retrieval.vector_index {theirs} disagree about the index")
    return vector_index


def index_form() -> str:
    """The form frozen retrieval queries — the only index worth building."""
    return _vector_index().frozen_form()


def index_sql(form: str | None = None) -> str:
    return _vector_index().index_sql(form or index_form(), INDEX_NAME, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION)


def live_index_form(conn: psycopg.Connection) -> str | None:
    """The form of the index that exists now (None: no index, or one of neither form)."""
    row = conn.execute("SELECT pg_get_indexdef(to_regclass(%s))", (INDEX_NAME,)).fetchone()
    return _vector_index().form_of(row[0] if row else None)


def header_recipe(key: str = TEXT_RECIPE):
    """The adopted text recipe, from the same module the bake-off measured it with."""
    _src_on_path()
    # `retrieval.recipe`, not `retrieval.bakeoff`: the renderer is the definition of the production recipe and the
    # embedder has no business importing a bake-off to get at it (E1 section 4.5).
    from retrieval.recipe import RECIPES_BY_KEY

    recipe = RECIPES_BY_KEY.get(key)
    if recipe is None:
        raise ValueError(f"unknown text recipe {key!r}; known: {', '.join(sorted(RECIPES_BY_KEY))}")
    return recipe


def guard_database(dsn: str) -> str:
    """Refuse a bulk write to the protected rollback corpus. Returns the dsn, never printing it."""
    _src_on_path()
    from app import safety

    found = safety.database_of(dsn)
    if found in safety.PROTECTED:
        raise SystemExit(
            f"refusing to embed into {found!r}: it is the rollback corpus and it has no embedding_recipe "
            f"column. This writes to {safety.TARGET!r}; set PG_DSN or pass --dsn."
        )
    return dsn


def clean_for_embedding(text: str | None) -> str | None:
    r"""Tidy whitespace, NULs and lone surrogates; None when nothing embeddable is left (None, blank, or no letter or
    digit). A lone surrogate is what a JSON body such as "\ud800" parses to, and it makes the tokenizer raise.
    """
    if text is None:
        return None
    text = text.encode("utf-8", "ignore").decode("utf-8")
    cleaned = " ".join(text.replace("\x00", "").split())
    return cleaned if any(ch.isalnum() for ch in cleaned) else None


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows as float32. A zero row stays zero (validate_vectors then refuses it) instead of becoming NaN.
    """
    m = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0).astype(np.float32)


def validate_vectors(vectors: np.ndarray, tolerance: float = 1e-3) -> None:
    """Refuse anything that must never reach the database: wrong shape, NaN/inf, or not unit length."""
    v = np.asarray(vectors)
    if v.ndim != 2 or v.shape[1] != DIMENSIONS:
        raise ValueError(f"bad shape {v.shape}, expected (n, {DIMENSIONS})")
    if not np.isfinite(v).all():
        raise ValueError("vectors contain non-finite values (NaN or inf); expected finite numbers")
    norms = np.linalg.norm(v.astype(np.float64), axis=1)
    if len(norms) and np.abs(norms - 1.0).max() > tolerance:
        raise ValueError(f"vectors are not unit length (norm range {norms.min():.4f} to {norms.max():.4f})")


# --------------------------------------------------------------------------- the model
# ---------------------------------------------------------------------------


class Encoder(Protocol):
    def encode_documents(self, texts: Sequence[str | None]) -> list[np.ndarray | None]: ...


def _load_model(device: str):
    """The pinned model from the local Hugging Face cache; downloaded only if it is not cached yet."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device, local_files_only=True)
    except OSError:  # a cache miss surfaces as OSError; anything else is a real error and propagates
        return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)


class Embedder:
    """bge-small-en-v1.5 on the GPU."""

    def __init__(self, device: str = "cuda:0", dtype: str = DEFAULT_DTYPE, batch_size: int = DEFAULT_BATCH_SIZE):
        import torch

        if dtype not in ("float16", "float32"):
            raise ValueError(f"dtype must be float16 or float32, not {dtype!r}")
        self._torch = torch
        self.device, self.dtype, self.batch_size = device, dtype, batch_size
        self.model = _load_model(device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        if dtype == "float16":
            self.model.half()
        self.model.eval()
        self.encode_documents(["warm up the kernels"])
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        with self._torch.inference_mode():
            tensor = self.model.encode(
                texts, batch_size=self.batch_size, convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=False
            )
        return l2_normalize(tensor.float().cpu().numpy())  # normalised in float64 here, whatever precision the model ran in

    def _encode_optional(self, texts: Sequence[str | None], prefix: str = "") -> list[np.ndarray | None]:
        cleaned = [clean_for_embedding(t) for t in texts]
        positions = [i for i, c in enumerate(cleaned) if c is not None]
        out: list[np.ndarray | None] = [None] * len(texts)
        if positions:
            for i, vector in zip(positions, self._encode([prefix + cleaned[i] for i in positions])):
                out[i] = vector
        return out

    def encode_documents(self, texts: Sequence[str | None]) -> list[np.ndarray | None]:
        """One unit float32 vector per text, or None where there is nothing to embed. Order is preserved."""
        return self._encode_optional(texts)

    def encode_queries(self, texts: Sequence[str | None]) -> list[np.ndarray | None]:
        """Like encode_documents, with BGE's retrieval instruction in front of each query."""
        return self._encode_optional(texts, QUERY_INSTRUCTION)

    def vram_mib(self) -> tuple[float, float]:
        """(currently allocated, peak allocated) by this process, in MiB."""
        if not self.device.startswith("cuda"):
            return 0.0, 0.0
        t = self._torch.cuda
        return t.memory_allocated(self.device) / 2**20, t.max_memory_allocated(self.device) / 2**20


# --------------------------------------------------------------------------- database
# ---------------------------------------------------------------------------


def connect(dsn: str = DEFAULT_DSN, autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=autocommit)
    register_vector(conn)
    return conn


# A chunk is pending when it has no vector, OR when the vector it has was made under a different recipe.
def _pending(alias: str = "") -> str:
    """The queue predicate, optionally qualified with a table alias for a joined query."""
    c = f"{alias}." if alias else ""
    return (f"NOT {c}is_boilerplate AND ({c}embedding IS NULL"
            f" OR {c}embedding_recipe IS DISTINCT FROM %(recipe)s)")


PENDING = _pending()


def count_pending(conn: psycopg.Connection, recipe: str = EMBEDDING_RECIPE) -> int:
    return conn.execute(f"SELECT count(*) FROM chunk WHERE {PENDING}", {"recipe": recipe}).fetchone()[0]


# The metadata the adopted header needs, joined once per page.
FETCH_SQL = f"""
SELECT c.id,
       c.body,
       c.debate_date::text AS date,
       c.house,
       d.heading,
       s.speaker_name,
       s.speaker_party AS party
  FROM chunk c
  LEFT JOIN speech s ON s.id = c.speech_id
  LEFT JOIN debate d ON d.id = c.debate_id
 WHERE {_pending("c")} AND c.id > %(after)s
 ORDER BY c.id
 LIMIT %(size)s
"""


def render_row(row: dict, recipe) -> str | None:
    """The exact text to embed for one chunk: the cleaned body under the recipe's header."""
    body = clean_for_embedding(row.get("body"))
    if body is None:
        return None
    return recipe.render({**row, "body": body})


def fetch_page(conn: psycopg.Connection, after_id: int, size: int,
               recipe: str = EMBEDDING_RECIPE, text_recipe: str = TEXT_RECIPE) -> list[tuple[int, str | None]]:
    """The next `size` pending chunks with id > after_id, as (id, the text to embed)."""
    rendered = header_recipe(text_recipe)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(FETCH_SQL, {"recipe": recipe, "after": after_id, "size": size})
        rows = cur.fetchall()
    return [(row["id"], render_row(row, rendered)) for row in rows]


def write_embeddings(conn: psycopg.Connection, ids: Sequence[int], vectors: np.ndarray,
                     model: str = MODEL_NAME, recipe: str = EMBEDDING_RECIPE) -> int:
    """Set embedding, embedding_model and embedded_at for `ids` in one round trip (binary COPY, then one UPDATE). A row
    that already has a vector is left alone.
    """
    if len(ids) == 0:
        return 0
    if len(ids) != len(vectors):
        raise ValueError(f"{len(ids)} ids but {len(vectors)} vectors")
    validate_vectors(vectors)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS _chunk_embedding (id bigint NOT NULL, embedding vector({DIMENSIONS}) NOT NULL)")
        cur.execute("TRUNCATE _chunk_embedding")
        with cur.copy("COPY _chunk_embedding (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int8", "vector"])
            for chunk_id, vector in zip(ids, vectors):
                copy.write_row((chunk_id, vector))
        # A row whose vector is already this model AND this recipe is left alone; a row carrying another recipe's vector
        # is overwritten, which is the point of the tag.
        cur.execute(
            "UPDATE chunk c SET embedding = e.embedding, embedding_model = %s,"
            " embedding_recipe = %s, embedded_at = now()"
            " FROM _chunk_embedding e WHERE c.id = e.id"
            " AND (c.embedding IS NULL OR c.embedding_recipe IS DISTINCT FROM %s)",
            (model, recipe, recipe),
        )
        return cur.rowcount


def index_exists(conn: psycopg.Connection) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL", (INDEX_NAME,)).fetchone()[0]


def drop_index(conn: psycopg.Connection) -> None:
    conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")


def maintenance_work_mem_for(rows: int, form: str | None = None) -> str:
    """How much memory to give the graph build for `rows` vectors of `form`, between the floor and the ceiling."""
    want = (rows * _vector_index().ROW_BYTES[form or index_form()]) // 2**20
    return f"{max(INDEX_MEM_FLOOR_MB, min(INDEX_MEM_CEILING_MB, want))}MB"


def create_index(conn: psycopg.Connection) -> float:
    """Build the HNSW index in one pass over the loaded table. Returns the build time in seconds."""
    rows = conn.execute("SELECT count(*) FROM chunk WHERE embedding IS NOT NULL").fetchone()[0]
    conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem_for(rows)}'")
    conn.execute(f"SET max_parallel_maintenance_workers = {INDEX_PARALLEL_WORKERS}")
    start = time.perf_counter()
    conn.execute(index_sql())
    return time.perf_counter() - start


def vacuum_analyze(conn: psycopg.Connection) -> None:
    conn.execute("VACUUM ANALYZE chunk")


def rebuild_index(dsn: str = DEFAULT_DSN, vacuum: bool = True) -> float:
    """Drop and rebuild the index from whatever vectors exist now. Returns the build time in seconds."""
    with connect(dsn, autocommit=True) as conn:
        drop_index(conn)
        seconds = create_index(conn)
        if vacuum:
            vacuum_analyze(conn)
    return seconds


@dataclass
class Status:
    total: int
    boilerplate: int
    embedded: int
    pending: int  # no vector at all, OR a vector under a different recipe: the real queue
    index_exists: bool
    index_bytes: int
    table_bytes: int
    models: dict[str, int] = field(default_factory=dict)
    never_embedded: int = 0  # the part of `pending` that has no vector at all
    stale: int = 0  # the part of `pending` carrying an earlier recipe's vector
    index_form: str | None = None  # the live index's form (`retrieval.vector_index`); None when neither
    wanted_form: str = ""  # the form frozen retrieval queries; a difference means every search is a sequential scan


def status(dsn: str = DEFAULT_DSN, recipe: str = EMBEDDING_RECIPE) -> Status:
    with connect(dsn, autocommit=True) as conn:
        # `pending` has to count a recipe change, not just a missing vector.
        total, boilerplate, embedded, never, stale = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE is_boilerplate),"
            " count(*) FILTER (WHERE embedding IS NOT NULL),"
            " count(*) FILTER (WHERE embedding IS NULL AND NOT is_boilerplate),"
            " count(*) FILTER (WHERE embedding IS NOT NULL AND NOT is_boilerplate"
            "                  AND embedding_recipe IS DISTINCT FROM %(recipe)s) FROM chunk",
            {"recipe": recipe},
        ).fetchone()
        pending = never + stale
        models = dict(conn.execute(
            "SELECT coalesce(embedding_recipe, embedding_model || '|?|?'), count(*)"
            " FROM chunk WHERE embedding IS NOT NULL GROUP BY 1").fetchall())
        exists = index_exists(conn)
        index_bytes = conn.execute("SELECT pg_relation_size(%s::regclass)", (INDEX_NAME,)).fetchone()[0] if exists else 0
        table_bytes = conn.execute("SELECT pg_total_relation_size('chunk')").fetchone()[0]
        form = live_index_form(conn) if exists else None
    return Status(total, boilerplate, embedded, pending, exists, index_bytes, table_bytes, models,
                  never_embedded=never, stale=stale, index_form=form, wanted_form=index_form())  # fmt: skip


# --------------------------------------------------------------------------- the worker
# ---------------------------------------------------------------------------


@dataclass
class PageStats:
    done: int  # chunks looked at so far (embedded + skipped)
    total: int  # chunks this run set out to do
    embedded: int
    skipped: int
    seconds: float
    vram_allocated: float
    vram_peak: float


@dataclass
class RunResult:
    embedded: int = 0
    skipped: int = 0  # nothing embeddable in the body; left NULL
    pages: int = 0
    seconds: float = 0.0
    index_rebuilt: bool = False
    index_seconds: float | None = None
    vacuumed: bool = False
    peak_vram_mib: float = 0.0


def run(
    dsn: str,
    encoder: Encoder,
    *,
    limit: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    index_mode: str = "auto",
    index_threshold: int = INDEX_THRESHOLD,
    vacuum: bool = True,
    on_page: Callable[[PageStats], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> RunResult:
    """Embed pending chunks page by page."""
    if index_mode not in ("auto", "always", "never"):
        raise ValueError(f"index_mode must be auto, always or never, not {index_mode!r}")
    guard_database(dsn)
    phase = on_phase or (lambda _name: None)
    result = RunResult()
    admin = connect(dsn, autocommit=True)
    interrupted = False
    failed = False
    try:
        pending = count_pending(admin)
        target = pending if limit is None else min(pending, limit)
        managed = target > 0 and (index_mode == "always" or (index_mode == "auto" and target >= index_threshold))
        if managed and index_exists(admin):
            phase("drop_index")
            drop_index(admin)
        work = connect(dsn)
        try:
            phase("embed")
            started = time.perf_counter()
            after_id = 0
            while result.embedded + result.skipped < target:
                rows = fetch_page(work, after_id, min(page_size, target - result.embedded - result.skipped))
                if not rows:
                    break
                after_id = rows[-1][0]
                # Already rendered under the adopted recipe by fetch_page; None where there was nothing embeddable,
                # which encode_documents passes through as None.
                vectors = encoder.encode_documents([text for _, text in rows])
                ok = [(cid, v) for (cid, _), v in zip(rows, vectors) if v is not None]
                if ok:
                    write_embeddings(work, [cid for cid, _ in ok], np.stack([v for _, v in ok]))
                work.commit()
                result.embedded += len(ok)
                result.skipped += len(rows) - len(ok)
                result.pages += 1
                result.seconds = time.perf_counter() - started
                allocated, peak = getattr(encoder, "vram_mib", lambda: (0.0, 0.0))()
                result.peak_vram_mib = max(result.peak_vram_mib, peak)
                if on_page:
                    on_page(PageStats(result.embedded + result.skipped, target, result.embedded, result.skipped, result.seconds, allocated, peak))
        except KeyboardInterrupt:
            interrupted = True
            raise
        except BaseException:
            failed = True
            raise
        finally:
            work.close()
            if index_mode != "never" and not interrupted:
                try:
                    if not index_exists(admin):
                        phase("build_index")
                        result.index_seconds = create_index(admin)
                        result.index_rebuilt = True
                except Exception as exc:
                    if not failed:
                        # Every vector landed and the index did not: NOT a success.
                        raise
                    # The embedding loop's own error is already on its way out; report this one without masking it.
                    print(f"WARNING: could not rebuild {INDEX_NAME} either: {exc}. Run with --index-only.",
                          file=sys.stderr)
        if vacuum and result.embedded > 0:
            phase("vacuum")
            vacuum_analyze(admin)
            result.vacuumed = True
    finally:
        admin.close()
    return result


# --------------------------------------------------------------------------- command line
# ---------------------------------------------------------------------------


def _mib(n: float) -> str:
    return f"{n / 2**20:,.0f} MiB"


def _print_status(s: Status, console) -> None:
    from rich.table import Table

    t = Table(title="chunk embeddings", show_header=False)
    t.add_row("chunks", f"{s.total:,} ({s.boilerplate:,} boilerplate, never embedded)")
    t.add_row("embedded", f"{s.embedded:,}  {dict(s.models) or ''}")
    t.add_row("pending", f"{s.pending:,}" + (
        f"  ({s.never_embedded:,} never embedded, {s.stale:,} under an older recipe)" if s.stale else ""))
    t.add_row("recipe", EMBEDDING_RECIPE)
    t.add_row("HNSW index", f"{INDEX_NAME}: {_mib(s.index_bytes)}, {s.index_form or 'unknown'} form"
              if s.index_exists else f"MISSING ({INDEX_NAME})")
    if s.index_exists and s.index_form != s.wanted_form:
        t.add_row("[bold red]index form[/bold red]",
                  f"[bold red]{s.index_form} but frozen retrieval queries {s.wanted_form}: every search is a sequential "
                  f"scan. Run: uv run python scripts/switch_vector_index.py run[/bold red]")
    t.add_row("chunk table + indexes", _mib(s.table_bytes))
    console.print(t)


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="database (default: $PG_DSN or the local container)")
    parser.add_argument("--limit", type=int, help="embed at most this many chunks (smoke test)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="GPU batch size (default: %(default)s)")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="rows fetched and written per commit (default: %(default)s)")
    parser.add_argument("--dtype", choices=["float16", "float32"], default=DEFAULT_DTYPE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index", choices=["auto", "always", "never"], default="auto", dest="index_mode",
                        help="auto: manage the HNSW index when the run writes >= %d vectors" % INDEX_THRESHOLD)  # fmt: skip
    parser.add_argument("--no-vacuum", action="store_true", help="skip VACUUM ANALYZE at the end")
    parser.add_argument("--status", action="store_true", help="show counts and index state, then exit")
    parser.add_argument("--index-only", action="store_true", help="drop and rebuild the index, then VACUUM ANALYZE")
    args = parser.parse_args(argv)
    console = Console()
    guard_database(args.dsn)

    if args.status:
        _print_status(status(args.dsn), console)
        return 0
    if args.index_only:
        with console.status("Building the HNSW index..."):
            seconds = rebuild_index(args.dsn, vacuum=not args.no_vacuum)
        console.print(f"[green]Index rebuilt in {seconds:.1f}s.[/green]")
        _print_status(status(args.dsn), console)
        return 0

    with console.status(f"Loading {MODEL_NAME} on {args.device} ({args.dtype})..."):
        encoder = Embedder(args.device, args.dtype, args.batch_size)
    console.print(f"Model loaded: {MODEL_NAME} @ {MODEL_REVISION[:8]}, {args.dtype}, batch {args.batch_size}, on {args.device}")

    progress = Progress(
        SpinnerColumn(), TextColumn("[bold]Embedding"), BarColumn(), MofNCompleteColumn(),
        TextColumn("{task.fields[rate]} chunks/s"), TimeElapsedColumn(), TimeRemainingColumn(),
        TextColumn("VRAM {task.fields[vram]}"), console=console,
    )  # fmt: skip
    labels = {
        "drop_index": f"Dropping {INDEX_NAME} before the bulk load",
        "build_index": f"Building {INDEX_NAME} (HNSW m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}) over "
                       "the finished table; this is the slow step and it reports no progress",
        "vacuum": "VACUUM ANALYZE chunk",
    }
    task_id = None

    def on_phase(name: str) -> None:
        if name in labels:
            console.print(f"[cyan]{labels[name]}[/cyan]")

    def on_page(p: PageStats) -> None:
        progress.update(task_id, completed=p.done, rate=f"{p.embedded / max(p.seconds, 1e-9):,.0f}", vram=f"{p.vram_allocated:,.0f}/{p.vram_peak:,.0f} MiB")

    try:
        with progress:
            task_id = progress.add_task("", total=1, rate="-", vram="-")
            before = status(args.dsn)
            target = before.pending if args.limit is None else min(before.pending, args.limit)
            progress.update(task_id, total=max(target, 1))
            result = run(
                args.dsn, encoder, limit=args.limit, page_size=args.page_size, index_mode=args.index_mode,
                vacuum=not args.no_vacuum, on_page=on_page, on_phase=on_phase,
            )  # fmt: skip
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/yellow] Finished pages are saved. The HNSW index was NOT rebuilt; re-run this command to "
            "resume (it rebuilds at the end) or use --index-only."
        )
        return 130

    summary = Table(title="run summary", show_header=False)
    summary.add_row("embedded", f"{result.embedded:,}")
    summary.add_row("skipped (nothing to embed)", f"{result.skipped:,}")
    summary.add_row("embedding time", f"{result.seconds:,.1f}s  ({result.embedded / max(result.seconds, 1e-9):,.0f} chunks/s)")
    summary.add_row("peak VRAM (PyTorch)", f"{result.peak_vram_mib:,.0f} MiB")
    summary.add_row("index rebuilt", f"yes, {result.index_seconds:.1f}s" if result.index_rebuilt else "no")
    summary.add_row("VACUUM ANALYZE", "yes" if result.vacuumed else "no")
    console.print(summary)
    after = status(args.dsn)
    _print_status(after, console)
    if args.index_mode != "never" and not after.index_exists:
        # The second line behind run: whatever happened, a run that manages the index does not end without one.
        console.print(f"[bold red]{INDEX_NAME} does not exist after the run.[/bold red] Run with --index-only.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
