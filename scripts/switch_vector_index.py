"""Switch the live HNSW index between its two forms without stopping searches."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from retrieval import vector_index as V  # noqa: E402

NEXT = "chunk_embedding_next"
# The graph is built in memory when it fits and an order of magnitude slower when it does not; same bounds as the
# embedder (gpu/embedder.py), whose build this mirrors.
MEM_FLOOR_MB, MEM_CEILING_MB = 512, 4_096
SWAP_LOCK_TIMEOUT = "10s"


def build_memory(rows: int, form: str) -> str:
    want = (rows * V.ROW_BYTES[form]) // 2**20
    return f"{max(MEM_FLOOR_MB, min(MEM_CEILING_MB, want))}MB"


def index_state(conn: psycopg.Connection, name: str) -> dict | None:
    """The named index on `chunk`: its form, validity and size; None when there is none."""
    row = conn.execute(
        "SELECT pg_get_indexdef(i.indexrelid), i.indisvalid, pg_relation_size(i.indexrelid) "
        "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = 'chunk'::regclass AND c.relname = %s", (name,)).fetchone()
    if row is None:
        return None
    definition, valid, size = row
    return {"name": name, "form": V.form_of(definition), "valid": bool(valid), "bytes": int(size),
            "definition": definition}


def status(conn: psycopg.Connection) -> dict:
    live, nxt = index_state(conn, V.INDEX_NAME), index_state(conn, NEXT)
    progress = conn.execute(
        "SELECT p.phase, p.blocks_done, p.blocks_total, p.tuples_done, p.tuples_total "
        "FROM pg_stat_progress_create_index p WHERE p.relid = 'chunk'::regclass").fetchone()
    return {"live": live, "next": nxt, "frozen_form": V.frozen_form(),
            "building": dict(zip(("phase", "blocks_done", "blocks_total", "tuples_done", "tuples_total"), progress))
            if progress else None}


def build(conn: psycopg.Connection, form: str, say=print) -> float:
    """CREATE INDEX CONCURRENTLY under NEXT. Returns seconds."""
    old = index_state(conn, NEXT)
    if old is not None:
        if old["valid"] and old["form"] == form:
            say(f"{NEXT} already exists, valid, form {form}: nothing to build")
            return 0.0
        say(f"dropping a leftover {NEXT} (form {old['form']}, valid {old['valid']})")
        conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {NEXT}")
    rows = conn.execute("SELECT count(*) FROM chunk WHERE embedding IS NOT NULL").fetchone()[0]
    memory = build_memory(rows, form)
    conn.execute(f"SET maintenance_work_mem = '{memory}'")
    conn.execute("SET max_parallel_maintenance_workers = 0")  # the container's 64 MB /dev/shm (gpu/embedder.py)
    conn.execute("SET statement_timeout = 0")
    notices: list[str] = []
    conn.add_notice_handler(lambda d: notices.append(d.message_primary or ""))
    say(f"building {NEXT} ({form}) over {rows:,} vectors with maintenance_work_mem {memory} ...")
    started = time.perf_counter()
    conn.execute(V.index_sql(form, name=NEXT, concurrently=True))
    seconds = time.perf_counter() - started
    for message in notices:
        say(f"  server notice: {message}")  # pgvector says so when the graph spilled out of memory
    state = index_state(conn, NEXT)
    if state is None or not state["valid"] or state["form"] != form:
        raise SystemExit(f"the build finished but {NEXT} is not a valid {form} index: {state}")
    say(f"built {NEXT} in {seconds:,.0f} s: {state['bytes'] / 2**20:,.0f} MiB ({state['bytes'] / max(rows, 1):,.0f} "
        f"B/vector)")
    return seconds


def swap(conn: psycopg.Connection, form: str, say=print) -> None:
    """Replace the live index with NEXT in one short transaction. Refuses unless NEXT is a valid index of `form`."""
    nxt = index_state(conn, NEXT)
    if nxt is None or not nxt["valid"] or nxt["form"] != form:
        raise SystemExit(f"nothing valid to swap in: {NEXT} is {nxt}. Run `build --form {form}` first.")
    with conn.transaction():
        conn.execute(f"SET LOCAL lock_timeout = '{SWAP_LOCK_TIMEOUT}'")
        conn.execute(f"DROP INDEX IF EXISTS {V.INDEX_NAME}")
        conn.execute(f"ALTER INDEX {NEXT} RENAME TO {V.INDEX_NAME}")
    live = index_state(conn, V.INDEX_NAME)
    if live is None or live["form"] != form or not live["valid"]:
        raise SystemExit(f"the swap committed but {V.INDEX_NAME} is {live}")
    say(f"{V.INDEX_NAME} is now the {form} index ({live['bytes'] / 2**20:,.0f} MiB)")


def run(conn: psycopg.Connection, form: str, say=print) -> bool:
    """Build and swap unless the live index is already `form`. Returns whether anything changed."""
    live = index_state(conn, V.INDEX_NAME)
    if live is not None and live["valid"] and live["form"] == form:
        say(f"{V.INDEX_NAME} is already the {form} index: nothing to do")
        return False
    build(conn, form, say)
    swap(conn, form, say)
    return True


def connect(database: str) -> psycopg.Connection:
    dsn = safety.refuse_protected(make_conninfo(settings.pg_dsn, dbname=database))
    return psycopg.connect(dsn, autocommit=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=("status", "build", "swap", "run"))
    p.add_argument("--form", choices=V.FORMS, default=None,
                   help="default: the form config/retrieval_frozen.json names (%(default)s means that)")
    p.add_argument("--database", default=safety.TARGET)
    args = p.parse_args(argv)
    form = args.form or V.frozen_form()
    with connect(args.database) as conn:
        if args.action == "status":
            for key, value in status(conn).items():
                print(f"{key}: {value}")
            return 0
        if args.action == "build":
            build(conn, form)
        elif args.action == "swap":
            swap(conn, form)
        else:
            run(conn, form)
    return 0


if __name__ == "__main__":
    sys.exit(main())
