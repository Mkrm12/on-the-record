"""Fixtures for the GPU worker's tests."""

import os
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

DB_DIR = Path(__file__).resolve().parents[2] / "db"
BASE_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/hansard")


@pytest.fixture(scope="session")
def scratch_dsn():
    admin_dsn = make_conninfo(BASE_DSN, dbname="postgres", connect_timeout=3)
    name = f"hansard_gpu_test_{os.getpid()}"
    try:
        admin = psycopg.connect(admin_dsn, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not reachable")
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.execute(f'CREATE DATABASE "{name}"')
    dsn = make_conninfo(BASE_DSN, dbname=name)
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
            for path in sorted((DB_DIR / "migrations").glob("*.sql")):
                conn.execute(path.read_text(encoding="utf-8"))
        yield dsn
    finally:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


@pytest.fixture
def db(scratch_dsn):
    """An autocommit connection to an empty scratch database: tables truncated and the HNSW index back to its baseline
    before each test.
    """
    from embedder import _vector_index

    V = _vector_index()
    form = V.frozen_form()
    conn = psycopg.connect(scratch_dsn, autocommit=True)
    conn.execute("TRUNCATE chunk, speech, debate, corpus_window CASCADE")
    conn.execute("DROP INDEX IF EXISTS chunk_embedding_idx")
    conn.execute(f"CREATE INDEX chunk_embedding_idx ON chunk USING hnsw ({V.index_expression(form)} {V.OPCLASS[form]})")
    yield conn
    conn.close()


# Every debate and speech belongs to a window, and a debate has a date, a House and a corpus version: the production
# schema requires them (migrations 014 and 017), so the fixtures give them, as the main suite's do.
WINDOW = "TGPU"


def _window(conn):
    conn.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                 "VALUES (%s, 'test', '1900-01-01', '2100-12-31', 'dev') ON CONFLICT DO NOTHING", (WINDOW,))


def add_chunks(conn, rows):
    """Insert chunks and return their ids in order. rows: (body, is_boilerplate) pairs, or bare strings."""
    _window(conn)
    conn.execute("INSERT INTO debate (id, date, house, window_id, corpus_version) "
                 "VALUES ('d1', '2024-03-14', 'Commons', %s, 'test') ON CONFLICT DO NOTHING", (WINDOW,))
    first = conn.execute("SELECT count(*) FROM speech").fetchone()[0]  # a second call in one test continues the numbering
    ids = []
    for i, row in enumerate(rows, start=first):
        body, boilerplate = (row, False) if isinstance(row, str) else row
        conn.execute("INSERT INTO speech (id, debate_id, seq, text, window_id) VALUES (%s, 'd1', %s, %s, %s) "
                     "ON CONFLICT DO NOTHING", (f"s{i}", i, body, WINDOW))
        ids.append(
            conn.execute(
                "INSERT INTO chunk (speech_id, debate_id, seq, body, sent_idx, parent_body, is_boilerplate)"
                " VALUES (%s, 'd1', %s, %s, 0, %s, %s) RETURNING id",
                (f"s{i}", i, body, body, boilerplate),
            ).fetchone()[0]
        )
    return ids


def add_chunk_with_context(conn, body, *, date="2024-03-14", house="Commons", heading="Space Industry",
                           speaker="Ms Smith", party="Labour", boilerplate=False, suffix=""):  # fmt: skip
    """Insert one chunk with the metadata the adopted header recipe reads. Returns its id."""
    debate_id, speech_id = f"dctx{suffix}", f"sctx{suffix}"
    _window(conn)
    # The DEBATE always has its date and House (the schema requires them); the header reads the CHUNK's copies, which
    # are what a thin header leaves NULL.
    conn.execute("INSERT INTO debate (id, heading, date, house, window_id, corpus_version) "
                 "VALUES (%s, %s, '2024-03-14', 'Commons', %s, 'test') "
                 "ON CONFLICT (id) DO UPDATE SET heading = excluded.heading",
                 (debate_id, heading, WINDOW))  # fmt: skip
    conn.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_party, text, window_id) "
                 "VALUES (%s, %s, 1, %s, %s, %s, %s)"
                 " ON CONFLICT (id) DO UPDATE SET speaker_name = excluded.speaker_name,"
                 " speaker_party = excluded.speaker_party", (speech_id, debate_id, speaker, party, body, WINDOW))  # fmt: skip
    return conn.execute(
        "INSERT INTO chunk (speech_id, debate_id, seq, body, sent_idx, parent_body, is_boilerplate,"
        " debate_date, house) VALUES (%s, %s, 1, %s, 0, %s, %s, %s, %s) RETURNING id",
        (speech_id, debate_id, body, body, boilerplate, date, house),
    ).fetchone()[0]
