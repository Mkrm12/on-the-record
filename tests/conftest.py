import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.conninfo import make_conninfo

from app.config import settings
from dbhelpers import connect

DB_DIR = Path(__file__).resolve().parents[1] / "db"


def pytest_addoption(parser):
    """--live-api opts in to the handful of tests that really talk to Parliament's APIs. Off by default so CI, and a
    fresh clone with no network, stay green.
    """
    parser.addoption("--live-api", action="store_true", default=False,
                     help="run the opt-in tests that make real API calls")


@pytest.fixture(scope="module")
def db():
    conn = connect()
    if conn is None:
        pytest.skip("Postgres is not reachable")
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def scratch_dsn():
    """A throw-away database built from the repo's own db/schema.sql and migrations. Tests that need to control every
    row (retrieval, ranking) use it, so they never depend on, or touch, the live data.
    """
    name = f"hansard_rt_test_{os.getpid()}"
    try:
        admin = psycopg.connect(make_conninfo(settings.pg_dsn, dbname="postgres", connect_timeout=3), autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not reachable")
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.execute(f'CREATE DATABASE "{name}"')
    dsn = make_conninfo(settings.pg_dsn, dbname=name)
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
def scratch(scratch_dsn):
    """An autocommit, vector-aware connection to the empty scratch database, emptied before every test."""
    conn = psycopg.connect(scratch_dsn, autocommit=True)
    register_vector(conn)
    # corpus_window is in the list because debate.window_id references it: leaving it behind would let one test's window
    # tag decide whether another test's insert is legal.
    conn.execute("TRUNCATE chunk, speech, debate, member_id_map, corpus_window, "
                 "hansard_correction, division, bill, lexeme_df, corpus_stat, load_report CASCADE")
    yield conn
    conn.close()
