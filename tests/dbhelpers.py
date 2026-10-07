"""Shared helpers for tests that read the LIVE database. They skip, never fail, when it is not there."""

import psycopg
import pytest

from app import safety
from app.config import settings


def connect():
    """The live corpus, read-only."""
    try:
        return psycopg.connect(safety.working_dsn(settings.pg_dsn), connect_timeout=3, autocommit=True,
                               options="-c default_transaction_read_only=on")
    except psycopg.OperationalError:
        return None


def scalar(db, sql, *params):
    return db.execute(sql, params or None).fetchone()[0]  # None, not : a bare "%" in LIKE is not a placeholder


def require(db, sql, message):
    """Skip unless a pipeline stage has produced data (so a fresh clone still gets a green unit-test run)."""
    if not scalar(db, sql):
        pytest.skip(message)
