"""Which database a write is allowed to touch."""

from __future__ import annotations

from psycopg.conninfo import conninfo_to_dict, make_conninfo

# Never written by the Corpus v2 loader, the orchestrator or any test.
PROTECTED = frozenset({"hansard"})

TARGET = "hansard_v2"

# Anything a test is allowed to create and drop.
TEST_PREFIXES = ("hansard_rt_test_", "hansard_scratch", "hansard_load_test_", "hansard_oracle_")


class WrongDatabase(RuntimeError):
    """The DSN does not name the database this code is allowed to write to."""


def database_of(dsn: str) -> str:
    """The dbname a DSN resolves to. A DSN with no dbname is refused rather than guessed: libpq would fall back to the
    OS user, which is nobody's intention here.
    """
    try:
        name = conninfo_to_dict(dsn).get("dbname")
    except Exception as exc:  # noqa: BLE001 - psycopg raises several types for a bad DSN
        raise WrongDatabase(f"could not read a database name out of the DSN ({type(exc).__name__})") from exc
    if not name:
        raise WrongDatabase("the DSN names no database; refusing to let libpq guess one")
    return str(name)


def require(dsn: str, expected: str = TARGET) -> str:
    """Return `dsn` if it points at `expected`; otherwise refuse, naming what it found."""
    found = database_of(dsn)
    # UNCONDITIONAL.
    if found in PROTECTED or expected in PROTECTED:
        raise WrongDatabase(
            f"refusing to write to {found!r}: {sorted(PROTECTED)} is the rollback corpus and no step writes to it. "
            f"Point PG_DSN at .../{TARGET} and try again.")
    if found != expected:
        raise WrongDatabase(f"this step writes to {expected!r}, but the DSN names {found!r}")
    return dsn


def require_test(dsn: str) -> str:
    """For tests: the database must be a throw-away one, whatever PG_DSN happens to say."""
    found = database_of(dsn)
    if not found.startswith(TEST_PREFIXES):
        raise WrongDatabase(
            f"a test tried to use {found!r}; tests may only use a database whose name starts "
            f"with one of {TEST_PREFIXES}")
    return dsn


def refuse_protected(dsn: str) -> str:
    """Return `dsn` unless it names a PROTECTED database (or none at all)."""
    found = database_of(dsn)
    if found in PROTECTED:
        raise WrongDatabase(
            f"refusing to write to {found!r}: {sorted(PROTECTED)} is the rollback corpus and no step writes to it. "
            f"Point PG_DSN at .../{TARGET} and try again.")
    return dsn


def refuse_for_v1(dsn: str) -> str:
    """For the retired v1 loaders (fill_gaps, forensic_fill, backfill_speech_meta, ingest_divisions, ingest_bills,
    ingest_members): never the rollback, and never TARGET either.
    """
    found = database_of(dsn)
    if found in PROTECTED:
        raise WrongDatabase(
            f"refusing to write to {found!r}: {sorted(PROTECTED)} is the rollback corpus and no step writes to it.")
    if found == TARGET:
        raise WrongDatabase(
            f"this is the retired v1 pipeline, and {TARGET!r} is built only from the Corpus v2 cache "
            "(scripts/fetch_corpus_v2.py, then scripts/load_corpus_v2.py or scripts/pipeline.py). "
            "Point PG_DSN at a scratch database to run it.")
    return dsn


def working_dsn(dsn: str) -> str:
    """The database a tool uses when nobody chose one: PG_DSN's, unless that is the rollback or names no database, in
    which case TARGET — with PG_DSN's host and credentials either way.
    """
    try:
        found = database_of(dsn)
    except WrongDatabase:
        found = None
    if found is None or found in PROTECTED:
        return with_database(dsn, TARGET)
    return dsn


def with_database(dsn: str, name: str) -> str:
    """The same DSN pointed at another database. Used to reach `postgres` for CREATE/DROP."""
    return make_conninfo(dsn, dbname=name)
