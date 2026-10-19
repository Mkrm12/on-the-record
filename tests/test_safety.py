"""Nothing writes to the rollback, and nothing reads it by accident believing it is the corpus."""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from app import safety

REPO = Path(__file__).resolve().parents[1]
URL = "postgresql://someone:s3cret@db.example:6543/{db}"


def dbname(dsn: str) -> str:
    return conninfo_to_dict(dsn)["dbname"]


# --- database_of ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dsn, name", [
    (URL.format(db="hansard_v2"), "hansard_v2"),
    ("host=127.0.0.1 dbname=hansard_rt_test_1 user=x", "hansard_rt_test_1"),
    ("postgresql:///hansard", "hansard"),
])  # fmt: skip
def test_database_of_reads_the_name_in_either_dsn_form(dsn, name):
    assert safety.database_of(dsn) == name


@pytest.mark.parametrize("dsn", ["host=127.0.0.1 user=x", "postgresql://u@h:5432", "this is = not a dsn ="])
def test_database_of_refuses_to_guess(dsn):
    """libpq would fall back to the OS user's name: nobody's intention, and possibly the rollback's."""
    with pytest.raises(safety.WrongDatabase):
        safety.database_of(dsn)


# --- require / refuse_protected --------------------------------------------------------------------------------------


def test_require_accepts_the_target_and_refuses_everything_else():
    assert safety.require(URL.format(db=safety.TARGET)) == URL.format(db=safety.TARGET)
    with pytest.raises(safety.WrongDatabase, match="writes to"):
        safety.require(URL.format(db="elsewhere"))


@pytest.mark.parametrize("protected", sorted(safety.PROTECTED))
def test_require_refuses_a_protected_database_even_when_asked_for_it(protected):
    with pytest.raises(safety.WrongDatabase, match="rollback"):
        safety.require(URL.format(db=protected), expected=protected)


def test_a_refusal_never_carries_the_password():
    with pytest.raises(safety.WrongDatabase) as err:
        safety.require(URL.format(db="hansard"))
    assert "s3cret" not in str(err.value)


@pytest.mark.parametrize("name", [safety.TARGET, "hansard_rt_test_9", "hansard_scratch", "anything_else"])
def test_refuse_protected_lets_every_unprotected_database_through(name):
    assert safety.refuse_protected(URL.format(db=name)) == URL.format(db=name)


@pytest.mark.parametrize("protected", sorted(safety.PROTECTED))
def test_refuse_protected_refuses_the_rollback(protected):
    with pytest.raises(safety.WrongDatabase, match="rollback"):
        safety.refuse_protected(URL.format(db=protected))


def test_refuse_protected_refuses_a_dsn_naming_no_database():
    with pytest.raises(safety.WrongDatabase):
        safety.refuse_protected("host=127.0.0.1 user=x")


# --- working_dsn: the default that can never be the rollback --------------------------------------------------------


@pytest.mark.parametrize("protected", sorted(safety.PROTECTED))
def test_working_dsn_replaces_the_rollback_with_the_target(protected):
    assert dbname(safety.working_dsn(URL.format(db=protected))) == safety.TARGET


def test_working_dsn_replaces_a_missing_database_with_the_target():
    assert dbname(safety.working_dsn("host=127.0.0.1 user=x")) == safety.TARGET


@pytest.mark.parametrize("name", [safety.TARGET, "hansard_rt_test_3", "hansard_scratch_x", "a_database_passed_down"])
def test_working_dsn_honours_a_database_named_explicitly(name):
    """A scratch database, or the orchestrator's `--database` override passed down as PG_DSN, must still work."""
    assert dbname(safety.working_dsn(URL.format(db=name))) == name


def test_working_dsn_keeps_host_port_user_and_password():
    parts = conninfo_to_dict(safety.working_dsn(URL.format(db="hansard")))
    assert (parts["host"], parts["port"], parts["user"], parts["password"]) == ("db.example", "6543", "someone", "s3cret")


@pytest.mark.parametrize("password", ["with space", "it's", 'dq"uote', "back\\slash", "=equals=", "ünïcödé"])
def test_a_password_with_awkward_characters_survives_the_database_swap(password):
    """`with_database` used to join key=value pairs unquoted: a space or a quote in the password broke the DSN."""
    dsn = psycopg.conninfo.make_conninfo(host="h", user="u", password=password, dbname="hansard")
    for swapped in (safety.working_dsn(dsn), safety.with_database(dsn, "postgres")):
        assert conninfo_to_dict(swapped)["password"] == password


def test_with_database_changes_only_the_database():
    before = conninfo_to_dict(URL.format(db="hansard_v2"))
    after = conninfo_to_dict(safety.with_database(URL.format(db="hansard_v2"), "postgres"))
    assert after.pop("dbname") == "postgres" and before.pop("dbname") == "hansard_v2"
    assert after == before


# --- the shared pool --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("configured", [URL.format(db="hansard"), "host=127.0.0.1 user=x",
                                        URL.format(db=safety.TARGET)])
def test_the_shared_pools_default_database_is_never_the_rollback(monkeypatch, configured):
    from app import db as db_mod
    from app.config import settings

    monkeypatch.setattr(settings, "pg_dsn", configured)
    assert dbname(db_mod.make_pool(served=False).conninfo) not in safety.PROTECTED


def test_the_pool_every_module_imports_is_not_on_the_rollback():
    """With PG_DSN as it really is in this checkout: the pool `ingest.rules` and `ingest.chunk` write through."""
    from ingest import chunk, rules

    for module in (rules, chunk):
        assert dbname(module.pool.conninfo) not in safety.PROTECTED, module.__name__


def test_an_explicit_dsn_is_used_as_given():
    from app import db as db_mod

    assert db_mod.make_pool(URL.format(db="hansard_rt_test_5"), served=False).conninfo == URL.format(db="hansard_rt_test_5")


def _write_through_pool(dsn: str) -> None:
    from app.db import make_pool
    from ingest.runtime import run

    async def go():
        pool = make_pool(dsn, min_size=1, max_size=1, served=False)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                assert (await (await conn.execute("SELECT 1")).fetchone())[0] == 1  # reads always work
                await conn.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                                   "VALUES ('RO', 'test', '2026-01-01', '2026-01-02', 'dev')")
        finally:
            await pool.close()

    run(go())


def test_a_pool_connection_to_a_protected_database_cannot_write(scratch, scratch_dsn, monkeypatch):
    """Server-enforced: the check is on `current_database`, so no DSN spelling gets round it."""
    monkeypatch.setattr(safety, "PROTECTED", safety.PROTECTED | {dbname(scratch_dsn)})
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        _write_through_pool(scratch_dsn)
    assert scratch.execute("SELECT count(*) FROM corpus_window WHERE window_id = 'RO'").fetchone()[0] == 0


def test_a_pool_connection_to_an_ordinary_database_can_write(scratch, scratch_dsn):
    """The guard must not make everything read-only: the loaders write through this pool."""
    _write_through_pool(scratch_dsn)
    assert scratch.execute("SELECT count(*) FROM corpus_window WHERE window_id = 'RO'").fetchone()[0] == 1


# --- the live-data tests' connection ----------------------------------------------------------------------------------


def test_the_live_test_connection_is_read_only(scratch, scratch_dsn, monkeypatch):
    import dbhelpers

    from app.config import settings

    monkeypatch.setattr(settings, "pg_dsn", scratch_dsn)
    conn = dbhelpers.connect()
    try:
        assert conn.execute("SELECT current_database()").fetchone()[0] == dbname(scratch_dsn)
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                         "VALUES ('RO', 'test', '2026-01-01', '2026-01-02', 'dev')")
    finally:
        conn.close()


def test_the_live_test_connection_reads_the_target_not_the_rollback(monkeypatch):
    import dbhelpers

    from app.config import settings

    monkeypatch.setattr(settings, "pg_dsn", settings.pg_dsn)  # whatever this checkout says, restored after
    conn = dbhelpers.connect()
    if conn is None:
        pytest.skip("Postgres is not reachable")
    try:
        assert conn.execute("SELECT current_database()").fetchone()[0] not in safety.PROTECTED
    finally:
        conn.close()


# --- the retired v1 loaders: never the rollback, never the v2 corpus -----------------------------------------------

V1_RAW = ["fill_gaps", "forensic_fill"]  # raw psycopg connections from PG_DSN
V1_POOLED = ["ingest_divisions", "ingest_bills", "ingest_members", "backfill_speech_meta"]  # the shared pool


def test_refuse_for_v1_refuses_the_rollback_and_the_target_and_nothing_else():
    for name in sorted(safety.PROTECTED):
        with pytest.raises(safety.WrongDatabase, match="rollback"):
            safety.refuse_for_v1(URL.format(db=name))
    with pytest.raises(safety.WrongDatabase, match="Corpus v2 cache"):
        safety.refuse_for_v1(URL.format(db=safety.TARGET))
    assert safety.refuse_for_v1(URL.format(db="hansard_scratch_v1")) == URL.format(db="hansard_scratch_v1")


def _load_script(script):
    import importlib.util

    spec = importlib.util.spec_from_file_location(script, REPO / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("database", ["hansard", safety.TARGET])
@pytest.mark.parametrize("script", V1_RAW)
def test_the_raw_v1_loaders_refuse_before_any_request(monkeypatch, script, database):
    import httpx

    from app.config import settings

    module = _load_script(script)
    monkeypatch.setattr(settings, "pg_dsn", URL.format(db=database))
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("a request was made before the refusal"))
    monkeypatch.setattr("sys.argv", [script])
    with pytest.raises(safety.WrongDatabase):
        module.main()


@pytest.mark.parametrize("database", ["hansard", safety.TARGET])
@pytest.mark.parametrize("script", V1_POOLED)
def test_the_pooled_v1_loaders_refuse_before_any_request(monkeypatch, script, database):
    """The shared pool now defaults to hansard_v2, so without their own guard these would have written a second, live-
    API copy of divisions, bills and members into the corpus that is built only from the cache.
    """
    import httpx

    from app import db as db_mod

    module = _load_script(script)
    monkeypatch.setattr(db_mod, "pool", db_mod.make_pool(URL.format(db=database), served=False))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: pytest.fail("a request was made before the refusal"))
    monkeypatch.setattr("sys.argv", [script])
    with pytest.raises(safety.WrongDatabase):
        module.main()


# --- no new raw connection takes its database from PG_DSN -------------------------------------------------------------

# A connection built straight from PG_DSN is how every one of these happened.
_RAW = re.compile(r"(?:psycopg\.connect|AsyncConnection\.connect|connect|make_pool)\(\s*settings\.pg_dsn\b")


def test_no_source_file_connects_with_pg_dsn_unguarded():
    offenders = []
    for root in ("src", "scripts", "tests"):
        for path in (REPO / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path.relative_to(REPO)}:{text.count(chr(10), 0, m.start()) + 1}"
                          for m in _RAW.finditer(text)]
    assert offenders == [], offenders
