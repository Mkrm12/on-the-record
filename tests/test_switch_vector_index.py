"""The index switch and migration 023, on the scratch database."""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
import pytest
from corpus import Corpus, vec

from app.safety import WrongDatabase
from retrieval import vector_index as V
from retrieval.hybrid import SearchFilters, fuse_sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import switch_vector_index as S  # noqa: E402

MIGRATION = (ROOT / "db" / "migrations" / "023_halfvec_index.sql").read_text(encoding="utf-8")


def live(conn) -> dict | None:
    return S.index_state(conn, V.INDEX_NAME)


def set_index(conn, form: str | None) -> None:
    conn.execute(f"DROP INDEX IF EXISTS {S.NEXT}")
    conn.execute(f"DROP INDEX IF EXISTS {V.INDEX_NAME}")
    if form:
        conn.execute(V.index_sql(form))


@pytest.fixture
def conn(scratch, scratch_dsn):
    """An autocommit connection to the emptied scratch database, its index reset to the full-precision form, and put
    back to halfvec afterwards.
    """
    connection = psycopg.connect(scratch_dsn, autocommit=True)
    set_index(connection, "vector")
    yield connection
    set_index(connection, "halfvec")
    connection.close()


def fill(scratch) -> None:
    c = Corpus(scratch)
    c.debate("d1")
    c.speech("d1", "s1", ["Housing benefit rules changed.", "Rents rose again.", "The council objected."],
             vectors=[vec((0, 1.0)), vec((1, 1.0)), vec((2, 1.0))])


# --- the switch --------------------------------------------------------------------------------------------------


def test_run_switches_the_index_and_the_second_run_has_nothing_to_do(conn, scratch):
    fill(scratch)
    said: list[str] = []
    assert S.run(conn, "halfvec", say=said.append) is True
    state = live(conn)
    assert state["form"] == "halfvec" and state["valid"] and state["bytes"] > 0
    assert S.index_state(conn, S.NEXT) is None, "the side index became the live one; nothing is left behind"
    assert S.run(conn, "halfvec", say=said.append) is False
    assert "nothing to do" in said[-1]


def test_build_leaves_the_live_index_serving_until_the_swap(conn, scratch):
    fill(scratch)
    with pytest.raises(SystemExit, match="nothing valid to swap in"):
        S.swap(conn, "halfvec", say=lambda _: None)
    S.build(conn, "halfvec", say=lambda _: None)
    assert live(conn)["form"] == "vector", "the build must not touch the live index"
    assert S.index_state(conn, S.NEXT)["form"] == "halfvec"
    with pytest.raises(SystemExit, match="nothing valid to swap in"):
        S.swap(conn, "vector", say=lambda _: None)  # the side index is the other form
    assert live(conn)["form"] == "vector"
    S.swap(conn, "halfvec", say=lambda _: None)
    assert live(conn)["form"] == "halfvec"


def test_the_switch_works_in_both_directions_which_is_the_rollback(conn, scratch):
    fill(scratch)
    S.run(conn, "halfvec", say=lambda _: None)
    S.run(conn, "vector", say=lambda _: None)
    assert live(conn)["form"] == "vector"


def test_a_leftover_side_index_of_the_wrong_form_is_dropped_and_rebuilt(conn, scratch):
    fill(scratch)
    conn.execute(V.index_sql("vector", name=S.NEXT))
    said: list[str] = []
    S.build(conn, "halfvec", say=said.append)
    assert S.index_state(conn, S.NEXT)["form"] == "halfvec"
    assert any("leftover" in line for line in said)


def test_an_invalid_side_index_from_a_failed_concurrent_build_is_dropped_and_rebuilt(conn, scratch):
    """A CONCURRENTLY build that fails leaves an INVALID index, which Postgres still maintains on every write."""
    fill(scratch)
    conn.execute(V.index_sql("halfvec", name=S.NEXT))
    try:
        conn.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass", (S.NEXT,))
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip("cannot mark an index invalid without superuser")
    assert S.index_state(conn, S.NEXT)["valid"] is False
    with pytest.raises(SystemExit, match="nothing valid"):
        S.swap(conn, "halfvec", say=lambda _: None)
    S.build(conn, "halfvec", say=lambda _: None)
    assert S.index_state(conn, S.NEXT)["valid"] is True


def test_after_the_switch_the_production_dense_leg_uses_the_index(conn, scratch):
    fill(scratch)
    S.run(conn, "halfvec", say=lambda _: None)
    from app import sealed

    # The production leg as served: with the sealed-window mask, which must not cost the index scan.
    leg = fuse_sql(SearchFilters(), exact=False, dense_index="halfvec").split("WITH", 1)[1].split("sparse AS")[0]
    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")
        plan = " ".join(r[0] for r in conn.execute(f"EXPLAIN WITH{leg.rstrip().rstrip(',')} SELECT id FROM dense",
                                                   {"qvec": str(vec((0, 1.0))), "dense_limit": 50,
                                                    **sealed.sql_exclusion("c.debate_date")[1]}).fetchall())
    assert f"Index Scan using {V.INDEX_NAME}" in plan, plan


def test_the_rollback_database_is_refused_before_connecting():
    with pytest.raises(WrongDatabase):
        S.connect("hansard")


def test_build_memory_is_sized_by_form_between_the_bounds():
    assert S.build_memory(90_040, "halfvec") == f"{S.MEM_FLOOR_MB}MB"
    assert int(S.build_memory(1_000_000, "vector")[:-2]) > int(S.build_memory(1_000_000, "halfvec")[:-2])
    assert S.build_memory(500_000_000, "halfvec") == f"{S.MEM_CEILING_MB}MB"


def test_status_names_the_live_index_and_the_frozen_form(conn):
    report = S.status(conn)
    assert report["live"]["form"] == "vector" and report["next"] is None
    assert report["frozen_form"] == V.frozen_form()


# --- migration 023 -------------------------------------------------------------------------------------------------


def test_the_migration_converts_the_index_of_an_empty_table(conn):
    conn.execute(MIGRATION)
    assert live(conn)["form"] == "halfvec"


def test_the_migration_does_nothing_when_the_index_is_already_halfvec(conn):
    set_index(conn, "halfvec")
    before = conn.execute("SELECT %s::regclass::oid", (V.INDEX_NAME,)).fetchone()[0]
    conn.execute(MIGRATION)
    assert conn.execute("SELECT %s::regclass::oid", (V.INDEX_NAME,)).fetchone()[0] == before, "not rebuilt"


def test_the_migration_refuses_a_rebuild_over_existing_vectors_and_names_the_command(conn, scratch):
    fill(scratch)
    with pytest.raises(psycopg.errors.RaiseException, match="switch_vector_index.py run"):
        conn.execute(MIGRATION)
    assert live(conn)["form"] == "vector", "nothing changed"


# --- the runtime says so at start-up ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("index", "warns"), [("halfvec", False), ("vector", True), (None, True)])
def test_the_runtime_warns_with_the_command_when_the_live_index_is_not_the_form_it_queries(conn, scratch_dsn, caplog,
                                                                                          index, warns):
    import logging

    from answer.pipeline import _check_index_form
    from app.db import make_pool
    from ingest.runtime import run as run_async

    set_index(conn, index)

    async def check():
        pool = make_pool(scratch_dsn, min_size=1, max_size=1, served=False)
        await pool.open(wait=True)
        try:
            return await _check_index_form(pool, "halfvec")
        finally:
            await pool.close()

    with caplog.at_level(logging.WARNING, logger="answer.pipeline"):
        live = run_async(check())
    assert live == index
    warned = [r.getMessage() for r in caplog.records if "sequential scan" in r.getMessage()]
    assert bool(warned) == warns
    if warns:
        assert "switch_vector_index.py run --form halfvec" in warned[0]


def test_the_build_is_concurrent_and_under_the_side_name(conn, scratch, monkeypatch):
    """CONCURRENTLY is the point of the script: the live index keeps serving every search while the new one builds. A
    plain CREATE INDEX would work just as well on a test table, so the call itself is checked.
    """
    fill(scratch)
    calls = []
    real = S.V.index_sql

    def spy(form, name=V.INDEX_NAME, **kw):
        calls.append((form, name, kw.get("concurrently")))
        return real(form, name, **kw)

    monkeypatch.setattr(S.V, "index_sql", spy)
    S.build(conn, "halfvec", say=lambda _: None)
    assert calls == [("halfvec", S.NEXT, True)]
