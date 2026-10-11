"""What the SOURCES lack, stored as data (migration 020): a division Hansard lists with no voter list anywhere."""

from __future__ import annotations

import pytest

from app import safety
from ingest import corpus_fetch as CF
from ingest import load as L
from ingest.corpus_fetch import DivisionTally, WindowSpec
from ingest.runtime import run


def tally(house, hansard, votes):
    return DivisionTally(house=house, hansard={("2015-06-01", n) for n in range(hansard)},
                         votes={("2015-06-01", n) for n in range(votes)})  # fmt: skip


@pytest.mark.parametrize("hansard, votes, want", [(30, 0, [("Commons", 30)]), (30, 29, []), (0, 0, []), (0, 5, [])])
def test_the_gap_rule_is_the_same_one_verify_applies(hansard, votes, want):
    assert L.voter_list_gaps({"Commons": tally("Commons", hansard, votes)}) == want


@pytest.fixture
def window_2015(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE source_gap")
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('C2015Q2', 'test', '2015-04-01', '2015-06-30', 'corpus')")
    return WindowSpec(id="C2015Q2", start=__import__("datetime").date(2015, 4, 1),
                      end=__import__("datetime").date(2015, 6, 30), role="corpus", active=True)


def record(dsn, window, tallies, monkeypatch):
    import psycopg

    monkeypatch.setattr(CF, "reconcile_divisions", lambda cache, w: tallies)

    class _Cache:
        root = "unused"

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
        try:
            return await L.record_source_gaps(conn, _Cache(), window)
        finally:
            await conn.close()
    return run(go())


def test_a_window_with_no_voter_lists_is_recorded_per_house(window_2015, scratch, scratch_dsn, monkeypatch):
    result = record(scratch_dsn, window_2015, {"Commons": tally("Commons", 30, 0), "Lords": tally("Lords", 12, 12)},
                    monkeypatch)  # fmt: skip
    assert result["no_voter_lists"] == 1
    rows = scratch.execute("SELECT house, source, kind, count FROM source_gap").fetchall()
    assert rows == [("Commons", "commons_votes_api", "no_voter_lists", 30)]


def test_a_gap_the_source_later_fills_is_cleared(window_2015, scratch, scratch_dsn, monkeypatch):
    record(scratch_dsn, window_2015, {"Commons": tally("Commons", 30, 0)}, monkeypatch)
    result = record(scratch_dsn, window_2015, {"Commons": tally("Commons", 30, 30)}, monkeypatch)
    assert result["cleared"] == 1
    assert scratch.execute("SELECT count(*) FROM source_gap").fetchone()[0] == 0


def test_recording_twice_changes_nothing(window_2015, scratch, scratch_dsn, monkeypatch):
    record(scratch_dsn, window_2015, {"Commons": tally("Commons", 30, 0)}, monkeypatch)
    before = scratch.execute("SELECT recorded_at FROM source_gap").fetchone()[0]
    record(scratch_dsn, window_2015, {"Commons": tally("Commons", 30, 0)}, monkeypatch)
    assert scratch.execute("SELECT recorded_at FROM source_gap").fetchone()[0] == before


def test_the_schema_refuses_a_gap_for_no_house(window_2015, scratch):
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        scratch.execute("INSERT INTO source_gap (window_id, source, house, kind, count, detail) "
                        "VALUES ('C2015Q2', 's', 'Both', 'k', 1, 'd')")
