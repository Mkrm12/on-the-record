"""Migration 024 stores the names each lobby's list records and holds `tally_reconciled` to them."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

MIGRATION = Path(__file__).resolve().parents[1] / "db" / "migrations" / "024_division_named_counts.sql"


def division(conn, did, house, ayes, noes, reconciled, anomalies=()):
    source = did - 1_000_000 if house == "Lords" else did
    conn.execute("INSERT INTO division (id, house, source_id, division_date, ayes, noes, tally_reconciled, anomalies) "
                 "VALUES (%s, %s, %s, '2026-09-08', %s, %s, %s, %s)",
                 (did, house, source, ayes, noes, reconciled, list(anomalies)))  # fmt: skip


def votes(conn, did, rows):
    for mnis_id, lobby, teller, flags in rows:
        conn.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, is_teller, flags) VALUES (%s, %s, %s, %s, %s)",
                     (did, mnis_id, lobby, teller, list(flags)))  # fmt: skip


def state(conn, did):
    return conn.execute("SELECT ayes_named, noes_named, tally_reconciled, anomalies FROM division WHERE id = %s",
                        (did,)).fetchone()  # fmt: skip


@pytest.fixture
def old_rule(scratch):
    """Four divisions as the old loader stored them."""
    # 1.
    division(scratch, 101, "Commons", 2, 3, True)
    votes(scratch, 101, [(1, "aye", False, ()), (2, "aye", False, ()), (3, "aye", True, ()), (4, "aye", True, ()),
                         (5, "no", False, ()), (6, "no", True, ()), (7, "no", True, ())])
    # 2.
    division(scratch, 102, "Commons", 2, 2, False,
             ["aye_count_mismatch:official=2,listed=1,with_tellers=3",
              "no_count_mismatch:official=2,listed=1,with_tellers=3", "conflicting_votes:1"])
    votes(scratch, 102, [(1, "aye", False, ()), (2, "no", False, ()), (3, "conflict", False, ("listed_aye", "listed_no")),
                         (4, "aye", True, ()), (5, "aye", True, ()), (6, "no", True, ()), (7, "no", True, ())])
    # 3.
    division(scratch, 1_000_103, "Lords", 2, 1, True)
    votes(scratch, 1_000_103, [(11, "aye", False, ()), (12, "aye", True, ()), (13, "no", False, ())])
    # 4. a figure not recorded
    division(scratch, 104, "Commons", None, 3, None)
    votes(scratch, 104, [(1, "no", False, ())])
    return scratch


def test_the_backfill_counts_the_named_members_and_settles_the_flag(old_rule):
    old_rule.execute(MIGRATION.read_text(encoding="utf-8"))
    assert state(old_rule, 101) == (2, 1, False, ["no_count_mismatch:official=3,named=1"])
    assert state(old_rule, 102) == (2, 2, True, ["conflicting_votes:1"])  # the stale mismatches go, the rest stays
    assert state(old_rule, 1_000_103) == (2, 1, True, [])
    assert state(old_rule, 104) == (0, 1, None, ["no_count_mismatch:official=3,named=1"])


def test_the_backfill_is_idempotent(old_rule):
    old_rule.execute(MIGRATION.read_text(encoding="utf-8"))
    before = old_rule.execute("SELECT id, xmin::text FROM division ORDER BY id").fetchall()
    old_rule.execute(MIGRATION.read_text(encoding="utf-8"))
    assert old_rule.execute("SELECT id, xmin::text FROM division ORDER BY id").fetchall() == before


def test_a_flag_the_named_counts_contradict_cannot_be_stored(old_rule):
    old_rule.execute(MIGRATION.read_text(encoding="utf-8"))
    with pytest.raises(psycopg.errors.CheckViolation):
        old_rule.execute("UPDATE division SET tally_reconciled = true WHERE id = 101")
    with pytest.raises(psycopg.errors.CheckViolation):
        old_rule.execute("UPDATE division SET ayes_named = -1 WHERE id = 102")


def test_the_backfill_refuses_a_commons_teller_named_in_the_other_lobby(old_rule):
    """The vote row does not say which side such a member told, so the count cannot be derived from it."""
    votes(old_rule, 101, [(8, "conflict", True, ("listed_aye", "listed_no"))])
    with pytest.raises(psycopg.errors.RaiseException, match="told and was named in the other lobby"):
        old_rule.execute(MIGRATION.read_text(encoding="utf-8"))
    old_rule.execute("ROLLBACK")  # the migration's own transaction, aborted by the refusal
    assert state(old_rule, 101) == (None, None, True, [])  # nothing was written


def test_a_division_stored_by_the_loader_already_agrees_with_the_backfill(scratch, scratch_dsn):
    """The loader (Python, from the payload's lists) and the backfill (SQL, from the vote rows) are two implementations
    of one definition. Stored by the first, the second must find nothing to change.
    """
    from ingest.divisions import parse_commons, parse_lords, store
    from ingest.runtime import run

    def m(mid):
        return {"MemberId": mid, "Name": f"M{mid}", "Party": "Labour"}

    def lm(mid):
        return {"memberId": mid, "name": f"L{mid}", "party": "Crossbench"}

    commons = {"DivisionId": 201, "Date": "2026-09-08T15:00:00", "Number": 9, "Title": "t", "AyeCount": 3,
               "NoCount": 4, "Ayes": [m(1), m(2), m(9)], "Noes": [m(3), m(9)], "AyeTellers": [m(4), m(5)],
               "NoTellers": [m(6), m(7)], "NoVoteRecorded": [m(8)]}
    lords = {"divisionId": 202, "date": "2026-09-08T00:00:00", "number": 1, "title": "t",
             "authoritativeContentCount": 2, "authoritativeNotContentCount": 2, "contents": [lm(11), lm(12)],
             "notContents": [lm(13)], "contentTellers": [lm(12)], "notContentTellers": [lm(14)]}

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn, autocommit=True) as conn:
            await store(conn, parse_commons(commons))
            await store(conn, parse_lords(lords))

    run(go())
    loaded = {did: state(scratch, did) for did in (201, 1_000_202)}
    assert loaded[201][:3] == (3, 2, False) and loaded[1_000_202][:3] == (2, 2, True)  # member 9 in each lobby
    before = scratch.execute("SELECT id, xmin::text FROM division ORDER BY id").fetchall()
    scratch.execute(MIGRATION.read_text(encoding="utf-8"))
    assert scratch.execute("SELECT id, xmin::text FROM division ORDER BY id").fetchall() == before
