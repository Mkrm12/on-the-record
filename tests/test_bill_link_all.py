"""`ingest.bills.link_all` against a real Postgres: the whole chain, idempotence, and the commit contract."""

from __future__ import annotations

from datetime import date

import psycopg
import pytest

from ingest import bills as B
from ingest.runtime import run as run_async

DAY = date(2026, 1, 22)


@pytest.fixture
def seeded(scratch, scratch_dsn):
    """One window; a renamed bill with a debated and a formal stage; a Lords division on it; distractors."""
    for table in ("division_vote", "division", "bill_stage_sitting", "bill_stage", "bill", "speech", "debate",
                  "corpus_window"):
        scratch.execute(f"TRUNCATE {table} CASCADE")
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('W4', 'v2', '2026-01-01', '2026-03-31', 'dev_test')")
    for debate_id, heading, house in (("dPS", "Pension Schemes Bill", "Lords"),
                                      ("dAB", "Arrangement of Business", "Lords"),
                                      ("dPC", "Pension Credit", "Lords"),
                                      ("dCO", "Pension Schemes Bill", "Commons")):
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                        "VALUES (%s, %s, %s, %s, %s, 'W4', 'test')",
                        (debate_id, DAY, house, heading, f"{house} Chamber"))
    # Stored under the Royal Assent title, as the bills API returns it.
    scratch.execute("INSERT INTO bill (id, title) VALUES (4001, 'Pension Schemes Act 2026')")
    scratch.execute("INSERT INTO bill (id, title) VALUES (4002, 'Unrelated Act 2026')")
    stages = ((501, 4001, "Committee stage", "Lords", "CS"), (502, 4001, "1st reading", "Lords", "1R"),
              (503, 4002, "2nd reading", "Lords", "2R"))
    for stage_id, bill_id, name, house, abbreviation in stages:
        scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house, abbreviation) VALUES (%s,%s,%s,%s,%s)",
                        (stage_id, bill_id, name, house, abbreviation))
    for sitting_id, stage_id, bill_id in ((901, 501, 4001), (902, 502, 4001), (903, 503, 4002)):
        scratch.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, house) "
                        "VALUES (%s, %s, %s, %s, 'Lords')", (sitting_id, stage_id, bill_id, DAY))
    scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, is_deferred, debate_id, debate_link) "
                    "VALUES (1007001, 7001, 'Lords', %s, 'Pension Schemes Bill', false, 'dPS', 'hansard_division_list')",
                    (DAY,))
    scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, is_deferred, debate_link) "
                    "VALUES (7002, 7002, 'Commons', %s, 'Draft Something Regulations 2026', true, "
                    "'hansard_link_outside_window')", (DAY,))
    yield scratch_dsn


def link(dsn: str, *, autocommit: bool, commit: bool) -> dict:
    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=autocommit)
        try:
            report = await B.link_all(conn)
            if commit and not autocommit:
                await conn.commit()
            return report
        finally:
            await conn.close()  # without a commit this rolls back, which is the point of one of the tests

    return run_async(go())


def read(dsn: str, sql: str, *args):
    with psycopg.connect(dsn, autocommit=True) as conn:  # a SECOND connection: only committed rows are visible
        return conn.execute(sql, args).fetchall()


def test_the_renamed_bills_sitting_links_to_the_bill_debate(seeded):
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT debate_id, debate_link FROM bill_stage_sitting WHERE id = 901") == [("dPS", "title_exact")]


def test_a_formal_stage_is_labelled_not_left_as_none(seeded):
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT debate_id, debate_link FROM bill_stage_sitting WHERE id = 902") == \
        [(None, "formal_no_debate")]  # fmt: skip


def test_an_unrelated_bill_does_not_steal_a_debate_that_shares_a_word(seeded):
    """"Unrelated Act 2026" has a sitting the same day; "Pension Credit" shares a word with the other bill."""
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT debate_id, debate_link FROM bill_stage_sitting WHERE id = 903") == [(None, "none")]


def test_the_stage_takes_its_first_linked_sittings_debate(seeded):
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT debate_id FROM bill_stage WHERE id = 501") == [("dPS",)]


def test_the_division_is_linked_to_its_bill_stage(seeded):
    """The step the v2 loader never ran. A Lords title names only the bill; the date and House decide."""
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT bill_stage_id, bill_link FROM division WHERE id = 1007001") == [(501, "title_date")]


def test_a_deferred_division_is_not_given_a_bill_stage(seeded):
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT bill_stage_id FROM division WHERE id = 7002") == [(None,)]


def test_a_second_run_changes_nothing(seeded):
    """Idempotence is what makes it safe to run after every window load."""
    link(seeded, autocommit=True, commit=False)
    again = link(seeded, autocommit=True, commit=False)
    assert again["sittings_changed"] == 0 and again["divisions_changed"] == 0 and again["stages_debate_set"] == 0


def test_the_report_says_what_it_did(seeded):
    report = link(seeded, autocommit=True, commit=False)
    assert report["sittings"] == {"linked": 1, "formal_no_debate": 1, "unlinked": 1}
    assert report["division_methods"]["title_date"] == 1


# --- the commit contract ------------------------------------------------------------------------------------


def test_on_a_non_autocommit_connection_nothing_is_kept_until_the_caller_commits(seeded):
    """The trap the first relink script fell into. `link_all`'s reads open an implicit transaction, so its own
    transaction blocks are savepoints; the caller must commit.
    """
    link(seeded, autocommit=False, commit=False)
    assert read(seeded, "SELECT count(bill_stage_id) FROM division") == [(0,)], "uncommitted work must not persist"


def test_and_the_caller_committing_keeps_it(seeded):
    link(seeded, autocommit=False, commit=True)
    assert read(seeded, "SELECT count(bill_stage_id) FROM division") == [(1,)]


def test_on_an_autocommit_connection_every_block_commits_by_itself(seeded):
    """How `scripts/relink.py` now connects, so its writes do not depend on what the linker read first."""
    link(seeded, autocommit=True, commit=False)
    assert read(seeded, "SELECT count(bill_stage_id) FROM division") == [(1,)]
