"""A bill stage's sittings are the days it sat."""

from __future__ import annotations

import psycopg

from answer import facts as F
from ingest.runtime import run as run_async


def test_a_stage_s_sittings_are_its_sitting_days_not_its_rows(scratch, scratch_dsn):
    scratch.execute("TRUNCATE bill, bill_stage, bill_stage_sitting CASCADE")
    scratch.execute("INSERT INTO bill (id, title) VALUES (7, 'Railways Bill')")
    scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house) VALUES (72, 7, 'Committee stage', 'Commons'), "
                    "(73, 7, 'Report stage', 'Commons')")  # fmt: skip
    for sid, stage, day in ((1, 72, "2026-01-13"), (2, 72, "2026-01-13"), (3, 72, "2026-01-15"), (4, 73, "2026-03-17"),
                            (5, 73, "2026-03-17")):
        scratch.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date) VALUES (%s, %s, 7, %s)",
                        (sid, stage, day))  # fmt: skip

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            return (await F.bill_stage_date(conn, 7, "Committee stage", "Commons"),
                    await F.bill_stage_date(conn, 7, "Report stage", "Commons"))

    committee, report = run_async(go())
    assert committee.value["sittings"] == 2 and committee.value["first_sitting"] == "2026-01-13"
    assert report.value["sittings"] == 1
    assert "sittings" not in report.sentence(), "one sitting day is not 'the first of 2 sittings'"
    scratch.execute("TRUNCATE bill, bill_stage, bill_stage_sitting CASCADE")


def test_a_sitting_after_the_record_s_own_as_of_date_is_not_a_stage_reached(scratch, scratch_dsn):
    """QA pass 4: the Bills API lists sittings scheduled ahead, and "reached 2nd reading on 15 October 2026" was said
    on the 14th. A sitting counts only on or before the bill's own last update (the API's as-of; the load time when
    it gives none).
    """
    scratch.execute("TRUNCATE bill, bill_stage, bill_stage_sitting CASCADE")
    scratch.execute("INSERT INTO bill (id, title, last_update) VALUES (8, 'Highways Bill', '2026-09-23T15:09:37Z'), "
                    "(9, 'Undated Bill', NULL)")  # fmt: skip
    scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house) VALUES (81, 8, '2nd reading', 'Commons'), "
                    "(82, 8, 'Committee stage', 'Commons'), (91, 9, '2nd reading', 'Commons')")  # fmt: skip
    for sid, stage, bill, day in ((1, 81, 8, "2026-10-15"), (2, 82, 8, "2026-09-15"), (3, 82, 8, "2026-10-20"),
                                  (4, 91, 9, "2999-01-01")):
        scratch.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date) VALUES (%s, %s, %s, %s)",
                        (sid, stage, bill, day))  # fmt: skip

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            return (await F.bill_stage_date(conn, 8, "2nd reading", "Commons"),
                    await F.bill_stage_date(conn, 8, "Committee stage", "Commons"),
                    await F.bill_stage_date(conn, 9, "2nd reading", "Commons"))

    second, committee, undated = run_async(go())
    assert not second.ok, "a sitting scheduled after the as-of date has not happened in the record"
    assert committee.ok and committee.value["sittings"] == 1 and "last_sitting" not in committee.value
    assert not undated.ok, "with no last update, the load time is the as-of"
    scratch.execute("TRUNCATE bill, bill_stage, bill_stage_sitting CASCADE")
