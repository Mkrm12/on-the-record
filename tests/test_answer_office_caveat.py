"""Which office periods are inferred."""

from __future__ import annotations

import psycopg

from answer import facts as F
from ingest.runtime import run as run_async


def test_only_a_period_inferred_from_hansard_says_it_is_inferred(scratch, scratch_dsn):
    scratch.execute("TRUNCATE member_office, member_id_map CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (1, 'Ms Recorded'), (2, 'Mr Inferred')")
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (1, 'Home Secretary', 'home secretary', '2025-09-05', NULL, 'mnis_biography', 0), "
                    "(2, 'The Speaker', 'speaker', '2020-01-01', NULL, 'hansard_attribution', 12)")  # fmt: skip

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            return (await F.office_held_at_date(conn, "home secretary", "2026-01-15"),
                    await F.office_held_at_date(conn, "speaker", "2026-01-15"))

    recorded, inferred = run_async(go())
    assert recorded.ok and recorded.caveat == ""
    assert inferred.ok and "inferred" in inferred.caveat and "Hansard" in inferred.caveat
    scratch.execute("TRUNCATE member_office, member_id_map CASCADE")


def test_a_recorded_appointment_is_preferred_to_an_inferred_period_for_the_same_holder(scratch, scratch_dsn):
    """The stronger evidence first: when both sources name the same holder, the answer rests on the recorded post.
    """
    scratch.execute("TRUNCATE member_office, member_id_map CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (3, 'Ms Both')")
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (3, 'The Chancellor', 'chancellor', '2024-07-05', NULL, 'hansard_attribution', 40), "
                    "(3, 'Chancellor', 'chancellor', '2024-07-05', NULL, 'mnis_biography', 0)")  # fmt: skip

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            return await F.office_held_at_date(conn, "chancellor", "2026-01-15")

    fact = run_async(go())
    assert fact.ok and fact.caveat == "", "the recorded appointment, not the inferred period"
    scratch.execute("TRUNCATE member_office, member_id_map CASCADE")
