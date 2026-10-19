"""An office held as the first part of a joint title."""

from __future__ import annotations

import pytest

from answer import facts as F
from app import safety
from ingest.runtime import run


@pytest.fixture
def offices(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE member_office CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (7101, 'Pat Former', 'Commons'), "
                    "(7102, 'Sam Current', 'Commons'), (7103, 'Lee Deputy', 'Commons'), (7104, 'Kim Chair', 'Commons'), "
                    "(7105, 'Ash Speaker', 'Commons')")
    rows = (
        (7101, "Prime Minister", "2020-01-01", "2026-09-09", "hansard_attribution", 40),
        (7102, "Prime Minister and First Lord of the Treasury", "2026-08-01", None, "mnis_biography", 0),
        (7102, "Prime Minister", "2026-09-10", "2026-09-12", "hansard_attribution", 3),
        (7103, "Deputy Speaker", "2024-07-01", "2026-12-31", "hansard_attribution", 9),
        (7104, "Deputy Speaker and Chairman of Ways and Means", "2024-07-20", None, "mnis_biography", 0),
        (7105, "Speaker", "2019-11-04", None, "mnis_biography", 0),
        (7101, "Speaker and Returning Officer", "2020-01-01", None, "mnis_biography", 0),
    )
    for mnis_id, office, start, end, source, evidence in rows:
        scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                        "VALUES (%s, %s, lower(%s), %s, %s, %s, %s)", (mnis_id, office, office, start, end, source,
                                                                        evidence))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE member_office CASCADE")


def _held(dsn, key, day):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await F.office_held_at_date(conn, key, day)
        finally:
            await conn.close()
    return run(go())


def test_the_first_part_of_a_joint_title_answers_where_nothing_is_recorded_under_its_own_name(offices):
    fact = _held(offices, "prime minister", "2026-09-17")
    assert fact.ok and fact.value["mnis_id"] == 7102
    assert fact.value["office"] == "Prime Minister and First Lord of the Treasury", "answered in full"
    assert fact.caveat == "", "a recorded appointment: no inference caveat"


def test_a_joint_title_and_an_inferred_period_that_agree_answer_once(offices):
    fact = _held(offices, "prime minister", "2026-09-11")
    assert fact.ok and fact.value["mnis_id"] == 7102
    assert fact.value["office"] == "Prime Minister and First Lord of the Treasury", "the recorded row, first"


def test_a_joint_title_and_an_inferred_period_that_disagree_are_asked_back(offices):
    fact = _held(offices, "prime minister", "2026-08-15")
    assert fact.needs_clarification, "Pat Former's inferred period and Sam Current's recorded post disagree that day"
    assert {c["mnis_id"] for c in fact.clarify} == {7101, 7102}


def test_a_joint_title_never_outranks_the_inferred_holders_of_its_first_part(offices):
    fact = _held(offices, "deputy speaker", "2025-03-04")
    assert fact.needs_clarification and {c["mnis_id"] for c in fact.clarify} == {7103, 7104}


def test_an_office_recorded_under_its_own_name_ignores_the_joint_titles_beginning_with_it(offices):
    fact = _held(offices, "speaker", "2025-03-04")
    assert fact.ok and fact.value["mnis_id"] == 7105 and fact.value["office"] == "Speaker"
