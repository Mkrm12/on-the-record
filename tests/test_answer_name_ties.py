"""Two members who share a name are told apart by who held a seat on the question's date."""

from __future__ import annotations

import json
from datetime import date

import pytest

from answer import resolve as RS
from app import safety
from ingest.runtime import run


def _bio(start, end):
    seat = {"house": 2, "startDate": start, "endDate": end}
    return json.dumps({"biography": {"representations": [], "houseMemberships": [seat], "partyAffiliations": []}})


@pytest.fixture
def bishops(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    rows = ((6001, "2011-02-01", "2021-11-30"),  # the earlier bishop
            (6002, "2023-01-10", None))  # the later one, same name
    for mnis_id, start, end in rows:
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house, raw) VALUES (%s, 'The Lord Bishop of "
                        "Newcastle', 'Lords', %s)", (mnis_id, _bio(start, end)))
    return scratch_dsn


def resolve(dsn, question, **kw):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await RS.mnis_id(conn, question, **kw)
        finally:
            await conn.close()
    return run(go())


QUESTION = "Which party was The Lord Bishop of Newcastle in?"


@pytest.mark.parametrize("on, want", [(date(2020, 3, 4), 6001), (date(2026, 9, 17), 6002)])
def test_a_shared_name_is_the_member_seated_on_the_date(bishops, on, want):
    found, why = resolve(bishops, QUESTION, on_date=on)
    assert found == want, why
    assert "seated" in why


def test_with_no_date_a_shared_name_is_still_refused(bishops):
    found, why = resolve(bishops, QUESTION)
    assert found is None and "2 members match that name" in why


def test_a_date_when_neither_sat_is_still_refused(bishops):
    found, _ = resolve(bishops, QUESTION, on_date=date(2022, 6, 1))
    assert found is None


def test_a_date_when_both_sat_is_still_refused(bishops, scratch):
    scratch.execute("UPDATE member_id_map SET raw = %s WHERE mnis_id = 6001", (_bio("2011-02-01", None),))
    found, _ = resolve(bishops, QUESTION, on_date=date(2024, 6, 1))
    assert found is None


def test_the_divisions_named_supply_the_date_when_the_question_does_not(bishops, scratch):
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                    "ayes_listed, noes_listed) VALUES (1000088, 88, 'Lords', 1, '2026-09-10', 'A Bill', 1, 0, 'agreed', "
                    "1, 0)")
    found, why = resolve(bishops, "How did The Lord Bishop of Newcastle vote on A Bill?", division_ids=(1_000_088,))
    assert found == 6002, why
