"""A member's party on a day they cast no vote."""

from __future__ import annotations

import json

import pytest

from answer import facts as F
from ingest.runtime import run as run_async


@pytest.fixture
def lists(scratch, scratch_dsn):
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (5424, 'Lord Docherty of Milngavie', "
                    "'Lords')")  # fmt: skip
    _member(scratch, 7, "Lord Steady", "Lords", seats=[(2, "2020-01-01", None)],
            parties=[("Crossbench", 2, "2020-01-01", None)])
    for did, day, party, steady in ((1, "2026-02-25", "Labour", "Crossbench"), (2, "2026-03-10", None, "Crossbench"),
                                    (3, "2026-09-08", "Non-affiliated", "Crossbench")):
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, ayes, noes) "
                        "VALUES (%s, %s, 'Lords', %s, 'x', 1, 1)", (did + 1_000_000, did, day))  # the Lords' id namespace
        did += 1_000_000
        if party:
            scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby, party) VALUES "
                            "(%s, 5424, 'Lord Docherty of Milngavie', 'aye', %s)", (did, party))  # fmt: skip
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby, party) VALUES "
                        "(%s, 7, 'Lord Steady', 'no', %s)", (did, steady))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")


def _member(scratch, mnis_id, name, house, *, seats, parties):
    """A member with the Members API's dated seats (Commons `representations`, Lords `houseMemberships`) and party
    history, as `member_id_map.raw->'biography'` stores them.
    """
    def when(d):
        return f"{d}T00:00:00" if d else None

    bio = {"representations": [{"house": 1, "name": "Somewhere", "startDate": when(a), "endDate": when(b)}
                               for h, a, b in seats if h == 1],
           "houseMemberships": [{"house": h, "name": "Commons" if h == 1 else "Lords", "startDate": when(a),
                                 "endDate": when(b)} for h, a, b in seats],
           "partyAffiliations": [{"name": n, "house": h, "startDate": when(a), "endDate": when(b)}
                                 for n, h, a, b in parties]}  # fmt: skip
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house, raw) VALUES (%s, %s, %s, %s::jsonb)",
                    (mnis_id, name, house, json.dumps({"biography": bio})))


def party(dsn, mnis_id, day):
    import psycopg

    async def go():
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            return await F.party_at_date(conn, mnis_id, day)

    return run_async(go())


def test_lists_that_disagree_either_side_of_the_day_state_no_party(lists):
    fact = party(lists, 5424, "2026-03-24")
    assert not fact.ok and "Labour" in fact.missing and "Non-affiliated" in fact.missing


def test_lists_that_agree_either_side_of_the_day_state_it_with_both_dates(lists):
    fact = party(lists, 7, "2026-03-24")
    assert fact.ok and fact.value["party"] == "Crossbench"
    assert "2026-03-10" in fact.value["basis"] and "2026-09-08" in fact.value["basis"]


def test_a_list_on_one_side_only_states_no_party(lists):
    fact = party(lists, 5424, "2026-10-01")  # after the last list that names him
    assert not fact.ok


def test_a_list_on_the_day_is_the_answer(lists):
    fact = party(lists, 5424, "2026-02-25")
    assert fact.ok and fact.value["party"] == "Labour"


@pytest.fixture
def gaps(scratch, scratch_dsn):
    """Two members the lists alone misreport (QA pass 4): one whose whip was removed between two lists that agree, and
    one who sat in neither House between leaving the Commons and joining the Lords.
    """
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")
    _member(scratch, 9, "Julian Gap", "Commons", seats=[(1, "2010-05-06", None)],
            parties=[("Conservative", 1, "1997-05-01", "2020-07-16"), ("Independent", 1, "2020-07-15", "2020-12-30"),
                     ("Conservative", 1, "2020-12-30", None)])  # fmt: skip
    _member(scratch, 10, "Lord Gap", "Lords", seats=[(1, "2010-05-06", "2024-05-30"), (2, "2024-09-10", None)],
            parties=[("Labour", 1, "2010-05-06", "2024-05-30"), ("Labour", 2, "2024-09-10", None)])  # fmt: skip
    for did, house, day, mid, party_ in ((1, "Commons", "2020-05-20", 9, "Conservative"),
                                         (2, "Commons", "2024-05-08", 9, "Conservative"),
                                         (3, "Commons", "2024-05-24", 10, "Labour"),
                                         (4, "Lords", "2025-10-13", 10, "Labour")):
        did = did + (1_000_000 if house == "Lords" else 0)
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, ayes, noes) "
                        "VALUES (%s, %s, %s, %s, 'x', 1, 1)", (did, did % 1_000_000, house, day))  # fmt: skip
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby, party) VALUES "
                        "(%s, %s, 'x', 'aye', %s)", (did, mid, party_))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")


def test_lists_that_agree_across_a_whip_removal_state_no_party(gaps):
    fact = party(gaps, 9, "2020-10-07")
    assert not fact.ok, "the history says Independent that day, whatever the lists either side say"


def test_a_day_the_member_sat_in_neither_house_states_no_party(gaps):
    fact = party(gaps, 10, "2024-08-15")
    assert not fact.ok and "seat" in fact.missing


def test_a_seat_a_history_and_both_lists_that_agree_state_the_party_once(gaps):
    fact = party(gaps, 10, "2025-01-15")
    assert fact.ok and fact.value["party"] == "Labour"
    sentence = fact.sentence()
    assert sentence.startswith("x sat as Labour on 15 January 2025.")
    assert "That is no division list" not in sentence, "the basis reads as English (QA pass 4, L2)"
    assert sentence.count("2024-05-24") <= 1 and sentence.count("2025-10-13") <= 1, "said once, not twice"


def test_a_boundary_day_of_a_party_is_not_settled(gaps):
    assert not party(gaps, 9, "2020-12-30").ok
