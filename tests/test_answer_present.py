"""Present-tense questions are answered as of the record's last sitting, and say so."""

from __future__ import annotations

import json
from datetime import date

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import present as PR
from answer import resolve as RS
from answer import router as R
from app import safety
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76",
            recipe="r", prompt_version="p", model_id="m")
LAST = date(2026, 9, 17)


@pytest.mark.parametrize("question", [
    "Who is the Chancellor of the Exchequer?",
    "Who's the Leader of the House of Commons?",
    "Who is the current Home Secretary?",
    "Who holds the office of Chancellor of the Exchequer now?",
    "Who currently holds the office of Chancellor of the Exchequer?",
    "Which party is Diane Abbott in?",
    "What party does Diane Abbott belong to?",
    "Which party is Diane Abbott in at present?",
])
def test_present_tense_is_recognised(question):
    assert PR.asks_present(question, R.route(question).filters)


@pytest.mark.parametrize("question", [
    "Who was the Chancellor of the Exchequer?",  # undated past: several holders, asked back
    "Who held the office of Chancellor of the Exchequer?",
    "Which party was Diane Abbott in?",
    "Who is the Chancellor of the Exchequer on 4 March 2020?",  # a date: the question's own time wins
    "Who was the Chancellor of the Exchequer in 2020?",
    "Which party was Diane Abbott in on 4 March 2020?",
])
def test_past_or_dated_questions_are_not_present(question):
    assert not PR.asks_present(question, R.route(question).filters)


@pytest.mark.parametrize("question", [
    "Who is the Chancellor of the Exchequer?",
    "Who is the current Chancellor of the Exchequer?",
    "Who holds the office of Chancellor of the Exchequer now?",
    "Who's the Chancellor of the Exchequer?",
])
def test_present_office_questions_route_to_the_office_fact(question):
    decision = R.route(question)
    assert (decision.route, decision.category) == ("structured", "office_held_at_date"), decision.why


@pytest.mark.parametrize("question, named", [
    ("Who is the current Chancellor of the Exchequer?", True),
    ("Who holds the office of Chancellor of the Exchequer now?", True),
    ("Who is the Chancellor of the Exchequer at present?", True),
    ("Who is the former Chancellor of the Exchequer?", False),  # "former" names another holder
])
def test_a_present_marker_does_not_qualify_the_office_but_former_does(question, named):
    assert RS._office_named_in_full(question, "Chancellor of the Exchequer") is named


# --- end to end on a scratch record
# ------------------------------------------------------------------------------------


def _bio(party, *, house=1, party_from="2015-05-07", seat_from="2015-05-07"):
    seat = {"house": house, "startDate": seat_from, "endDate": None}
    return {"biography": {"representations": [seat] if house == 1 else [], "houseMemberships": [seat],
                          "partyAffiliations": [{"name": party, "house": house, "startDate": party_from,
                                                 "endDate": None}]}}


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE coverage, member_office CASCADE")
    for window_id, start, end, last in (("WA", "2020-03-01", "2020-05-31", "2020-05-21"),
                                        ("W0", "2026-09-01", "2026-09-17", "2026-09-17")):
        scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                        "VALUES (%s, 'v2', %s, %s, 'dev')", (window_id, start, end))
        for house, section, last_day in (("Commons", "Commons Chamber", "2026-09-15" if window_id == "W0" else last),
                                         ("Lords", "Lords Chamber", last)):
            scratch.execute("INSERT INTO coverage (window_id, house, section, role, declared_start, declared_end, "
                            "first_sitting, last_sitting, sitting_days) VALUES (%s, %s, %s, 'dev', %s, %s, %s, %s, 9)",
                            (window_id, house, section, start, end, start, last_day))
    members = ((5001, "Ann Holder", _bio("Labour")), (5002, "Ben Earlier", _bio("Labour")),
               (5003, "Cat Member", _bio("Labour")), (5004, "Dan Switcher", _bio("Reform UK", party_from="2026-09-16")))
    for mnis_id, name, raw in members:
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house, raw) VALUES (%s, %s, 'Commons', %s)",
                        (mnis_id, name, json.dumps(raw)))
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5002, 'Chancellor of the Exchequer', 'chancellor of the exchequer', '2019-07-24', "
                    "'2026-07-05', 'mnis_biography', 0), (5001, 'Chancellor of the Exchequer', "
                    "'chancellor of the exchequer', '2026-07-05', NULL, 'mnis_biography', 0)")
    # the record's last Commons division: 15 September; both members listed as Labour
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                    "ayes_listed, noes_listed) VALUES (91, 91, 'Commons', 1, '2026-09-15', 'Housing', 2, 0, 'agreed', "
                    "2, 0)")
    for mnis_id, name in ((5003, "Cat Member"), (5004, "Dan Switcher")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                        "VALUES (91, %s, 'aye', %s, 'Labour')", (mnis_id, name))
    yield scratch_dsn
    scratch.execute("TRUNCATE coverage, member_office CASCADE")


def ask(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()
    return run(go())


def test_who_is_the_holder_is_answered_as_of_the_records_last_sitting(record):
    answer, notes = ask(record, "Who is the Chancellor of the Exchequer?")
    assert answer is not None and answer.abstention is None, notes
    fact = answer.structured_facts[0].value
    assert fact["mnis_id"] == 5001 and fact["date"] == LAST.isoformat()
    assert answer.rendered.startswith("As of the record up to 17 September 2026"), answer.rendered
    assert answer.facets.get("as_of") == LAST.isoformat()
    assert "cannot say what has changed since" in answer.coverage_note


def test_who_was_the_holder_with_no_date_is_not_answered_as_of_now(record):
    answer, notes = ask(record, "Who was the Chancellor of the Exchequer?")
    assert answer is None or not answer.rendered.startswith("As of the record"), notes


def test_party_now_at_the_records_edge_needs_the_history_and_the_latest_list_to_agree(record):
    answer, notes = ask(record, "Which party is Cat Member in?")
    assert answer is not None and answer.abstention is None, notes
    fact = answer.structured_facts[0].value
    assert (fact["mnis_id"], fact["party"]) == (5003, "Labour")
    assert answer.rendered.startswith("As of the record up to 17 September 2026")
    assert "latest division list" in fact["basis"] and "Members API" in fact["basis"]


def test_party_now_is_not_stated_when_the_history_has_moved_since_the_latest_list(record):
    """Dan Switcher was listed Labour on 15 September; the history gives Reform UK from the 16th. Neither is stated as
    his party on the 17th: both are shown with their sources and dates, and the reader is asked which they mean.
    """
    answer, notes = ask(record, "Which party is Dan Switcher in?")
    assert answer is not None, notes
    assert not answer.structured_facts, "no party asserted"
    offered = {c.get("party") for c in answer.facets.get("clarify") or []}
    assert offered == {"Labour", "Reform UK"}, answer.facets
    note = answer.coverage_note
    assert "15 September 2026" in note and "Members API" in note and "Reform UK" in note and "Labour" in note


def test_a_commons_office_is_asked_as_of_the_records_last_sitting_not_the_commons_own(record, scratch):
    """"House of Commons" inside an office's name is not a House filter, and the record's present is one day: the
    Commons last sat on the 15th, the record runs to the 17th.
    """
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5003, 'Leader of the House of Commons', 'leader of the house of commons', '2024-07-05', "
                    "NULL, 'mnis_biography', 0)")
    answer, notes = ask(record, "Who is the Leader of the House of Commons?")
    assert answer is not None and answer.abstention is None, notes
    assert answer.structured_facts[0].value["date"] == LAST.isoformat()
    assert answer.rendered.startswith("As of the record up to 17 September 2026")


def test_the_edge_rule_is_only_the_edge_a_gap_inside_the_record_keeps_the_bracket_rule(record, scratch):
    """A day with lists after it is a gap, where the latest list alone says nothing of a change in between."""
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                    "ayes_listed, noes_listed) VALUES (92, 92, 'Commons', 1, '2026-09-17', 'Later', 1, 0, 'agreed', "
                    "1, 0)")
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                    "VALUES (92, 5004, 'aye', 'Dan Switcher', 'Reform UK')")
    answer, notes = ask(record, "Which party was Cat Member in on 16 September 2026?")
    facts = answer.structured_facts if answer is not None else ()
    assert not facts, f"16 September has a list after it (none naming her): the bracket rule applies; {notes}"


def test_a_division_question_in_the_present_tense_is_not_stamped(record):
    answer, _ = ask(record, "What is the result of the Commons division on Housing?")
    assert answer is None or not answer.rendered.startswith("As of the record")


def test_a_present_question_with_two_holders_is_asked_back_with_the_stamp_and_both_names(record, scratch):
    """Review pass 1 (S06): a clarification's text is built at the exit only when empty; a stamp alone made it non-
    empty, so the reader saw "As of the record up to …:" and nothing more.
    """
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5001, 'Joint Minister for Housing', 'joint minister for housing', '2025-01-01', NULL, "
                    "'mnis_biography', 0), (5003, 'Joint Minister for Housing', 'joint minister for housing', "
                    "'2025-06-01', NULL, 'mnis_biography', 0)")
    answer, notes = ask(record, "Who is the Joint Minister for Housing?")
    assert answer is not None and answer.facets.get("clarify"), notes
    shown = PL.reader_ready(answer).rendered
    assert shown.startswith("As of the record up to 17 September 2026")
    assert "Ann Holder" in shown and "Cat Member" in shown, shown
