"""The structured route end to end, without a model."""

from __future__ import annotations

from datetime import date

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import router as R
from app import safety
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76",
            recipe="r", prompt_version="p", model_id="m")
DAY = date(2026, 9, 9)
BILL = "Financial Services and Markets Bill [HL]"


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (3308, 'Viscount Chandos', 'Lords'), "
                    "(9, 'Lord Both', 'Lords'), (10, 'Lord Split', 'Lords'), (11, 'Lord Nobody', 'Lords')")
    for vid, number, ayes, noes in ((1_003_713, 1, 248, 163), (1_003_714, 2, 188, 125)):
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed) VALUES (%s, %s, 'Lords', %s, %s, %s, %s, %s, 'agreed', %s, %s)",
                        (vid, vid - 1_000_000, number, DAY, BILL, ayes, noes, ayes, noes))
    # Two same-titled divisions on another day whose RESULTS differ.
    for vid, number, ayes, noes, result in ((1_003_146, 1, 212, 144, "agreed"), (1_003_147, 2, 91, 192, "negatived")):
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed) VALUES (%s, %s, 'Lords', %s, '2024-05-21', 'Victims and Prisoners "
                        "Bill', %s, %s, %s, %s, %s)", (vid, vid - 1_000_000, number, ayes, noes, result, ayes, noes))
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (12, 'Lord Proxy', 'Lords')")
    votes = [(1_003_714, 3308, "no", "Viscount Chandos", None),
             (1_003_713, 9, "no", "Lord Both", None), (1_003_714, 9, "no", "Lord Both", None),
             (1_003_713, 10, "aye", "Lord Split", None), (1_003_714, 10, "no", "Lord Split", None),
             (1_003_146, 12, "aye", "Lord Proxy", "Some Whip")]
    for division_id, mnis_id, lobby, name, proxy in votes:
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party, proxy_name) "
                        "VALUES (%s, %s, %s, %s, 'Labour', %s)", (division_id, mnis_id, lobby, name, proxy))
    # A private bill's stage the Bills API gives no House for.
    scratch.execute("INSERT INTO bill (id, title) VALUES (3381, 'Royal Albert Hall Bill [HL]')")
    scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house, abbreviation) "
                    "VALUES (77, 3381, 'Commons Examiners', NULL, 'CE')")
    scratch.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, house, debate_link) "
                    "VALUES (78, 77, 3381, '2025-04-08', NULL, 'none')")
    return scratch_dsn


def ask(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()
    return run(go())


def vote_question(member):
    return f"Was {member} an Aye or a No on {BILL} on 9 September 2026?"


def test_a_member_who_voted_in_one_of_two_same_titled_divisions_is_answered(record):
    answer, notes = ask(record, vote_question("Viscount Chandos"))
    assert answer is not None and answer.abstention is None, notes
    assert "Viscount Chandos" in answer.rendered and "voted No" in answer.rendered
    assert "2 divisions on that day match this description; this member is recorded voting in 1 of them" \
        in answer.rendered, "the reader is told the question matched two divisions"
    assert answer.structured_facts[0].value["division_id"] == 1_003_714, "the division he voted in, not the first"


def test_a_member_who_voted_the_same_way_in_both_is_answered_once(record):
    answer, _ = ask(record, vote_question("Lord Both"))
    assert "voted No" in answer.rendered and "they give the same answer" in answer.rendered


def test_a_member_who_voted_differently_is_asked_which_division_by_name(record):
    """A clarification used to render as NOTHING — six dev answers the judge saw as empty. The divisions are NAMED so
    the reader can choose; their differing answers are not asserted.
    """
    answer, _ = ask(record, vote_question("Lord Split"))
    assert answer.abstention is None and answer.facets.get("clarify")
    assert {c["division_id"] for c in answer.facets["clarify"]} == {1_003_713, 1_003_714}
    assert "Say which division you mean: division 1" in answer.rendered and "division 2" in answer.rendered
    assert "voted" not in answer.rendered and not answer.structured_facts


def test_results_that_differ_are_asked_about_not_asserted(record):
    answer, _ = ask(record, "Was Victims and Prisoners Bill passed or rejected when the Lords divided on 21 May 2024?")
    assert answer.facets.get("clarify") and "Say which division you mean" in answer.rendered
    assert "212" not in answer.rendered and "rejected" not in answer.rendered and not answer.structured_facts


def test_a_proxy_vote_says_who_cast_it(record):
    answer, _ = ask(record, "Was Lord Proxy an Aye or a No on Victims and Prisoners Bill on 21 May 2024?")
    assert "voted Aye" in answer.rendered and "Cast by proxy by Some Whip." in answer.rendered


def test_a_member_recorded_in_neither_is_told_so_not_told_the_answers_differ(record):
    answer, _ = ask(record, vote_question("Lord Nobody"))
    assert answer.abstention == K.Abstention.NOT_IN_RECORD
    assert "not recorded in the division lists of any of the 2 divisions" in answer.abstention_detail
    assert "do not share the same answer" not in (answer.coverage_note or "")
    # named as resolved, so a name resolved to someone else shows
    assert answer.abstention_detail.startswith("Lord Nobody is not recorded"), answer.abstention_detail


def test_a_result_both_divisions_share_is_answered_with_every_tally(record):
    """The result is stated once, and no division's numbers stand for another's."""
    answer, _ = ask(record, f"On 9 September 2026, was the motion on {BILL} carried in the Lords?")
    assert answer.rendered.startswith("The Lords divided 2 times on 9 September 2026")
    assert "agreed to each time (division 1, 248 to 163; division 2, 188 to 125)" in answer.rendered
    assert len(answer.structured_facts) == 2


def test_tallies_that_differ_are_never_guessed_and_never_asserted(record):
    answer, _ = ask(record, f"What were the Ayes and Noes when the Lords divided on {BILL} on 9 September 2026?")
    assert answer.facets.get("clarify"), "which division is still asked"
    assert "248" not in answer.rendered and "division 2" in answer.rendered


def test_a_stage_with_no_recorded_house_answers_a_question_naming_one(record):
    answer, notes = ask(record, "What was the date of the Commons Examiners of the Royal Albert Hall Bill [HL]?")
    assert answer is not None and answer.abstention is None, notes
    assert "8 April 2025" in answer.rendered


def test_an_unresolvable_question_falls_back_rather_than_guessing(record):
    answer, notes = ask(record, vote_question("Lord Unknown"))
    assert answer is None and notes
