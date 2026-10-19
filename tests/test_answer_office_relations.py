"""An office qualified by a party or a relation is answered from the record, never with the office's own holder."""

from __future__ import annotations

import json

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import router as R
from app import safety
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76",
            recipe="r", prompt_version="p", model_id="m")


def _bio(party):
    seat = {"house": 1, "startDate": "2010-05-06", "endDate": None}
    return json.dumps({"biography": {"representations": [seat], "houseMemberships": [seat],
                                     "partyAffiliations": [{"name": party, "house": 1, "startDate": "2010-05-06",
                                                            "endDate": None}]}})


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE member_office CASCADE")
    for mnis_id, name, party in ((5101, "Amy Before", "Conservative"), (5102, "Bob After", "Conservative"),
                                 (5103, "Cal Second", "Conservative"), (5104, "Dee Unlisted", "Labour")):
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house, raw) VALUES (%s, %s, 'Commons', %s)",
                        (mnis_id, name, _bio(party)))
    offices = (
        (5101, "Chancellor of the Exchequer", "2015-05-08", "2020-02-13", "mnis_biography"),
        (5102, "Chancellor of the Exchequer", "2020-02-13", None, "mnis_biography"),
        (5102, "Prime Minister", "2019-07-24", None, "mnis_biography"),
        (5103, "Deputy Prime Minister", "2019-07-24", None, "mnis_biography"),
        # an inferred period is never a recorded predecessor
        (5104, "Paymaster General", "2018-01-01", "2019-12-31", "hansard_attribution"),
        (5103, "Paymaster General", "2020-01-01", None, "mnis_biography"),
        (5104, "Lord Privy Seal", "2019-01-01", None, "mnis_biography"),
    )
    for mnis_id, office, start, end, source in offices:
        scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, "
                        "evidence) VALUES (%s, %s, lower(%s), %s, %s, %s, 0)", (mnis_id, office, office, start, end, source))
    # the lists that give their parties on 4 March 2020
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                    "ayes_listed, noes_listed) VALUES (95, 95, 'Commons', 1, '2020-03-04', 'Budget', 3, 0, 'agreed', "
                    "3, 0)")
    for mnis_id, name, party, abbreviation in ((5102, "Bob After", "Conservative", "Con"),
                                               (5103, "Cal Second", "Conservative", "Con")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party, party_abbreviation) "
                        "VALUES (95, %s, 'aye', %s, %s, %s)", (mnis_id, name, party, abbreviation))
    # parties the record knows, from other members' lists
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (5201, 'G One', 'Commons'), "
                    "(5202, 'S Two', 'Commons'), (5203, 'L Three', 'Commons')")
    for mnis_id, party, abbreviation in ((5201, "Green Party", "Green"), (5202, "Scottish National Party", "SNP"),
                                         (5203, "Liberal Democrat", "LD")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party, party_abbreviation) "
                        "VALUES (95, %s, 'no', 'Someone', %s, %s)", (mnis_id, party, abbreviation))
    yield scratch_dsn
    scratch.execute("TRUNCATE member_office CASCADE")


def ask(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()
    return run(go())


def holders(answer):
    return [sf.value.get("mnis_id") for sf in answer.structured_facts if sf.value]


# --- a party
# -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "Who was the Green Party Chancellor of the Exchequer on 4 March 2020?",
    "Who was the Chancellor of the Exchequer in the SNP on 4 March 2020?",
    "Who was the Chancellor of the Exchequer on 4 March 2020 for the Liberal Democrats?",
    "Who was the Scottish National Party Chancellor of the Exchequer on 4 March 2020?",
])
def test_an_office_held_for_another_party_is_a_corrected_premise_never_the_holder_as_the_answer(record, question):
    answer, notes = ask(record, question)
    assert answer is not None, notes
    assert answer.abstention == K.Abstention.FALSE_PREMISE and not answer.structured_facts
    assert "Bob After" in answer.abstention_detail and "Conservative" in answer.abstention_detail, \
        "the correction says who held it, and for which party"


def test_an_office_held_for_the_party_named_is_answered(record):
    answer, notes = ask(record, "Who was the Conservative Chancellor of the Exchequer on 4 March 2020?")
    assert answer is not None and answer.abstention is None, notes
    assert holders(answer) == [5102]


def test_a_holder_whose_party_the_record_does_not_settle_is_neither_confirmed_nor_denied(record):
    """Dee Unlisted is in no list that day: whether she is the Green Party one is not in the record."""
    answer, notes = ask(record, "Who was the Green Party Lord Privy Seal on 4 March 2020?")
    assert answer is not None and not answer.structured_facts, notes
    assert answer.abstention == K.Abstention.INSUFFICIENT_EVIDENCE


# --- a relation
# ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "Who was the predecessor of the Chancellor of the Exchequer on 4 March 2020?",
    "Who was the Chancellor of the Exchequer's predecessor on 4 March 2020?",
])
def test_the_predecessor_is_the_recorded_holder_before_the_holder_that_day(record, question):
    answer, notes = ask(record, question)
    assert answer is not None and answer.abstention is None, notes
    assert holders(answer) == [5101]
    assert "Amy Before" in answer.rendered and "Bob After" in answer.rendered


def test_the_successor_is_the_recorded_holder_after_the_holder_that_day(record):
    answer, notes = ask(record, "Who was the successor of the Chancellor of the Exchequer on 1 June 2019?")
    assert answer is not None and answer.abstention is None, notes
    assert holders(answer) == [5102]


def test_no_recorded_predecessor_is_declined_without_claiming_there_was_none(record):
    answer, notes = ask(record, "Who was the predecessor of the Chancellor of the Exchequer on 1 June 2019?")
    assert answer is not None and not answer.structured_facts, notes
    assert answer.abstention == K.Abstention.INSUFFICIENT_EVIDENCE, "not NOT_IN_RECORD: that would be an absence claim"


def test_an_inferred_period_is_never_a_predecessor(record):
    answer, notes = ask(record, "Who was the predecessor of the Paymaster General on 4 March 2020?")
    assert answer is not None and 5104 not in holders(answer), notes


def test_a_deputy_is_an_office_of_its_own_offered_when_the_record_holds_it(record):
    answer, notes = ask(record, "Who was the Prime Minister's deputy on 4 March 2020?")
    assert answer is not None and not answer.structured_facts, notes
    offered = [c.get("office") for c in answer.facets.get("clarify") or []]
    assert offered == ["Deputy Prime Minister"]


def test_a_deputy_the_record_holds_no_office_for_is_declined_without_an_absence_claim(record):
    answer, notes = ask(record, "Who was the Chancellor of the Exchequer's deputy on 4 March 2020?")
    assert answer is not None and not answer.structured_facts, notes
    assert answer.abstention == K.Abstention.INSUFFICIENT_EVIDENCE


# --- anything else is not read here
# ----------------------------------------------------------------------------------------


def test_an_unknown_qualifier_is_still_refused(record):
    answer, notes = ask(record, "Who was the disgraced Chancellor of the Exchequer on 4 March 2020?")
    assert answer is None or 5102 not in holders(answer), notes


def test_the_office_named_in_full_is_answered_as_before(record):
    answer, notes = ask(record, "Who held the office of Chancellor of the Exchequer on 4 March 2020?")
    assert holders(answer) == [5102], notes


# --- the router: an office is a title, whatever its head word (MR-07: 86 misroutes; MR-16: 20)
# ---------------------------


@pytest.mark.parametrize("question", [
    "Who was the Green Party Captain of the Honourable Corps of Gentlemen-at-Arms (HM Household) (Chief Whip, House of "
    "Lords) on 4 March 2020?",
    "Who was the Parliamentary Under-Secretary (Ministry of Defence) (Minister for Veterans and People)'s deputy on "
    "4 March 2020?",
    "Who was the predecessor of the National campaign co-ordinator on 4 March 2020?",
    "Who is the Leader of HM Official Opposition?",
    "Who was the President of the Board of Trade on 4 March 2020?",
    "Who is the Plaid Cymru Westminster Leader?",
    "Who was the Shadow Advocate-General for Scotland on 4 March 2020?",
    "who was the parliamentary under-secretary (department for international development) on 4 march 2020?",
    "Who was the Lord Privy Seal on 4 March 2020?",
])
def test_an_office_question_routes_to_the_office_fact_whatever_the_offices_head_word(question):
    decision = R.route(question)
    assert (decision.route, decision.category) == ("structured", "office_held_at_date"), decision.why


@pytest.mark.parametrize("question", [
    "Who was the first speaker in the debate on Flooding on 4 March 2020?",
    "What did the Prime Minister say about flooding on 4 March 2020?",
    "Who was the member for Hull West on 4 March 2020?",
    "who was the first member to speak on 4 March 2020?",
])
def test_questions_that_are_not_about_an_office_do_not_route_to_it(question):
    assert R.route(question).category != "office_held_at_date"


# --- the office strings as stored ---------------------------------------------------------


def test_an_office_stored_with_stray_whitespace_is_matched_like_any_other(record, scratch):
    """32 biography posts end in a space and 3 hold a double space: matched raw, "… House of Lords)'s deputy" could
    never name them, while "… House of Lords) on 4 March" did by luck.
    """
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5101, 'Captain of the  Guard (Deputy Chief Whip) ', 'captain of the guard (deputy chief "
                    "whip)', '2019-01-01', NULL, 'mnis_biography', 0)")
    answer, notes = ask(record, "Who held the office of Captain of the Guard (Deputy Chief Whip) on 4 March 2020?")
    assert holders(answer) == [5101], notes
    answer, notes = ask(record, "Who was the Captain of the Guard (Deputy Chief Whip)'s deputy on 4 March 2020?")
    assert answer is not None and not answer.structured_facts, notes
    assert answer.abstention == K.Abstention.INSUFFICIENT_EVIDENCE


def test_a_recorded_appointment_outranks_a_period_inferred_from_hansards_labels(record, scratch):
    """Housing, 13 October 2025: the biography records one Secretary of State; a period inferred from how Hansard
    printed another member ran on past it. The recorded appointment is the answer; the inference is a fallback.
    """
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5104, 'The Chancellor of the Exchequer', 'chancellor of the exchequer', '2019-01-01', "
                    "'2026-09-07', 'hansard_attribution', 40)")
    answer, notes = ask(record, "Who held the office of Chancellor of the Exchequer on 4 March 2020?")
    assert holders(answer) == [5102], notes


def test_an_inferred_period_still_answers_where_no_appointment_is_recorded(record, scratch):
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5104, 'The Lord Chancellor', 'lord chancellor', '2020-01-01', '2020-12-31', "
                    "'hansard_attribution', 12)")
    answer, notes = ask(record, "Who held the office of Lord Chancellor on 4 March 2020?")
    assert holders(answer) == [5104], notes


def test_an_office_is_shown_without_the_leading_the_hansard_prints(record, scratch):
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5104, 'The Lord Chancellor', 'lord chancellor', '2020-01-01', '2020-12-31', "
                    "'hansard_attribution', 12)")
    answer, _ = ask(record, "Who held the office of Lord Chancellor on 4 March 2020?")
    assert "held the office of Lord Chancellor" in answer.rendered and "The Lord" not in answer.rendered


def test_a_successor_whose_tenure_is_a_day_or_two_is_still_named(record, scratch):
    """A post held from 8 to 9 July (then renamed) has no day strictly inside it; its own start day is still one on
    which the record names that holder alone (S06's MR-07 run: the last successor trial).
    """
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5101, 'Shadow Paymaster', 'shadow paymaster', '2023-09-04', '2024-05-30', "
                    "'mnis_biography', 0), (5103, 'Shadow Paymaster', 'shadow paymaster', '2024-07-08', '2024-07-09', "
                    "'mnis_biography', 0)")
    answer, notes = ask(record, "Who was the successor of the Shadow Paymaster on 8 May 2024?")
    assert answer is not None and holders(answer) == [5103], notes


def test_the_party_labels_are_read_once_per_connection(record, monkeypatch):
    """S06's timing: re-reading every vote's party label cost 240 ms a question."""
    import psycopg

    from answer import office_relations as OR
    from answer import resolve as RS

    seen = []
    real = RS._all

    async def counting(conn, sql, params=()):
        if "DISTINCT party" in sql:
            seen.append(sql)
        return await real(conn, sql, params)

    monkeypatch.setattr(RS, "_all", counting)

    async def go():
        conn = await psycopg.AsyncConnection.connect(record)
        try:
            for q in ("Who was the Green Party Chancellor of the Exchequer on 4 March 2020?",
                      "Who was the Chancellor of the Exchequer in the SNP on 4 March 2020?"):
                await OR.qualified(conn, q)
        finally:
            await conn.close()
    run(go())
    assert len(seen) == 1


# --- a change of holder across the sealed window (QA pass 9, High 2)
# ---------------------------------------------------


@pytest.mark.parametrize("ended, began, question", [
    # clipped both sides by the mask: the outgoing post ends inside the window, the incoming begins inside it
    ("2026-05-05", "2026-05-06", "Who was the successor of the Paymaster General on 2 March 2026?"),
    # neither clipped, but the gap between them spans the window: a post wholly inside it could sit between
    ("2026-03-20", "2026-08-03", "Who was the successor of the Paymaster General on 2 March 2026?"),
    ("2026-03-20", "2026-08-03", "Who was the predecessor of the Paymaster General on 10 September 2026?"),
])
def test_a_change_of_holder_that_touches_the_sealed_window_is_not_asserted(record, scratch, ended, began, question):
    """Migration 027 drops a post wholly inside the sealed window, so the record cannot know whether someone held the
    office between two holders whose change touches it: "John Healey held the office … after Rachel Reeves" was
    asserted across it (QA pass 9).
    """
    scratch.execute("DELETE FROM member_office WHERE office_key = 'paymaster general'")
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5101, 'Paymaster General', 'paymaster general', '2025-01-01', %s, 'mnis_biography', 0), "
                    "(5102, 'Paymaster General', 'paymaster general', %s, NULL, 'mnis_biography', 0)", (ended, began))
    answer, notes = ask(record, question)
    assert answer is not None and not answer.structured_facts, notes
    assert answer.abstention == K.Abstention.INSUFFICIENT_EVIDENCE
    assert "not yet available" in answer.abstention_detail


def test_a_change_of_holder_clear_of_the_sealed_window_is_still_named(record, scratch):
    scratch.execute("DELETE FROM member_office WHERE office_key = 'paymaster general'")
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (5101, 'Paymaster General', 'paymaster general', '2025-01-01', '2026-02-10', "
                    "'mnis_biography', 0), (5102, 'Paymaster General', 'paymaster general', '2026-02-10', NULL, "
                    "'mnis_biography', 0)")
    answer, notes = ask(record, "Who was the successor of the Paymaster General on 2 January 2026?")
    assert answer is not None and holders(answer) == [5102], notes
