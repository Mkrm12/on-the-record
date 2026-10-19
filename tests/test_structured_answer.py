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


# Every value stated is the record's, and the probes check each one.


def test_a_member_who_voted_differently_has_each_vote_stated(record):
    answer, _ = ask(record, vote_question("Lord Split"))
    assert answer.abstention is None and not answer.facets.get("clarify"), "an answer, not a question back"
    assert {sf.value["division_id"]: sf.value["lobby"] for sf in answer.structured_facts} == {1_003_713: "aye",
                                                                                               1_003_714: "no"}
    table = answer.rendered
    assert "| 1 |" in table and "| 2 |" in table and "Content" in table and "Not Content" in table, table


def test_results_that_differ_are_each_stated_with_their_figures(record):
    answer, _ = ask(record, "What was the result of the Lords division on the Victims and Prisoners Bill on 21 May 2024?")
    assert not answer.facets.get("clarify")
    assert {sf.value["division_id"]: sf.value["result"] for sf in answer.structured_facts} == {1_003_146: "agreed",
                                                                                                1_003_147: "negatived"}
    for needed in ("| Contents | Not-Contents |", "| 1 |", "agreed to", "212", "144", "| 2 |", "rejected", "91", "192"):
        assert needed in answer.rendered, needed


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


def test_tallies_that_differ_are_never_guessed_each_is_stated_against_its_own_division(record):
    answer, _ = ask(record, f"What were the Ayes and Noes when the Lords divided on {BILL} on 9 September 2026?")
    assert not answer.facets.get("clarify")
    figures = {sf.value["division_id"]: (sf.value["ayes"], sf.value["noes"]) for sf in answer.structured_facts}
    assert figures == {1_003_713: (248, 163), 1_003_714: (188, 125)}
    rows = [line for line in answer.rendered.splitlines() if line.startswith("| 1 ") or line.startswith("| 2 ")]
    assert ["248" in rows[0] and "163" in rows[0], "188" in rows[1] and "125" in rows[1]] == [True, True], rows


# --- MR-06: a day's divisions asked about with no business named ----------------------------------------------


@pytest.fixture
def commons_day(record, scratch):
    """20 May 2020's shape: four Commons divisions, different business, results that differ."""
    for vid, number, title, ayes, noes, result in (
            (47, 47, "Business of the House (20 May)", 320, 220, "agreed"),
            (48, 48, "Trade Bill: Second Reading", 340, 251, "agreed"),
            (49, 49, "Trade Bill: Reasoned Amendment", 240, 342, "negatived"),
            (50, 50, "Agriculture Bill: Programme", 300, 10, "agreed")):
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed) VALUES (%s, %s, 'Commons', %s, '2020-05-20', %s, %s, %s, %s, %s, %s)",
                        (vid, vid, number, title, ayes, noes, result, ayes, noes))  # fmt: skip
    return record


@pytest.mark.parametrize("question", ["What was the result of the Commons division on 20 May 2020?",
                                      "What were the Ayes and Noes in the Commons division on 20 May 2020?",
                                      # the noun in the plural, with and without a House (QA pass 6)
                                      "What was the result of the divisions on 20 May 2020?",
                                      "What were the results of the Commons divisions on 20 May 2020?"])
def test_a_question_naming_no_business_gets_every_division_that_day(commons_day, question):
    answer, notes = ask(commons_day, question)
    assert answer is not None, f"fell through to text search: {notes}"
    assert {sf.value["division_id"] for sf in answer.structured_facts} == {47, 48, 49, 50}
    assert answer.rendered.startswith("The Commons divided 4 times on 20 May 2020:"), answer.rendered
    for number in (47, 48, 49, 50):
        assert f"| {number} |" in answer.rendered
    assert "| Ayes | Noes |" in answer.rendered


def test_business_named_that_no_division_that_day_holds_is_never_answered_with_the_days_divisions(commons_day):
    answer, notes = ask(commons_day, "What was the result of the Commons division on Fisheries on 20 May 2020?")
    assert answer is None or not answer.structured_facts, (answer.rendered if answer else notes)


def test_a_stage_with_no_recorded_house_answers_a_question_naming_one(record):
    answer, notes = ask(record, "What was the date of the Commons Examiners of the Royal Albert Hall Bill [HL]?")
    assert answer is not None and answer.abstention is None, notes
    assert "8 April 2025" in answer.rendered


def test_an_unresolvable_question_falls_back_rather_than_guessing(record):
    answer, notes = ask(record, vote_question("Lord Unknown"))
    assert answer is None and notes


def test_a_vote_table_counts_every_division_matched_not_only_those_the_member_is_listed_in(commons_day, scratch):
    """Review pass 1 (S05): the lead counted the table's rows, so a member missing from one of three lists read as
    "divided 2 times".
    """
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (77, 'Ann Member', 'Commons')")
    for division_id, lobby in ((47, "aye"), (48, "no")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                        "VALUES (%s, 77, %s, 'Ann Member', 'Labour')", (division_id, lobby))
    answer, notes = ask(commons_day, "How did Ann Member vote in the Commons division on 20 May 2020?")
    assert answer is not None, notes
    assert answer.rendered.startswith("The Commons divided 4 times on 20 May 2020:"), answer.rendered
    assert answer.rendered.count("is not recorded in the published list") == 2


def test_a_member_listed_in_2_of_11_divisions_reads_divided_11_times_not_2(record, scratch):
    """The lead is how often the House divided on the matching business, never how many rows the member fills: a day of
    11 divisions on which the member is listed twice read "divided 2 times" (review pass 1, S05).
    """
    for number in range(1, 12):
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed) VALUES (%s, %s, 'Commons', %s, '2025-03-11', %s, 300, 200, 'agreed', "
                        "300, 200)", (500 + number, 500 + number, number, f"Example Bill: New Clause {number}"))  # fmt: skip
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (78, 'Bea Member', 'Commons')")
    for division_id, lobby in ((503, "aye"), (509, "no")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                        "VALUES (%s, 78, %s, 'Bea Member', 'Labour')", (division_id, lobby))
    answer, notes = ask(record, "How did Bea Member vote in the Commons division on 11 March 2025?")
    assert answer is not None and answer.abstention is None, notes
    lead = answer.rendered.splitlines()[0]
    assert lead == "The Commons divided 11 times on 11 March 2025:", lead
    assert "2 times" not in lead
    rows = [line for line in answer.rendered.splitlines() if line.startswith("| ") and "New Clause" in line]
    assert len(rows) == 2 and "| 3 |" in rows[0] and "| 9 |" in rows[1], rows
    assert answer.rendered.count("Bea Member is not recorded in the published list") == 9
    assert answer.facets["divisions"] == list(range(501, 512)), "every matching division is offered, not the two rows"


# --- a vote question naming no House, on a day both Houses divided (QA pass 6, S05)
# ------------------------------------ A member sits in one House at a time, so the other House's divisions cannot be
# the ones asked about for them: a Commons MP's table carried "is not recorded in the published list for the Lords
# division …" for every Lords division that day.


@pytest.fixture
def both_houses(record, scratch):
    """9 March 2026: three Commons divisions and two Lords divisions; 10 March 2026: one of each."""
    import json

    for vid, house, number, day in ((601, "Commons", 101, "2026-03-09"), (602, "Commons", 102, "2026-03-09"),
                                    (603, "Commons", 103, "2026-03-09"), (1_000_611, "Lords", 1, "2026-03-09"),
                                    (1_000_612, "Lords", 2, "2026-03-09"), (604, "Commons", 104, "2026-03-10"),
                                    (1_000_613, "Lords", 3, "2026-03-10")):
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed) VALUES (%s, %s, %s, %s, %s, %s, 300, 200, 'agreed', 300, 200)",
                        (vid, vid - 1_000_000 if house == "Lords" else vid, house, number, day,
                         f"Example Bill: Amendment {number}"))  # fmt: skip
    seat = {"representations": [{"house": 1, "startDate": "2024-07-04T00:00:00", "endDate": None}],
            "houseMemberships": [{"house": 1, "startDate": "2024-07-04T00:00:00", "endDate": None}]}
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (79, 'Cy Member', 'Commons'), "
                    "(80, 'Lord Dee', 'Lords')")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house, raw) VALUES (81, 'Eve Member', 'Commons', %s)",
                    (json.dumps({"biography": seat}),))
    for division_id, mnis_id, lobby, name in ((601, 79, "aye", "Cy Member"), (602, 79, "no", "Cy Member"),
                                              (604, 79, "aye", "Cy Member"), (1_000_611, 80, "aye", "Lord Dee"),
                                              (1_000_612, 80, "no", "Lord Dee")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                        "VALUES (%s, %s, %s, %s, 'Labour')", (division_id, mnis_id, lobby, name))
    return record


def test_an_mps_vote_table_on_a_day_both_houses_divided_holds_only_the_commons(both_houses):
    answer, notes = ask(both_houses, "How did Cy Member vote in the division on 9 March 2026?")
    assert answer is not None and answer.abstention is None, notes
    assert answer.rendered.startswith("The Commons divided 3 times on 9 March 2026:"), answer.rendered
    assert "Lords" not in answer.rendered, answer.rendered
    assert answer.rendered.count("Cy Member is not recorded in the published list") == 1  # division 103 only
    assert answer.facets["divisions"] == [601, 602, 603]


def test_a_peers_vote_table_on_a_day_both_houses_divided_holds_only_the_lords(both_houses):
    answer, notes = ask(both_houses, "How did Lord Dee vote in the division on 9 March 2026?")
    assert answer is not None and answer.abstention is None, notes
    assert answer.rendered.startswith("The Lords divided 2 times on 9 March 2026:"), answer.rendered
    assert "Commons" not in answer.rendered and "not recorded in the published list" not in answer.rendered
    assert answer.facets["divisions"] == [1_000_611, 1_000_612]


def test_one_division_left_in_the_members_house_is_answered_as_that_division(both_houses):
    answer, notes = ask(both_houses, "How did Cy Member vote in the division on 10 March 2026?")
    assert answer is not None and answer.abstention is None, notes
    assert [sf.value["division_id"] for sf in answer.structured_facts] == [604]
    assert answer.structured_facts[0].value["house"] == "Commons" and "voted Aye" in answer.rendered, answer.rendered
    assert "Lords" not in answer.rendered and "divisions on that day match" not in answer.rendered, answer.rendered


def test_a_member_listed_nowhere_is_counted_against_the_divisions_of_the_house_they_sat_in(both_houses):
    answer, notes = ask(both_houses, "How did Eve Member vote in the division on 9 March 2026?")
    assert answer is not None and answer.abstention == K.Abstention.NOT_IN_RECORD, notes
    assert "any of the 3 Commons divisions" in answer.abstention_detail, answer.abstention_detail


def test_a_member_listed_in_one_of_their_houses_divisions_is_told_how_many_of_that_houses_matched(both_houses, scratch):
    """QA pass 7 (L6): after the other House's divisions are set aside, the count names the House it counts."""
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (82, 'Fay Member', 'Commons')")
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                    "VALUES (601, 82, 'aye', 'Fay Member', 'Labour')")
    answer, notes = ask(both_houses, "How did Fay Member vote in the division on 9 March 2026?")
    assert "3 Commons divisions on that day match this description; this member is recorded voting in 1 of them." \
        in answer.rendered, answer.rendered


def test_a_question_naming_the_other_house_is_still_answered_for_that_house(both_houses):
    """The House the question names is never overridden: an MP asked about the Lords divisions is told they are not in
    those lists, not answered from the Commons.
    """
    answer, notes = ask(both_houses, "How did Cy Member vote in the Lords division on 9 March 2026?")
    assert answer is not None, notes
    assert not any(sf.value.get("house") == "Commons" for sf in answer.structured_facts), answer.rendered


def test_the_table_lead_names_the_house_of_every_division_matched_not_only_of_the_rows():
    """The lead counts every matching division, so it names every matching division's House: rows from the Commons
    alone, of five divisions two of which were the Lords', read "The Commons divided 5 times" (QA pass 6, S05).
    """
    from answer import facts as F

    facts = [F.Fact("division_vote", {}, value={"division_id": i, "division_number": i, "house": "Commons",
                                                "member": "A Member", "lobby": "aye"}) for i in (1, 2)]  # fmt: skip
    meta = {i: (i, f"Division {i}") for i in (1, 2)}
    lead = F.summary_table("division_vote", facts, meta, house="Commons", date="2026-03-09", divided=5,
                           houses=("Commons", "Commons", "Commons", "Lords", "Lords")).splitlines()[0]
    assert lead.startswith("The two Houses divided 5 times on 9 March 2026"), lead
    alone = F.summary_table("division_vote", facts, meta, house="Commons", date="2026-03-09", divided=2,
                            houses=("Commons", "Commons")).splitlines()[0]
    assert alone.startswith("The Commons divided 2 times on 9 March 2026"), alone
