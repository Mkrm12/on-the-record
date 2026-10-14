"""A different division, or a different member, must never be answered as the one asked (QA pass 3, findings 1 and 2).
"""

from __future__ import annotations

import pytest

from answer import resolve as RS

AGRI_3R = "Agriculture Bill: Third Reading"
AGRI_NC2 = "Agriculture Bill: Report Stage: New Clause 2 (Neil Parish)"


@pytest.mark.parametrize("question, title", [
    ("What was the result of the Lords division on the Rwanda Bill on 23 May 2024?",
     "Post Office (Horizon System) Offences Bill"),
    ("How did Alan Mak vote on the Fisheries Bill: Third Reading on 13 May 2020?", AGRI_3R),
    ("What was the result of the Commons division on Finance (No. 2) Bill: Second Reading on 23 May 2024?",
     "Finance (No. 2) Bill: Third Reading"),
    ("How did Alan Mak vote on Agriculture Bill: Third Reading on 13 May 2020?", AGRI_NC2),
    ("Was the Agriculture Bill's report stage new clause carried on 13 May 2020?", AGRI_3R),
])  # fmt: skip
def test_a_contradicted_bill_or_stage_disqualifies_the_division(question, title):
    assert RS.title_score(question, title) == 0.0
    assert RS.contradicts(question, {"title": title, "motion_text": ""})


@pytest.mark.parametrize("question, title", [
    ("How did Alan Mak vote on Agriculture Bill: Third Reading on 13 May 2020?", AGRI_3R),
    ("What was the result of the vote on the immigration bill on 18 May 2020?",
     "Immigration and Social Security Co-ordination (EU Withdrawal) Bill: Third Reading"),
    ("What was the result of the vote on the Rwanda bill?", "Safety of Rwanda (Asylum and Immigration) Bill: 2nd reading"),
    ("Was the Finance Bill's second reading carried?", "Finance (No. 2) Bill: 2nd Reading"),
    ("How did Bill Cash vote on the Agriculture Bill: Third Reading?", AGRI_3R),
    ("How did Sir Bill Cash vote on the Agriculture Bill: Third Reading?", AGRI_3R),
    ("What was the result of the Lords division on Lords Amendment 23 to the Employment Rights Bill?",
     "Employment Rights Bill"),
])  # fmt: skip
def test_a_matching_or_silent_question_is_no_contradiction(question, title):
    assert not RS.contradicts(question, {"title": title, "motion_text": ""})


def test_a_bill_named_only_in_the_motion_is_not_contradicted():
    row = {"title": "Business of the House", "motion_text": "That the Safety of Rwanda (Asylum and Immigration) Bill be "
                                                            "proceeded with as follows"}  # fmt: skip
    assert not RS.contradicts("the vote on the Rwanda Bill on 23 April 2024", row)


@pytest.mark.parametrize("question, fits", [
    ("What was the result of the Commons division on 18 May 2020?", True),  # names no business: the date does
    ("What was the result of the vote on the immigration bill on 18 May 2020?", True),
    ("What was the result of the Commons vote on the smoking ban on 18 May 2020?", False),
    ("What was the result of the Commons vote on the Rwanda Bill on 18 May 2020?", False),
    ("How did Alan Mak vote on 18 May 2020?", True),
])  # fmt: skip
def test_the_only_division_that_day_is_taken_only_when_the_question_fits_it(question, fits):
    row = {"title": "Immigration and Social Security Co-ordination (EU Withdrawal) Bill: Third Reading",
           "motion_text": ""}  # fmt: skip
    assert RS.fits_only_division(question, row) is fits


@pytest.mark.parametrize("question, subject", [
    ("Did Neil Parish's new clause pass on 13 May 2020, and how did Alan Mak vote on " + AGRI_NC2 + "?", "Alan Mak"),
    ("Did Boris Johnson's government win, and how did Alan Mak vote on " + AGRI_3R + " on 13 May 2020?", "Alan Mak"),
    ("Which way did Alan Mak vote on " + AGRI_3R + "?", "Alan Mak"),
    ("Was Alan Mak an Aye or a No on " + AGRI_3R + "?", "Alan Mak"),
    ("Did Alan Mak vote for the " + AGRI_3R + "?", "Alan Mak"),
    ("Did Alan Mak back the Agriculture Bill?", "Alan Mak"),
    ("Which lobby did Alan Mak go into on 13 May 2020?", "Alan Mak"),
])  # fmt: skip
def test_the_voter_is_the_subject_of_the_vote_verb(question, subject):
    assert RS.vote_subject(question) == subject


def listed(*names):
    return [{"mnis_id": i, "printed": n, "name": n} for i, n in enumerate(names, start=1)]


def test_a_subject_naming_two_members_is_refused():
    found, why = RS._pick_listed(listed("Keir Starmer", "Alan Mak"), "Keir Starmer and Alan Mak")
    assert found is None and "more than one member" in why


def test_a_name_inside_a_longer_name_is_one_member():
    found, why = RS._pick_listed(listed("Ian Paisley", "Ian Paisley Jnr"), "Ian Paisley Jnr")
    assert found == 2, why


def test_two_members_of_the_record_named_in_the_subject_are_refused():
    rows = [{"mnis_id": 1, "name": "Keir Starmer"}, {"mnis_id": 2, "name": "Alan Mak"}]
    assert RS._distinct_people("Keir Starmer and Alan Mak", rows) == {1, 2}
    rows = [{"mnis_id": 1, "name": "Ian Paisley"}, {"mnis_id": 2, "name": "Ian Paisley Jnr"}]
    assert RS._distinct_people("Ian Paisley Jnr", rows) == {2}


# A question that names only the bill, or a kind without its number -----------------------------------

DAY = [{"id": 42, "title": AGRI_NC2, "motion_text": ""},
       {"id": 43, "title": "Agriculture Bill: Report Stage: New Clause 7 (Luke Pollard)", "motion_text": ""},
       {"id": 44, "title": "Agriculture Bill: Report Stage: Amendment 39 (Deidre Brock)", "motion_text": ""},
       {"id": 45, "title": AGRI_3R, "motion_text": ""}]  # fmt: skip


def test_a_kind_named_without_its_number_keeps_the_divisions_of_that_kind():
    rows, why = RS._narrow("How did Alan Mak vote on the Agriculture Bill amendment on 13 May 2020?", DAY)
    assert [r["id"] for r in rows] == [44], why
    rows, _ = RS._narrow("How did Alan Mak vote on the Agriculture Bill new clause on 13 May 2020?", DAY)
    assert [r["id"] for r in rows] == [42, 43]


def test_a_kind_no_division_that_day_holds_leaves_nothing():
    rows, why = RS._narrow("How did Alan Mak vote on the Agriculture Bill schedule on 13 May 2020?", DAY)
    assert rows == [] and "schedule" in why


def test_a_contradicted_division_is_no_candidate():
    rows, _ = RS._narrow("How did Alan Mak vote on Agriculture Bill: Third Reading on 13 May 2020?", DAY)
    assert [r["id"] for r in rows] == [45]


def test_a_question_naming_only_the_bill_does_not_choose_between_its_divisions():
    """The Third Reading's short title scored 0.5 and the new clauses' long ones less, so "the Agriculture Bill" was
    answered as its Third Reading: a guess from title length. Every division that day holds all the question names.
    """
    question = "How did Alan Mak vote on the Agriculture Bill on 13 May 2020?"
    assert {r["id"] for r in RS._equally_covered(question, DAY[3], DAY)} == {42, 43, 44}
    assert RS._equally_covered("How did Alan Mak vote on Agriculture Bill: Third Reading on 13 May 2020?", DAY[3],
                               DAY) == []  # fmt: skip


def test_a_numberless_kind_inside_a_name_is_not_a_kind_question():
    rows = [{"id": 1, "title": "Draft Medical Devices (Fees Amendment) Regulations 2026", "motion_text": ""},
            {"id": 2, "title": "Trade Bill: Second Reading", "motion_text": ""}]  # fmt: skip
    kept, _ = RS._narrow("Did the Commons agree to the Draft Medical Devices (Fees Amendment) Regulations 2026?", rows)
    assert [r["id"] for r in kept] == [1]


def test_a_division_named_in_full_is_chosen_over_one_whose_title_also_holds_its_words():
    """Named in full is named."""
    rows = [{"id": 2223, "title": "Finance (No. 2) Bill: Second Reading", "motion_text": ""},
            {"id": 2222, "title": "Finance (No. 2) Bill: Reasoned Amendment to Second Reading (Opposition)",
             "motion_text": ""}]  # fmt: skip
    question = "Give the voting figures for the Commons division on Finance (No. 2) Bill: Second Reading on 16 December 2025."
    assert RS._equally_covered(question, rows[0], rows) == []


# --- found by the rebuilt probes ---------------------------------------------------------------------


@pytest.mark.parametrize("question, title", [
    # the question names Third Reading; the division is the reasoned amendment, whose title names no stage
    ("What was the result of the Commons division on Northern Ireland Troubles Bill: Third Reading on 18 November 2025?",
     "Northern Ireland Troubles Bill: Opposition Reasoned Amendment"),
    ("What was the result of the Commons division on Universal Credit (Removal of Two Child Limit) Bill: Second Reading?",
     "Universal Credit (Removal of Two Child Limit) Bill Committee: New Clause 3"),
])  # fmt: skip
def test_a_stage_the_question_names_must_be_the_division_s_own(question, title):
    assert RS.contradicts(question, {"title": title, "motion_text": title})


def test_a_stage_the_lords_print_in_the_motion_is_shown():
    row = {"title": "Railways Bill", "motion_text": "Lord Hendy moved that the Bill be now read a second time."}
    assert not RS.contradicts("the Lords division on the Railways Bill: Second Reading", row)


@pytest.mark.parametrize("asked", ["Bus Services (No. 2) Bill", "Representation of the People Bill", "Criminal Justice Bill"])
def test_a_bill_name_is_compared_with_the_division_s_bill_not_its_motion_s_words(asked):
    """The Lords print a long motion; "services", "people" and "justice" turn up in it by chance."""
    row = {"title": "Public Authorities (Fraud, Error and Recovery) Bill",
           "motion_text": "Lord Vaux moved amendment 67, in clause 76, to insert the costs incurred by the services of "
                          "the people who administer criminal justice."}  # fmt: skip
    assert RS.contradicts(f"What was the result of the Lords division on {asked} on 21 October 2025?", row)


def test_a_division_on_no_bill_is_not_one_the_question_names_a_bill_for():
    assert RS.contradicts("the vote on the Railways Bill on 9 December 2025",
                          {"title": "UK-EU customs union (duty to negotiate): Ten Minute Rule Motion", "motion_text": ""})


def test_a_division_carries_a_question_only_if_it_carries_every_reference_named():
    rows = [{"title": "Planning and Infrastructure Bill: Government amendment (a) to Lords Amendment 1", "number": 1,
             "motion_text": ""}]  # fmt: skip
    assert RS._carrying("Government amendment (a) to Lords Amendment 900", rows) == []
    assert RS._carrying("Government amendment (a) to Lords Amendment 1", rows) == rows


def test_a_hyphenated_name_is_one_name():
    """"Lord Booth-Smith" was answered as Lord Booth: the hyphen read as the end of a word."""
    rows = [{"mnis_id": 5016, "printed": "Lord Booth", "name": "Lord Booth"}]
    assert RS._pick_listed(rows, "Which party was Lord Booth-Smith in on 10 March 2026?") == (None, None)
    assert RS._pick_listed(rows, "Which party was Lord Booth in on 10 March 2026?")[0] == 5016
    assert RS._pick_listed([{"mnis_id": 1, "printed": "Alan Mak", "name": "Alan Mak"}], "Was Alan Mak's vote Aye?")[0] == 1


def test_without_amendment_is_not_a_question_about_an_amendment():
    assert "amendment" not in RS._kinds_unnumbered("Did the Commons pass the Agriculture Bill without amendment?")
    assert "amendment" in RS._kinds_unnumbered("How did Alan Mak vote on the Agriculture Bill amendment?")


# --- found by MR-10, the typo probe ---------------------------------------------------------------------


@pytest.fixture
def peers(scratch, scratch_dsn):
    scratch.execute("TRUNCATE member_id_map CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (924, 'Lord Moynihan', 'Lords'), "
                    "(5007, 'Lord Moynihan of Chelsea', 'Lords'), (4970, 'Lord Weir of Ballyholme', 'Lords'), "
                    "(4145, 'Lord Weir', 'Lords')")  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE member_id_map CASCADE")


def member(dsn, text):
    import psycopg

    from ingest.runtime import run as run_async

    async def go():
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            return await RS._member_record(conn, text)

    return run_async(go())


@pytest.mark.parametrize("text", ["How did Lord Moynihan of Chlesea vote?", "How did Lord Weir of Balylholme vote?"])
def test_a_misspelt_designation_never_falls_back_to_a_shorter_peer_s_name(peers, text):
    """"Lord Moynihan of Chlesea" was answered for Lord Moynihan, another peer: the member record matched names as
    substrings, without the designation rule the division's own list already applied.
    """
    found, why = member(peers, text)
    assert found is None, why


def test_the_designation_rule_still_finds_the_peer_asked_about(peers):
    assert member(peers, "How did Lord Moynihan of Chelsea vote?")[0] == 5007
    assert member(peers, "How did Lord Moynihan vote?")[0] == 924
