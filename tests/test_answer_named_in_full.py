"""Everything a question says about an entity must be accounted for by the entity matched, or the route refuses."""

from __future__ import annotations

import pytest

from answer import resolve as RS

# A date is not a division number ------------------------------------------------------------------------


@pytest.mark.parametrize("question, number", [
    ("What were the Contents and Not-Contents in the Lords division 5 November 2025 on the duty to deport?", None),
    ("How did Lord Cameron of Lochiel vote in the Lords division 5th November 2025 on deportation?", None),
    ("What was the result of the Lords division 2 March 2026 on the Owen amendment?", None),
    ("What was the result of division 5 on 5 November 2025?", 5),
    ("How did Alan Mak vote in Division No. 45 on 13 May 2020?", 45),
    ("What was the result of the third division on 13 May 2020?", 3),
])  # fmt: skip
def test_a_day_of_the_month_is_never_read_as_a_division_number(question, number):
    assert RS.division_number(question) == number


# An office named in full ----------------------------------------------------------------------------------


@pytest.mark.parametrize("question, office", [
    ("Who was the Shadow Chancellor of the Exchequer on 1 November 2025?", "Shadow Chancellor of the Exchequer"),
    ("Who held the office of Chancellor of the Exchequer on 1 November 2025?", "The Chancellor of the Exchequer"),
    ("Who was Prime Minister on 15 January 2026?", "The Prime Minister"),
    ("On 1 November 2025, who was the Home Secretary?", "Home Secretary"),
    ("Which minister was Home Secretary on 10 March 2026?", "Home Secretary"),
    ("Who held the post of Lord Chamberlain (HM Household) on 1 September 2026?", "Lord Chamberlain (HM Household)"),
    ("On 1 September 2026, who was serving as Opposition Deputy Chief Whip (Lords)?", "Opposition Deputy Chief Whip (Lords)"),
    ("Who held the post of Party Chair, Labour Party on 1 May 2024?", "Party Chair, Labour Party"),
])  # fmt: skip
def test_an_office_the_question_names_in_full_is_named(question, office):
    assert RS._office_named_in_full(question, office)


@pytest.mark.parametrize("question, office", [
    ("Who was the Liberal Democrat Shadow Chancellor of the Exchequer on 1 November 2025?",
     "Shadow Chancellor of the Exchequer"),
    ("Who was the SNP Shadow Home Secretary on 1 November 2025?", "Shadow Home Secretary"),
    ("Who was the Green Party Shadow Home Secretary on 15 January 2026?", "Shadow Home Secretary"),
    ("Who was the Home Secretary's deputy on 1 November 2025?", "Home Secretary"),
    ("Who was the Prime Minister's Parliamentary Private Secretary on 15 January 2026?", "The Prime Minister"),
    ("Who was the predecessor of the Chancellor of the Exchequer on 1 November 2025?", "The Chancellor of the Exchequer"),
    ("Who was deputy to the Home Secretary on 1 November 2025?", "Home Secretary"),
    ("Who was the former Home Secretary on 1 November 2025?", "Home Secretary"),
    ("Who was Minister of State (Home Office) on 1 November 2025?", "Minister of State"),
    # found by the probes' qualified forms: the possessive ends a longer office the match sits inside
    ("Who was the Prime Minister, First Lord of the Treasury, Minister for the Civil Service, and Minister for the "
     "Union's deputy on 20 May 2020?", "The Prime Minister"),
    ("Who was the Minister of State (Home Office) (Security)'s deputy on 8 May 2024?", "Minister of State (Home Office)"),
    ("Who was the Speaker of the House of Commons on 1 November 2025?", "Speaker"),
])  # fmt: skip
def test_an_office_the_question_qualifies_is_not_that_office(question, office):
    assert not RS._office_named_in_full(question, office)


# A bill named in full -------------------------------------------------------------------------------------


@pytest.mark.parametrize("question, title, named", [
    ("When did the Social Housing Bill have its second reading?", "Housing Bill", False),
    ("When did the Social Housing Bill have its second reading?", "Social Housing Bill [HL]", True),
    ("When did the Housing Bill have its second reading?", "Housing Bill", True),
    ("When did the Menstrual and Gynaecological Health Bill have its first reading?", "Health Bill", False),
    ("When did the Menstrual and Gynaecological Health Bill have its first reading?",
     "Menstrual and Gynaecological Health Bill", True),
    ("When did the Health and Care Bill have its second reading?", "Care Bill", False),
    ("When did the Finance (No. 2) Bill have its third reading?", "Finance (No. 2) Bill", True),
    ("When did the Finance (No. 2) Bill have its third reading?", "Finance Bill", False),
    ("When did the Planning and Infrastructure Act 2025 have its Committee stage?", "Planning and Infrastructure Act 2025", True),
])  # fmt: skip
def test_a_bill_is_matched_only_where_the_question_names_it_whole(question, title, named):
    assert RS._bill_named_in_full(question, title) is named


# The stage asked, never another of its kind -----------------------------------------------------------


@pytest.mark.parametrize("question, stage, ok", [
    ("When did the X Bill have its second reading?", "2nd reading", True),
    ("When did the X Bill have its second reading?", "1st reading", False),
    ("When did the X Bill have its first reading?", "3rd reading", False),
    ("When did the X Bill have its report stage?", "Committee stage", False),
    ("When did the X Bill have its committee stage?", "Committee of the whole House", True),
    ("When did the X Bill have its committee stage?", "Committee stage", True),
    ("When did the X Bill have its Royal Assent?", "Royal Assent", True),
    ("When did the X Bill have its second reading?", "Second reading committee", True),
])  # fmt: skip
def test_a_stage_of_another_ordinal_or_kind_scores_nothing(question, stage, ok):
    assert (RS._stage_score(RS._stage_terms(question), stage) > 0) is ok


# A possessive or a relation names someone else -------------------------------------------------------


@pytest.mark.parametrize("text", [
    "How did Keir Starmer's deputy vote on the Agriculture Bill",
    "Did Rishi Sunak's PPS vote for the Agriculture Bill",
    "Which party was Keir Starmer's deputy in on 13 May 2020",
    "Which way did the MP who succeeded Theresa May vote",
    "How did the successor to Theresa May vote",
    "How did the deputy of Keir Starmer vote",
])  # fmt: skip
def test_a_member_named_only_to_describe_someone_else_is_not_matched(text):
    forms = ("Keir Starmer", "Rishi Sunak", "Theresa May")
    assert RS._name_span(text, forms) is None


@pytest.mark.parametrize("text", [
    "What was Keir Starmer's vote on the Agriculture Bill",
    "What was Keir Starmer's party on 13 May 2020",
    "How did Keir Starmer vote on the Agriculture Bill",
    "Neil O'Brien",
])  # fmt: skip
def test_a_possessive_of_the_attribute_asked_still_names_the_member(text):
    assert RS._name_span(text, ("Keir Starmer", "Neil O'Brien")) is not None


ROWS = [{"mnis_id": 1, "printed": "Keir Starmer", "name": "Keir Starmer"},
        {"mnis_id": 2, "printed": "Alan Mak", "name": "Mr Alan Mak", "name_bare": "Alan Mak"},
        {"mnis_id": 3, "printed": "Lord McConnell of Glenscorrodale", "name": "Lord McConnell of Glenscorrodale"}]


@pytest.mark.parametrize("subject, found", [
    ("Keir Starmer", 1),
    ("Mr Alan Mak MP", 2),
    ("the Rt Hon. Alan Mak", 2),
    ("The Rt Hon. the Lord McConnell of Glenscorrodale", 3),
    ("Conservative MP Alan Mak", 2),
    ("the MP who succeeded Alan Mak", None),
    ("Alan Mak's chief whip", None),
    ("the shadow chancellor under Keir Starmer", None),
])  # fmt: skip
def test_a_vote_subject_must_be_the_member_and_address_words_alone(subject, found):
    assert RS._pick_listed(ROWS, subject, whole=True)[0] == found


# Dashes and curly apostrophes are the same characters -----------------------------------------------


@pytest.mark.parametrize("dash", ["‐", "‑", "‒", "–", "—", "―", "−"])
def test_any_dash_continues_a_name(dash):
    assert RS._name_span(f"Which lobby did Lord Booth{dash}Smith go into", ("Lord Booth",)) is None
    assert RS._name_span(f"Which lobby did Lord Booth{dash}Smith go into", ("Lord Booth-Smith",)) is not None


def test_a_curly_apostrophe_matches_a_straight_one():
    assert RS._name_span("How did Neil O’Brien vote", ("Neil O'Brien",)) is not None


# --- L5: the pronoun "I" ---------------------------------------------------------------------------------------


def test_the_pronoun_i_after_amendment_is_no_reference():
    assert RS._asked_marks("Was the amendment I asked about on the Agriculture Bill on 13 May 2020 carried?") == {}
    assert RS._asked_marks("Was the amendment I mentioned carried in the Commons?") == {}
    assert RS._asked_marks("Was Amendment A carried on 13 May 2020?") == {"amendment": {"A"}}
    assert RS._asked_marks("Was Amendment I carried?") == {"amendment": {"I"}}
