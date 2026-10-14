"""What a division was on, and what a question asks of it."""

from __future__ import annotations

import pytest

from answer import facts as F
from answer import resolve as RS

MURRAY = {"title": "Tobacco and Vapes Bill", "motion_text": "Lord Murray of Blidworth moved amendment 1, after clause 1, "
          "insert the following new clause. The House divided:"}  # fmt: skip
BENNETT = {"title": "Public Order Act 2023 (Interference With Use or Operation of Key National Infrastructure) "
           "Regulations 2025", "motion_text": "Baroness Bennett of Manor Castle moved, as an amendment to the motion "
           "that this House approve the Regulations, at end insert “but this House declines to approve”. "
           "The House divided:"}  # fmt: skip
SECOND_READING_LORDS = {"title": "Assisted Dying Bill [HL]",
                        "motion_text": "Lord Falconer moved that the Bill be now read a second time. The House divided:"}
AGRI_3R = {"title": "Agriculture Bill: Third Reading", "motion_text": ""}
REASONED = {"title": "Agriculture Bill: Reasoned Amendment to Second Reading", "motion_text": ""}
DRAFT_SI = {"title": "Draft Public Order Act 2023 (Interference With Use or Operation of Key National Infrastructure) "
            "Regulations 2025", "motion_text": ""}  # fmt: skip


@pytest.mark.parametrize("question, row", [
    ("Did the Lords pass the Tobacco and Vapes Bill on 24 February 2026?", MURRAY),
    ("Was the Tobacco and Vapes Bill passed by the Lords on 24 February 2026?", MURRAY),
    ("Did the Lords pass Tobacco and Vapes Bill when it divided on 24 February 2026?", MURRAY),
    ("Did the Lords reject the Tobacco and Vapes Bill on 24 February 2026?", MURRAY),
    ("Did the House of Lords approve the Public Order Act 2023 (Interference With Use or Operation of Key National "
     "Infrastructure) Regulations 2025 on 4 February 2026?", BENNETT),
    ("Did the Commons pass the Agriculture Bill at Second Reading on 13 May 2020?", REASONED),
])  # fmt: skip
def test_a_question_about_passing_the_business_is_contradicted_by_a_division_on_an_amendment(question, row):
    assert RS.contradicts(question, row)
    assert RS.title_score(question, row["title"], row["motion_text"]) == 0.0


@pytest.mark.parametrize("question, row", [
    ("Did the Commons pass the Agriculture Bill at Third Reading on 13 May 2020?", AGRI_3R),
    ("Did the Lords pass the Assisted Dying Bill [HL] at second reading on 1 March 2026?", SECOND_READING_LORDS),
    ("Did the Commons approve the Draft Public Order Act 2023 (Interference With Use or Operation of Key National "
     "Infrastructure) Regulations 2025 on 4 February 2026?", DRAFT_SI),
    # not a passage question: it asks about the division, and the answer says what the division was on
    ("What was the result of the Lords division on the Tobacco and Vapes Bill on 24 February 2026?", MURRAY),
    # it names the amendment: the division on it is the one asked about
    ("Did the Lords pass amendment 1 to the Tobacco and Vapes Bill on 24 February 2026?", MURRAY),
])  # fmt: skip
def test_a_division_on_the_business_itself_or_a_question_about_the_division_is_not_contradicted(question, row):
    assert not RS.contradicts(question, row)


GOV_382 = {"title": "Employment Rights Bill: Government motion to insist on disagreement to LA23 and LA106 to LA120, "
           "not to insist on Commons Amendment 120C, 120D and 120E but to propose Gov (a) to (f) in lieu of LA23 and "
           "LA106 to LA120", "motion_text": ""}  # fmt: skip


def test_gov_lettered_amendments_are_read():
    held = RS._held(GOV_382)["amendment"]
    assert {"(a)", "(b)", "(c)", "(d)", "(e)", "(f)"} <= held and "23" in held
    assert RS._held({"title": "X Bill: Government motion to propose Gov (a) in lieu of LA62"})["amendment"] >= {"(a)"}
    assert RS._carrying("Was Government amendment (d) on the Employment Rights Bill agreed on 8 December 2025?",
                        [GOV_382]) == [GOV_382]


def test_a_lettered_reference_left_unread_in_a_title_makes_the_day_unreadable():
    row = {"title": "X Bill: Government motion to propose (a) and (b) in lieu", "motion_text": ""}
    assert not RS._readable(row)
    assert not RS._absence_claimable({"amendment": {"(d)"}}, [row])
    # a Lords motion quotes the amendment's own text, whose "(a)" is no reference: it stays readable
    lords = {"title": "Crime and Policing Bill", "motion_text": "Baroness X moved amendment 12, in clause 5, at end "
             "insert “(a) the duty applies; (b) it does not”. The House divided:"}  # fmt: skip
    assert RS._readable(lords)


PARKINSON = {"title": "Planning and Infrastructure Bill", "motion_text": "Lord Parkinson of Whitley Bay moved amendment "
             "7A, as an amendment to amendment 4, after subsection (6) to insert the following. The House divided:"}
COFFEY = {"title": "Planning and Infrastructure Bill", "motion_text": "Baroness Coffey moved amendment 3, in clause 2, "
          "page 3, line 33, to leave out subsection (3). The House divided:"}  # fmt: skip


def test_the_target_of_an_amendment_is_not_what_its_division_was_on():
    assert RS._held(PARKINSON) == {"amendment": {"7A"}}
    assert RS._marks("Motion A1, as an amendment to Motion A") == {"motion": {"A1"}}
    asked = RS._asked_marks("What was the result of the Lords division on amendment 4 to the Planning and "
                            "Infrastructure Bill on 20 October 2025?")  # fmt: skip
    assert RS._carrying("… amendment 4 …", [COFFEY, PARKINSON]) == []
    # QA pass 5 (H2): a reference the day's divisions mention, as the target of an amendment, is not claimed absent: "no
    # division on amendment 4" would hide the division on 7A, an amendment to it
    assert not RS._absence_claimable(asked, [COFFEY, PARKINSON])
    assert RS._carrying("the division on amendment 7A", [COFFEY, PARKINSON]) == [PARKINSON]


# --- sentences: what the division was on, several results, ties -------------------------


def _value(**kw):
    base = {"house": "Lords", "date": "2026-02-24", "division_number": 1, "title": "Tobacco and Vapes Bill",
            "motion": MURRAY["motion_text"], "result": "negatived", "ayes": 78, "noes": 246}
    return {**base, **kw}


def test_a_result_sentence_says_what_the_division_was_on():
    sentence = F._s_division_result(_value())
    assert "Lord Murray of Blidworth moved amendment 1" in sentence
    assert "division 1" in sentence and "78 votes to 246" in sentence
    commons = F._s_division_result(_value(house="Commons", title="Agriculture Bill: Third Reading", motion=""))
    assert "(division 1: Agriculture Bill: Third Reading)" in commons


def test_several_results_never_imply_every_division_on_the_business():
    facts = [F.Fact("division_result", {}, value=_value(division_number=n, motion=f"Lord X moved amendment {n}."))
             for n in (1, 2)]
    sentence = F.several_results_sentence(facts)
    assert "on business matching this description" in sentence
    assert "on Tobacco and Vapes Bill," not in sentence
    assert "amendment 1" in sentence and "amendment 2" in sentence


def test_a_tie_says_what_the_record_says_of_it_and_infers_nothing():
    commons = F._s_division_result(_value(house="Commons", title="Some motion", result="tied", ayes=100, noes=100,
                                          motion="Tied vote: casting vote with the Ayes"))  # fmt: skip
    assert "100 votes to 100" in commons and "casting vote with the Ayes" in commons
    lords = F._s_division_result(_value(result="tied", ayes=138, noes=138,
                                        motion="Viscount Younger of Leckie moved amendment 87."))  # fmt: skip
    assert "138 votes to 138" in lords and "agreed" not in lords and "rejected" not in lords


BERTIN = {"title": "Crime and Policing Bill", "motion_text": "Baroness Bertin moved amendment 297AA, as an amendment to "
          "amendment 297, in subsection (1), to leave out from second “related”. The House divided:"}  # fmt: skip


def test_an_amendment_to_an_amendment_is_not_a_later_round_of_it():
    """The full probe run found it: "297AA" read as another round of amendment 297 (as "48B" is of 48), so the division
    on 297AA still answered for 297. An amendment TO another is not a round OF it.
    """
    asked = RS._asked_marks("What was the result of the Lords division on amendment 297 to the Crime and Policing Bill "
                            "on 2 March 2026?")  # fmt: skip
    assert RS._carrying("… amendment 297 …", [BERTIN]) == []
    assert not RS._absence_claimable(asked, [BERTIN]), "mentioned as the target of 297AA: not claimed absent (QA pass 5)"
    # a later round in ping-pong still counts as the amendment's: "Lords Amendment 48B" for "Lords Amendment 48"
    round_b = {"title": "Employment Rights Bill: motion to disagree with Lords Amendment 48B", "motion_text": ""}
    assert RS._carrying("the division on Lords Amendment 48", [round_b]) == [round_b]


# --- a passage question on a day of amendments only: the divisions named, no result asserted ---

MURRAY2 = {"id": 2, "number": 2, "house": "Lords", "title": "Tobacco and Vapes Bill",
           "motion_text": "Lord Moylan moved amendment 2, after clause 1, insert the following new clause."}
OTHER_BILL = {"id": 3, "number": 3, "house": "Lords", "title": "Crime and Policing Bill",
              "motion_text": "Baroness X moved amendment 9, after clause 2, insert the following new clause."}
Q_PASS = "Did the Lords pass the Tobacco and Vapes Bill on 24 February 2026?"


def test_a_passage_question_on_a_day_of_amendments_names_the_bill_s_divisions():
    rows = [{**MURRAY, "id": 1, "number": 1, "house": "Lords"}, MURRAY2, OTHER_BILL]
    assert [r["id"] for r in RS.passage_divisions(Q_PASS, rows)] == [1, 2], "the bill's divisions, not another bill's"
    third = {"id": 4, "number": 4, "house": "Commons", "title": "Agriculture Bill: Third Reading", "motion_text": ""}
    assert RS.passage_divisions("Did the Commons pass the Agriculture Bill on 13 May 2020?", [third]) == [], \
        "a division on the bill itself is answered, not listed"


def test_the_listed_divisions_sentence_asserts_no_result_and_says_why():
    rows = [{**MURRAY, "id": 1, "number": 1, "house": "Lords"}, MURRAY2]
    from datetime import date

    sentence = F.passage_sentence("Lords", date(2026, 2, 24), rows)
    assert "division 1" in sentence and "Lord Murray of Blidworth moved amendment 1" in sentence
    assert "division 2" in sentence and "amendment 2" in sentence
    for word in ("agreed", "rejected", "negatived", "passed"):
        assert word not in sentence.replace("on passing it", ""), word
    assert "does not show any of them as a vote on passing it" in sentence
