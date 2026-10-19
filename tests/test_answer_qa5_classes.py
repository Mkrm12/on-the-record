"""The classes QA pass 5 found still open, fixed by their structure, not by the wording that found them."""

from __future__ import annotations

import pytest

from answer import resolve as RS
from answer import router as RO

REGS = "Public Order Act 2023 (Interference With Use or Operation of Key National Infrastructure) Regulations 2025"
BENNETT = {"title": REGS, "motion_text": "Baroness Bennett of Manor Castle moved, as an amendment to the motion that this "
           "House approve the Regulations, at end insert “but this House declines to approve”."}  # fmt: skip
MURRAY = {"title": "Tobacco and Vapes Bill", "motion_text": "Lord Murray of Blidworth moved amendment 1, in clause 1, "
          "page 1, line 5, to leave out."}  # fmt: skip
CALLANAN = {"title": "Diego Garcia Military Base and British Indian Ocean Territory Bill",
            "motion_text": "Lord Callanan moved, as an amendment to the motion that this Bill do now pass, at end to "
                           "insert “but that this House regrets”."}  # fmt: skip
INSTRUCTION = {"title": "High Speed Rail (Crewe - Manchester) Bill: Instruction (No. 3)", "motion_text": ""}
THIRD = {"title": "High Speed Rail (Crewe - Manchester) Bill: Third Reading", "motion_text": ""}
SECOND = {"title": "High Speed Rail (Crewe - Manchester) Bill: Second Reading", "motion_text": ""}
DRAFT = {"title": f"Draft {REGS}", "motion_text": ""}


# --- H1, M1: a fate question about the business -----------------------------------------------------------------


@pytest.mark.parametrize("question, row", [
    (f"Did the Lords agree to the {REGS} on 4 February 2026?", BENNETT),
    (f"Were the {REGS} agreed in the Lords on 4 February 2026?", BENNETT),
    (f"Were the {REGS} carried in the Lords on 4 February 2026?", BENNETT),
    (f"Was the motion to approve the {REGS} agreed in the Lords on 4 February 2026?", BENNETT),
    ("Was the Tobacco and Vapes Bill carried in the Lords on 24 February 2026?", MURRAY),
    ("Was the Tobacco and Vapes Bill agreed to in the Lords on 24 February 2026?", MURRAY),
    ("Was the Tobacco and Vapes Bill lost in the Lords on 24 February 2026?", MURRAY),
    ("Did the Lords agree to the Tobacco and Vapes Bill on 24 February 2026?", MURRAY),
    ("Was the Diego Garcia Military Base and British Indian Ocean Territory Bill carried in the Lords on 12 January "
     "2026?", CALLANAN),
    ("Did the Commons pass the High Speed Rail (Crewe - Manchester) Bill on 21 May 2024?", INSTRUCTION),
    ("Was the High Speed Rail (Crewe - Manchester) Bill passed by the Commons on 21 May 2024?", INSTRUCTION),
    ("Did the Commons pass the High Speed Rail (Crewe - Manchester) Bill on 21 May 2024?", SECOND),
])  # fmt: skip
def test_a_fate_question_about_the_business_is_answered_only_by_a_division_that_decided_it(question, row):
    assert RS.contradicts(question, row), (question, row["title"])


@pytest.mark.parametrize("question, row", [
    ("Did the Commons pass the High Speed Rail (Crewe - Manchester) Bill on 21 May 2024?", THIRD),
    ("Did the Commons back the High Speed Rail (Crewe - Manchester) Bill at Second Reading on 21 May 2024?", SECOND),
    (f"Did the Commons approve the Draft {REGS} on 4 February 2026?", DRAFT),
    ("What was the result of the Lords division on the Tobacco and Vapes Bill on 24 February 2026?", MURRAY),
])  # fmt: skip
def test_a_division_that_decided_the_business_or_a_question_about_the_division_is_not_contradicted(question, row):
    assert not RS.contradicts(question, row), (question, row["title"])


# --- H2, M3: an amendment reached through another, and a motion about an amendment --------------------------------

MOTION_31A = {"title": "Bus Services (No. 2) Bill [HL]", "motion_text": "Lord Moynihan moved Motion 31A, as an amendment "
              "to the motion that this House do agree with the Commons in their Amendment 31, to leave out from "
              "“House” to end and insert “do disagree with the Commons in their Amendment 31.”"}  # fmt: skip
YOUNG = {"title": "Crime and Policing Bill", "motion_text": "Lord Young of Acton moved amendment 387B, as an amendment "
         "to replace amendment 387, in clause 140, page 180."}  # fmt: skip
MOTION_B1 = {"title": "Employment Rights Bill", "motion_text": "Lord Sharpe of Epsom moved Motion B1, as an amendment to "
             "Motion B, at end to insert “and do propose Amendment 120N as an amendment to Commons Amendment "
             "120G”."}  # fmt: skip
GOV_A_TO_LA2 = {"title": "Planning and Infrastructure Bill: Government amendment (a) to Lords Amendment 2",
                "motion_text": ""}
INSIST_1B = {"title": "Employment Rights Bill: Government motion to insist on disagreement to Lords Amendment 1B but to "
             "propose Government amendments (a) and (b) in lieu of Lords Amendment 1B", "motion_text": ""}  # fmt: skip


@pytest.mark.parametrize("asked, row", [
    ("What was the result of the Lords division on Commons Amendment 31 to the Bus Services (No. 2) Bill on 13 October "
     "2025?", MOTION_31A),
    ("What was the result of the Lords division on amendment 387 to the Crime and Policing Bill on 11 March 2026?", YOUNG),
    ("What was the result of the Lords division on Commons Amendment 120G to the Employment Rights Bill on 10 December "
     "2025?", MOTION_B1),
    ("What was the result of the Commons division on Lords Amendment 2 to the Planning and Infrastructure Bill on 13 "
     "November 2025?", GOV_A_TO_LA2),
])  # fmt: skip
def test_an_amendment_reached_through_an_amending_relation_is_not_what_the_division_decided(asked, row):
    marks = RS._asked_marks(asked)
    assert RS._carrying(asked, [row]) == [], row["motion_text"] or row["title"]
    assert not RS._absence_claimable(marks, [row]), "it is mentioned, so 'no division was on it' is not claimed"


def test_the_compound_reference_and_the_moved_one_are_still_carried():
    assert RS._carrying("Government amendment (a) to Lords Amendment 2 on 13 November 2025", [GOV_A_TO_LA2]) == [GOV_A_TO_LA2]
    # a Lords motion's own letter is a reference kind `_carrying` reads
    assert RS._carrying("the division on Motion 31A on 13 October 2025", [MOTION_31A]) == [MOTION_31A]
    assert RS._carrying("the division on amendment 387B on 11 March 2026", [YOUNG]) == [YOUNG]
    # "(a) to Lords Amendment 900" is still no division on "(a) to Lords Amendment 2"
    assert RS._carrying("Government amendment (a) to Lords Amendment 900 on 13 November 2025", [GOV_A_TO_LA2]) == []


def test_a_fate_question_about_an_amendment_is_not_answered_from_a_motion_about_it():
    assert RS.contradicts("Was Lords Amendment 1B to the Employment Rights Bill agreed in the Commons on 8 December 2025?",
                          INSIST_1B)
    assert RS.contradicts("Did the Lords agree to Commons Amendment 31 to the Bus Services (No. 2) Bill on 13 October "
                          "2025?", MOTION_31A)  # fmt: skip
    assert not RS.contradicts("What was the result of the Commons division on Lords Amendment 1B to the Employment Rights "
                              "Bill on 8 December 2025?", INSIST_1B), "a question about the division, answered with it"


# --- H3: a qualified tally ------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What was the Conservative tally on the Agriculture Bill Third Reading on 13 May 2020?",
    "What were the Labour ayes and noes on New Clause 2 to the Agriculture Bill on 13 May 2020?",
    "What was the SNP vote count on the Agriculture Bill Third Reading on 13 May 2020?",
    "What were the Conservative voting figures on the Agriculture Bill Third Reading on 13 May 2020?",
    "How many voted against the Agriculture Bill at Third Reading on 13 May 2020 among Conservative MPs?",
    "How many MPs voted against the Agriculture Bill at Third Reading on 13 May 2020, of the Conservatives?",
    "What were the Contents and Not-Contents among crossbench peers on amendment 1 on 24 February 2026?",
])  # fmt: skip
def test_a_tally_of_part_of_the_house_is_never_the_house_s_tally(question):
    assert RO.route(question).category != "division_tally", question


@pytest.mark.parametrize("question", [
    "What was the tally on the Agriculture Bill Third Reading on 13 May 2020?",
    "What were the ayes and noes on New Clause 2 to the Agriculture Bill on 13 May 2020?",
    "What were the Contents and Not-Contents in the Lords division on the Tobacco and Vapes Bill on 24 February 2026?",
])  # fmt: skip
def test_a_whole_house_tally_is_still_one(question):
    assert RO.route(question).category == "division_tally"


# --- H4: an office qualified anywhere ---------------------------------------------------------------------------


@pytest.mark.parametrize("question, office", [
    ("Who was the Shadow Home Secretary in the SNP on 1 November 2025?", "Shadow Home Secretary"),
    ("Who was the Shadow Chancellor of the Exchequer at the Liberal Democrats on 1 November 2025?",
     "Shadow Chancellor of the Exchequer"),
    ("Who was the Shadow Home Secretary from the Green Party on 15 January 2026?", "Shadow Home Secretary"),
    ("Who was the Shadow Chancellor of the Exchequer on 1 November 2025 for the Liberal Democrats?",
     "Shadow Chancellor of the Exchequer"),
    ("In the SNP, who was the Shadow Home Secretary on 1 November 2025?", "Shadow Home Secretary"),
])  # fmt: skip
def test_a_qualifier_after_the_office_after_the_date_or_before_the_opener_names_another_office(question, office):
    assert not RS._office_named_in_full(question, office)


@pytest.mark.parametrize("question, office", [
    ("On 1 November 2025, who was the Home Secretary?", "Home Secretary"),
    ("Who was the Home Secretary on 1 November 2025?", "Home Secretary"),
    ("Who held the post of Party Chair, Labour Party on 1 May 2024?", "Party Chair, Labour Party"),
    ("On 1 October 2025, who was serving as Minister of State (Minister for Women and Equalities) ?",
     "Minister of State (Minister for Women and Equalities)"),
    ("Which minister was Home Secretary on 10 March 2026?", "Home Secretary"),
])  # fmt: skip
def test_ordinary_office_questions_still_name_their_office(question, office):
    assert RS._office_named_in_full(question, office)


# --- H5: a party question's subject is the member alone ---------------------------------------------------------


@pytest.mark.parametrize("question, subject", [
    ("Which party was the brother of Ed Miliband in on 13 May 2020?", "the brother of Ed Miliband"),
    ("Which party was the MP who took over from Theresa May in on 13 May 2020?", "the MP who took over from Theresa May"),
    ("What was Catherine Atkinson's party on 22 July 2024?", "Catherine Atkinson"),
    ("On 7 September 2026, which party was Baroness Featherstone a member of?", "Baroness Featherstone"),
    ("Which party did Brian Leishman sit for on 13 January 2026?", "Brian Leishman"),
])  # fmt: skip
def test_a_party_question_has_a_subject(question, subject):
    assert RS.party_subject(question) == subject


def test_a_party_question_about_someone_described_by_a_member_is_refused():
    rows = [{"mnis_id": 1, "printed": "Edward Miliband", "name": "Ed Miliband"},
            {"mnis_id": 2, "printed": "Theresa May", "name": "Mrs Theresa May"}]
    for q in ("Which party was the brother of Ed Miliband in on 13 May 2020?",
              "Which party was the MP who took over from Theresa May in on 13 May 2020?"):
        assert RS._pick_listed(rows, RS.party_subject(q), whole=True)[0] is None, q
    assert RS._pick_listed(rows, RS.party_subject("Which party was Ed Miliband in on 13 May 2020?"), whole=True)[0] == 1


# --- H6, M2 -----------------------------------------------------------------------------------------------------


def test_a_lettered_reference_in_capitals_is_the_same_reference():
    assert RS._asked_marks("Was Government amendment (D) on the Employment Rights Bill agreed?") == {"amendment": {"(d)"}}
    assert "(d)" in RS._held({"title": "X Bill: Government motion to propose Gov (A) to (F) in lieu of LA23"})["amendment"]


@pytest.mark.parametrize("title, stage", [
    ("Commonwealth Parliamentary Association and International Committee of the Red Cross (Status) Bill", "1st reading"),
    ("Access to Elected Office Fund (Report) Bill", "2nd reading"),
    ("Bank Resolution (Recapitalisation) Act 2025", "3rd reading"),
])  # fmt: skip
def test_a_bill_s_own_title_is_no_stage_word(title, stage):
    question = f"When did the {title} have its {stage}?"
    assert RS._stage_score(RS._stage_terms(RS.stage_question(question, [title])), stage) == 1.0


# --- found re-running pass 5's inputs --------------------------------------------------------------

BUSCOMBE = {"title": "Crime and Policing Bill", "motion_text": "Baroness Buscombe moved amendment 387A, after clause 139, "
            "to insert the new clause."}  # fmt: skip


def test_a_lettered_lords_report_amendment_is_its_own_amendment_not_a_round():
    """"387A" moved at report is a different amendment from 387; only an amendment exchanged between the Houses has
    rounds ("Lords Amendment 48B" is a later round of Lords Amendment 48).
    """
    assert RS._carrying("the Lords division on amendment 387 to the Crime and Policing Bill on 11 March 2026",
                        [BUSCOMBE]) == []  # fmt: skip
    ping_pong = {"title": "Employment Rights Bill: motion to disagree with Lords Amendment 48B", "motion_text": ""}
    assert RS._carrying("the division on Lords Amendment 48 on 8 December 2025", [ping_pong]) == [ping_pong]
    la = {"title": "Employment Rights Bill: Government motion to insist on disagreement to LA48B", "motion_text": ""}
    assert RS._carrying("the division on Lords Amendment 48 on 8 December 2025", [la]) == [la]


def test_a_stage_question_back_reads_as_english():
    from answer import facts as F

    fact = F.Fact("bill_stage_date", {}, clarify=[{"stage_id": 1, "house": "Commons", "first_sitting": "2023-12-12",
                                                   "sittings": 1}, {"stage_id": 2, "house": "Lords",
                                                                    "first_sitting": "2024-04-29", "sittings": 1}])
    text = fact.render()
    assert "{" not in text and "stage_id" not in text and "Commons" in text and "12 December 2023" in text


def test_a_reference_a_division_holds_exactly_is_never_claimed_absent():
    """The CI probes caught it: with rounds narrowed to inter-House amendments, an exactly held "New Clause 65" no
    longer stopped "no division is recorded on New Clause 65" when the voter's name did not resolve.
    """
    nc65 = {"title": "Representation of the People Bill: Report Stage: New Clause 65", "motion_text": ""}
    assert not RS._absence_claimable({"new clause": {"65"}}, [nc65])


# --- pass 1 on the pass-5 fixes --------------------------------------------------------------------


@pytest.mark.parametrize("title, like", [
    ("Draft Medical Devices (Fees Amendment) Regulations 2026", False),
    ("Ministerial Salaries (Amendment) Act 2026", False),
    ("Children's Wellbeing (Amendment) Bill: Second Reading", False),
    ("Agriculture Bill: Report Stage: Amendment 39 (Deidre Brock)", True),
    ("Agriculture Bill: Reasoned Amendment to Second Reading", True),
])  # fmt: skip
def test_an_amendment_in_an_instrument_s_or_a_bill_s_own_name_is_no_amendment(title, like):
    assert RS.amendment_like({"title": title, "motion_text": ""}) is like


def test_an_instrument_whose_name_says_amendment_is_answered_on_its_approval():
    row = {"title": "Draft Medical Devices (Fees Amendment) Regulations 2026", "motion_text": ""}
    q = "Were the Draft Medical Devices (Fees Amendment) Regulations 2026 approved by the Commons on 28 January 2026?"
    assert RS.decides_passage(q, row) and not RS.contradicts(q, row)


def test_a_quoted_reference_in_a_question_keeps_its_kind():
    assert RS._asked_marks('What was the result of the division on "New Clause 2" on 13 May 2020?') == {
        "new clause": {"2"}}
    assert RS._targets({"title": "x", "motion_text": "Lord X moved Motion A1, as an amendment to Motion A."}) == {
        "motion": {"A"}}


def test_a_question_that_names_the_motion_asks_the_motion_s_fate():
    """Only a question asking the AMENDMENT's fate is not answered from a motion about it."""
    row = {"title": "National Insurance Contributions (Employer Pensions Contributions) Bill: motion to disagree with Lords "
                    "Amendment 5", "motion_text": ""}  # fmt: skip
    asks_motion = ("Was National Insurance Contributions (Employer Pensions Contributions) Bill: motion to disagree with "
                   "Lords Amendment 5 passed or rejected when the Commons divided on 23 March 2026?")  # fmt: skip
    asks_amendment = "Was Lords Amendment 5 to the National Insurance Contributions Bill agreed on 23 March 2026?"
    assert not RS.contradicts(asks_motion, row)
    assert RS.contradicts(asks_amendment, row)


# --- the full probe run on the pass-5 fixes ------------------------------------------------------

HOEY = {"title": "Greenhouse Gas Emissions Trading Scheme (Amendment) (Extension to Maritime Activities) Order 2026",
        "motion_text": "Baroness Hoey moved, as an amendment to the motion to approve the Order, to leave out all the "
                       "words after \"that\" and insert \"this House declines to approve the draft Order\"."}  # fmt: skip
TRUE_AMENDS = {"title": "Business of the House", "motion_text": "Baroness Smith of Basildon moved that Standing Order "
               "38(1) be dispensed with. Then Lord True moved, as an amendment to the motion, to leave out."}  # fmt: skip


@pytest.mark.parametrize("question, row", [
    ("Did the Lords pass Greenhouse Gas Emissions Trading Scheme (Amendment) (Extension to Maritime Activities) Order "
     "2026 when it divided on 12 March 2026?", HOEY),
    ("Was Greenhouse Gas Emissions Trading Scheme (Amendment) (Extension to Maritime Activities) Order 2026 carried in "
     "the Lords on 12 March 2026?", HOEY),
    ("Did the Lords agree to Business of the House when it divided on 14 October 2025?", TRUE_AMENDS),
])  # fmt: skip
def test_an_instrument_named_amendment_or_any_named_business_is_still_a_passage_question(question, row):
    assert RS.asks_passage(question), question
    assert RS.contradicts(question, row)


def test_amendment_in_a_name_is_no_kind_word():
    assert RS._kinds_unnumbered("Were the Draft Radio Equipment (Amendment) (Northern Ireland) Regulations 2025 "
                                "approved?") == set()  # fmt: skip


@pytest.mark.parametrize("question", [
    "What was the Ulster Unionist Party tally in the Commons division on Draft Radio Equipment (Amendment) (Northern "
    "Ireland) Regulations 2025 on 19 November 2025?",
    "What were the Alba ayes and noes on the Agriculture Bill on 13 May 2020?",
    "What was the Workers Party vote count on the Agriculture Bill on 13 May 2020?",
])  # fmt: skip
def test_any_qualifier_of_a_tally_makes_it_no_whole_house_tally(question):
    assert RO.route(question).category != "division_tally", question


def test_a_tally_qualified_only_by_whole_house_words_is_still_one():
    assert RO.route("What was the final tally in the Commons division on the Agriculture Bill on 13 May "
                    "2020?").category == "division_tally"  # fmt: skip
