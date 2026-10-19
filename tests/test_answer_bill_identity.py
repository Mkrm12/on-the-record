"""Which bill a question names is decided by the record's own bills, not by shared words (S05's close, QA pass 8, H1).
"""

from __future__ import annotations

import pytest

from answer import resolve as RS

NAMES = RS.BillNames.from_rows([
    {"id": 1, "title": "Sentencing Act 2026"},
    {"id": 2, "title": "Sentencing (Women) Bill"},
    {"id": 3, "title": "Public Authorities (Fraud, Error and Recovery) Act 2025"},
    {"id": 4, "title": "Rule of Law (Enforcement by Public Authorities) Bill"},
    {"id": 5, "title": "Terminally Ill Adults (End of Life) Bill"},
    {"id": 6, "title": "Pensions (Special Rules for End of Life) Bill"},
    {"id": 7, "title": "Finance (No. 2) Act 2024"},
    {"id": 8, "title": "Student Finance (Review) Bill"},
    {"id": 9, "title": "Health Bill"},
    {"id": 10, "title": "Children’s Wellbeing and Schools Bill"},
    {"id": 11, "title": "Wellbeing of Future Generations Bill [HL]"},
    {"id": 12, "title": "Wellbeing of Future Generations (No. 2) Bill"},
    {"id": 13, "title": "Finance Act 2025"},
    {"id": 14, "title": "Finance Act 2026"},
    {"id": 15, "title": "Universal Credit (Removal of Two Child Limit) Bill"},
    {"id": 16, "title": "Water (Special Measures) Act 2025"},
    {"id": 17, "title": "Crime and Policing Bill"},
    {"id": 18, "title": "Tobacco and Vapes Bill"},
])


def _row(title, bill_id=None, motion=""):
    return {"title": title, "motion_text": motion, "bill_id": bill_id}


def _contradicted(question, row):
    RS.annotate_bills(NAMES, question, [row])
    return RS.contradicts(question, row)


@pytest.mark.parametrize("question, row", [
    ("What was the result of the division on the Sentencing (Women) Bill on 20 January 2026?",
     _row("Sentencing Bill: motion to disagree with Lords Amendment 7", 1)),
    ("What was the result of the division on the Rule of Law (Enforcement by Public Authorities) Bill on 5 November 2025?",
     _row("Public Authorities (Fraud, Error and Recovery) Bill: motion to disagree with Lords Amendment 43", 3)),
    ("What was the result of the division on the Pensions (Special Rules for End of Life) Bill on 11 September 2026?",
     _row("Terminally Ill Adults (End of Life) Bill: Second Reading", 5)),
    ("What was the result of the division on the Student Finance (Review) Bill on 23 May 2024?",
     _row("Finance (No. 2) Bill: Third Reading", 7)),
    # a division on no bill at all is never the one a question naming a Bill asks about
    ("What was the result of the division on the Health Bill on 9 September 2026?",
     _row("Draft Plant Health, Seeds, Seed Potatoes and Plant Propagating Material (Amendment) Regulations 2026")),
    ("How many MPs voted for the Tax Reliefs Bill on 2 December 2025?", _row("Budget Resolution No. 50")),
    # two bills whose names hold the same words are still two bills
    ("What was the result of the division on the Wellbeing of Future Generations (No. 2) Bill?",
     _row("Wellbeing of Future Generations Bill [HL]: Second Reading", 11)),
    # a name the record does not hold is compared by its words: all of them must be in the division's bill
    ("What was the result of the division on the Courts and Tribunals (Victims) Bill on 10 March 2026?",
     _row("Victims and Courts Bill")),
])  # fmt: skip
def test_a_question_naming_one_bill_is_not_answered_from_another_s_division(question, row):
    assert _contradicted(question, row)


@pytest.mark.parametrize("question, row", [
    ("What was the result of the division on the Sentencing Bill on 20 January 2026?",
     _row("Sentencing Bill: motion to disagree with Lords Amendment 7", 1)),
    # the words before the name ("Becky Gittins support", "voted against") are not part of it
    ("Did Becky Gittins support Sentencing Bill: motion to disagree with Lords Amendment 7 on 20 January 2026?",
     _row("Sentencing Bill: motion to disagree with Lords Amendment 7", 1)),
    ("How many voted against Universal Credit (Removal of Two Child Limit) Bill: Second Reading on 3 February 2026?",
     _row("Universal Credit (Removal of Two Child Limit) Bill: Second Reading", 15)),
    ("How many voted against Water (Special Measures) Bill: Third Reading on 1 May 2025?",
     _row("Water (Special Measures) Bill: Third Reading", 16)),
    ("Did Ann Member support Crime and Policing Bill: Third Reading on 18 June 2025?",
     _row("Crime and Policing Bill: Third Reading", 17)),
    ("How many members voted in favour of Tobacco and Vapes Bill: Second Reading?",
     _row("Tobacco and Vapes Bill: Second Reading")),

    # the apostrophe as typed, straight or curly, or the possessive left off (QA pass 8, Med 4)
    ("What was the result of the division on the Children's Wellbeing and Schools Bill on 14 January 2026?",
     _row("Children’s Wellbeing and Schools Bill", 10)),
    ("What was the result of the division on the Childrens Wellbeing and Schools Bill on 14 January 2026?",
     _row("Children’s Wellbeing and Schools Bill", 10)),
    # a bill of the same name in another session is the same name: the link decides nothing against the question
    ("Was the Finance Bill's third reading carried?", _row("Finance Bill: Third Reading", 14)),
    # a short name the record does not hold whole is the bill's own words
    ("What was the result of the vote on the Rwanda bill?",
     _row("Safety of Rwanda (Asylum and Immigration) Bill: 2nd reading")),
    # the division's bill named only in its motion
    ("What was the result of the vote on the Rule of Law (Enforcement by Public Authorities) Bill?",
     _row("Business of the House", None, "That the Rule of Law (Enforcement by Public Authorities) Bill be considered")),
])  # fmt: skip
def test_a_question_naming_the_division_s_own_bill_is_no_contradiction(question, row):
    assert not _contradicted(question, row)


def test_an_act_cited_with_its_number_before_the_year_is_the_numbered_bill():
    """The record titles it "Finance Act (No. 2) 2024"; it was the Finance (No. 2) Bill."""
    names = RS.BillNames.from_rows([{"id": 3690, "title": "Finance Act (No. 2) 2024"}, {"id": 4042, "title": "Finance Act 2026"}])
    question = "How many voted against Finance (No. 2) Bill: Third Reading on 23 May 2024?"
    row = _row("Finance (No. 2) Bill: Third Reading", 3690)
    RS.annotate_bills(names, question, [row])
    assert row["bills"][0] == {3690} and not RS.contradicts(question, row)


def test_the_longest_name_found_wins_and_a_name_inside_it_is_not_named():
    assert NAMES.named("the Sentencing (Women) Bill") == {2}
    assert NAMES.named("the Wellbeing of Future Generations Bill") == {11}
    assert NAMES.named("the Wellbeing of Future Generations (No. 2) Bill") == {12}
    assert NAMES.named("Finance Bill") == {13, 14}  # both sessions' Finance Bills: the same name


def test_without_the_record_s_bills_the_division_s_printed_name_in_the_question_decides():
    """Without the record's bills, the words in front of a name ("voted against") are told from it by the division's
    own printed bill name found whole in the question; another bill's division is still refused.
    """
    question = "How many voted against Water (Special Measures) Bill: Third Reading on 1 May 2025?"
    assert not RS.contradicts(question, _row("Water (Special Measures) Bill: Third Reading"))
    assert RS.contradicts(question, _row("Water Industry Bill: Third Reading"))


def test_without_the_record_s_bills_the_word_rule_still_refuses_two_bills_sharing_a_word():
    """A caller that never annotated its rows (no registry) keeps the word rule: containment, never overlap."""
    row = _row("Courts and Tribunals Bill: Second Reading")
    assert RS.contradicts("What was the result of the Commons division on Victims and Courts Bill on 10 March 2026?", row)


# --- on the record itself (skips where hansard_v2 is not loaded)
# -------------------------------------------------------


def _live(question: str, when: str):
    import datetime as dt

    import psycopg
    from psycopg.conninfo import make_conninfo

    from app import safety, sealed
    from app.config import settings
    from ingest.runtime import run as run_async

    async def go():
        conn = await psycopg.AsyncConnection.connect(make_conninfo(settings.pg_dsn, dbname=safety.TARGET),
                                                     options="-c default_transaction_read_only=on")
        try:
            await conn.execute(sealed.serving_sql())
            day = dt.date.fromisoformat(when)
            if not (await (await conn.execute("SELECT 1 FROM division WHERE division_date = %s", (day,))).fetchall()):
                pytest.skip(f"no division on {when} in this database")
            return await RS.division_id(conn, question, house="Commons", when=day)
        finally:
            await conn.close()

    try:
        return run_async(go())
    except (psycopg.OperationalError, psycopg.errors.InvalidCatalogName) as exc:
        pytest.skip(f"hansard_v2 is not reachable ({type(exc).__name__})")


@pytest.mark.parametrize("question, when, expected", [
    ("What was the result of the division on the Sentencing (Women) Bill on 20 January 2026?", "2026-01-20", None),
    ("What was the result of the division on the Rule of Law (Enforcement by Public Authorities) Bill on 5 November "
     "2025?", "2025-11-05", None),
    # the positive controls: the same days, the bills that did divide
    ("What was the result of the division on the Sentencing Bill motion to disagree with Lords Amendment 7 on 20 "
     "January 2026?", "2026-01-20", 2239),
    ("What was the result of the division on the Public Authorities (Fraud, Error and Recovery) Bill motion to disagree "
     "with Lords Amendment 43 on 5 November 2025?", "2025-11-05", 2172),
])  # fmt: skip
def test_on_the_record_another_bill_s_division_is_never_the_answer(question, when, expected):
    found, why = _live(question, when)
    assert found == expected, why


def test_the_title_score_judges_the_same_annotated_row_as_the_filter(monkeypatch):
    """The scorer built a bare row and so fell back to words: 198 tally questions on days of several divisions fell
    through in the final probes. It must judge the caller's own annotated row.
    """
    seen = []
    monkeypatch.setattr(RS, "contradicts", lambda question, row: seen.append(row) or False)
    row = _row("Victims and Courts Bill", 1)
    RS.title_score("How many members voted for and against Victims and Courts Bill?", row["title"], None, row=row)
    assert seen and seen[0] is row


@pytest.mark.parametrize("question, title", [
    # the record's bills name it otherwise ("Bus Services Act 2025"), so words decide; the title's own bill name is in
    # the question whole, so it is the bill asked about, whatever the words read before it ("pass")
    ("Did Lord Example support Bus Services (No. 2) Bill [HL] on 13 October 2025?", "Bus Services (No. 2) Bill [HL]"),
    ("How many members voted for and against Bus Services (No. 2) Bill [HL] in the Lords on 13 October 2025?",
     "Bus Services (No. 2) Bill [HL]"),
    # a title the record misprints with "Bill" twice
    ("Did the Commons pass Passenger Railway Services Bill (Public Ownership) Bill: Second Reading when it divided on "
     "29 July 2024?", "Passenger Railway Services Bill (Public Ownership) Bill: Second Reading"),
])  # fmt: skip
def test_a_division_whose_own_bill_name_is_in_the_question_whole_is_the_bill_asked(question, title):
    names = RS.BillNames.from_rows([{"id": 1, "title": "Bus Services Act 2025"},
                                    {"id": 2, "title": "Passenger Railway Services (Public Ownership) Act 2024"}])
    row = _row(title, 1 if "Bus" in title else 2)
    RS.annotate_bills(names, question, [row])
    assert not RS.contradicts(question, row)
