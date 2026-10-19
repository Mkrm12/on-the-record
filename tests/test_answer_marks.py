"""Reading the clause, schedule and amendment references a division carries."""

from __future__ import annotations

import pytest

from answer import resolve as RS


@pytest.mark.parametrize("text, kind, refs", [
    ("Lords Amendment 23", "amendment", {"23"}),
    ("That this House do insist on its Amendments 23 and 106 to 120", "amendment", {"23"} | {str(n) for n in range(106, 121)}),
    ("Government motion to insist on disagreement to LA23 and LA106 to LA120", "amendment",
     {"23"} | {str(n) for n in range(106, 121)}),
    ("motion to disagree with Lords Amendment 1B", "amendment", {"1B"}),
    ("not to insist on Commons Amendment 120C, 120D and 120E", "amendment", {"120C", "120D", "120E"}),
    ("LA72D to LA72H and to propose", "amendment", {"72D", "72E", "72F", "72G", "72H"}),
    ("to propose Government amendments (a) to (c) in lieu", "amendment", {"(a)", "(b)", "(c)"}),
    # the target of "as an amendment to" is not what the division decided
    ("Amendment A1, as an amendment to Amendment A", "amendment", {"A1"}),
    ("Agriculture Bill: Report Stage: New Clause 2 (Neil Parish)", "new clause", {"2"}),
    ("New Clauses 3 and 4", "new clause", {"3", "4"}),
    ("Clause 8 stand part", "clause", {"8"}),
    ("Finance (No. 2) Bill: Third Reading", "no", {"2"}),
    ("Lords Amendment 48B", "amendment", {"48B"}),
])  # fmt: skip
def test_the_record_s_reference_forms_are_read(text, kind, refs):
    assert RS._marks(text).get(kind) == refs


@pytest.mark.parametrize("text", [
    "Amendment 12, in clause 5, page 3, line 10, at end insert",
    "Amendment 7, after clause 2, to insert the new clause",
    "Amendment 4, to leave out clause 9",
])  # fmt: skip
def test_a_location_is_not_what_the_division_is_on(text):
    marks = RS._marks(text)
    assert "clause" not in marks, marks
    assert marks.get("amendment"), marks


def test_a_list_stops_at_words_that_are_not_references():
    assert RS._marks("Amendment 5, page 3, line 10").get("amendment") == {"5"}
    assert RS._marks("Lords Amendment 23 to the Employment Rights Bill").get("amendment") == {"23"}
    assert RS._marks("amendments and other business") == {}


def test_a_lowercase_letter_is_not_read_as_a_lettered_amendment():
    assert RS._marks("the amendment a member moved") == {}


def test_a_huge_range_is_not_expanded():
    assert RS._marks("Amendments 1 to 100000").get("amendment") == {"1", "100000"}


ERB = "Employment Rights Bill"


def row(title, motion="", number=1):
    return {"title": title, "motion_text": motion, "number": number}


def test_a_division_carrying_the_amendment_in_a_list_is_found():
    rows = [row(ERB, "That this House do insist on its Amendments 23 and 106 to 120", 2),
            row(ERB, "That this House do not insist on its Amendment 1B", 1)]  # fmt: skip
    held = RS._carrying("What was the result of the Lords division on Lords Amendment 23 to the Employment Rights Bill "
                        "on 28 October 2025?", rows)  # fmt: skip
    assert [r["number"] for r in held] == [2]
    assert RS._carrying("... on Lords Amendment 110 ...", rows)[0]["number"] == 2


def test_a_later_round_of_the_same_amendment_is_found_when_the_exact_one_is_not():
    rows = [row(f"{ERB}: motion to disagree with LA48B", number=383), row(f"{ERB}: Third Reading", number=384)]
    assert [r["number"] for r in RS._carrying("the Commons division on Lords Amendment 48 on 8 December 2025", rows)] \
        == [383]
    assert RS._carrying("on Lords Amendment 48B", rows)[0]["number"] == 383


def test_the_exact_reference_is_preferred_to_another_round():
    rows = [row(f"{ERB}: motion to disagree with LA48B", number=383), row(f"{ERB}: LA48", number=380)]
    assert [r["number"] for r in RS._carrying("on Lords Amendment 48", rows)] == [380]


def test_absence_is_claimed_only_when_every_division_can_be_read():
    readable = [row(f"{ERB}: Third Reading", number=1), row(f"{ERB}: Lords Amendment 5", number=2)]
    assert RS._absence_claimable({"amendment": {"23"}}, readable)
    bare_lords = [*readable, row(ERB, "", 3)]  # a Lords division with no motion: what was it on?
    assert not RS._absence_claimable({"amendment": {"23"}}, bare_lords)
    unnumbered = [*readable, row(ERB, "That this House do not insist on its amendment", 3)]
    assert not RS._absence_claimable({"amendment": {"23"}}, unnumbered)
    family = [*readable, row(f"{ERB}: LA23B", number=3)]
    assert not RS._absence_claimable({"amendment": {"23"}}, family), "a later round of 23 is not an absence of 23"


def test_an_inserted_new_clause_does_not_make_a_division_unreadable():
    rows = [row("Bill: Amendment 7, after clause 2, to insert the new clause", number=1)]
    assert RS._absence_claimable({"amendment": {"23"}}, rows)


def test_a_conflicting_reference_still_disqualifies_and_a_later_round_does_not():
    assert RS.title_score("on New Clause 3 of the Agriculture Bill", "Agriculture Bill: New Clause 2") == 0.0
    assert RS.title_score("on Lords Amendment 48 to the Employment Rights Bill", f"{ERB}: motion to disagree with LA48B") > 0
