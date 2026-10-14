"""A division reaches its bill stage the way Parliament titles it."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from ingest import bills as B
from ingest.divisions import DebateInfo

DAY = date(2025, 11, 19)


def stage(stage_id, title, abbreviation, *, house="Commons", days=(DAY,), debates=(), name="", former=None):
    forms = B.bill_name_forms({"title": title, "former_short_title": former})
    return B.StageRef(stage_id, 100 + stage_id, title, abbreviation, house, frozenset(days), frozenset(debates),
                      bill_titles=frozenset(B.normalise_title(t) for t in forms), stage_name=name)


def division(title, *, house="Commons", day=DAY, debate_id=None, deferred=False, debate_link="hansard_division_list"):
    return {"house": house, "division_date": day, "title": title, "is_deferred": deferred,
            "debate_id": debate_id, "debate_link": debate_link}


def link(title, stages, **kw):
    return B.link_division_stage(division(title, **kw), stages)


# --- the stage before the colon
# ----------------------------------------------------------------------------------------

REMAINING = [stage(1, "Armed Forces Act 2026", "COTWH"), stage(2, "Armed Forces Act 2026", "RS"),
             stage(3, "Armed Forces Act 2026", "3R")]


@pytest.mark.parametrize("title, want", [
    ("Armed Forces Bill Committee: New Clause 13", (1, "title_stage")),
    ("Armed Forces Bill Committee: Amendment 4", (1, "title_stage")),
    ("Armed Forces Bill Committee: Clause 3 stand part", (1, "title_stage")),
    ("Armed Forces Bill Report Stage: Amendment 11", (2, "title_stage")),
    ("Armed Forces Bill Report Stage: New Clause 22", (2, "title_stage")),
    ("Armed Forces Bill: Report Stage: New Clause 4", (2, "title_stage")),
    ("Armed Forces Bill: Third Reading", (3, "title_stage")),
    ("Armed Forces Bill Third Reading", (3, "title_stage")),
])  # fmt: skip
def test_the_stage_is_read_wherever_the_commons_writes_it(title, want):
    assert link(title, REMAINING) == want


def test_a_bill_name_followed_by_other_words_is_another_bill():
    """"Health Bill Reform" is not the Health Bill: before the colon only nothing or the stage may follow the name.
    """
    stages = [stage(1, "Health Bill", "RS")]
    assert link("Health Bill Reform: Report Stage", stages) == (None, "none")
    assert link("Health Bill Report Stage", stages) == (1, "title_stage")


def test_only_the_first_stage_phrase_counts():
    """What follows the stage is the question: a new clause about a committee is still report stage."""
    stages = [stage(1, "Standards Act 2026", "CS"), stage(2, "Standards Act 2026", "RS")]
    assert link("Standards Bill: Report Stage: New Clause 5 (Committee on Standards)", stages) == (2, "title_stage")


def test_a_stage_the_title_names_that_is_not_sitting_that_day_is_not_replaced_by_one_that_is():
    stages = [stage(1, "Standards Act 2026", "CS")]
    assert link("Standards Bill: Report Stage: New Clause 5 (Committee on Standards)", stages) == (None, "none")


def test_a_bill_named_after_a_stage_word_is_not_that_stage():
    stages = [stage(1, "Second Reading Reform Bill", "RS"), stage(2, "Second Reading Reform Bill", "2R")]
    assert link("Second Reading Reform Bill: Report Stage", stages) == (1, "title_stage")


# --- the amendment rule, remaining stages, reasoned amendments, ping-pong ---------------------------------------------


def test_a_new_clause_with_no_stage_named_is_never_second_or_third_reading():
    stages = [stage(1, "Housing Act 2026", "RS"), stage(2, "Housing Act 2026", "3R")]
    assert link("Housing Bill: New Clause 7", stages) == (1, "title_amendment")
    assert link("Housing Bill: Amendment 12", stages) == (1, "title_amendment")


def test_an_amendment_with_two_amending_stages_that_day_is_refused():
    stages = [stage(1, "Housing Act 2026", "COTWH"), stage(2, "Housing Act 2026", "RS")]
    assert link("Housing Bill: New Clause 7", stages) == (None, "ambiguous")


@pytest.mark.parametrize("title, want", [
    ("Housing Bill Remaining Stages: New Clause 1", 1),
    ("Housing Bill Remaining Stages: Amendment 3", 1),
    ("Housing Bill: Remaining Stages: Third Reading", 2),
])  # fmt: skip
def test_remaining_stages_are_resolved_by_what_was_put(title, want):
    stages = [stage(1, "Housing Act 2026", "RS"), stage(2, "Housing Act 2026", "3R")]
    assert link(title, stages)[0] == want


@pytest.mark.parametrize("title", ["Social Housing Bill [Lords]: Reasoned Amendment to Second Reading",
                                   "Social Housing Bill: Opposition Reasoned Amendment",
                                   "Social Housing Bill: Reasoned amendment to Second Reading (Keir Starmer)"])
def test_a_reasoned_amendment_is_a_reading(title):
    stages = [stage(1, "Social Housing Bill [HL]", "2R"), stage(2, "Social Housing Bill [HL]", "PM")]
    assert link(title, stages) == (1, "title_stage")


@pytest.mark.parametrize("title", [
    "Border Security Bill: Motion to disagree with Lords Amendment 7",
    "Border Security Bill: Government motion to insist on disagreement",
    "Border Security Bill: Motion relating to Lords Reason 12B",
    "Border Security Bill: Government Amendment (a) in lieu of Lords Amendment 3",
    "Border Security Bill: Consideration of Lords Message: manuscript amendment",
])  # fmt: skip
def test_ping_pong_is_consideration_of_lords_amendments_or_message(title):
    stages = [stage(1, "Border Security Act 2026", "COLM"), stage(2, "Border Security Act 2026", "PM")]
    assert link(title, stages) == (1, "title_stage")


def test_an_instruction_is_its_own_stage():
    stages = [stage(1, "Planning Act 2026", "I"), stage(2, "Planning Act 2026", "2R")]
    assert link("Planning Bill: Instruction (No. 2)", stages) == (1, "title_stage")


# --- +numbered: "(No.


def test_the_api_writes_the_number_after_act_and_the_bill_name_keeps_it():
    assert "Finance (No. 2) Bill" in B.bill_name_forms({"title": "Finance Act (No. 2) 2024"})
    assert "Finance (No. 2) Bill" in B.bill_name_forms({"title": "Finance (No. 2) Act 2024"})


def test_a_numbered_bill_renamed_without_its_number_links_on_the_numbered_tier():
    stages = [stage(1, "Finance Act 2026", "CS")]
    assert link("Finance (No. 2) Bill Committee: New Clause 5", stages) == (1, "title_stage+numbered")
    assert link("Finance (No.2) Bill Committee: Amendment 1", stages) == (1, "title_stage+numbered")


def test_the_exact_numbered_name_wins_over_the_numberless_fallback():
    """Two private members' bills: "(Repeal) Bill" and "(Repeal) (No. 2) Bill". An exact name is never overridden.
    """
    stages = [stage(1, "Climate Change Act 2008 (Repeal) Bill", "2R"),
              stage(2, "Climate Change Act 2008 (Repeal) (No. 2) Bill", "2R")]
    assert link("Climate Change Act 2008 (Repeal) (No. 2) Bill: Second Reading", stages) == (2, "title_stage")
    assert link("Climate Change Act 2008 (Repeal) Bill: Second Reading", stages) == (1, "title_stage")


def test_the_numbered_tier_refuses_two_bills_that_differ_only_by_number():
    stages = [stage(1, "Water Act 2026", "2R"), stage(2, "Water (No. 3) Bill", "2R")]
    assert link("Water (No. 2) Bill: Second Reading", stages) == (None, "ambiguous")


def test_a_budget_resolution_number_is_not_a_bill_number():
    stages = [stage(1, "Finance Act 2026", "WAMR")]
    assert link("Budget Resolution No. 28: Capital gains tax", stages) == (None, "not_a_bill")


# --- +fuzzy: Hansard's own typos
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("bill, title", [
    ("Pension Schemes Act 2026", "Pensions Scheme Bill: Report Stage: New Clause 1"),
    ("Pension Schemes Act 2026", "Pensions Schemes Bill: Report Stage: New Clause 1"),
    ("Children's Wellbeing and Schools Bill", "Children's School and Wellbeing Bill: Report Stage: New Clause 1"),
    ("Post Office (Horizon System) Offences Act 2024", "Post Office (Horizon Systems) Offences Bill: Report Stage: New Clause 1"),
    ("Passenger Railway Services (Public Ownership) Act 2024",
     "Passenger Railway Services Bill (Public Ownership) Bill: Report Stage: New Clause 1"),
])  # fmt: skip
def test_the_same_words_in_another_order_or_number_are_the_same_bill(bill, title):
    assert link(title, [stage(1, bill, "RS")]) == (1, "title_stage+fuzzy")


@pytest.mark.parametrize("title", ["Pension Bill: Report Stage", "Pension Schemes Reform Bill: Report Stage",
                                   "Schemes Bill: Report Stage", "Pension Schemes (Scotland) Bill: Report Stage"])
def test_different_words_are_a_different_bill(title):
    assert link(title, [stage(1, "Pension Schemes Act 2026", "RS")]) == (None, "none")


def test_curly_and_straight_apostrophes_are_one_name():
    stages = [stage(1, "Children's Wellbeing and Schools Act 2026", "RS")]
    assert link("Children’s Wellbeing and Schools Bill: Report Stage: New Clause 1", stages) == (1, "title_stage")


# --- labels: none vs not_a_bill, deferred, the debate fallback
# ---------------------------------------------------------


@pytest.mark.parametrize("title", ["Opposition Day: Fuel Duty", "Budget Resolution No. 4: Income tax (dividend rates)",
                                   "Business of the House (25 May)", "Motion to sit in private",
                                   "Liaison Committee membership: Amendment (a)"])
def test_a_title_that_names_no_bill_is_not_a_bill(title):
    assert link(title, REMAINING) == (None, "not_a_bill")


def test_a_title_that_names_a_bill_not_sitting_is_a_real_miss():
    assert link("Water Bill: Second Reading", REMAINING) == (None, "none")


def test_a_deferred_division_is_not_a_bill():
    assert link("Armed Forces Bill: Report Stage", REMAINING, deferred=True) == (None, "not_a_bill")


def test_a_closure_motion_still_reaches_its_stage_through_its_debate():
    stages = [stage(1, "Housing Act 2026", "RS", debates=("d1",))]
    assert link("Closure motion", stages, debate_id="d1") == (1, "debate")
    assert link("Closure motion", stages, debate_id="d1", debate_link="time_window") == (1, "debate_weak")


def test_a_formal_stage_is_never_the_one_divided_on():
    stages = [stage(1, "Housing Act 2026", "1R", name="1st reading"), stage(2, "Housing Act 2026", "PM")]
    assert link("Housing Bill: Programme motion", stages) == (2, "title_stage")
    assert link("Housing Bill", stages) == (2, "title_date")


# --- the index is only an index
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_the_index_gives_the_same_answer_as_a_scan(seed):
    rng = random.Random(seed)
    names = ["Housing Act 2026", "Finance Act 2026", "Water Bill", "Pension Schemes Act 2026", "Health Bill"]
    abbrs = ["2R", "RS", "3R", "CS", "COTWH", "PM", "COLA"]
    days = [DAY + timedelta(days=i) for i in range(5)]
    stages = [stage(i, rng.choice(names), rng.choice(abbrs), house=rng.choice(["Commons", "Lords"]),
                    days=tuple(rng.sample(days, rng.randint(1, 3))), debates=(f"d{rng.randint(0, 4)}",))
              for i in range(1, 30)]
    titles = ["Housing Bill: Report Stage: New Clause 1", "Finance (No. 2) Bill Committee: Amendment 3",
              "Water Bill", "Pensions Scheme Bill: Third Reading", "Closure motion", "Health Bill: New Clause 2"]
    index = B.StageIndex(stages)
    for _ in range(40):
        d = division(rng.choice(titles), house=rng.choice(["Commons", "Lords"]), day=rng.choice(days),
                     debate_id=f"d{rng.randint(0, 4)}")
        assert B.link_division_stage(d, stages, index=index) == B.link_division_stage(d, stages)


# --- sittings: the numbered fallback
# ------------------------------------------------------------------------------------


def debate(did, heading, day=DAY, house="Commons"):
    return DebateInfo(did, day, house, heading, "Commons Chamber")


def test_a_sitting_of_a_renumbered_bill_finds_its_debate():
    forms = B.bill_name_forms({"title": "Finance Act 2026"})
    assert B.link_sitting(forms, "Commons", DAY, [debate("a", "Finance (No. 2) Bill")]) == ("a", "title_exact+numbered")


def test_a_sitting_prefers_an_exact_heading_to_a_numberless_one():
    forms = B.bill_name_forms({"title": "Finance Act 2026"})
    found = B.link_sitting(forms, "Commons", DAY, [debate("a", "Finance (No. 2) Bill"), debate("b", "Finance Bill")])
    assert found == ("b", "title_exact")
