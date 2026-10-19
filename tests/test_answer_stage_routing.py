"""A question about when a bill reached a stage, routed to the stage fact whatever the stage is called."""

from __future__ import annotations

import pytest

from answer import router as R


@pytest.mark.parametrize("stage", [
    "Programme motion", "Money resolution", "Consideration of Commons amendments and / or reasons",
    "Ways and Means resolution", "Carry-over motion", "Committee of the whole House", "Committee negatived",
    "Instruction", "Allocation of time motion", "Bill reintroduced", "Committal (to a Select Committee)",
])
def test_a_stage_question_routes_to_the_stage_fact_whatever_the_stage_is_called(stage):
    question = f"When did the High Speed Rail (West Midlands-Crewe) Act 2021 have its {stage} in the Commons?"
    decision = R.route(question)
    assert (decision.route, decision.category) == ("structured", "bill_stage_date"), decision.why


@pytest.mark.parametrize("question", [
    "When did the Climate Bill reach its Money resolution?",
    "When did the Climate Bill receive its Programme motion?",
    "When did the Climate Bill go through its Committee of the whole House?",
    "When did the Climate Bill have its 2nd reading?",  # as before
])
def test_the_shapes_of_a_stage_question(question):
    assert R.route(question).category == "bill_stage_date"


@pytest.mark.parametrize("question", [
    "What did MPs say when the Climate Bill was debated?",
    "When did the Prime Minister speak about the Climate Bill?",
    "Who spoke in the debate on the Climate Bill on 4 March 2020?",
])
def test_questions_that_are_not_about_a_stage_do_not_route_to_it(question):
    assert R.route(question).category != "bill_stage_date"


def test_a_second_question_after_the_stage_name_is_still_read_by_the_why_guard():
    assert R.route("When did the Climate Bill have its 2nd reading and why was it delayed?").category != \
        "bill_stage_date"


# --- a year inside a title is a name, not the question's time --------


@pytest.mark.parametrize("question", [
    "When did the High Speed Rail (West Midlands-Crewe) Act 2021 have its Money resolution in the Commons?",
    "When did the Finance (No. 2) Act 2023 receive Royal Assent?",
    "What did MPs say about the Draft Electricity Supply (Amendment) Regulations 2024?",
])
def test_a_year_in_a_legislations_title_is_not_the_questions_time(question):
    filters = R.route(question).filters
    assert (filters.year, filters.on_date, filters.from_date) == (None, None, None), filters


def test_the_questions_own_time_still_wins_beside_a_titled_year():
    assert R.route("What did MPs say about the Online Safety Act 2023 in 2024?").filters.year == 2024
    assert R.route("What was the result of the division on the Draft Electricity Supply (Amendment) Regulations 2024 "
                   "on 4 March 2020?").filters.on_date.isoformat() == "2020-03-04"


def test_a_year_on_its_own_is_still_the_questions_time():
    assert R.route("What did MPs say about flooding in 2012?").filters.year == 2012


# --- the title rule is the weakest signal, so a miss falls to the next shape (QA pass 9, Med 4)
# --------------------------

DESCRIBED_BY_AN_ACT = [
    "Who was the Conservative MP who opened the debate on the Pension Schemes Bill on 3 March 2026?",
    "Who was the MP that spoke first in the debate on the Pension Schemes Bill on 19 March 2026?",
    "Who was the Bishop who spoke in the debate on assisted dying on 4 March 2026?",
]


@pytest.mark.parametrize("question", DESCRIBED_BY_AN_ACT)
def test_without_the_title_rule_a_person_described_by_an_act_is_the_set_route_s(question):
    """"Who was the <Capitalised>..." takes these as an office; the record holds no such office, and the structured
    miss had fallen to text search, never back to the exact SQL of the set route (23 of 390 questions moved).
    """
    from answer import router as R

    assert R.route(question).category == "office_held_at_date"
    assert R.route(question, titles=False).route == "set"


@pytest.mark.parametrize("question", DESCRIBED_BY_AN_ACT)
def test_after_a_structured_miss_on_the_title_rule_the_next_shape_is_taken(question):
    from answer import pipeline as PL
    from answer import router as R

    following = PL.after_structured_miss(question, R.route(question))
    assert following is not None and following.route == "set"


def test_a_structured_miss_on_any_other_shape_has_no_second_route():
    from answer import pipeline as PL
    from answer import router as R

    question = "Who held the office of Chancellor of the Exchequer on 4 March 2020?"
    assert PL.after_structured_miss(question, R.route(question)) is None
    title = "Who was the Chancellor of the Exchequer on 4 March 2020?"
    assert PL.after_structured_miss(title, R.route(title)) is None, "no other shape: text search, as before"


def test_after_a_structured_miss_only_a_route_still_dispatched_is_taken(monkeypatch):
    from answer import pipeline as PL
    from answer import router as R

    question = DESCRIBED_BY_AN_ACT[0]
    decision = R.route(question)
    for route in ("clarify", "abstain", "retrieval"):
        monkeypatch.setattr(PL.RO, "route", lambda q, category=None, titles=True, _r=route: R.Decision(_r, why="x"))
        assert PL.after_structured_miss(question, decision) is None, route
