"""The router: rules first, filters from the text, clarify when the question names nobody."""

from __future__ import annotations

from datetime import date

import pytest

from answer import router as R


def routed(question, category=None):
    return R.route(question, category)


# --- routes -----------------------------------------------------------------------------------------------


def test_the_route_names_are_the_ones_the_brief_lists():
    assert set(R.ROUTES) == {"structured", "set", "retrieval", "overview", "multi_hop", "timeline",
                             "clarify", "abstain"}  # fmt: skip


def test_an_unknown_route_cannot_be_constructed():
    """A typo'd route name would silently answer nothing."""
    with pytest.raises(ValueError, match="unknown route"):
        R.Decision("lookup")


@pytest.mark.parametrize("question", [
    "How many divisions were there in the Commons in September 2026?",
    "How did Diane Abbott vote on the Bill?",
    "What was the result of the division on 12 May 2026?",
    "Who was the Secretary of State for Scotland on 1 June 2026?",
    "When did the Football Governance Bill receive its second reading?",
])
def test_questions_with_one_exact_answer_go_to_structured(question):
    """These are SQL questions. Sending them to retrieval would produce a plausible sentence instead of the recorded
    figure, which is the whole failure the structured route exists to prevent.
    """
    assert routed(question).route == "structured"


@pytest.mark.parametrize("question", [
    "Who spoke in the debate on housing?",
    "Which members took part in the Westminster Hall debate?",
    "List the peers who contributed on 3 June 2026.",
    "Who opened the debate on the Budget?",
])
def test_questions_about_who_took_part_go_to_set(question):
    assert routed(question).route == "set"


@pytest.mark.parametrize("question", [
    "What were the main arguments about the visitor levy?",
    "Summarise what MPs said about fracking.",
    "What are the competing views on jury trials?",
    "Where do the parties stand on water ownership?",
])
def test_questions_about_the_shape_of_a_debate_go_to_overview(question):
    assert routed(question).route == "overview"


def test_a_timeline_question_is_its_own_route():
    assert routed("What happened to the Football Governance Bill?").route == "timeline"
    assert routed("Give me a timeline of the Budget debates.").route == "timeline"


def test_anything_else_falls_to_retrieval():
    """The default must be the route that searches text, not an abstention: a question the rules cannot characterise is
    usually an ordinary one.
    """
    assert routed("Was there any argument that households cannot change supplier?").route == "retrieval"
    assert routed("did the tories U-turn on fracking").route == "retrieval"


def test_an_empty_question_abstains():
    assert routed("   ").route == "abstain"


def test_a_gold_structured_category_is_never_trusted_over_the_words():
    """It pinned the router trusting the gold category — which production never has — so E3 measured a structured route
    that only the answer key could reach.
    """
    decision = routed("Tell me about the vote.", category="division_tally")
    assert decision.route == "retrieval" and decision.category is None


def test_the_gold_category_does_not_decide_any_other_route():
    """An evaluation that routed by gold category would be measuring the router against its own answer key."""
    decision = routed("What were the main arguments about housing?", category="multi_speech")
    assert decision.route == "overview", "the pattern decided this, not the category"


def test_every_decision_records_why_it_fired():
    """A misroute is only diagnosable if the rule that fired says so."""
    for question in ("How many divisions were there in May 2026?", "Who spoke about housing?",
                     "Summarise the debate on fracking.", "did the tories U-turn"):  # fmt: skip
        assert routed(question).why, question


def test_the_rules_alone_consult_no_model():
    assert routed("Who spoke in the debate?").by == "rules"


# --- the office-with-no-date case -------------------------------------------------------------------------


def test_an_office_with_no_date_asks_instead_of_guessing():
    """The corpus holds four different Prime Ministers — Boris Johnson in W3, Rishi Sunak and Keir Starmer inside W1,
    Andy Burnham in W0 — so this question names nobody. Picking the most recent holder would be a confident answer
    about a person the asker did not mean.
    """
    decision = routed("What did the Prime Minister say about housing?")
    assert decision.route == "clarify"
    assert decision.confidence == 0.0
    assert "no date" in decision.why
    assert decision.clarify


def test_an_office_with_a_date_is_answerable():
    decision = routed("What did the Prime Minister say about housing on 12 May 2026?")
    assert decision.route != "clarify"
    assert decision.filters.office == "prime minister"
    assert decision.filters.on_date == date(2026, 5, 12)


def test_an_office_with_a_year_is_enough_to_proceed():
    """A year narrows the corpus to one Parliament in most cases; it is not perfect, and it is not nothing."""
    assert routed("What did the Chancellor say about tax in 2024?").route != "clarify"


@pytest.mark.parametrize("office", ["the Prime Minister", "the Chancellor of the Exchequer",
                                   "the Home Secretary", "the Leader of the Opposition",
                                   "the Secretary of State for Scotland"])  # fmt: skip
def test_the_offices_that_change_hands_are_recognised(office):
    assert routed(f"What did {office} say?").filters.office is not None


# --- filters --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("question", "house"), [
    ("What did the Commons decide?", "Commons"),
    ("Which MPs voted against?", "Commons"),
    ("What did the Lords say?", "Lords"),
    ("Which peers spoke?", "Lords"),
    ("What was said about housing?", None),
])
def test_the_house_is_taken_from_the_text_or_left_unset(question, house):
    """Left unset, never defaulted. Defaulting to the Commons would answer a Lords question from the wrong chamber and
    never say so.
    """
    assert R.extract_filters(question).house == house


def test_a_named_section_implies_its_house():
    """"Grand Committee" is a Lords thing and "Public Bill Committee" a Commons one; the question often names the
    section and not the House.
    """
    assert R.extract_filters("What was said in Grand Committee?").house == "Lords"
    assert R.extract_filters("What happened in the Public Bill Committee?").house == "Commons"
    assert R.extract_filters("Who spoke in Westminster Hall?").section == "Westminster Hall"


@pytest.mark.parametrize(("question", "expected"), [
    ("on 2026-05-12", date(2026, 5, 12)),
    ("on 12 May 2026", date(2026, 5, 12)),
    ("on 12th May 2026", date(2026, 5, 12)),
    ("on May 12, 2026", date(2026, 5, 12)),
])
def test_an_exact_date_is_read_in_every_form_the_question_might_use(question, expected):
    assert R.extract_filters(f"What happened {question}?").on_date == expected


def test_a_month_becomes_a_range_not_a_day():
    filters = R.extract_filters("What was debated in May 2026?")
    assert filters.on_date is None
    assert filters.from_date == date(2026, 5, 1) and filters.to_date == date(2026, 5, 31)


def test_december_does_not_overflow_the_year():
    """The month-end calculation is the kind of thing that is wrong exactly once a year."""
    filters = R.extract_filters("What was debated in December 2025?")
    assert filters.to_date == date(2025, 12, 31)


def test_a_year_alone_becomes_the_whole_year():
    filters = R.extract_filters("What was said about fracking in 2024?")
    assert filters.from_date == date(2024, 1, 1) and filters.to_date == date(2024, 12, 31)


def test_an_impossible_date_is_no_date_rather_than_a_crash():
    assert R.extract_filters("What happened on 31 February 2026?").on_date is None


def test_the_narrower_expression_wins():
    """A question naming both a day and its year means the day."""
    filters = R.extract_filters("What happened on 12 May 2026, during 2026?")
    assert filters.on_date == date(2026, 5, 12)


def test_a_relative_expression_is_reported_and_not_resolved():
    """"Last month" must be resolved against the coverage table, not the wall clock: the newest sitting in the corpus
    is September 2026, so resolving against today would answer about a month the corpus may not hold.
    """
    filters = R.extract_filters("What was debated last month?")
    assert filters.relative == "last month"
    assert filters.on_date is None and filters.from_date is None


def test_filters_carry_only_what_was_found():
    """An empty filter must not reach the cache key as a null, or two identical questions would hash differently
    depending on which fields the extractor happened to touch.
    """
    assert R.extract_filters("What was said about housing?").as_dict() == {}
    found = R.extract_filters("What did the Lords say on 12 May 2026?").as_dict()
    assert found == {"house": "Lords", "on_date": "2026-05-12", "year": 2026}


def test_dates_serialise_as_iso_strings_for_the_cache_key():
    found = R.extract_filters("What happened in May 2026?").as_dict()
    assert found["from_date"] == "2026-05-01" and isinstance(found["from_date"], str)


# --- the LLM fallback -----------------------------------------------------------------------------------


class FakeRouterClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def chat(self, prompt, *, system="", max_tokens=0, temperature=0.0, json_object=False):
        self.calls += 1

        class R2:
            content = self.payload if isinstance(self.payload, str) else __import__("json").dumps(self.payload)
        if isinstance(self.payload, Exception):
            raise self.payload
        return R2()


def run(coro):
    from ingest.runtime import run as run_async

    return run_async(coro)


def test_the_model_is_not_consulted_when_a_rule_fired():
    """The whole point of rules first: a question a regex settles must not cost a call, a latency budget and a failure
    mode.
    """
    client = FakeRouterClient({"route": "overview"})
    decision = run(R.route_with_model("Who spoke in the debate on housing?", client))
    assert decision.route == "set" and client.calls == 0


def test_the_model_is_consulted_only_for_plain_retrieval():
    client = FakeRouterClient({"route": "overview", "why": "asks for the shape of a debate"})
    decision = run(R.route_with_model("Tell me about the fracking business.", client))
    assert client.calls == 1
    assert decision.route == "overview" and decision.by == "llm"
    assert decision.confidence < 1.0


def test_a_model_route_outside_the_list_is_ignored():
    client = FakeRouterClient({"route": "web_search"})
    assert run(R.route_with_model("Tell me about fracking.", client)).route == "retrieval"


def test_a_model_failure_leaves_the_rules_answer_standing():
    """A broken classifier must not be able to stop the system answering."""
    client = FakeRouterClient(RuntimeError("429"))
    decision = run(R.route_with_model("Tell me about fracking.", client))
    assert decision.route == "retrieval" and decision.by == "rules"


def test_unparseable_model_output_leaves_the_rules_answer_standing():
    client = FakeRouterClient("I think this is an overview question, probably.")
    assert run(R.route_with_model("Tell me about fracking.", client)).route == "retrieval"


def test_no_client_means_rules_only():
    assert run(R.route_with_model("Tell me about fracking.", None)).by == "rules"
