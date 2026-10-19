"""A question about a time the record does not cover is answered with what the record does cover."""

from __future__ import annotations

from datetime import date

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import router as R
from answer import scope as SC
from app import safety, sealed
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76",
            recipe="r", prompt_version="p", model_id="m")

# Two windows years apart, the shape of the real corpus: a date between them belongs to neither.
WINDOWS = (("WA", "2020-03-01", "2020-05-31"), ("WB", "2024-05-01", "2024-07-31"))


# --- the question's own time ----------------------------------------------------------------------------------------


def test_the_asked_span_is_the_day_the_range_or_the_year():
    assert SC.asked_span(R.Filters(on_date=date(2012, 3, 4))) == (date(2012, 3, 4), date(2012, 3, 4))
    assert SC.asked_span(R.Filters(from_date=date(2012, 1, 1), to_date=date(2012, 6, 30))) == \
        (date(2012, 1, 1), date(2012, 6, 30))
    assert SC.asked_span(R.Filters(year=2012)) == (date(2012, 1, 1), date(2012, 12, 31))
    assert SC.asked_span(R.Filters()) is None, "an undated question has no time to be outside"
    assert SC.asked_span({"on_date": "2012-03-04"}) == (date(2012, 3, 4), date(2012, 3, 4)), "a trace's filters too"


def test_an_open_ended_range_is_bounded_only_on_its_given_side():
    since = SC.asked_span(R.Filters(from_date=date(2012, 1, 1)))
    assert since[0] == date(2012, 1, 1) and since[1] >= date(2100, 1, 1)
    until = SC.asked_span(R.Filters(to_date=date(2012, 1, 1)))
    assert until[0] <= date(1900, 1, 1) and until[1] == date(2012, 1, 1)


def _periods(*extra):
    return [SC.Period(date.fromisoformat(s), date.fromisoformat(e)) for _, s, e in WINDOWS] + list(extra)


@pytest.mark.parametrize("span, relation", [
    ((date(2012, 3, 4), date(2012, 3, 4)), "outside"),  # before every window
    ((date(2022, 6, 1), date(2022, 6, 1)), "outside"),  # between two windows: the corpus is not a range
    ((date(2030, 1, 1), date(2030, 12, 31)), "outside"),  # after every window
    ((date(2020, 3, 4), date(2020, 3, 4)), "inside"),
    ((date(2020, 5, 31), date(2020, 5, 31)), "inside"),  # a window's last declared day
    ((date(2020, 6, 1), date(2020, 6, 1)), "outside"),  # the day after it
    ((date(2020, 1, 1), date(2020, 12, 31)), "partial"),  # "in 2020": a quarter of it is held
    ((date(2019, 1, 1), date(2025, 1, 1)), "partial"),  # spans both windows and the hole between
])
def test_a_span_is_inside_partly_inside_or_outside_the_windows(span, relation):
    assert SC.relate(span, _periods()) == relation


def test_the_sealed_window_counts_as_held_so_a_question_inside_it_is_never_called_outside():
    held_back = SC.Period(date(2026, 4, 1), date(2026, 7, 31), available=False)
    assert SC.relate((date(2026, 5, 5), date(2026, 5, 5)), _periods(held_back)) == "inside"
    assert SC.relate((date(2026, 8, 5), date(2026, 8, 5)), _periods(held_back)) == "outside"


# --- which answers the windows bound ----------------------------------------------------------------------------------


def test_every_structured_category_names_its_source_explicitly():
    """A category added later must say where its answers come from; a default would decide it silently."""
    assert set(R.STRUCTURED_CATEGORIES) <= set(SC.SOURCE), set(R.STRUCTURED_CATEGORIES) - set(SC.SOURCE)
    assert set(SC.SOURCE.values()) <= {SC.SITTINGS, SC.MEMBERS, SC.BILLS}


@pytest.mark.parametrize("category", ["division_result", "division_tally", "division_vote", "division_debate", "count"])
def test_the_sittings_categories_are_bounded_by_the_windows(category):
    assert SC.SOURCE[category] == SC.SITTINGS


@pytest.mark.parametrize("category", ["office_held_at_date", "bill_stage_date"])
def test_the_api_categories_are_not(category):
    assert SC.SOURCE[category] != SC.SITTINGS


def test_an_unknown_category_is_treated_as_the_sittings():
    """The safe direction: a true statement of scope, never an answer from a source nobody classified."""
    assert SC.bounded_by_windows("a_category_added_in_2030")


# --- what the reader is told
# --------------------------------------------------------------------------------------------


def _scope(span, *, houses=("Commons", "Lords")):
    held_back = SC.Period(date(2026, 4, 1), date(2026, 7, 31), available=False)
    periods = tuple(_periods(held_back))
    return SC.Scope(asked=span, relation=SC.relate(span, periods), periods=periods, houses=houses)


def test_the_statement_names_the_time_asked_and_every_period_held_in_order():
    text = _scope((date(2012, 3, 4), date(2012, 3, 4))).statement()
    assert text.startswith("The record does not cover 4 March 2012.")
    assert "1 March to 31 May 2020" in text and "1 May to 31 July 2024" in text
    assert text.index("2020") < text.index("2024"), "periods in date order"
    assert "Commons and Lords" in text
    assert sealed.SCOPE_STATEMENT in text, "the sealed window is named as not yet available, never as covered"
    assert "WA" not in text and "WB" not in text, "window ids are internal"


def test_a_range_and_a_year_are_named_as_asked():
    assert _scope((date(2012, 1, 1), date(2012, 12, 31))).statement().startswith("The record does not cover 2012.")
    assert _scope((date(2011, 2, 1), date(2012, 6, 30))).statement().startswith(
        "The record does not cover 1 February 2011 to 30 June 2012.")


def test_the_statement_says_what_the_record_is_made_of_and_what_it_leaves_out():
    """A decline names the kind of record as well as its dates."""
    text = _scope((date(2012, 3, 4), date(2012, 3, 4))).statement()
    assert "spoken proceedings" in text
    assert "written questions" in text.lower()


def test_the_left_out_sentence_matches_what_the_fetcher_actually_collects():
    """A fetcher that adds written questions or statements fails this until the sentence is changed with it."""
    import re
    from pathlib import Path

    fetcher = (Path(__file__).resolve().parents[1] / "src" / "ingest" / "corpus_fetch.py").read_text(encoding="utf-8")
    endpoints = set(re.findall(r'_URL = f"\{HANSARD\}/search/contributions/(\w+)\.json"', fetcher))
    assert endpoints == {"Spoken", "Corrections"}, endpoints
    assert SC.KIND == "spoken proceedings" and "Written questions" in SC.LEFT_OUT


def test_the_sections_are_named_from_the_coverage_rows_chambers_first():
    span = (date(2012, 3, 4), date(2012, 3, 4))
    scope = SC.Scope(asked=span, relation="outside", periods=tuple(_periods()), houses=("Commons", "Lords"),
                     sections=("Commons Chamber", "Westminster Hall", "Lords Chamber", "Grand Committee"))
    text = scope.statement()
    assert "Commons Chamber, Westminster Hall, Lords Chamber and Grand Committee" in text
    assert SC._section_order("A Section Added In 2030") > SC._section_order("Grand Committee")


@pytest.mark.parametrize("span, label", [
    ((date(2012, 3, 4), date(2012, 3, 4)), "later"),  # every period held is after it
    ((date(2030, 1, 1), date(2030, 1, 1)), "earlier"),  # every period held is before it
    ((date(2022, 6, 1), date(2022, 6, 1)), "other"),  # periods both sides
])
def test_found_text_is_labelled_by_when_it_was_said_relative_to_the_time_asked(span, label):
    assert label in _scope(span).remarks_label().lower()


# --- the structured route: a sittings question outside every window --------------------------------------------------


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE coverage, member_office CASCADE")
    for window_id, start, end in WINDOWS:
        scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                        "VALUES (%s, 'v2', %s, %s, 'dev')", (window_id, start, end))
        for house, section in (("Commons", "Commons Chamber"), ("Lords", "Lords Chamber")):
            scratch.execute("INSERT INTO coverage (window_id, house, section, role, declared_start, declared_end, "
                            "first_sitting, last_sitting, sitting_days) VALUES (%s, %s, %s, 'dev', %s, %s, %s, %s, 9)",
                            (window_id, house, section, start, end, start, end))
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (4005, 'A Long-Serving Member', 'Commons'), "
                    "(4006, 'A Later Member', 'Commons')")
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (4005, 'Chancellor of the Exchequer', 'chancellor of the exchequer', '2010-05-11', "
                    "'2016-07-13', 'mnis_biography', 0), (4006, 'Chancellor of the Exchequer', "
                    "'chancellor of the exchequer', '2020-02-13', NULL, 'mnis_biography', 0)")
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                    "ayes_listed, noes_listed) VALUES (77, 77, 'Commons', 1, '2020-03-04', 'Flooding', 200, 100, "
                    "'agreed', 200, 100)")
    yield scratch_dsn
    scratch.execute("TRUNCATE coverage, member_office CASCADE")


def ask(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()
    return run(go())


def test_a_division_question_dated_outside_every_window_is_answered_with_the_scope(record):
    """MR-04's form: the division exists, on the same day of another year. Answering it would be a harm; text search
    would answer from later speeches; "no division is recorded" would imply the House did not vote.
    """
    answer, notes = ask(record, "What was the result of the Commons division on Flooding on 4 March 2012?")
    assert answer is not None, notes
    assert answer.abstention == K.Abstention.OUTSIDE_WINDOW
    assert not answer.structured_facts
    assert "does not cover 4 March 2012" in answer.abstention_detail
    assert "1 March to 31 May 2020" in answer.abstention_detail
    assert answer.rendered == answer.abstention_detail or answer.abstention_detail in answer.rendered


def test_the_same_question_inside_a_window_is_answered_as_before(record):
    answer, notes = ask(record, "What was the result of the Commons division on Flooding on 4 March 2020?")
    assert answer is not None and answer.abstention is None, notes
    assert answer.structured_facts[0].value["division_id"] == 77


def test_a_date_between_two_windows_is_outside_too(record):
    answer, _ = ask(record, "What was the result of the Commons division on Flooding on 4 March 2022?")
    assert answer is not None and answer.abstention == K.Abstention.OUTSIDE_WINDOW


def test_an_office_question_outside_the_windows_is_answered_from_the_members_api(record):
    """The Members API holds posts the sittings do not: refusing them would be wrong in the other direction."""
    answer, notes = ask(record, "Who held the office of Chancellor of the Exchequer on 4 March 2012?")
    assert answer is not None and answer.abstention is None, notes
    assert answer.structured_facts[0].value["mnis_id"] == 4005
    assert "4 March 2012" in answer.coverage_note and "Members API" in answer.coverage_note, \
        "the reader is told the sittings for that date are not in the record and where the answer comes from"


def test_an_office_question_inside_a_window_carries_no_scope_note(record):
    answer, notes = ask(record, "Who held the office of Chancellor of the Exchequer on 4 March 2020?")
    assert answer is not None and answer.structured_facts[0].value["mnis_id"] == 4006, notes
    assert "does not cover" not in answer.coverage_note


def test_a_who_spoke_question_outside_every_window_is_the_scope_not_an_absence(record):
    """The set route: "no debate in the record matches" about 2012 reads as Parliament not debating it."""
    import psycopg

    from answer import routes as RT

    question = "Who spoke in the debate on Flooding on 4 March 2012?"

    async def go():
        conn = await psycopg.AsyncConnection.connect(record)
        try:
            return await RT.set_route(conn, question, K.Provenance(**PROV), filters=R.route(question).filters)
        finally:
            await conn.close()
    answer = run(go()).answer
    assert answer.abstention == K.Abstention.OUTSIDE_WINDOW
    assert "does not cover 4 March 2012" in answer.abstention_detail


# --- text found anyway: labelled, never presented as an account of the time asked
# ----------------------------------------


def _claimed(text="The member said those floods were the worst in memory."):
    citation = K.Citation(speech_id="s1", quote="those floods were the worst in memory", chunk_ids=(1,))
    claim = K.Claim(id="c1", kind="said", text=text, date=date(2024, 6, 1), citations=(citation,),
                    verification=K.Verification(quote_ok=True, citation_ok=True))
    return K.Answer(question="q", provenance=K.Provenance(**PROV), rendered=text, claims=(claim,),
                    retrieved_chunk_ids=frozenset({1}))


def test_found_text_about_an_uncovered_time_is_labelled_as_later_remarks():
    scope = _scope((date(2012, 3, 4), date(2012, 3, 4)))
    out = SC.apply(_claimed(), scope)
    assert out.claims, "the remarks are kept: they are what the record holds"
    assert out.rendered.startswith("The record does not cover 4 March 2012.")
    assert "later" in out.rendered.lower()
    assert out.facets.get("scope") == "outside"


def test_labelling_is_idempotent():
    scope = _scope((date(2012, 3, 4), date(2012, 3, 4)))
    once = SC.apply(_claimed(), scope)
    assert SC.apply(once, scope) == once


def test_a_decline_about_an_uncovered_time_becomes_the_scope_statement():
    scope = _scope((date(2012, 3, 4), date(2012, 3, 4)))
    declined = K.Answer(question="q", provenance=K.Provenance(**PROV), abstention=K.Abstention.NOT_IN_RECORD,
                        abstention_detail="Nothing in the record answers this.")
    out = SC.apply(declined, scope)
    assert out.abstention == K.Abstention.OUTSIDE_WINDOW
    assert out.abstention_detail.startswith("The record does not cover 4 March 2012.")


def test_a_question_inside_the_windows_is_left_alone():
    scope = _scope((date(2020, 3, 4), date(2020, 3, 4)))
    answer = _claimed()
    assert SC.apply(answer, scope) == answer


def test_the_statement_never_names_a_date_inside_the_sealed_window_as_covered():
    text = _scope((date(2012, 3, 4), date(2012, 3, 4))).statement()
    assert sealed.sealed_days_in(text) == [], text


# --- production's one exit: `ask` places every route's answer --------------------------------------------------------


class _EmptySearcher:
    """Finds nothing; records what it was asked."""

    def __init__(self):
        self.calls = 0

    async def search(self, *a, **k):
        import types

        self.calls += 1
        return types.SimpleNamespace(hits=[])


def test_ask_turns_a_text_question_about_an_uncovered_time_into_the_scope(record):
    """Text search finds nothing of 2012; the reader is told why, not "not in the record"."""
    import psycopg

    searcher = _EmptySearcher()
    runtime = PL.Runtime(searcher=searcher, client=None, frozen=PL.Frozen.load(), model_id="test-model")

    async def go():
        async with await psycopg.AsyncConnection.connect(record) as conn:
            return await PL.ask(runtime, conn, "What did MPs say about flooding on 4 March 2012?", use_cache=False,
                                store=False)

    asked = run(go())
    assert asked.answer.abstention is K.Abstention.OUTSIDE_WINDOW, asked.answer
    assert asked.answer.rendered.startswith("The record does not cover 4 March 2012.")


def test_ask_places_answers_at_its_one_exit():
    import inspect

    source = inspect.getsource(PL.ask)
    finish = source[source.index("async def finish("):source.index("# 2. the cache")]
    assert "SCOPE.apply(" in finish, "finish() places every answer it is handed, on every route"
    assert finish.index("SCOPE.apply(") < finish.index("SG.guard("), "placed before the sealed guard reads it"


# A party outside the windows is the Members API's ----------------------------------------------------


def _member_with_history(scratch, mnis_id, name):
    bio = {"biography": {
        "representations": [{"house": 1, "startDate": "2005-05-05", "endDate": None}],
        "houseMemberships": [{"house": 1, "startDate": "2005-05-05", "endDate": None}],
        "partyAffiliations": [{"name": "Labour", "house": 1, "startDate": "2005-05-05", "endDate": "2015-03-30"},
                              {"name": "Conservative", "house": 1, "startDate": "2015-03-30", "endDate": None}]}}
    import json

    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house, raw) VALUES (%s, %s, 'Commons', %s)",
                    (mnis_id, name, json.dumps(bio)))
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                    "VALUES (77, %s, 'aye', %s, 'Conservative')", (mnis_id, name))


def test_a_party_outside_the_windows_comes_from_the_members_api_history_and_says_so(record, scratch):
    """No division list can name anyone on a day outside every window; the history is the record's only statement of
    it, so it is the answer, with its basis and the source note.
    """
    _member_with_history(scratch, 4010, "Eve Switched")
    answer, notes = ask(record, "Which party was Eve Switched in on 4 March 2012?")
    assert answer is not None and answer.abstention is None, notes
    fact = answer.structured_facts[0].value
    assert (fact["mnis_id"], fact["party"]) == (4010, "Labour"), "the party that day, not the party in the 2020 list"
    assert "Members API" in fact["basis"] and "no division list" in fact["basis"]
    assert "sittings do not cover 4 March 2012" in answer.coverage_note


def test_a_party_outside_the_windows_on_a_day_with_no_seat_is_not_stated(record, scratch):
    _member_with_history(scratch, 4011, "Fay Unseated")
    answer, notes = ask(record, "Which party was Fay Unseated in on 4 March 2001?")
    assert answer is None or not answer.structured_facts, notes


def test_a_party_on_a_changeover_day_outside_the_windows_is_not_stated(record, scratch):
    """The history's own boundary day: whether the start or the end counts is not in the payload."""
    _member_with_history(scratch, 4012, "Gus Boundary")
    answer, notes = ask(record, "Which party was Gus Boundary in on 30 March 2015?")
    assert answer is None or not answer.structured_facts, notes


# Questions about the record itself are answered from its coverage, never from speeches ------------------------


@pytest.mark.parametrize("question", [
    "What does this record cover?",
    "Which periods does the record cover?",
    "Does the record include written questions?",
    "How far back does the database go?",
    "What is in this record?",
])
def test_a_question_about_the_record_itself_is_recognised(question):
    assert SC.asks_about_the_record(question)


@pytest.mark.parametrize("question", [
    "What is Diane Abbott's voting record on housing?",
    "Did the Minister put on record her thanks to the staff?",
    "What did MPs say about the record of the Government on flooding?",
    "Does the record show who voted against the Bill on 4 March 2020?",
])
def test_questions_about_anything_else_are_not(question):
    assert not SC.asks_about_the_record(question)


def _ask_full(dsn, question):
    import psycopg

    runtime = PL.Runtime(searcher=_EmptySearcher(), client=None, frozen=PL.Frozen.load(), model_id="test-model")

    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            return await PL.ask(runtime, conn, question, use_cache=False, store=False)
    return run(go())


def test_what_the_record_covers_is_its_periods_sections_and_what_it_leaves_out(record):
    asked = _ask_full(record, "What does this record cover?")
    text = asked.answer.rendered
    assert asked.answer.abstention is None and "1 March to 31 May 2020" in text and "1 May to 31 July 2024" in text
    assert "spoken proceedings" in text and "Written questions" in text and sealed.SCOPE_STATEMENT in text
    assert not asked.retrieval_hits, "nothing is searched: the coverage table is the answer"


def test_does_it_cover_a_date_outside_is_the_scope_and_inside_is_yes(record):
    outside = _ask_full(record, "Does the record cover 4 March 2012?")
    assert outside.answer.abstention == K.Abstention.OUTSIDE_WINDOW
    inside = _ask_full(record, "Does the record cover 4 March 2020?")
    assert inside.answer.abstention is None and inside.answer.rendered.startswith("Yes: the record covers 4 March 2020")


@pytest.mark.parametrize("question", [
    "Does the record include Diane Abbott's vote on the Climate Bill on 4 March 2020?",
    "Does the record include Lords debates on the Climate Bill?",
    "Does the record cover what the Prime Minister said about flooding?",
])
def test_a_question_about_something_in_the_record_is_not_taken_for_one_about_the_record(question):
    """Review pass 1 (S06): "does the record include <an item>" asks for the item, and the front door must not answer
    it with the coverage.
    """
    assert not SC.asks_about_the_record(question)


@pytest.mark.parametrize("question", [
    "Does the record cover 4 March 2012?", "Does the record include written questions?",
    "Does the record include Westminster Hall debates?", "When does the record start?",
])
def test_a_yes_no_question_about_a_time_or_a_kind_of_record_is_one(question):
    assert SC.asks_about_the_record(question)


# --- after the record's present (QA pass 9, High 1) -------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "Who held the office of Chancellor of the Exchequer on 1 January 2030?",
    "Who held the office of Chancellor of the Exchequer on 1 September 2024?",
    "Which party was Eve Switched in on 1 January 2030?",
])
def test_nothing_is_stated_for_a_day_after_the_record_s_present_whatever_its_source(record, scratch, question):
    """An open post or an open party affiliation is open as of the data, not forever: "John Healey held the office of
    Chancellor of the Exchequer on 1 January 2030" (QA pass 9). The record's present is its most recent sitting, so
    a day after it is answered with the scope and that day, from the Members API as from the sittings.
    """
    _member_with_history(scratch, 4010, "Eve Switched")
    answer, notes = ask(record, question)
    assert answer is not None and answer.abstention == K.Abstention.OUTSIDE_WINDOW, notes
    assert not answer.structured_facts
    assert "31 July 2024" in answer.abstention_detail, "the reader is told where the record ends"


def test_a_day_before_the_record_s_present_outside_the_windows_is_still_the_members_api_s(record):
    answer, notes = ask(record, "Who held the office of Chancellor of the Exchequer on 4 March 2023?")
    assert answer is not None and answer.abstention is None and answer.structured_facts, notes


# --- a span asked of the record itself (QA pass 9, Low 9)
# --------------------------------------------------------------


def test_does_it_cover_a_span_reads_the_whole_span():
    """"Does the record cover 1 January 2026 to 31 July 2026?" was answered "Yes: the record covers 1 January 2026":
    the router reads one date, so the meta answer reads every date the question names.
    """
    assert SC.span_named("Does the record cover 1 January 2026 to 31 July 2026?") == (date(2026, 1, 1),
                                                                                         date(2026, 7, 31))
    assert SC.span_named("Does the record cover 4 March 2020?") is None, "one date is the router's span"


def test_a_span_across_a_gap_between_windows_is_covered_in_part(record):
    inside = _ask_full(record, "Does the record cover 1 March 2020 to 31 May 2020?")
    assert inside.answer.rendered.startswith("Yes: the record covers 1 March 2020 to 31 May 2020"), inside.answer.rendered
    part = _ask_full(record, "Does the record cover 1 April 2020 to 31 July 2024?")
    assert part.answer.rendered.startswith("The record covers part of"), part.answer.rendered


def test_a_span_the_sealed_window_alone_holds_is_never_yes(record):
    """The window not yet available holds no day a reader can be given: it never makes a span covered."""
    asked = _ask_full(record, "Does the record cover 1 March 2026 to 31 July 2026?")
    assert not asked.answer.rendered.startswith(("Yes", "The record covers part")), asked.answer.rendered


def test_after_the_record_s_present_the_words_fit_every_source(record):
    """The sentence covers a post, a party and a bill's stage alike: no source's words in another's answer."""
    answer, _ = ask(record, "Who held the office of Chancellor of the Exchequer on 1 January 2030?")
    assert "states nothing about a later day such as 1 January 2030" in answer.abstention_detail
    assert "party" not in answer.abstention_detail and "post " not in answer.abstention_detail, answer.abstention_detail
