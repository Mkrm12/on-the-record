"""Retrieval is scoped by the question's own date, and a House word alone never scopes it."""

from __future__ import annotations

from datetime import date

import pytest

from answer import pipeline as PL
from answer import router as RO


def scope_of(question: str):
    return PL.retrieval_scope(RO.extract_filters(question))


def test_a_dated_question_is_scoped_to_that_day():
    scope = scope_of("What did the Minister say about housing on 12 May 2026?")
    assert scope.date_from == scope.date_to == date(2026, 5, 12)


def test_a_date_range_scopes_to_the_range():
    scope = scope_of("What was said about ferries between 1 May 2024 and 31 July 2024?")
    assert (scope.date_from, scope.date_to) == (date(2024, 5, 1), date(2024, 7, 31))


def test_a_dated_question_naming_a_house_is_scoped_to_both():
    scope = scope_of("What was debated when the Lords divided on 5 November 2025?")
    assert scope.houses == ("Lords",) and scope.date_from == date(2025, 11, 5)


@pytest.mark.parametrize("question", [
    "What did MPs say about the Lords amendments to the Bill?",
    "What did the Lords say about housing?",
    "How did the peers vote on assisted dying?",
    "What was debated in Westminster Hall about potholes?",
])
def test_a_house_or_section_alone_never_scopes(question):
    """Both losses in the measurement were House-only scopes: a House word is often the TOPIC of the question."""
    assert not scope_of(question)


def test_an_undated_question_is_not_scoped():
    assert not scope_of("What has been said about the two-child benefit cap?")


def test_a_bare_year_scopes_to_that_calendar_year():
    scope = PL.retrieval_scope({"year": 2025})
    assert (scope.date_from, scope.date_to) == (date(2025, 1, 1), date(2025, 12, 31))


def test_the_mapping_accepts_the_filters_dict_the_trace_stores():
    scope = PL.to_search_filters({"on_date": "2026-05-12", "house": "Commons"})
    assert scope.date_from == date(2026, 5, 12) and scope.houses == ("Commons",)


def test_no_filters_map_to_an_empty_scope():
    assert not PL.to_search_filters(None)
    assert not PL.to_search_filters({})


def test_the_widening_threshold_is_the_measured_one():
    assert PL.WIDEN_BELOW == 3


# A call the budget cannot price is never free ----------------------------------------------------------


class Unpriceable:
    total_tokens = 1_000

    def cost_usd(self):
        raise KeyError("no rate for this model")


def test_an_unpriceable_call_is_counted_and_charged_not_ignored():
    """`Budget.record` used to `pass` on a pricing failure, so the call counted as $0 and the spend cap was blind to
    it. It is now charged at a deliberately pessimistic rate and counted.
    """
    from answer import generate as G

    budget = G.Budget()
    budget.record(Unpriceable())
    assert budget.unpriced_calls == 1
    assert budget.spent_usd > 0, "an unpriceable call must not look free to the spend cap"
    assert budget.calls == 1


def test_the_pessimistic_rate_is_above_every_real_rate():
    """Pessimistic means it errs towards stopping early, never towards overspending blind."""
    from answer import generate as G
    from goldset.llm import RATES_USD_PER_MTOK

    highest = max(max(rate.values()) for rate in RATES_USD_PER_MTOK.values()) / 1_000_000
    assert G._PESSIMISTIC_USD_PER_TOKEN > highest
