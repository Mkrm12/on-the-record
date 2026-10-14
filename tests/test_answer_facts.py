"""The structured-fact engine. Pure logic here; the live scoring is `scripts/score_facts.py`."""

from __future__ import annotations

import pytest

from answer import facts as F


def division(house="Commons", ayes=100, noes=90, ayes_listed=102, noes_listed=92, **kw):
    row = {"id": 1, "house": house, "number": 3, "division_date": "2026-05-12", "title": "t",
           "motion_text": "m", "result": "agreed", "ayes": ayes, "noes": noes,
           "ayes_listed": ayes_listed, "noes_listed": noes_listed, "no_vote_count": 0,
           "is_deferred": False, "debate_id": "d1"}  # fmt: skip
    row.update(kw)
    return row


def test_the_route_states_the_stored_reconciliation_and_derives_none_of_its_own():
    """196 and 267 listed with the tellers against 195 and 266. The old overhang rule said reconciled; the names,
    tellers apart, are 194 and 265, one short on each side.
    """
    row = division(ayes=195, noes=266, ayes_listed=196, noes_listed=267, ayes_named=194, noes_named=265,
                   tally_reconciled=False)
    assert F._reconciled(row) is False
    assert F._reconciled(division(tally_reconciled=True)) is True
    assert F._reconciled(division(ayes_listed=None, tally_reconciled=None)) is None
    assert not hasattr(F, "TELLER_ALLOWANCE")


def test_a_discrepancy_produces_a_caveat_naming_the_names_and_the_figures():
    caveat = F._tally_caveat(division(ayes=77, noes=317, ayes_listed=78, noes_listed=319, ayes_named=76,
                                      noes_named=317, tally_reconciled=False))
    assert "do not reconcile" in caveat and "I-8" in caveat
    assert "76 Ayes and 317 Noes, tellers apart" in caveat and "77 and 317" in caveat
    assert "tellers' figures are the result" in caveat


def test_a_lords_caveat_counts_the_tellers_in():
    caveat = F._tally_caveat(division(house="Lords", ayes=194, noes=138, ayes_named=193, noes_named=138,
                                      tally_reconciled=False))
    assert "193 Contents and 138 Not-Contents, tellers included" in caveat and "194 and 138" in caveat


def test_a_discrepancy_without_named_counts_still_says_so_and_invents_no_figure():
    caveat = F._tally_caveat(division(ayes=77, noes=317, tally_reconciled=False))
    assert "do not reconcile" in caveat and "77 / 317" in caveat and "tellers apart" not in caveat


def test_a_reconciled_division_carries_no_caveat():
    assert F._tally_caveat(division(ayes=11, noes=182, ayes_named=11, noes_named=182, tally_reconciled=True)) == ""


def test_unrecorded_lists_say_only_the_tellers_can_be_cited():
    assert "only the tellers' figures" in F._tally_caveat(division(ayes_listed=None, noes_listed=None,
                                                                   tally_reconciled=None))


def test_an_unknown_reconciliation_never_claims_the_lists_are_missing_when_they_are_not():
    """Found by the full suite: a row with name lists but no stored flag was told "No name lists are published", which
    is false. Unknown is not absent, and with no known discrepancy there is nothing to warn of.
    """
    assert F._tally_caveat(division(tally_reconciled=None)) == ""
    assert "only the tellers' figures" in F._tally_caveat(division(ayes_listed=None, tally_reconciled=None))


def test_unrecorded_figures_are_said_to_be_unrecorded():
    assert F._tally_caveat(division(ayes=None, tally_reconciled=None)) == (
        "The tellers' figures are not recorded for this division.")


def test_both_routes_word_the_caveat_identically():
    """Two modules describing one discrepancy differently would let a division read as sound in one answer and unsound
    in another.
    """
    from answer import graph as G

    for row in (division(ayes=77, noes=317, ayes_named=76, noes_named=317, tally_reconciled=False),
                division(tally_reconciled=True), division(ayes_listed=None, noes_listed=None, tally_reconciled=None),
                division(ayes=None, tally_reconciled=None)):
        assert G._division_caveat(row) == F._tally_caveat(row)


def test_render_never_drops_the_caveat():
    """The caveat is the difference between a true answer and a misleading one, so it is not optional."""
    fact = F.Fact("division_tally", {}, value={"ayes": 77, "noes": 317},
                  caveat="The published name lists do not reconcile (I-8).")  # fmt: skip
    rendered = fact.render()
    assert "77" in rendered and "do not reconcile" in rendered


# --- clarify is a third outcome -------------------------------------------------------------------------


def test_clarify_is_neither_an_answer_nor_a_miss():
    """Ambiguity returns clarify, never a guess. A caller must distinguish 'ask' from 'abstain'."""
    fact = F.Fact("office_held_at_date", {}, clarify=[{"holder": "A"}, {"holder": "B"}])
    assert fact.needs_clarification is True
    assert fact.ok is False
    assert "could mean 2 different things" in fact.render()


def test_a_miss_is_not_a_clarify():
    fact = F.Fact("division_result", {}, missing="no division with id 9")
    assert fact.ok is False and fact.needs_clarification is False
    assert fact.render() == "no division with id 9"


# --- routing ---------------------------------------------------------------------------------------------


def test_every_gold_structured_fact_category_has_a_template():
    """A category with no template is a category nothing can answer. Asserted rather than discovered when a question
    arrives.
    """
    from goldset.schema import STRUCTURED_FACT

    missing = sorted(set(STRUCTURED_FACT) - set(F.DISPATCH))
    assert missing == [], f"no template answers {missing}"
    assert set(F.DISPATCH) == set(F.TEMPLATES)


def test_an_unknown_category_raises_rather_than_returning_an_empty_fact():
    """A silent empty would score as a data miss and send the reader looking in the database."""
    import asyncio

    with pytest.raises(F.FactError, match="no template answers"):
        asyncio.run(F.answer(None, "no_such_category", {}))


def test_a_count_item_with_no_routable_parameters_raises():
    """`count` is three templates wearing one category. Each extra shape was found by this raising."""
    import asyncio

    with pytest.raises(F.FactError, match="month .*date.*mnis_id|must carry"):
        asyncio.run(F.count_any(None, house="Commons"))


@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"house": "Commons", "month": "2026-09-01"}, "count_divisions"),
    ({"house": "Commons", "on_date": "2026-09-08"}, "count_debates"),
    ({"mnis_id": 1940, "from_date": "2024-05-01", "to_date": "2024-07-31"}, "count_contributions"),
])
def test_count_routes_on_the_parameters_the_item_carries(kwargs, expected, monkeypatch):
    import asyncio

    called = {}

    async def spy(name):
        async def inner(*args, **kw):
            called["name"] = name
            return F.Fact("count", {})
        return inner

    for name in ("count_divisions", "count_debates", "count_contributions"):
        monkeypatch.setattr(F, name, asyncio.run(spy(name)))
    asyncio.run(F.count_any(None, **kwargs))
    assert called["name"] == expected


# --- date handling ----------------------------------------------------------------------------------------


def test_dates_are_accepted_as_strings_or_dates():
    from datetime import date

    assert F._as_date("2026-05-12") == date(2026, 5, 12)
    assert F._as_date(date(2026, 5, 12)) == date(2026, 5, 12)
    assert F._as_date("2026-05-12 00:00:00+00") == date(2026, 5, 12), "a timestamp truncates to its date"


def test_iso_renders_none_as_none_not_as_the_string():
    assert F._iso(None) is None
    assert F._iso("2026-05-12 00:00:00") == "2026-05-12"
