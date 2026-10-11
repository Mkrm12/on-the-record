"""Database-generated fact items are graded deterministically — and everything else still goes to the judge."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_answers_e05", ROOT / "scripts" / "eval_answers.py")
EA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EA)


def item(item_id, category, facts, forbidden=()):
    return SimpleNamespace(id=item_id, category=category, expected_structured_facts=facts,
                           forbidden_claims=tuple(forbidden), nuggets=(), expected_behaviour="answer")


def answer(rendered, *, facts=True, clarify=None, abstention=None):
    return SimpleNamespace(rendered=rendered, structured_facts=("fact",) if facts else (), abstention=abstention,
                           facets={"clarify": clarify} if clarify else {}, claims=(), coverage_note="")


VOTE = {"division_id": 1, "member": "Jamie Stone", "lobby": "aye", "party_at_vote": "Liberal Democrat",
        "is_teller": False, "proxy_name": "Will Forster", "mnis_id": 9}


def test_a_vote_stated_exactly_with_its_proxy_is_correct():
    got = EA.structured_correct(item("db0071", "division_vote", VOTE),
                                answer("Jamie Stone (Liberal Democrat) voted Aye. Cast by proxy by Will Forster."))
    assert got is True


def test_a_vote_without_the_proxys_name_goes_to_the_judge():
    """The six dev answers the judge marked "missing vital" said "Cast by proxy." and no more."""
    got = EA.structured_correct(item("db0071", "division_vote", VOTE),
                                answer("Jamie Stone (Liberal Democrat) voted Aye. Cast by proxy."))
    assert got is False


@pytest.mark.parametrize("rendered", ["Jamie Stone (Liberal Democrat) voted No. Cast by proxy by Will Forster.",
                                      "Jamie Stone (Labour) voted Aye. Cast by proxy by Will Forster."])
def test_a_wrong_lobby_or_party_is_never_correct(rendered):
    assert EA.structured_correct(item("db0071", "division_vote", VOTE), answer(rendered)) is False


def test_a_tally_whose_lists_disagree_must_carry_the_caveat():
    facts = {"division_id": 2, "ayes": 77, "noes": 317, "ayes_listed": 76, "noes_listed": 317,
             "tally_reconciled": False, "must_cite": "official tellers' figures"}
    bare = "77 voted Aye and 317 voted No, as counted by the tellers."
    assert EA.structured_correct(item("db0041", "division_tally", facts), answer(bare)) is False
    caveated = bare + (" The name lists record one vote the tellers' count does not: 76 Ayes and 317 Noes are listed "
                       "against the official 77 and 317. The tellers' figures are the result (I-8).")
    assert EA.structured_correct(item("db0041", "division_tally", facts), answer(caveated)) is True


def test_a_result_is_stated_with_its_tally():
    facts = {"division_id": 3, "house": "Lords", "division_number": 2, "date": "2026-09-09", "result": "agreed",
             "ayes": 194, "noes": 138}
    good = "The Lords divided on 9 September 2026 (division 2) and the motion was agreed to, 194 votes to 138."
    assert EA.structured_correct(item("db0001", "division_result", facts), answer(good)) is True
    assert EA.structured_correct(item("db0001", "division_result", facts),
                                 answer(good.replace("194", "195"))) is False


@pytest.mark.parametrize("kw", [dict(clarify=[{"division_id": 1}]), dict(facts=False), dict(abstention="x")])
def test_a_clarification_an_unstructured_answer_or_a_decline_goes_to_the_judge(kw):
    rendered = "Jamie Stone (Liberal Democrat) voted Aye. Cast by proxy by Will Forster."
    assert EA.structured_correct(item("db0071", "division_vote", VOTE), answer(rendered, **kw)) is False


def test_only_database_generated_items_are_graded_this_way():
    assert EA.structured_correct(item("hum0071", "division_vote", VOTE),
                                 answer("Jamie Stone (Liberal Democrat) voted Aye. Cast by proxy by Will Forster.")) is False


def test_a_category_without_a_template_goes_to_the_judge():
    facts = {"debate_id": "x", "division_id": 1, "heading": "Health Bill", "result": "agreed"}
    assert EA.structured_correct(item("db0266", "multi_speech", facts), answer("The debate was Health Bill.")) is False


def test_an_opener_is_correct_when_the_answer_leads_with_them():
    facts = {"debate_id": "d", "heading": "Arrangement of Business", "mnis_id": 1, "opened_by": "Lord Kennedy of Southwark",
             "party": "Labour"}
    good = ("Lord Kennedy of Southwark (Lab) opened the debate on Arrangement of Business in the Lords on 2025-11-05. "
            "4 members spoke in all.")
    assert EA.structured_correct(item("db0226", "who_spoke", facts), answer(good, facts=False)) is True
    assert EA.structured_correct(item("db0226", "who_spoke", facts),
                                 answer("Baroness Smith (Lab) opened the debate on Arrangement of Business.",
                                        facts=False)) is False


def test_an_opener_answer_that_also_lists_the_roster_goes_to_the_judge():
    """Validated against the final run: the judge disputed two such answers. A deterministic grader must never be
    looser than the judge it replaces; the dispute itself is for the maintainer's calibration.
    """
    facts = {"opened_by": "Wes Streeting"}
    listed = ("Wes Streeting (Labour) opened the debate on Ukraine and Russia in the Commons on 2026-09-10. "
              "Speakers, in order: Wes Streeting, Mr Mark Francois, Alex Baker.")
    assert EA.structured_correct(item("db0261", "who_spoke", facts), answer(listed, facts=False)) is False
