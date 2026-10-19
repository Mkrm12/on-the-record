"""Ties are probed, and judged against the payload's own words for the tie."""

from __future__ import annotations

from datetime import date

from answer.contract import Answer, Provenance, StructuredFact
from eval.oracles import divisions as O
from eval.probes import core
from eval.probes import divisions as PD

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip


def commons_payload(description="Tied vote: casting vote with the Ayes"):
    return {"DivisionId": 2215, "Date": "2025-12-09T00:00:00", "Number": 386, "IsDeferred": False,
            "Title": "UK-EU customs union (duty to negotiate): Ten Minute Rule Motion", "AyeCount": 100, "NoCount": 100,
            "Ayes": [], "Noes": [], "AyeTellers": [], "NoTellers": [], "FriendlyDescription": description}


def test_the_oracle_keeps_the_payloads_words_for_a_tie_and_reads_the_casting_vote_from_them():
    truth = O.parse_commons(commons_payload())
    assert truth.result == "tie" and truth.described == "Tied vote: casting vote with the Ayes"
    assert truth.casting == "agreed"
    assert O.parse_commons(commons_payload("Tied vote: casting vote with the Noes")).casting == "negatived"
    assert O.parse_commons(commons_payload("")).casting is None


def _trial(truth):
    return core.Trial(trial_id="t", cls="ST-division_result", category="division_result", question="q?",
                      form="canonical", subject={"division_id": truth.division_id})


def _answer(truth, result, text):
    value = {"division_id": truth.division_id, "house": truth.house, "date": truth.date.isoformat(), "result": result,
             "ayes": truth.ayes, "noes": truth.noes, "division_number": truth.number, "title": truth.title}
    return Answer(question="q?", provenance=PROV, rendered=text,
                  structured_facts=(StructuredFact(template="division_result", value=value),))


def test_a_tie_said_to_be_tied_with_the_records_words_passes():
    truth = O.parse_commons(commons_payload())
    verdict = PD.verdict_for([truth])
    text = ("The Commons divided on 9 December 2025 (division 386) and the vote was tied, 100 votes to 100. The record "
            "describes the division as: “Tied vote: casting vote with the Ayes”.")
    assert verdict(_trial(truth), _answer(truth, "tied", text))[0] == "pass"


def test_a_tie_stated_as_an_outcome_the_records_words_do_not_give_is_a_harm():
    truth = O.parse_commons(commons_payload())
    verdict = PD.verdict_for([truth])
    assert verdict(_trial(truth), _answer(truth, "negatived", "The motion was rejected, 100 to 100."))[0] == "harm"
    assert verdict(_trial(truth), _answer(truth, "agreed", "agreed, 100 to 100. Tied vote: casting vote with the Ayes"))[
        0] == "pass", "the record's own words give the casting vote with the Ayes"


def test_a_tie_without_the_records_words_on_it_is_incomplete():
    truth = O.parse_commons(commons_payload())
    verdict = PD.verdict_for([truth])
    assert verdict(_trial(truth), _answer(truth, "tied", "The vote was tied, 100 votes to 100."))[0] == "incomplete"


def test_a_lords_tie_with_no_words_on_it_may_only_be_called_tied():
    truth = O.DivisionTruth(division_id=1_003_408, house="Lords", date=date(2025, 10, 21), number=5,
                            title="Public Authorities (Fraud, Error and Recovery) Bill", ayes=138, noes=138,
                            ayes_listed=0, noes_listed=0, ayes_named=0, noes_named=0, tally_reconciled=None, votes=())
    verdict = PD.verdict_for([truth])
    assert verdict(_trial(truth), _answer(truth, "tied", "The vote was tied, 138 votes to 138."))[0] == "pass"
    assert verdict(_trial(truth), _answer(truth, "agreed", "The amendment was agreed to."))[0] == "harm"


def test_a_tie_is_judged_in_the_routes_own_word_and_never_as_an_outcome():
    """The route can only say "tied" (the database's `division_result_consistent_ck` admits no other word for a tie),
    so the verdict judges that word, in the route's own case; "tie" is the oracle's word and no answer of the
    route's (QA pass 7, L7: widening the verdict to "tie" fitted fixtures written in the oracle's vocabulary, so
    they now use the route's). Any outcome but the record's own casting vote is a harm.
    """
    truth = O.DivisionTruth(division_id=1_003_408, house="Lords", date=date(2025, 10, 21), number=5,
                            title="Public Authorities (Fraud, Error and Recovery) Bill", ayes=138, noes=138,
                            ayes_listed=0, noes_listed=0, ayes_named=0, noes_named=0, tally_reconciled=None, votes=())
    verdict = PD.verdict_for([truth])
    assert verdict(_trial(truth), _answer(truth, "tied", "The vote was tied, 138 votes to 138."))[0] == "pass"
    for word in ("tie", "agreed", "negatived", "carried"):
        assert verdict(_trial(truth), _answer(truth, word, "The vote was tied, 138 votes to 138."))[0] == "harm", word
