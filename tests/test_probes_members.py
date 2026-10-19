"""ST-party_at_date: each member's party on a day they voted, asked by the name the list printed."""

from __future__ import annotations

from datetime import date

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import divisions as O
from eval.probes import members as PM

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip


def division(did, day, votes):
    return O.DivisionTruth(did, "Commons", day, 1, "t", 2, 1, 2, 1, 2, 1, True, tuple(votes))


VOTES = [O.Vote(11, "aye", False, None, "Labour", "Alan Mak"), O.Vote(12, "no", False, None, "Conservative", "Ann Member"),
         O.Vote(13, "aye", True, None, "Labour", "A Teller"), O.Vote(14, "no_vote", False, None, "Labour", "Not Voting")]
TRUTHS = [division(1, date(2026, 9, 7), VOTES), division(2, date(2026, 9, 8), VOTES[:2])]
TRIALS = {t.subject["mnis_id"]: t for t in PM.party_trials(TRUTHS) if t.form == "canonical"}  # QA pass 5's relational form aside
V = PM.verdict_for(TRUTHS)


def party(mid=11, name="Labour"):
    return StructuredFact(template="party_at_date", params={}, value={"mnis_id": mid, "party": name, "date": "x"})


def answer(*facts, abstention=None):
    return Answer(question="q?", provenance=PROV, structured_facts=tuple(facts), abstention=abstention,
                  rendered="An answer." if facts else "")  # fmt: skip


def test_one_trial_per_member_who_voted_on_a_day_they_voted_asked_by_their_printed_name():
    assert set(TRIALS) == {11, 12, 13}, "voters and tellers; a member who cast no vote has no party on record that day"
    t = TRIALS[11]
    assert t.question.startswith("Which party was Alan Mak in on ") and t.question.endswith(" 2026?")
    assert t.subject["party"] == "Labour" and t.cls == "ST-party_at_date"


def test_the_listed_party_is_a_pass_and_another_party_or_member_a_harm():
    t = TRIALS[11]
    assert V(t, answer(party()))[0] == "pass"
    assert V(t, answer(party(name="Conservative")))[0] == "harm"
    assert V(t, answer(party(mid=12)))[0] == "harm"
    assert V(t, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "declined"
    assert V(t, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "harm", "they sat that day; an absence is false"


def test_the_day_chosen_does_not_depend_on_the_order_of_the_truths():
    assert {k: v.subject["date"] for k, v in TRIALS.items()} == {
        k: v.subject["date"] for k, v in {t.subject["mnis_id"]: t for t in PM.party_trials(list(reversed(TRUTHS)))}.items()}
