"""MR-08: every debate the Chair opens, asked "who opened it?"."""

from __future__ import annotations

from datetime import date

from answer.contract import Abstention, Answer, Provenance
from eval.oracles import openers as OO
from eval.probes import core
from eval.probes import openers as PO

PROV = Provenance(route="set", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip


def debate(did="D1", heading="Housing", day=date(2026, 9, 8), first="Madam Deputy Speaker (Caroline Nokes)"):
    return OO.DebateTruth(did, "Commons", day, heading, "Commons Chamber", (
        OO.Contribution(1, 4048, first, "Order."),
        OO.Contribution(2, 5001, "The Minister for Housing (Ann Minister)", "I beg to move..."),
        OO.Contribution(3, 5002, "Jim Shannon (Strangford) (DUP)", "Words.")))


def answer(opener=None, clarify=None, abstention=None):
    facets = {}
    if opener:
        facets["opener"] = opener
    if clarify:
        facets["clarify"] = clarify
    return Answer(question="q?", provenance=PROV, facets=facets, abstention=abstention,
                  rendered="An answer." if opener else "")  # fmt: skip


def op(mnis, name="X", did="D1"):
    return {"speech_id": "s", "mnis_id": mnis, "speaker_name": name, "debate_id": did}


TRIALS = PO.mr08_trials([debate(), debate("D2", "Health", first="Jim Shannon (Strangford) (DUP)")])
V = PO.verdict_for([debate(), debate("D2", "Health")])
T = TRIALS[0]


def test_only_debates_the_chair_opens_are_asked_about_and_the_question_is_how_people_ask_it():
    assert [t.subject["debate_id"] for t in TRIALS] == ["D1"]
    assert T.question == "Who opened the debate on Housing in the Commons on 8 September 2026?"
    assert (T.cls, T.route, T.subject["opener"], T.subject["presiding"]) == ("MR-08", "set", 5001, [4048])


def test_naming_the_record_s_first_member_is_a_pass():
    assert V(T, answer(op(5001)))[0] == "pass"


def test_naming_the_chair_is_a_harm_by_its_member_or_by_its_printed_form():
    assert V(T, answer(op(4048)))[0] == "harm"
    assert V(T, answer(op(None, "Madam Deputy Speaker")))[0] == "harm"


def test_another_non_presiding_member_is_a_mismatch_for_triage_never_a_pass():
    out = core.Outcome(T, *V(T, answer(op(5002))))
    assert out.outcome == "mismatch" and not out.acceptable


def test_another_debate_answered_is_a_harm_unless_the_heading_is_shared_that_day():
    assert V(T, answer(op(5001, did="D9")))[0] == "harm"
    shared = PO.mr08_trials([debate(), debate("D3")])[0]
    assert shared.ambiguous and V(shared, answer(op(5001, did="D3")))[0] == "substituted"


def test_a_question_back_or_a_decline():
    assert V(T, answer(clarify=[{"debate_id": "D1"}, {"debate_id": "D5"}]))[0] == "clarified"
    assert V(T, answer(clarify=[{"debate_id": "D5"}]))[0] == "declined"
    assert V(T, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "declined"
    assert V(T, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "harm", "a member's speech opens it"


def test_no_trial_inside_the_sealed_window():
    import pytest

    with pytest.raises(core.ProbeError):
        PO.mr08_trials([debate(day=date(2026, 5, 20))])


def test_a_debate_trial_is_its_own_cluster():
    """Found on the first MR-08 run: every debate trial fell into one cluster, so the per-subject bound meant nothing.
    """
    assert T.cluster == ("debate", "D1")


def test_a_member_who_opens_and_later_presides_is_still_the_opener():
    """Election of Speaker, 9 July 2024 (MR-08's triage): Sir Lindsay Hoyle spoke first as a candidate, and presided
    only after his election. The named speech is judged, not everything its member did in the debate.
    """
    election = OO.DebateTruth("E1", "Commons", date(2024, 7, 9), "Election of Speaker", "Commons Chamber", (
        OO.Contribution(1, 345, "Sir Edward Leigh (in the Chair)", "Under the terms of Standing Order No. 1A..."),
        OO.Contribution(3, 467, "Sir Lindsay Hoyle (Chorley) (Ind)", "First, I thank my constituents..."),
        OO.Contribution(9, 467, "Mr Speaker-Elect", "I thank the House...")))
    (t,) = PO.mr08_trials([election])
    v = PO.verdict_for([election])
    assert v(t, answer(op(467, "Sir Lindsay Hoyle (Chorley) (Ind)", did="E1")))[0] == "pass"
    assert v(t, answer(op(345, "Sir Edward Leigh (in the Chair)", did="E1")))[0] == "harm"
