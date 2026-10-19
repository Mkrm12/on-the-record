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


# --- S05: MR-08 asks about threads, and an answer may list every same-heading item -------------------------


def record(did, order, heading="Sovereign Grant Bill", first="Madam Deputy Speaker (Caroline Nokes)", opener=5001):
    return OO.DebateTruth(did, "Commons", date(2026, 9, 14), heading, "Commons Chamber", (
        OO.Contribution(1, 4048, first, "Order."),
        OO.Contribution(2, opener, "A Member (Somewhere) (Lab)", "I beg to move...")), section_order=order)


def test_a_debate_split_across_adjacent_records_is_one_trial_about_the_thread():
    """Second and Third Reading under the bill's heading, next to each other: one debate. Naming its opener from either
    record of the thread passes; the question is not ambiguous.
    """
    truths = [record("SR", 27), record("TR", 28, opener=5003)]
    (t,) = PO.mr08_trials(truths)
    v = PO.verdict_for(truths)
    assert t.subject["debate_ids"] == ["SR", "TR"] and t.subject["opener"] == 5001 and not t.ambiguous
    assert v(t, answer(op(5001, did="SR")))[0] == "pass"
    assert v(t, answer(op(5003, did="TR")))[0] == "mismatch", "the second record's opener is not the thread's"


def test_recurring_items_under_one_heading_are_ambiguous_and_an_answer_listing_every_ones_opener_is_an_aggregate():
    truths = [record("A1", 3, "Arrangement of Business", opener=11), record("X", 4, "Something Else"),
              record("A2", 8, "Arrangement of Business", opener=12)]
    trials = {t.subject["debate_id"]: t for t in PO.mr08_trials(truths)}
    v = PO.verdict_for(truths)
    t = trials["A2"]
    assert t.ambiguous
    both = Answer(question="q?", provenance=PROV, facets={"openers": [op(11, did="A1"), op(12, did="A2")]},
                  rendered="Two items.")
    assert v(t, both)[0] == "aggregate"
    wrong = both.model_copy(update={"facets": {"openers": [op(11, did="A1"), op(4048, "Madam Deputy Speaker",
                                                                                       did="A2")]}})  # fmt: skip
    assert v(t, wrong)[0] == "harm"
    other_wrong = both.model_copy(update={"facets": {"openers": [op(4048, did="A1"), op(12, did="A2")]}})
    assert v(t, other_wrong)[0] == "harm", "a wrong opener for the other item is a false claim too"
    omitted = both.model_copy(update={"facets": {"openers": [op(11, did="A1")]}})
    assert v(t, omitted)[0] == "harm", "a listing that leaves out the asked item claims a completeness it lacks"


# --- QA pass 6 (S05): the Chair's own statement, and contributions that say nothing ----------------------------------


def statement(did="SS", heading="Speaker’s Statement", order=4):
    return OO.DebateTruth(did, "Commons", date(2026, 9, 14), heading, "Commons Chamber", (
        OO.Contribution(1, 467, "Mr Speaker", "I wish to make a short statement."),
        OO.Contribution(2, 7, "Sir Desmond Swayne (New Forest West) (Con)", "On a point of order, Mr Speaker."),
        OO.Contribution(3, 467, "Mr Speaker", "I thank him.")), section_order=order)


def test_the_chairs_own_statement_is_asked_about_and_naming_the_member_after_it_is_a_harm():
    (t,) = PO.mr08_trials([statement()])
    v = PO.verdict_for([statement()])
    assert t.subject["chair_business"] and t.subject["opener"] is None and t.subject["statement"] == 467
    assert v(t, answer(op(7, "Sir Desmond Swayne (New Forest West) (Con)", did="SS")))[0] == "harm"
    said = Answer(question="q?", provenance=PROV, rendered="A statement by the Chair.",
                  facets={"chair_statement": {"speech_id": "s1", "mnis_id": 467, "speaker_name": "Mr Speaker",
                                              "debate_id": "SS"}})  # fmt: skip
    assert v(t, said)[0] == "pass"
    elsewhere = said.model_copy(update={"facets": {"chair_statement": {**said.facets["chair_statement"],
                                                                       "debate_id": "XX"}}})  # fmt: skip
    assert v(t, elsewhere)[0] != "pass"
    assert v(t, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "declined", "no member opened it: that is true"


def test_a_listing_must_call_a_chairs_statement_item_the_chairs():
    truths = [statement("S1", order=3), record("X", 4, "Something Else"), statement("S2", order=8)]
    v = PO.verdict_for(truths)
    t = next(x for x in PO.mr08_trials(truths) if x.subject["debate_id"] == "S2")
    chair = {"chair_statement": True, "speech_id": "s", "mnis_id": 467, "speaker_name": "Mr Speaker"}
    good = Answer(question="q?", provenance=PROV, rendered="Two items.",
                  facets={"openers": [{**chair, "debate_id": "S1"}, {**chair, "debate_id": "S2"}]})
    assert v(t, good)[0] == "aggregate"
    bad = good.model_copy(update={"facets": {"openers": [{**chair, "debate_id": "S1"}, op(7, did="S2")]}})
    assert v(t, bad)[0] == "harm"


def test_a_member_who_only_rose_is_not_the_truth_and_naming_them_is_not_a_pass():
    rose = OO.DebateTruth("R1", "Commons", date(2026, 9, 14), "Housing", "Commons Chamber", (
        OO.Contribution(1, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order."),
        OO.Contribution(2, 7, "Sir Desmond Swayne (New Forest West) (Con)", "<em>rose—</em>"),
        OO.Contribution(3, 5001, "The Minister for Housing (Ann Minister)", "I beg to move...")))
    (t,) = PO.mr08_trials([rose])
    v = PO.verdict_for([rose])
    assert t.subject["opener"] == 5001
    assert v(t, answer(op(5001, did="R1")))[0] == "pass"
    assert v(t, answer(op(7, did="R1")))[0] == "mismatch"


# --- QA pass 7: who made the statement, and an absence of an opener is a claim
# -----------------------------------------


def test_a_chairs_statement_answer_must_name_the_chair_who_made_it():
    """Med-1: the verdict passed any `chair_statement` for the thread, even one naming a member as the Chair."""
    (t,) = PO.mr08_trials([statement()])
    v = PO.verdict_for([statement()])
    wrong = Answer(question="q?", provenance=PROV, rendered="A statement.",
                   facets={"chair_statement": {"speech_id": "s2", "mnis_id": 7, "debate_id": "SS",
                                               "speaker_name": "Sir Desmond Swayne (New Forest West) (Con)"}})
    assert v(t, wrong)[0] == "harm"
    by_mnis = wrong.model_copy(update={"facets": {"chair_statement": {"mnis_id": 467, "debate_id": "SS",
                                                                      "speaker_name": None}}})  # fmt: skip
    assert v(t, by_mnis)[0] == "pass", "the statement's maker, by member"


def _no_opener(did="D1", reason="said_nothing"):
    return Answer(question="q?", provenance=PROV, rendered="No member's speech opens it.",
                  facets={"no_opener": {"reason": reason, "debate_id": did}})


def test_saying_no_member_opened_a_debate_a_member_opened_is_a_harm():
    """Med-2: "No member's speech opens …" was a claim no verdict could see (it carried no abstention)."""
    assert V(T, _no_opener())[0] == "harm"
    assert V(T, _no_opener(reason="only_chair"))[0] == "harm"


def silent(did="R2", heading="Rail Fares"):
    return OO.DebateTruth(did, "Commons", date(2026, 9, 14), heading, "Commons Chamber", (
        OO.Contribution(1, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order."),
        OO.Contribution(2, 7, "Sir Desmond Swayne (New Forest West) (Con)", "<em>rose\u2014</em>")))


def test_a_debate_no_member_opened_is_asked_about_and_judged_by_its_truth():
    (t,) = PO.mr08_trials([silent()])
    v = PO.verdict_for([silent()])
    assert t.subject["opener"] is None and t.subject["no_opener"]
    assert v(t, _no_opener("R2"))[0] == "pass"
    assert v(t, answer(op(7, "Sir Desmond Swayne (New Forest West) (Con)", did="R2")))[0] == "mismatch"
    assert v(t, answer(op(4048, "Madam Deputy Speaker", did="R2")))[0] == "harm"


def test_a_listing_that_says_an_item_has_no_opener_is_judged_against_its_truth():
    truths = [record("A1", 3, "Arrangement of Business", opener=11), record("X", 4, "Something Else"),
              record("A2", 8, "Arrangement of Business", opener=12)]
    v = PO.verdict_for(truths)
    t = next(x for x in PO.mr08_trials(truths) if x.subject["debate_id"] == "A2")
    false_none = Answer(question="q?", provenance=PROV, rendered="Two items.",
                        facets={"openers": [op(11, did="A1"), {"debate_id": "A2", "no_opener": "said_nothing"}]})
    assert v(t, false_none)[0] == "harm"
