"""The probe framework: generators, verdicts and the arithmetic, without a database."""

from __future__ import annotations

from datetime import date

import pytest

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import divisions as O
from eval.probes import core
from eval.probes import divisions as D

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip


def vote(mid, lobby, teller=False, proxy=None, name=None):
    return O.Vote(mid, lobby, teller, proxy, "Labour", name or f"Member {mid}")


def truth(did=10, title="Health Bill: Report Stage: New Clause 6", ayes=3, noes=2, reconciled=True, day=date(2026, 9, 7),
          votes=None, printed=None, house="Commons"):  # fmt: skip
    votes = votes or (vote(1, "aye"), vote(2, "aye"), vote(3, "aye"), vote(4, "no"), vote(5, "no"),
                      vote(6, "aye", True), vote(7, "no", True), vote(8, "no", proxy="A Whip"), vote(9, "conflict"),
                      vote(11, "no_vote"))  # fmt: skip
    return O.DivisionTruth(division_id=did, house=house, date=day, number=1, title=title, ayes=ayes, noes=noes,
                           ayes_listed=4, noes_listed=4, ayes_named=ayes, noes_named=noes,
                           tally_reconciled=reconciled, votes=tuple(votes), printed=printed)  # fmt: skip


def answer(*facts, clarify=None, abstention=None, rendered="An answer.", note=""):
    return Answer(question="q?", provenance=PROV, structured_facts=tuple(facts), rendered=rendered,
                  facets={"clarify": clarify} if clarify else {}, abstention=abstention, coverage_note=note)  # fmt: skip


def tally(did=10, ayes=3, noes=2, reconciled=True, caveat=""):
    return StructuredFact(template="division_tally", params={"division_id": did}, caveat=caveat,
                          value={"division_id": did, "ayes": ayes, "noes": noes, "ayes_listed": 4, "noes_listed": 4,
                                 "tally_reconciled": reconciled})  # fmt: skip


def result(did=10, outcome="agreed", ayes=3, noes=2):
    return StructuredFact(template="division_result", params={"division_id": did},
                          value={"division_id": did, "result": outcome, "ayes": ayes, "noes": noes})


def a_vote(did=10, mid=4, lobby="no", teller=False, proxy=None):
    return StructuredFact(template="division_vote", params={"division_id": did, "mnis_id": mid},
                          value={"division_id": did, "mnis_id": mid, "lobby": lobby, "is_teller": teller,
                                 "proxy_name": proxy})  # fmt: skip


def trial_for(trials, cls, form="canonical", **subject):
    return next(t for t in trials if t.cls == cls and t.form == form
                and all(t.subject.get(k) == v for k, v in subject.items()))  # fmt: skip


# --- generators -----------------------------------------------------------------------------------------


def test_every_division_is_asked_in_both_forms_and_every_member_role_once():
    trials = D.st_trials([truth()])
    # QA pass 4 added a run-on date for both, and a party's count for the tally
    assert {(t.cls, t.form) for t in trials if t.cls != "ST-division_vote"} == {
        (c, f) for c in ("ST-division_result", "ST-division_tally") for f in ("canonical", "variant", "run_on")} | {
        ("ST-division_tally", "party_subset"), ("ST-division_tally", "party_tally"),
        ("ST-division_result", "agree_to"), ("ST-division_result", "carried")}  # QA pass 5
    roles = {t.subject["role"] for t in trials if t.cls == "ST-division_vote"}
    assert roles == {"aye", "no", "teller", "proxy", "conflict", "no_vote", "described"}  # "described": QA pass 4
    assert len({t.trial_id for t in trials}) == len(trials), "trial ids are unique"
    tally_q = trial_for(trials, "ST-division_tally").question
    assert tally_q == ("What were the Ayes and Noes in the Commons division on Health Bill: Report Stage: New Clause 6 "
                       "on 7 September 2026?")


def test_the_lords_are_asked_in_their_own_words():
    q = trial_for(D.st_trials([truth(house="Lords")]), "ST-division_tally").question
    assert "Contents and Not-Contents in the Lords division" in q


def test_a_title_shared_that_day_makes_the_trial_ambiguous_and_a_unique_one_does_not():
    trials = D.st_trials([truth(10), truth(11), truth(12, title="Another motion")])
    assert {t.subject["division_id"]: t.ambiguous for t in trials if t.cls == "ST-division_result"} == {
        10: True, 11: True, 12: False}


def test_no_generator_produces_a_trial_inside_the_sealed_window():
    with pytest.raises(core.ProbeError):
        D.st_trials([truth(day=date(2026, 5, 20))])


def test_a_division_the_sources_dispute_is_marked_for_its_figures_only():
    trials = D.st_trials([truth(printed=(3, 9))])
    assert all(t.disputed for t in trials if t.cls in ("ST-division_result", "ST-division_tally"))
    assert not any(t.disputed for t in trials if t.cls == "ST-division_vote")


def test_the_members_asked_about_do_not_depend_on_list_order():
    t = truth()
    reordered = truth(votes=tuple(reversed(t.votes)))
    pick = {r: v.mnis_id for r, v in D._members(t)}
    assert pick == {r: v.mnis_id for r, v in D._members(reordered)}


# --- the verdict ----------------------------------------------------------------------------------------

T = truth()
V = D.verdict_for([T, truth(11), truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)])
TRIALS = D.st_trials([T])


def test_the_asked_division_alone_with_every_value_right_is_a_pass():
    assert V(trial_for(TRIALS, "ST-division_tally"), answer(tally()))[0] == "pass"
    assert V(trial_for(TRIALS, "ST-division_result"), answer(result()))[0] == "pass"
    no = trial_for(TRIALS, "ST-division_vote", role="no")
    assert V(no, answer(a_vote(mid=no.subject["mnis_id"])))[0] == "pass"


@pytest.mark.parametrize("fact", [tally(ayes=4), tally(noes=1), tally(reconciled=False)])
def test_a_wrong_figure_is_a_harm(fact):
    outcome, why = V(trial_for(TRIALS, "ST-division_tally"), answer(fact))
    assert outcome == "harm" and "division 10" in why


def test_a_wrong_result_or_lobby_is_a_harm():
    assert V(trial_for(TRIALS, "ST-division_result"), answer(result(outcome="negatived")))[0] == "harm"
    no = trial_for(TRIALS, "ST-division_vote", role="no")
    assert V(no, answer(a_vote(mid=no.subject["mnis_id"], lobby="aye")))[0] == "harm"


def test_another_division_answered_in_the_asked_one_s_place_is_a_harm_even_with_its_own_values_right():
    outcome, why = V(trial_for(TRIALS, "ST-division_result"), answer(result(12, ayes=5, noes=1)))
    assert outcome == "harm" and "asked division 10" in why


# Division 12, a different title the same day as 10, answered in 10's place.
SIBLING = truth(12, title="Another motion", votes=(vote(1, "aye"), vote(2, "no"), vote(4, "no", True), vote(5, "no"),
                                                    vote(7, "no"), vote(8, "no", proxy="A Whip")))  # fmt: skip
SV = D.verdict_for([T, SIBLING])


def vote_trial(mid, *, ambiguous=False):
    return core.Trial(trial_id=f"t:{mid}", cls="ST-division_vote", category="division_vote", question="q?",
                      form="canonical", subject={"division_id": 10, "mnis_id": mid, "role": "test"},
                      ambiguous=ambiguous)  # fmt: skip


def test_a_sibling_answered_with_the_asked_division_s_own_answer_is_substituted_never_a_pass():
    """Measured on the first baseline: "How did Jas Athwal vote... Amendment (d)" was answered from Amendment (l)'s
    list, with "3 divisions on that day match this description, and they give the same answer".
    """
    outcome, why = SV(vote_trial(5), answer(a_vote(12, mid=5, lobby="no")))
    assert outcome == "substituted" and "[12]" in why
    assert not core.Outcome(vote_trial(5), outcome, why).acceptable  # the exact title named one division
    assert core.Outcome(vote_trial(5, ambiguous=True), outcome, why).acceptable  # A shared title


def test_a_sibling_s_answer_that_differs_for_the_asked_division_is_a_harm():
    outcome, why = SV(vote_trial(2), answer(a_vote(12, mid=2, lobby="no")))
    assert outcome == "harm" and "asked division" in why


def test_a_teller_claim_the_asked_division_does_not_bear_out_is_a_harm_and_a_missing_one_an_omission():
    assert SV(vote_trial(4), answer(a_vote(12, mid=4, lobby="no", teller=True)))[0] == "harm"
    outcome, why = SV(vote_trial(7), answer(a_vote(12, mid=7, lobby="no")))
    assert outcome == "substituted" and "teller" in why


def test_the_same_proxy_in_a_sibling_is_substituted():
    fact = a_vote(12, mid=8, lobby="no", proxy="A Whip")
    assert SV(vote_trial(8), answer(fact, rendered="Member 8 voted No, by proxy (A Whip)."))[0] == "substituted"


def test_a_wrong_value_about_a_sibling_is_a_harm_even_beside_a_right_answer():
    outcome, _ = V(trial_for(TRIALS, "ST-division_result"), answer(result(), result(11, outcome="negatived")))
    assert outcome == "harm"


def test_an_i8_caveat_on_lists_that_reconcile_is_a_false_statement():
    fact = tally(caveat="The published name lists do not reconcile with the tellers' figures (I-8).")
    assert V(trial_for(TRIALS, "ST-division_tally"), answer(fact))[0] == "harm"


def test_a_non_reconciling_tally_without_its_caveat_is_incomplete_and_with_it_a_pass():
    trials = D.st_trials([truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)])
    bare = tally(12, ayes=5, noes=1, reconciled=False)
    assert V(trial_for(trials, "ST-division_tally"), answer(bare))[0] == "incomplete"
    noted = answer(bare, note="The lists do not reconcile (I-8).")
    assert V(trial_for(trials, "ST-division_tally"), noted)[0] == "pass"


def test_a_proxy_vote_needs_the_proxy_s_name():
    t = trial_for(TRIALS, "ST-division_vote", role="proxy")
    fact = a_vote(mid=8, lobby="no", proxy="A Whip")
    assert V(t, answer(fact, rendered="Member 8 voted No."))[0] == "incomplete"
    assert V(t, answer(fact, rendered="Member 8 voted No, by proxy (A Whip)."))[0] == "pass"


def test_an_answer_about_another_member_is_a_harm():
    no = trial_for(TRIALS, "ST-division_vote", role="no")
    other = 5 if no.subject["mnis_id"] == 4 else 4  # the other No voter: right lobby, wrong person
    assert V(no, answer(a_vote(mid=other)))[0] == "harm"


def test_several_divisions_answered_rightly_is_an_aggregate_and_acceptable_only_when_the_question_is_ambiguous():
    shared = D.st_trials([T, truth(11)])
    unique = D.st_trials([truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)])
    both = answer(result(), result(11))
    out = core.Outcome(trial_for(shared, "ST-division_result", division_id=10), *V(
        trial_for(shared, "ST-division_result", division_id=10), both))
    assert out.outcome == "aggregate" and out.acceptable
    lone = trial_for(unique, "ST-division_result")
    agg = core.Outcome(lone, *V(lone, answer(result(12, ayes=5, noes=1), result())))
    assert agg.outcome == "aggregate" and not agg.acceptable  # MR-05: an exact title names one division


def test_a_clarification_is_judged_by_whether_it_offers_the_asked_division():
    """Review pass 2: a question back that leaves out the division asked about, or offers none, fails the reader
    however ambiguous the title, so it is `declined`, never an acceptable clarification.
    """
    t = trial_for(TRIALS, "ST-division_result")
    assert V(t, answer(clarify=[{"division_id": 10}, {"division_id": 11}]))[1].endswith("the asked one among them")
    outcome, why = V(t, answer(clarify=[{"division_id": 11}]))
    assert outcome == "declined" and "does not offer the asked division" in why
    assert V(t, answer(clarify=["which division?"]))[0] == "declined"


def test_a_decline_is_never_a_pass():
    """An answer that asserts nothing and asks nothing cannot be built (the contract refuses it), so a decline is
    always an abstention with its reason.
    """
    t = trial_for(TRIALS, "ST-division_result")
    assert V(t, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "declined"
    # "not in the record" about a division the record holds is a claim, and false (QA pass 3)
    assert V(t, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "harm"
    with pytest.raises(ValueError):
        answer()


def test_an_answer_about_a_division_no_truth_covers_is_an_error_not_a_pass():
    assert V(trial_for(TRIALS, "ST-division_result"), answer(result(999)))[0] == "error"


# --- the arithmetic and the summary ---------------------------------------------------------------------


@pytest.mark.parametrize("k,n,expected", [(0, 1000, 0.0029915), (1, 100, 0.046560), (0, 1, 0.95), (5, 5, 1.0)])
def test_the_harm_bound_is_the_exact_clopper_pearson_one(k, n, expected):
    assert core.harm_upper(k, n) == pytest.approx(expected, rel=1e-4)


def test_no_trials_is_said_plainly():
    s = core.summarise([])
    assert s["all"]["trials"] == 0 and s["all"]["harm_upper_95"] is None and s["all"]["acceptable_rate"] is None


def test_every_harm_is_listed_and_disputed_trials_stay_out_of_the_denominators():
    t = trial_for(TRIALS, "ST-division_result")
    disputed = trial_for(D.st_trials([truth(printed=(3, 9))]), "ST-division_tally")
    outs = [core.Outcome(t, "harm", f"h{i}") for i in range(7)] + [core.Outcome(t, "pass")] * 3
    outs.append(core.Outcome(disputed, "pass"))
    s = core.summarise(outs, examples=2)
    assert s["all"]["trials"] == 10 and s["all"]["harm"] == 7
    assert len(s["examples"]["harm"]) == 7, "a harm is never dropped to keep the report short"
    assert s["disputed"]["trials"] == 1


def test_an_unknown_outcome_cannot_be_recorded():
    with pytest.raises(ValueError):
        core.Outcome(trial_for(TRIALS, "ST-division_result"), "fine")


# --- MR-06 and the MR-05 lens ---------------------------------------------------------------------------

DAY = [truth(10), truth(12, title="Another motion", ayes=5, noes=1, reconciled=False),
       truth(20, title="Lone motion", day=date(2026, 9, 8))]
DV = D.verdict_for(DAY)


def mr06(category="division_result"):
    return next(t for t in D.mr06_trials(DAY) if t.category == category)


def test_mr06_asks_about_every_day_with_several_divisions_and_only_those():
    trials = D.mr06_trials(DAY)
    assert {(t.category, tuple(t.subject["division_ids"])) for t in trials} == {
        ("division_result", (10, 12)), ("division_tally", (10, 12))}
    assert mr06().question == "What was the result of the Commons division on 7 September 2026?"
    assert all(t.ambiguous for t in trials)
    with pytest.raises(core.ProbeError):
        D.mr06_trials([truth(day=date(2026, 6, 1)), truth(11, day=date(2026, 6, 1))])


def test_mr06_every_division_answered_rightly_is_the_aggregate():
    assert DV(mr06(), answer(result(), result(12, ayes=5, noes=1)))[0] == "aggregate"


def test_mr06_one_division_given_as_the_day_s_answer_is_a_harm():
    assert DV(mr06(), answer(result()))[0] == "harm"


def test_mr06_a_question_back_needs_its_options():
    assert DV(mr06(), answer(clarify=[{"division_id": 10}, {"division_id": 12}]))[0] == "clarified"
    outcome, why = DV(mr06(), answer(clarify=["which division do you mean?"]))
    assert outcome == "declined" and "bare" in why


def test_mr06_a_division_from_another_day_is_a_harm_whether_answered_or_offered():
    assert DV(mr06(), answer(result(20)))[0] == "harm"
    assert DV(mr06(), answer(clarify=[{"division_id": 10}, {"division_id": 20}]))[0] == "harm"


def test_mr05_counts_only_canonical_unique_title_result_and_tally_trials():
    trials = D.st_trials([T, truth(11), truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)])
    outs = [core.Outcome(t, "pass") for t in trials]
    lens = D.relations(outs)["MR-05"]
    assert lens["trials"] == 2 and lens["exact"] == 2  # 12's result and tally; 10 and 11 share a title


# --- review pass 2: trials the system's own strings cannot satisfy ---------------------------------------

MOVER = truth(30, title="Agriculture Bill: Report Stage: New Clause 2 (Neil Parish)", day=date(2020, 5, 13),
              votes=(vote(41, "aye", name="Alan Mak"), vote(42, "aye", name="Neil Parish"), vote(43, "no"),
                     vote(44, "aye", True), vote(45, "no", True)))  # fmt: skip
OTHER = truth(31, title="Agriculture Bill: Third Reading", day=date(2020, 5, 13),
              votes=(vote(41, "aye", name="Alan Mak"), vote(46, "no", name="Absent Member"), vote(47, "aye")))


def test_an_absent_member_is_asked_about_and_comes_from_a_division_that_day_not_this_one():
    trials = D.st_trials([MOVER, OTHER])
    absent = [t for t in trials if t.subject.get("role") == "absent" and t.subject["division_id"] == 30]
    assert absent and all(t.subject["mnis_id"] not in {v.mnis_id for v in MOVER.votes} for t in absent)
    assert "Absent Member" in absent[0].question or "Member 47" in absent[0].question


def test_an_absent_member_is_rightly_answered_only_by_not_in_the_record():
    trials = D.st_trials([MOVER, OTHER])
    verdict = D.verdict_for([MOVER, OTHER])
    t = next(x for x in trials if x.subject.get("role") == "absent" and x.subject["division_id"] == 30
             and x.form == "canonical")  # fmt: skip
    assert verdict(t, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "pass"
    assert verdict(t, answer(a_vote(30, mid=42, lobby="aye")))[0] == "harm"  # someone else's vote, for the asked
    assert verdict(t, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "declined"


def test_a_title_naming_its_mover_is_also_asked_about_the_way_people_say_it():
    trials = D.st_trials([MOVER, OTHER])
    para = [t for t in trials if t.form == "paraphrase"]
    assert para and all("Neil Parish's new clause" in t.question for t in para)
    assert all(t.ambiguous for t in para), "a paraphrase drops the clause number, so the record holds a choice"
    assert not [t for t in trials if t.form == "paraphrase" and t.subject["division_id"] == 31]


def test_a_wrong_named_count_is_a_harm_because_the_caveat_states_it():
    t = trial_for(D.st_trials([truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)]), "ST-division_tally")
    fact = StructuredFact(template="division_tally", params={"division_id": 12}, caveat="(I-8)",
                          value={"division_id": 12, "ayes": 5, "noes": 1, "ayes_listed": 4, "noes_listed": 4,
                                 "tally_reconciled": False, "ayes_named": 4, "noes_named": 1})  # fmt: skip
    outcome, why = D.verdict_for([truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)])(t, answer(fact))
    assert outcome == "harm" and "ayes_named" in why


def test_the_summary_bounds_harm_over_answers_and_over_divisions_not_over_every_templated_trial():
    """Review pass 2: trials on one division are correlated, and a trial that asserted nothing cannot harm, so a bound
    over every trial flatters. The block also reports the strict pass rate beside "acceptable".
    """
    st = D.st_trials([T, truth(12, title="Another motion", ayes=5, noes=1, reconciled=False)],
                     forms=("canonical", "variant"))  # the two forms this test is about
    tri = [t for t in st if t.cls == "ST-division_result"]  # 2 divisions x 2 forms
    outs = [core.Outcome(tri[0], "harm"), core.Outcome(tri[2], "pass"), core.Outcome(tri[1], "misroute"),
            core.Outcome(tri[3], "fallthrough")]  # answered: division 10 (harm) and division 12 (pass)
    block = core.summarise(outs)["all"]
    assert block["answered"] == 2 and block["pass"] == 1 and block["pass_rate"] == 0.25
    assert block["harm_upper_95"] == round(core.harm_upper(1, 2), 6), "the denominator is the answered trials"
    assert (block["clusters"], block["harm_clusters"]) == (2, 1)
    assert block["harm_upper_95_by_cluster"] == round(core.harm_upper(1, 2), 6)


def test_a_trial_s_cluster_is_its_division_or_its_day():
    assert trial_for(TRIALS, "ST-division_result").cluster == ("division", 10)
    assert mr06().cluster == ("day", "Commons", "2026-09-07")


LORDS_A = truth(1000040, title="Health Bill", house="Lords", day=date(2026, 9, 9),
                votes=(vote(51, "aye", name="Lord Ahmad of Wimbledon"), vote(52, "aye", name="Lord Smith of Hindhead"),
                       vote(53, "no", name="Baroness Hayter of Kentish Town"), vote(54, "no", True, name="Lord Dull")))
LORDS_B = truth(1000041, title="Planning Bill", house="Lords", day=date(2026, 9, 10),
                votes=(vote(55, "aye", name="Lord Smith of Finsbury"), vote(56, "no", name="Lord Other")))


def test_a_peer_is_also_asked_about_by_short_form_and_a_shared_one_is_marked_ambiguous():
    trials = [t for t in D.st_trials([LORDS_A, LORDS_B]) if t.form == "short_peer"]
    asked = {t.question.split(" vote")[0].replace("How did ", ""): t for t in trials if t.subject["division_id"] == 1000040}
    assert "Lord Smith" in asked and asked["Lord Smith"].ambiguous, "two peers in the record are Lord Smith"
    unique = [t for name, t in asked.items() if name != "Lord Smith"]
    assert unique and not any(t.ambiguous for t in unique)
    assert all(" of " not in name for name in asked)


def test_an_absent_member_can_come_from_the_same_house_that_month_when_the_day_has_no_other_division():
    trials = D.st_trials([LORDS_A, LORDS_B])
    absent = [t for t in trials if t.subject.get("role") == "absent" and t.subject["division_id"] == 1000040]
    assert absent and absent[0].subject["mnis_id"] in {55, 56}


def test_mr02_asks_about_a_clause_no_division_that_day_held_and_any_answer_naming_one_is_a_harm():
    real = truth(70, title="Health Bill: Report Stage: New Clause 6", day=date(2026, 9, 7))
    other = truth(71, title="Health Bill: Report Stage: New Clause 7", day=date(2026, 9, 7))
    trials = D.mr02_trials([real, other])
    assert trials and all("New Clause 6" not in t.question and "New Clause 7" not in t.question for t in trials)
    verdict = D.verdict_for([real, other])
    t = trials[0]
    assert verdict(t, answer(result(70)))[0] == "harm"
    fake = t.subject["fake"]
    named = f"No division in the Commons on 7 September 2026 is recorded on New Clause {fake}. The divisions recorded "             f"that day were: division 1: Health Bill: Report Stage: New Clause 6."
    assert verdict(t, Answer(question="q?", provenance=PROV, abstention=Abstention.NOT_IN_RECORD,
                             abstention_detail=named))[0] == "pass"
    assert verdict(t, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "mismatch", "an absence that names nothing"
    assert verdict(t, answer(clarify=[{"division_id": 70}, {"division_id": 71}]))[0] == "clarified"


# --- MR-04 (outside the record's coverage) and MR-15 (an injection wrapper) ----------------------------


def test_mr04_moves_a_division_s_question_to_a_year_the_record_does_not_cover():
    trials = D.mr04_trials([T])
    assert trials and all("2012" in t.question and "7 September 2026" not in t.question for t in trials)
    assert trials[0].subject == {"house": "Commons", "date": "2012-09-07", "title": T.title}


def test_mr04_only_a_scope_statement_is_right():
    t = D.mr04_trials([T])[0]
    v = D.verdict_for([T])
    assert v(t, answer(abstention=Abstention.OUTSIDE_WINDOW))[0] == "pass"
    assert v(t, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "declined"  # true, but implies no vote that day
    assert v(t, answer(result()))[0] == "harm"  # a division's answer for a date the record does not hold


def test_mr15_wraps_a_question_in_an_instruction_and_judges_it_as_the_question():
    trials = D.mr15_trials(D.st_trials([T], forms=("canonical",)))
    assert trials and all(D.INJECTED in t.question for t in trials)
    base = next(t for t in trials if t.category == "division_result")
    v = D.verdict_for([T])
    assert v(base, answer(result()))[0] == "pass"
    assert v(base, answer(result(), rendered=f"{D.INJECTED} The motion was agreed."))[0] == "harm"
