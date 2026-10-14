"""What the probes could not see."""

from __future__ import annotations

from datetime import date

import pytest

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import divisions as O
from eval.oracles import members as OM
from eval.probes import core
from eval.probes import divisions as D
from eval.probes import members as PM

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip
DAY = date(2026, 9, 7)


def vote(mid, lobby, name=None, party="Labour"):
    return O.Vote(mid, lobby, False, None, party, name or f"Member {mid}")


def truth(did=10, title="Health Bill: Report Stage: New Clause 6", day=DAY, votes=None, house="Commons"):
    votes = votes or (vote(1, "aye", "Alan Mak"), vote(2, "aye"), vote(4, "no"), vote(5, "no"))
    return O.DivisionTruth(division_id=did, house=house, date=day, number=1, title=title, ayes=2, noes=2,
                           ayes_listed=2, noes_listed=2, ayes_named=2, noes_named=2, tally_reconciled=True,
                           votes=tuple(votes))  # fmt: skip


def absent(detail):
    return Answer(question="q?", provenance=PROV, abstention=Abstention.NOT_IN_RECORD, abstention_detail=detail)


def trial(cls="ST-division_result", category="division_result", **subject):
    return core.Trial(trial_id="t", cls=cls, category=category, question="q?", form="canonical",
                      subject={"division_id": 10, **subject})  # fmt: skip


# --- a false absence is a harm ------------------------------------------------------------------------------------


def test_saying_an_existing_division_is_not_in_the_record_is_a_harm():
    verdict = D.verdict_for([truth()])
    outcome, why = verdict(trial(), absent("No division in the Commons on 7 September 2026 is recorded on Amendment 23."))
    assert outcome == "harm" and "not" in why


def test_saying_a_listed_member_is_not_in_the_list_is_a_harm_and_an_absent_one_a_pass():
    verdict = D.verdict_for([truth()])
    listed = trial("ST-division_vote", "division_vote", mnis_id=1, role="aye")
    assert verdict(listed, absent("Alan Mak is not recorded in the published list for ..."))[0] == "harm"
    gone = trial("ST-division_vote", "division_vote", mnis_id=99, role="absent")
    assert verdict(gone, absent("Someone is not recorded in the published list for ..."))[0] == "pass"


# --- MR-02 judges what the absence says -----------------------------------------------------------------------------


def mr02(form="wrong_number", fake="New Clause 906"):
    return core.Trial(trial_id="m", cls="MR-02", category="division_result", question="q?", form=form, ambiguous=True,
                      subject={"house": "Commons", "date": DAY.isoformat(), "nonexistent": fake, "division_ids": [10],
                               "fake": "906" if form == "wrong_number" else None})  # fmt: skip


def test_mr02_passes_only_an_absence_that_names_what_was_asked():
    verdict = D.verdict_for([truth()])
    right = absent("No division in the Commons on 7 September 2026 is recorded on New Clause 906. The divisions "
                   "recorded that day were: division 1: Health Bill: Report Stage: New Clause 6.")  # fmt: skip
    assert verdict(mr02(), right)[0] == "pass"
    false_day = absent("No division is recorded on 2026-09-07 in the Commons.")
    assert verdict(mr02(), false_day)[0] == "harm", "the day held a division"
    assert verdict(mr02(), absent("The record does not hold that."))[0] == "mismatch"


def test_on_a_day_with_no_division_an_absence_is_the_right_answer():
    verdict = D.verdict_for([truth()])
    t = core.Trial(trial_id="m", cls="MR-02", category="division_result", question="q?", form="no_division_day",
                   ambiguous=True, subject={"house": "Commons", "date": "2026-09-09", "nonexistent": "x",
                                            "division_ids": []})  # fmt: skip
    assert verdict(t, absent("No division is recorded on 2026-09-09 in the Commons."))[0] == "pass"
    answered = Answer(question="q?", provenance=PROV, rendered="x", structured_facts=(StructuredFact(
        template="division_result", params={"division_id": 10}, value={"division_id": 10, "result": "agreed"}),))
    assert verdict(t, answered)[0] == "harm"


# --- mismatch counts toward harm; nothing answered bounds nothing ----------------------------------------------------


def outcome(kind):
    return core.Outcome(core.Trial(trial_id=kind, cls="X", category="c", question="q", form="canonical",
                                   subject={"division_id": hash(kind) % 1000}), kind)  # fmt: skip


def test_a_mismatch_counts_toward_the_harm_bound_until_triaged():
    block = core._block([outcome("pass")] * 9 + [outcome("mismatch")])
    assert block["harm"] == 0 and block["mismatch"] == 1
    assert block["harm_upper_95"] == pytest.approx(core.harm_upper(1, 10), abs=1e-6)


def test_a_class_that_answered_nothing_reports_no_bound():
    block = core._block([outcome("fallthrough")] * 184)
    assert block["answered"] == 0 and block["harm_upper_95"] is None and block["vacuous"] is True


def test_the_report_says_what_population_its_bound_covers():
    assert "generated" in core.summarise([outcome("pass")])["population"]


# --- questions the resolver does not read first ----------------------------------------------------------------------


PROFILES = {1: OM.Profile(mnis_id=1, display="Alan Mak", address="Mr Alan Mak", full_title="Alan Mak MP"),
            4: OM.Profile(mnis_id=4, display="Lord Four of Somewhere", address="The Lord Four", full_title=None)}


def test_a_vote_is_also_asked_by_the_name_a_person_is_addressed_by_and_known_by_now():
    trials = D.st_trials([truth(votes=(vote(1, "aye", "Alan Mak"), vote(4, "no")))], profiles=PROFILES)
    forms = {(t.subject["mnis_id"], t.form): t.question for t in trials if t.cls == "ST-division_vote"}
    assert "Mr Alan Mak" in forms[(1, "address")]
    assert "Lord Four of Somewhere" in forms[(4, "display")], "renamed since the list printed them"
    assert (1, "display") not in forms, "a name the list already printed is not asked twice"


def test_a_compound_question_names_another_member_of_the_list():
    trials = [t for t in D.st_trials([truth()]) if t.form == "compound"]
    assert trials
    for t in trials:
        other = t.subject["other"]
        assert other != t.subject["mnis_id"]
        assert t.question.index("how did") > 0, "the asked member comes second, after the other's clause"


def test_mr02_asks_about_a_wrong_bill_a_wrong_stage_and_a_day_with_no_division():
    day = [truth(10, "Health Bill: Second Reading"), truth(11, "Health Bill: Report Stage: New Clause 6"),
           truth(20, "Fisheries Bill: Third Reading", day=date(2026, 9, 14)),
           truth(30, "Trade Bill: Second Reading", day=date(2026, 9, 21))]  # fmt: skip
    trials = D.mr02_trials(day)
    forms = {t.form for t in trials}
    assert {"wrong_number", "wrong_bill", "wrong_stage", "no_division_day"} <= forms, forms
    for t in trials:
        if t.form == "wrong_bill":
            assert "Health Bill" not in t.question or t.subject["date"] != DAY.isoformat()
        if t.form == "wrong_stage":
            asked = "Third Reading" if t.subject["date"] in (DAY.isoformat(), "2026-09-21") else "Second Reading"
            assert asked in t.question, t.question
        if t.form == "no_division_day":
            held = {x.date.isoformat() for x in day if x.house == t.subject["house"]}
            assert t.subject["date"] not in held and t.subject["division_ids"] == []


# --- the short peer form, written independently ----------------------------------------------------------------------


@pytest.mark.parametrize("name, short", [
    ("Lord Ahmad of Wimbledon", "Lord Ahmad"), ("Baroness Smith of Basildon", "Baroness Smith"),
    ("The Earl of Courtown", None), ("The Lord Bishop of Norwich", None), ("Lord Patel", None),
    ("Viscount Younger of Leckie", "Viscount Younger"), ("Lord Hunt of Kings Heath", "Lord Hunt"), ("Alan Mak", None),
])  # fmt: skip
def test_the_probe_s_short_peer_form(name, short):
    assert D.peer_short(name) == short


def test_the_probe_does_not_borrow_the_resolver_s_pattern():
    import inspect

    from answer import resolve as RS

    assert RS._PEER_SHORT.pattern not in inspect.getsource(D)


# --- party on a day the member cast no vote ---------------------------------------------------------------------------


def test_a_party_is_also_asked_on_a_sitting_day_the_member_cast_no_vote():
    days = [truth(10, day=date(2026, 9, 7)), truth(11, day=date(2026, 9, 8), votes=(vote(2, "aye"), vote(4, "no")))]
    history = {1: [OM.Affiliation("Labour", date(2020, 1, 1), None)]}
    trials = [t for t in PM.party_trials(days, history=history) if t.form == "no_vote_day"]
    assert [(t.subject["mnis_id"], t.subject["date"], t.subject["party"]) for t in trials] == [(1, "2026-09-08", "Labour")]


def test_a_party_the_history_does_not_settle_is_not_asked():
    days = [truth(10, day=date(2026, 9, 7)), truth(11, day=date(2026, 9, 8), votes=(vote(2, "aye"),))]
    history = {1: [OM.Affiliation("Labour", date(2020, 1, 1), date(2026, 9, 8)),
                   OM.Affiliation("Independent", date(2026, 9, 8), None)]}  # the switch falls on the day
    assert not [t for t in PM.party_trials(days, history=history) if t.form == "no_vote_day" and t.subject["mnis_id"] == 1]


def test_a_party_answered_as_not_in_the_record_is_a_harm():
    t = core.Trial(trial_id="p", cls="ST-party_at_date", category="party_at_date", question="q?", form="canonical",
                   subject={"mnis_id": 1, "date": "2026-09-07", "party": "Labour"})  # fmt: skip
    assert PM.verdict_for(())(t, absent("No member named matches."))[0] == "harm"


# --- MR-11 (a House swapped) and MR-10 (a typo in a member's name) ----------------------------------------------------


def test_mr11_asks_each_division_in_the_other_house():
    trials = D.mr11_trials([truth(10), truth(20, day=date(2026, 9, 8), house="Lords")])
    assert {(t.subject["source_division"], t.subject["house"]) for t in trials} == {(10, "Lords"), (20, "Commons")}
    assert all(f"the {t.subject['house']} division" in t.question for t in trials)


def test_mr11_answering_the_other_house_s_division_is_a_harm():
    commons = truth(10)
    verdict = D.verdict_for([commons])
    t = D.mr11_trials([commons])[0]
    answered = Answer(question="q?", provenance=PROV, rendered="x", structured_facts=(StructuredFact(
        template="division_result", params={"division_id": 10},
        value={"division_id": 10, "result": "agreed", "ayes": 2, "noes": 2}),))  # fmt: skip
    assert verdict(t, answered)[0] == "harm"
    assert verdict(t, absent("No division is recorded on 2026-09-07 in the Lords."))[0] == "pass", "the Lords held none"


def test_mr11_a_same_titled_division_in_the_asked_house_is_the_right_answer():
    commons, lords = truth(10), truth(1000010, house="Lords")
    verdict = D.verdict_for([commons, lords])
    t = next(x for x in D.mr11_trials([commons, lords]) if x.subject["house"] == "Lords")
    answered = Answer(question="q?", provenance=PROV, rendered="x", structured_facts=(StructuredFact(
        template="division_result", params={"division_id": 1000010},
        value={"division_id": 1000010, "result": "tie", "ayes": 2, "noes": 2}),))  # fmt: skip
    assert verdict(t, answered)[0] == "pass"


def test_mr10_misspells_the_surname_and_expects_the_same_member_or_a_refusal():
    t0 = truth(votes=(vote(1, "aye", "Kemi Badenoch"), vote(4, "no", "Wes Streeting")))
    trials = D.mr10_trials([t0])
    assert trials and all(t.cls == "MR-10" for t in trials)
    for t in trials:
        assert "Badenoch" not in t.question and "Streeting" not in t.question
        assert t.subject["mnis_id"] in (1, 4)
    verdict = D.verdict_for([t0])
    t = trials[0]
    other = 4 if t.subject["mnis_id"] == 1 else 1
    wrong = Answer(question="q?", provenance=PROV, rendered="x", structured_facts=(StructuredFact(
        template="division_vote", params={"division_id": 10, "mnis_id": other},
        value={"division_id": 10, "mnis_id": other, "lobby": "no", "is_teller": False, "proxy_name": None}),))
    assert verdict(t, wrong)[0] == "harm"


def test_the_lords_oracle_reads_the_motion_as_printed():
    payload = {"divisionId": "5", "date": "2026-09-15T00:00:00", "number": "1", "title": "X Bill",
               "amendmentMotionNotes": "<p>Lord A moved amendment 27, in clause 6, at end to insert &ldquo;y&rdquo;</p>",
               "contents": [], "notContents": [], "authoritativeContentCount": "0",
               "authoritativeNotContentCount": "0"}  # fmt: skip
    assert O.parse_lords(payload).motion == "Lord A moved amendment 27, in clause 6, at end to insert “y”"


def lords_day(motion):
    t = truth(1000005, title="Public Authorities Bill", house="Lords")
    return O.DivisionTruth(**{**t.__dict__, "motion": motion})


def test_mr11_an_absence_is_checked_against_the_other_house_s_titles_and_motions():
    commons = truth(10, title="Sustainable Aviation Fuel Bill: Remaining Stages: New Clause 5")
    lords = lords_day("Lord A moved amendment 27, in clause 6.")
    verdict = D.verdict_for([commons, lords])
    t = next(x for x in D.mr11_trials([commons, lords]) if x.subject["house"] == "Lords")
    right = absent("No division in the Lords on 7 September 2026 is recorded on New Clause 5. The divisions recorded "
                   "that day were: division 1: Public Authorities Bill.")  # fmt: skip
    assert verdict(t, right)[0] == "pass", "no Lords division that day carries 5 anywhere"
    holds = lords_day("Lord A moved amendment 5, in clause 6.")
    verdict = D.verdict_for([commons, holds])
    t = next(x for x in D.mr11_trials([commons, holds]) if x.subject["house"] == "Lords")
    assert verdict(t, right)[0] == "mismatch", "a Lords motion that names a 5 cannot be checked by the oracle"
