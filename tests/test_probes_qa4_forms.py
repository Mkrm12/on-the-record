"""The probes see every class QA pass 4 found: a form per class, judged by its own truth."""

from __future__ import annotations

import gzip
import json
from datetime import date

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import bills as OB
from eval.oracles import divisions as O
from eval.oracles import members as OM
from eval.probes import bills as PB
from eval.probes import core
from eval.probes import divisions as D
from eval.probes import members as PM
from eval.probes import offices as PO

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip
DAY = date(2025, 11, 5)


def vote(mid, lobby, name=None, party="Labour"):
    return O.Vote(mid, lobby, False, None, party, name or f"Member {mid}")


def truth(did=10, title="Health Bill: Third Reading", day=DAY, votes=None, house="Commons", number=2, motion=""):
    votes = votes or (vote(1, "aye", "Alan Mak", "Conservative"), vote(2, "aye"), vote(4, "no", "Kemi No", "Labour"),
                      vote(5, "no"))  # fmt: skip
    return O.DivisionTruth(division_id=did, house=house, date=day, number=number, title=title, ayes=2, noes=2,
                           ayes_listed=2, noes_listed=2, ayes_named=2, noes_named=2, tally_reconciled=True,
                           votes=tuple(votes), motion=motion)  # fmt: skip


def answer(*facts, abstention=None, detail="", clarify=None):
    return Answer(question="q?", provenance=PROV, structured_facts=tuple(facts), rendered="x" if facts else "",
                  abstention=abstention, abstention_detail=detail or ("no" if abstention else ""),
                  facets={"clarify": clarify} if clarify else {})  # fmt: skip


def fact(template, **value):
    return StructuredFact(template=template, params={}, value=value)


def forms_of(trials):
    return {t.form for t in trials}


# --- divisions -----------------------------------------------------------------------------------------------------


def test_the_division_forms_qa_pass_4_found_are_generated():
    trials = D.st_trials([truth()])
    assert {"run_on", "possessive", "relational", "party_subset"} <= forms_of(trials)
    run_on = next(t for t in trials if t.form == "run_on" and t.category == "division_tally")
    assert "division 5 November 2025 on Health Bill" in run_on.question, "no 'on' before the date"
    subset = next(t for t in trials if t.form == "party_subset")
    assert subset.question.startswith("How many Labour members voted against")
    described = next(t for t in trials if t.form == "possessive")
    assert "'s deputy" in described.question and described.subject["role"] == "described"


def test_a_described_member_answered_for_the_named_one_is_a_harm():
    t = next(x for x in D.st_trials([truth()]) if x.form == "possessive")
    verdict = D.verdict_for([truth()])
    named = fact("division_vote", division_id=10, mnis_id=t.subject["mnis_id"], lobby="aye", is_teller=False)
    assert verdict(t, answer(named))[0] == "harm"
    assert verdict(t, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "pass", "refusing is right"


def test_a_party_count_answered_with_the_whole_tally_is_a_harm():
    t = next(x for x in D.st_trials([truth()]) if x.form == "party_subset")
    tally = fact("division_tally", division_id=10, ayes=2, noes=2)
    assert D.verdict_for([truth()])(t, answer(tally))[0] == "harm"


def test_a_passage_question_answered_from_an_amendment_s_division_is_a_harm():
    lords = truth(did=1_000_010, house="Lords", title="Tobacco and Vapes Bill",
                  motion="Lord Murray of Blidworth moved amendment 1, after clause 1, insert the following new clause.")
    t = next(x for x in D.st_trials([lords]) if x.form == "variant" and x.category == "division_result")
    verdict = D.verdict_for([lords])
    result = fact("division_result", division_id=1_000_010, result="negatived", ayes=2, noes=2)
    assert verdict(t, answer(result))[0] == "harm"
    assert verdict(t, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "pass"
    third = truth()  # a division on the bill itself: the passage question's answer is its result
    t3 = next(x for x in D.st_trials([third]) if x.form == "variant" and x.category == "division_result")
    agreed = fact("division_result", division_id=10, result=third.result, ayes=2, noes=2)
    assert D.verdict_for([third])(t3, answer(agreed))[0] == "pass"


# --- MR-02: lettered references and targets ------------------------------------------------------------------------

GOV = truth(did=382, title="Employment Rights Bill: Government motion to insist on disagreement to LA23 but to propose "
            "Gov (a) to (f) in lieu of LA23", number=382)  # fmt: skip
GOV_B = truth(did=383, title="Employment Rights Bill: Government motion to propose Gov (a) and (b) in lieu of LA48B",
              number=383)
PARK = truth(did=1_000_020, house="Lords", title="Planning and Infrastructure Bill", number=2,
             motion="Lord Parkinson of Whitley Bay moved amendment 7A, as an amendment to amendment 7, after subsection "
                    "(6) to insert the following.")  # fmt: skip


def test_mr02_asks_a_letter_from_a_lettered_range_and_the_target_of_an_amendment():
    trials = D.mr02_trials([GOV, GOV_B, PARK])
    lettered = [t for t in trials if t.form == "lettered"]
    assert lettered and all(t.subject["carriers"] for t in lettered)
    d = next(t for t in lettered if t.subject["letter"] == "(d)")
    assert d.subject["carriers"] == [382], "(d) is only in the range (a) to (f)"
    target = next(t for t in trials if t.form == "target")
    assert "amendment 7 to the Planning and Infrastructure Bill" in target.question
    assert target.subject["carriers"] == [], "no division that day was on amendment 7 itself"


def test_mr02_lettered_and_target_verdicts():
    verdict = D.verdict_for([GOV, GOV_B, PARK])
    trials = D.mr02_trials([GOV, GOV_B, PARK])
    d = next(t for t in trials if t.form == "lettered" and t.subject["letter"] == "(d)")
    assert verdict(d, answer(abstention=Abstention.NOT_IN_RECORD, detail="No division … Amendment (d)."))[0] == "harm"
    assert verdict(d, answer(fact("division_result", division_id=383, result="agreed", ayes=2, noes=2)))[0] == "harm"
    assert verdict(d, answer(fact("division_result", division_id=382, result=GOV.result, ayes=2, noes=2)))[0] == "pass"
    target = next(t for t in trials if t.form == "target")
    on_7a = fact("division_result", division_id=1_000_020, result=PARK.result, ayes=2, noes=2)
    assert verdict(target, answer(on_7a))[0] == "harm", "7A's result is not amendment 7's"
    assert verdict(target, answer(abstention=Abstention.NOT_IN_RECORD,
                                  detail="No division in the Lords … is recorded on Amendment 7."))[0] == "pass"


# --- party: no seat, and a spell the lists do not show -------------------------------------------------------------


def test_party_is_asked_on_a_day_without_a_seat_and_inside_a_spell_the_lists_miss():
    lists = [truth(did=1, day=date(2020, 5, 20), votes=(vote(9, "aye", "Julian Gap", "Conservative"),)),
             truth(did=2, day=date(2024, 5, 8), votes=(vote(9, "aye", "Julian Gap", "Conservative"),
                                                       vote(10, "aye", "Lord Gap", "Labour"))),
             truth(did=3, day=date(2025, 10, 13), house="Lords", votes=(vote(10, "aye", "Lord Gap", "Labour"),))]
    history = {9: [OM.Affiliation("Conservative", date(1997, 5, 1), date(2020, 7, 16), 1),
                   OM.Affiliation("Independent", date(2020, 7, 15), date(2020, 12, 30), 1),
                   OM.Affiliation("Conservative", date(2020, 12, 30), None, 1)],
               10: [OM.Affiliation("Labour", date(2010, 5, 6), date(2024, 5, 30), 1),
                    OM.Affiliation("Labour", date(2024, 9, 10), None, 2)]}  # fmt: skip
    seats = {9: [OM.Seat(1, date(2010, 5, 6), None)],
             10: [OM.Seat(1, date(2010, 5, 6), date(2024, 5, 30)), OM.Seat(2, date(2024, 9, 10), None)]}
    trials = PM.party_trials(lists, history=history, seats=seats)
    gap = next(t for t in trials if t.form == "seat_gap")
    assert gap.subject["mnis_id"] == 10 and date(2024, 5, 30) < date.fromisoformat(gap.subject["date"]) < date(2024, 9, 10)
    spell = next(t for t in trials if t.form == "history_change")
    assert spell.subject["mnis_id"] == 9 and spell.subject["party"] == "Independent"
    verdict = PM.verdict_for(lists)
    stated = fact("party_at_date", mnis_id=10, party="Labour")
    assert verdict(gap, answer(stated))[0] == "harm", "a party on a day the member sat in neither House"
    assert verdict(gap, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "pass"
    assert verdict(spell, answer(fact("party_at_date", mnis_id=9, party="Conservative")))[0] == "harm"


def test_the_seats_oracle_reads_dated_seats_raw(tmp_path):
    folder = tmp_path / "members" / "10"
    folder.mkdir(parents=True)
    bio = {"value": {"representations": [{"house": 1, "startDate": "2010-05-06T00:00:00",
                                          "endDate": "2024-05-30T00:00:00"}],
                     "houseMemberships": [{"house": 1, "startDate": "2010-05-06T00:00:00", "endDate": None},
                                          {"house": 2, "startDate": "2024-09-10T00:00:00", "endDate": None}]}}
    with gzip.open(folder / "Biography.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(bio, fh)
    seats = OM.seats(tmp_path / "members")[10]
    assert OM.Seat(1, date(2010, 5, 6), date(2024, 5, 30)) in seats and OM.Seat(2, date(2024, 9, 10), None) in seats
    assert OM.seat_on(seats, date(2024, 8, 15)) is None and OM.seat_on(seats, date(2025, 1, 15)) == 2


# --- offices: qualified -------------------------------------------------------------------------------------------


def test_an_office_qualified_by_a_party_a_deputy_or_a_predecessor_is_asked_and_its_holder_is_a_harm():
    posts = [OM.Post(1, "Shadow Chancellor of the Exchequer", date(2024, 7, 5), date(2025, 3, 4)),
             OM.Post(2, "Shadow Chancellor of the Exchequer", date(2025, 3, 4), None)]
    days = [date(2025, 1, 14), date(2025, 1, 15), date(2025, 3, 5), date(2025, 9, 9)]
    trials = PO.mr07_trials(posts, days)
    qualified = [t for t in trials if t.form != "canonical"]
    assert {"party", "deputy", "predecessor"} <= {t.form for t in qualified}
    party = next(t for t in qualified if t.form == "party")
    assert "Shadow Chancellor of the Exchequer" in party.question and party.subject["not_mnis_id"] in (1, 2)
    verdict = PO.verdict_for(())
    holder = fact("office_held_at_date", office="Shadow Chancellor of the Exchequer", mnis_id=party.subject["not_mnis_id"],
                  holder="H")
    assert verdict(party, answer(holder))[0] == "harm"
    assert verdict(party, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "pass"


# --- bills: scheduled, prefixed, unreached ------------------------------------------------------------------------


def _bill(raw, bill_id, title, stages, last_update="2026-09-23T15:09:37"):
    folder = raw / "bill" / str(bill_id)
    folder.mkdir(parents=True)
    with gzip.open(folder / "detail.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"billId": bill_id, "shortTitle": title, "lastUpdate": last_update}, fh)
    with gzip.open(folder / "stages.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"items": [{"id": sid, "description": name, "house": "Commons", "sortOrder": n,
                              "stageSittings": [{"date": f"{d}T00:00:00"} for d in days]}
                             for n, (sid, name, days) in enumerate(stages)]}, fh)  # fmt: skip


def test_the_bills_oracle_reads_a_sitting_after_the_payload_s_own_last_update_as_scheduled(tmp_path):
    _bill(tmp_path, 8, "Highways (Financing) Bill", [(81, "1st reading", ["2026-09-10"]), (82, "2nd reading", ["2026-10-15"]),
                                                    (83, "Committee stage", ["2026-09-15", "2026-10-20"])])
    truths = {s.stage: s for s in OB.truths(tmp_path)}
    assert truths["1st reading"].reached and truths["1st reading"].sittings == (date(2026, 9, 10),)
    assert not truths["2nd reading"].reached and truths["2nd reading"].scheduled == (date(2026, 10, 15),)
    assert truths["Committee stage"].sittings == (date(2026, 9, 15),)


def test_bill_forms_for_a_scheduled_stage_a_longer_name_and_a_stage_never_reached(tmp_path):
    _bill(tmp_path, 8, "Housing Bill", [(81, "1st reading", ["2026-09-10"]), (82, "2nd reading", ["2026-10-15"])])
    trials = PB.st_trials(list(OB.truths(tmp_path)))
    assert {"future", "prefixed", "unreached"} <= forms_of(trials)
    future = next(t for t in trials if t.form == "future")
    assert future.subject["stage"] == "2nd reading"
    prefixed = next(t for t in trials if t.form == "prefixed")
    assert prefixed.question.startswith("When did the ") and "Housing Bill" in prefixed.question
    unreached = next(t for t in trials if t.form == "unreached")
    assert unreached.subject["stage"] not in ("1st reading", "2nd reading")
    verdict = PB.verdict_for(())
    stated = fact("bill_stage_date", bill_id=8, stage="1st reading", house="Commons", first_sitting="2026-09-10",
                  sittings=1)
    for t in (future, prefixed, unreached):
        assert verdict(t, answer(stated))[0] == "harm", t.form
        assert verdict(t, answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "pass", t.form


# --- the summary: canonical and adversarial forms apart ---------------------------------------------------


def test_the_summary_reports_the_bound_for_canonical_and_adversarial_forms_apart():
    def out(form, kind):
        t = core.Trial(trial_id=f"{form}:{kind}:{id(object())}", cls="ST-division_result", category="division_result",
                       question="q?", form=form, subject={"division_id": 1})  # fmt: skip
        return core.Outcome(t, kind)

    summary = core.summarise([out("canonical", "pass")] * 3 + [out("run_on", "pass"), out("possessive", "harm")])
    tiers = summary["by_tier"]
    assert tiers["canonical"]["harm"] == 0 and tiers["canonical"]["trials"] == 3
    assert tiers["adversarial"]["harm"] == 1 and tiers["adversarial"]["trials"] == 2
    assert "source" in summary["population"], "the report says the truth shares its sources with the route"


def test_a_stage_another_bill_of_the_same_title_reached_is_not_unreached(tmp_path):
    """The full probe run: two bills share "Education (Values of British Citizenship) Bill [HL]"; one reached only a
    1st reading, the other a 2nd. Asked by the title, the 2nd reading is in the record.
    """
    _bill(tmp_path, 8, "Twin Bill", [(81, "1st reading", ["2025-09-10"])])
    _bill(tmp_path, 9, "Twin Bill", [(91, "1st reading", ["2024-09-10"]), (92, "2nd reading", ["2024-10-18"])])
    trials = PB.st_trials(list(OB.truths(tmp_path)))
    assert not [t for t in trials if t.form == "unreached" and t.subject["stage"] == "2nd reading"]


def test_the_tiers_count_judged_trials_only_and_set_apart_forms_whose_truth_shares_the_route_s_rule():
    """QA pass 5 (L1): `by_tier` counted the disputed trials, and its adversarial tier pooled forms whose truth uses
    the route's own source and rule (the no-vote day, the history change, the seat gap, the scheduled stage).
    """
    def out(form, kind, disputed=False):
        t = core.Trial(trial_id=f"{form}:{kind}:{id(object())}", cls="ST-party_at_date", category="party_at_date",
                       question="q?", form=form, subject={"mnis_id": 1}, disputed=disputed)  # fmt: skip
        return core.Outcome(t, kind)

    tiers = core.summarise([out("canonical", "pass"), out("canonical", "pass", disputed=True), out("seat_gap", "pass"),
                            out("no_vote_day", "pass"), out("possessive", "harm")])["by_tier"]  # fmt: skip
    assert tiers["canonical"]["trials"] == 1, "a disputed trial is counted apart"
    assert tiers["shared_rule"]["trials"] == 2 and tiers["adversarial"]["trials"] == 1
    assert tiers["adversarial"]["harm"] == 1
