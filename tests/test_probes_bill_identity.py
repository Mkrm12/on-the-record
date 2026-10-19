"""MR-02-bill: a real bill of the record asked about on a day it did not divide (S05's close, QA pass 8, Med 2)."""

from __future__ import annotations

from datetime import date

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import divisions as O
from eval.probes import divisions as D

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip
DAY = date(2026, 1, 20)


def truth(did, title, day=DAY, motion=None):
    t = O.DivisionTruth(division_id=did, house="Commons", date=day, number=did, title=title, ayes=2, noes=1,
                        ayes_listed=2, noes_listed=1, ayes_named=2, noes_named=1, tally_reconciled=True, votes=())
    return O.DivisionTruth(**{**t.__dict__, "motion": motion}) if motion else t


BILLS = ["Sentencing Act 2026", "Sentencing (Women) Bill", "Health Bill", "Diego Garcia Military Base Bill [HL]",
         "Children’s Wellbeing and Schools Bill"]


def test_a_bill_sharing_a_word_with_the_one_that_divided_is_asked_and_judged_absent():
    trials = D.mr02_bill_trials([truth(1, "Sentencing Bill: motion to disagree with Lords Amendment 7")], BILLS)
    assert [(t.form, t.subject["nonexistent"]) for t in trials] == [("sharing", "Sentencing (Women) Bill")]
    assert trials[0].cls == "MR-02-bill" and trials[0].subject["division_ids"] == [1]


def test_the_bill_that_divided_is_never_asked_as_absent_however_it_is_printed():
    """"Sentencing Act 2026" is the Sentencing Bill; a bill named in a motion that day divided too."""
    trials = D.mr02_bill_trials([truth(1, "Sentencing Bill: Third Reading"),
                                 truth(2, "Business of the House", motion="That the Sentencing (Women) Bill be read")],
                                ["Sentencing Act 2026", "Sentencing (Women) Bill"])  # fmt: skip
    assert trials == []


def test_a_day_whose_divisions_name_no_bill_asks_any_bill():
    trials = D.mr02_bill_trials([truth(1, "Draft Plant Health (Amendment) Regulations 2026", day=date(2026, 9, 9))], BILLS)
    assert [t.form for t in trials] == ["no_bill_day"] and trials[0].subject["nonexistent"] in BILLS


def test_an_answer_naming_any_division_is_a_harm_and_a_true_absence_passes():
    verdict = D.verdict_for([truth(1, "Sentencing Bill: motion to disagree with Lords Amendment 7")])
    trial = D.mr02_bill_trials([truth(1, "Sentencing Bill: motion to disagree with Lords Amendment 7")], BILLS)[0]
    answered = Answer(question="q?", provenance=PROV, rendered="x", structured_facts=(StructuredFact(
        template="division_result", params={"division_id": 1}, value={"division_id": 1, "result": "agreed"}),))
    assert verdict(trial, answered)[0] == "harm"
    absent = Answer(question="q?", provenance=PROV, abstention=Abstention.NOT_IN_RECORD,
                    abstention_detail="No division in the Commons on 20 January 2026 is recorded on Sentencing (Women) "
                                      "Bill. The divisions recorded that day were: division 1: Sentencing Bill: motion to "
                                      "disagree with Lords Amendment 7.")  # fmt: skip
    assert verdict(trial, absent)[0] == "pass"
