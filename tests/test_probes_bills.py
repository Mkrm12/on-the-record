"""ST-bill_stage_date: every dated stage of every Bill, asked when it happened."""

from __future__ import annotations

from datetime import date

import pytest

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import bills as OB
from eval.probes import core
from eval.probes import bills as PB

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip
S2R = OB.StageTruth(7, "Railways Bill", 71, "2nd reading", "Commons", (date(2025, 12, 9),))
SCOM = OB.StageTruth(7, "Railways Bill", 72, "Committee stage", "Commons",
                     (date(2026, 1, 13), date(2026, 1, 15)))  # fmt: skip
SRA = OB.StageTruth(7, "Railways Bill", 79, "Royal Assent", "", (date(2026, 3, 20),))
TRUTHS = [S2R, SCOM, SRA]
TRIALS = {t.subject["stage_id"]: t for t in PB.st_trials(TRUTHS) if t.form == "canonical"}  # QA pass 4's forms aside
V = PB.verdict_for(TRUTHS)


def fact(stage="2nd reading", house="Commons", first="2025-12-09", sittings=1, bill_id=7):
    return StructuredFact(template="bill_stage_date", params={},
                          value={"bill_id": bill_id, "bill": "Railways Bill", "stage": stage, "house": house,
                                 "first_sitting": first, "sittings": sittings})  # fmt: skip


def answer(*facts, clarify=None, abstention=None):
    return Answer(question="q?", provenance=PROV, structured_facts=tuple(facts), rendered="An answer." if facts else "",
                  facets={"clarify": clarify} if clarify else {}, abstention=abstention)  # fmt: skip


def test_every_stage_is_asked_about_in_the_words_people_use():
    assert TRIALS[71].question == "When did the Railways Bill have its 2nd reading in the Commons?"
    assert TRIALS[79].question == "When did the Railways Bill receive Royal Assent?"
    assert all(t.cls == "ST-bill_stage_date" and t.category == "bill_stage_date" for t in TRIALS.values())


def test_the_record_s_first_sitting_and_its_sitting_days_are_a_pass():
    assert V(TRIALS[71], answer(fact()))[0] == "pass"
    assert V(TRIALS[72], answer(fact("Committee stage", first="2026-01-13", sittings=2)))[0] == "pass"


@pytest.mark.parametrize("wrong", [fact(first="2025-12-10"), fact(sittings=2), fact(bill_id=8)])
def test_a_wrong_date_a_wrong_count_or_another_bill_is_a_harm(wrong):
    assert V(TRIALS[71], answer(wrong))[0] == "harm"


def test_another_stage_answered_is_a_harm():
    assert V(TRIALS[71], answer(fact("Committee stage", first="2026-01-13", sittings=2)))[0] == "harm"


def test_a_question_back_or_a_decline_is_never_a_pass():
    assert V(TRIALS[71], answer(clarify=[{"stage_id": 71}, {"stage_id": 99}]))[0] == "clarified"
    assert V(TRIALS[71], answer(abstention=Abstention.INSUFFICIENT_EVIDENCE))[0] == "declined"
    assert V(TRIALS[71], answer(abstention=Abstention.NOT_IN_RECORD))[0] == "harm", "the stage is in the record"


def test_a_title_two_bills_share_makes_the_trial_ambiguous():
    twin = OB.StageTruth(8, "Railways Bill", 81, "2nd reading", "Commons", (date(2024, 5, 1),))
    trials = [t for t in PB.st_trials([S2R, twin]) if t.form == "canonical"]
    assert trials and all(t.ambiguous for t in trials)


def test_no_trial_from_the_sealed_window():
    with pytest.raises(core.ProbeError):
        PB.st_trials([OB.StageTruth(7, "Railways Bill", 73, "Report stage", "Commons", (date(2026, 5, 20),))])


def test_a_stage_trial_is_its_own_cluster():
    assert TRIALS[71].cluster == ("stage", 71)
