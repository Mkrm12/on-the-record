"""The E3 gate report must never print PASS for a bar the run could not evaluate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e3_gate_report_t", ROOT / "scripts" / "e3_gate_report.py")
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)

ANSWER, DECLINE = "answer", "abstain_not_in_record"


def gold(rows):
    return {r["id"]: SimpleNamespace(expected_behaviour=r.pop("_behaviour", ANSWER)) for r in rows}


def row(i, outcome, by="deterministic", behaviour=ANSWER, vital=None):
    r = {"id": f"x{i:03d}", "outcome": outcome, "by": by, "route": "retrieval", "_behaviour": behaviour}
    if vital:
        r["vital_present"], r["vital_total"] = vital
    return r


ROUTER = {"fact_questions": 10, "right_fact": 10, "accuracy": 1.0, "sent_to_a_fact": 0}


def report(rows, *, judged):
    g = gold(rows)
    return {"rows": rows, "items": len(rows), "gate_evaluated": judged, "router": ROUTER,
            "trace": {"overall": {"p95_ms": 4800}}}, g


def verdicts(rep, g):
    return {name: (verdict, value) for name, _, value, verdict in GR.bars(rep, g)}


def no_judge_run():
    rows = ([row(i, "correct", by="deterministic-structured", vital=(1, 1)) for i in range(40)]
            + [row(40 + i, "over_abstention") for i in range(10)]
            + [row(50 + i, "correct_abstention", behaviour=DECLINE) for i in range(10)]
            + [row(60 + i, None, by="awaiting-judge") for i in range(40)])
    return report(rows, judged=False)


def test_a_no_judge_run_does_not_pass_the_harmful_bar():
    v = verdicts(*no_judge_run())
    assert v["Harmful answers"][0] == "NOT EVALUATED"
    assert "40 answers await the judge" in v["Harmful answers"][1]


def test_over_abstention_is_counted_over_every_item_asked_not_over_the_few_graded():
    v = verdicts(*no_judge_run())
    verdict, value = v["Over-abstention"]
    assert "10 of 100 (10.0%)" in value, value  # not 10 of 60 (16.7%) — the graded rows alone
    assert verdict == "PASS"


def test_vital_coverage_waits_for_the_judge_rather_than_reading_the_structured_items_alone():
    verdict, value = verdicts(*no_judge_run())["Vital-nugget coverage (answered, judged)"]
    assert verdict == "PENDING JUDGE" and "40/40" in value


def test_correct_abstention_and_the_router_need_no_judge():
    v = verdicts(*no_judge_run())
    assert v["Correct abstention on should-decline items"][0] == "PASS"
    assert v["Router accuracy (fact questions, from the words)"][0] == "PASS"


def test_a_should_decline_item_awaiting_the_judge_stays_in_the_correct_abstention_denominator():
    """A should-decline item the system ANSWERED waits for the judge, and used to drop out of the denominator — 8
    declined of 10 read "8/8 PASS". Bounded instead: PASS only if it passes however the judge grades the rest.
    """
    rows = ([row(i, "correct", by="deterministic-structured", vital=(1, 1)) for i in range(20)]
            + [row(20 + i, "correct_abstention", behaviour=DECLINE) for i in range(8)]
            + [row(28 + i, None, by="awaiting-judge", behaviour=DECLINE) for i in range(2)])
    verdict, value = verdicts(*report(rows, judged=False))["Correct abstention on should-decline items"]
    assert verdict == "PENDING JUDGE" and "8/10" in value and "2 await the judge" in value


def test_correct_abstention_fails_when_even_the_best_case_misses_the_bar():
    rows = ([row(i, "correct_abstention", behaviour=DECLINE) for i in range(7)]
            + [row(7 + i, "harmful", by="gemini", behaviour=DECLINE) for i in range(2)]
            + [row(9, None, by="awaiting-judge", behaviour=DECLINE)])
    assert verdicts(*report(rows, judged=False))["Correct abstention on should-decline items"][0] == "FAIL"


def test_a_decline_awaiting_the_judge_may_still_be_an_over_abstention():
    """F6: a decline in the model's own words goes to the judge, which may grade it over_abstention. 13 of 100 passes
    the 15% bar only if the 5 waiting declines are not all over-abstentions — not known until the judge has read
    them.
    """
    rows = ([row(i, "correct", by="deterministic-structured", vital=(1, 1)) for i in range(82)]
            + [row(82 + i, "over_abstention") for i in range(13)]
            + [dict(row(95 + i, None, by="awaiting-judge"), abstention="false_premise") for i in range(5)])
    verdict, value = verdicts(*report(rows, judged=False))["Over-abstention"]
    assert verdict == "PENDING JUDGE" and "13 of 100" in value and "5 declines await the judge" in value


def test_a_judged_run_with_an_ungraded_answer_is_incomplete_not_passed():
    rows = [row(i, "correct", by="gemini", vital=(1, 1)) for i in range(20)] + [row(20, None, by="judge-failed")]
    v = verdicts(*report(rows, judged=True))
    assert v["Harmful answers"][0] == "INCOMPLETE"


def test_a_complete_judged_run_is_judged_on_its_numbers():
    rows = [row(i, "correct", by="gemini", vital=(1, 1)) for i in range(20)] + [row(20, "harmful", by="gemini")]
    v = verdicts(*report(rows, judged=True))
    assert v["Harmful answers"][0] == "FAIL"
    clean = [row(i, "correct", by="gemini", vital=(1, 1)) for i in range(20)]
    assert verdicts(*report(clean, judged=True))["Harmful answers"][0] == "PASS"


def test_the_exit_code_says_failed_not_evaluated_or_passed(tmp_path, monkeypatch):
    import json

    def code(rep, g):
        path = tmp_path / "run.json"
        path.write_text(json.dumps(rep), encoding="utf-8")
        monkeypatch.setattr(GR, "load_gold", lambda _: [SimpleNamespace(id=k, **vars(v)) for k, v in g.items()])
        return GR.main(["e3_gate_report.py", str(path)])

    assert code(*no_judge_run()) == 2, "not evaluated is not a pass"
    failing = [row(i, "correct", by="gemini", vital=(1, 1)) for i in range(20)] + [row(20, "harmful", by="gemini")]
    assert code(*report(failing, judged=True)) == 1
    clean = ([row(i, "correct", by="gemini", vital=(1, 1)) for i in range(20)]
             + [row(20 + i, "correct_abstention", behaviour=DECLINE) for i in range(3)])
    assert code(*report(clean, judged=True)) == 0
