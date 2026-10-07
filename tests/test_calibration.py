"""E3's last maintainer step, made structural: the answer-judge calibration packet."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from goldset import calibration as CAL
from goldset.blind import BlindnessError


def row(i, outcome, by="groq", rendered="An answer."):
    return {"id": f"x{i:03d}", "outcome": outcome, "by": by, "rendered": rendered, "why": f"judge said {outcome}"}


ROWS = ([row(i, "harmful") for i in range(3)] + [row(i, "wrong_but_benign") for i in range(3, 5)]
        + [row(i, "missing_vital_nugget") for i in range(5, 25)] + [row(i, "correct") for i in range(25, 125)]
        + [row(900, "correct", by="deterministic-structured"), row(901, "over_abstention", by="deterministic")])
GOLD = {r["id"]: SimpleNamespace(question=f"Question number {n}", nuggets=(SimpleNamespace(text="N", vital=True),),
                                 forbidden_claims=()) for n, r in enumerate(ROWS)}


def test_only_the_judges_own_verdicts_can_calibrate_it():
    judged = CAL.judged_rows(ROWS)
    assert all(r["by"] == "groq" for r in judged) and len(judged) == 125


def test_the_sample_carries_every_rare_verdict():
    chosen = CAL.select(CAL.judged_rows(ROWS), n=50)
    outcomes = [r["outcome"] for r in chosen]
    assert len(chosen) == 50 and outcomes.count("harmful") == 3 and outcomes.count("wrong_but_benign") == 2
    assert outcomes.count("missing_vital_nugget") == 20 and outcomes.count("correct") == 25


def test_the_sample_is_never_all_one_kind():
    """The first version took every missing-vital answer before any correct one: 41 of 50, and no correct answer."""
    rows = [row(i, "missing_vital_nugget") for i in range(60)] + [row(i, "correct") for i in range(60, 200)]
    outcomes = [r["outcome"] for r in CAL.select(CAL.judged_rows(rows), n=50)]
    assert outcomes.count("correct") == 25 and outcomes.count("missing_vital_nugget") == 25


def test_the_sample_is_the_same_every_time():
    judged = CAL.judged_rows(ROWS)
    assert [r["id"] for r in CAL.select(judged, seed=7)] == [r["id"] for r in CAL.select(judged, seed=7)]


def test_the_worksheet_carries_no_verdict_and_no_real_id():
    items, key = CAL.worksheet(CAL.select(CAL.judged_rows(ROWS), n=10), GOLD)
    text = repr(items)
    assert "judge" not in text and "x0" not in text and "harmful" not in text.replace("wrong_if_asserted", "")
    assert set(key) == {i["cid"] for i in items} and all("judge_outcome" in v for v in key.values())


def test_a_leak_into_the_worksheet_is_refused():
    with pytest.raises(BlindnessError):
        CAL.assert_blind({"items": [{"cid": "c001", "judge_outcome": "correct"}]})


def test_perfect_agreement_passes():
    key = {"c001": {"item_id": "a", "judge_outcome": "harmful", "judge_why": ""},
           "c002": {"item_id": "b", "judge_outcome": "correct", "judge_why": ""},
           "c003": {"item_id": "c", "judge_outcome": "missing_vital_nugget", "judge_why": ""}}
    got = CAL.score({c: v["judge_outcome"] for c, v in key.items()}, key)
    assert got["passes"] and got["kappa_outcomes"]["kappa"] == 1.0 and got["harmful"]["recall"] == 1.0


def test_a_judge_that_misses_a_harmful_answer_fails_however_good_its_kappa():
    key = {f"c{i:03d}": {"item_id": str(i), "judge_outcome": "correct", "judge_why": ""} for i in range(20)}
    key["c100"] = {"item_id": "h", "judge_outcome": "correct", "judge_why": ""}
    owner = {c: "correct" for c in key} | {"c100": "harmful"}
    got = CAL.score(owner, key)
    assert got["harmful"]["recall"] == 0.0 and not got["passes"]
    assert got["disagreements"] == [{"cid": "c100", "item_id": "h", "owner": "harmful", "judge": "correct",
                                      "judge_why": ""}]


def test_the_owners_verdicts_on_every_judge_harmful_answer_are_the_runs_harmful_count():
    key = {"c001": {"item_id": "h1", "judge_outcome": "harmful", "judge_why": ""},
           "c002": {"item_id": "h2", "judge_outcome": "harmful", "judge_why": ""},
           "c003": {"item_id": "ok", "judge_outcome": "correct", "judge_why": ""}}
    owner = {"c001": "harmful", "c002": "correct", "c003": "correct"}
    got = CAL.score(owner, key, judge_harmful_in_run=2)["adjudicated"]
    assert got["every_judge_harmful_answer_labelled"] and got["harmful_after_owner_review"] == ["h1"]


def test_no_adjudicated_count_when_a_judge_harmful_answer_was_left_out():
    key = {"c001": {"item_id": "h1", "judge_outcome": "harmful", "judge_why": ""}}
    got = CAL.score({"c001": "correct"}, key, judge_harmful_in_run=3)["adjudicated"]
    assert not got["every_judge_harmful_answer_labelled"] and got["harmful_after_owner_review"] is None


def test_labels_outside_the_vocabulary_or_the_sample_are_refused():
    key = {"c001": {"item_id": "a", "judge_outcome": "correct", "judge_why": ""}}
    with pytest.raises(ValueError):
        CAL.score({"c001": "fine"}, key)
    with pytest.raises(ValueError):
        CAL.score({"c999": "correct"}, key)


# The maintainer grades what the judge graded, in the judge's words -----------------------------


def gold_item(question="Who said it in the Commons?", behaviour="correct_false_premise"):
    return SimpleNamespace(question=question, nuggets=(SimpleNamespace(text="It was said in the Lords", vital=True),),
                           forbidden_claims=(), expected_behaviour=behaviour)


def test_a_decline_the_judge_graded_is_shown_to_the_owner_in_its_own_words():
    """F6: Gemini reads a decline's words ("(the system declined: …)"). The page used to show `rendered`, which a
    decline leaves empty — "(nothing was shown)" — so the maintainer's verdict on a decline Gemini called harmful,
    the verdict the run's adjudicated harmful count is made of, would have been given on no information.
    """
    decline = {"id": "adv0020", "outcome": "harmful", "by": "gemini", "rendered": "", "abstention": "false_premise",
               "abstention_detail": "That was said in the Lords by Lord Hanson, not in the Commons.", "why": "w"}
    items, _ = CAL.worksheet([decline], {"adv0020": gold_item()})
    assert items[0]["answer"] == "(the system declined: That was said in the Lords by Lord Hanson, not in the Commons.)"


def test_the_owner_is_told_what_a_right_response_does_in_the_judges_words():
    from goldset import rubric as R

    items, _ = CAL.worksheet([row(1, "correct", by="gemini")], {"x001": gold_item()})
    assert items[0]["expected"] == R.EXPECTED_IN_WORDS["correct_false_premise"]


def test_the_owner_chooses_from_the_judges_own_seven_outcomes():
    from goldset import rubric as R

    assert CAL.OWNER_OUTCOMES == R.OUTCOMES
    key = {"c001": {"item_id": "a", "judge_outcome": "over_abstention", "judge_why": ""}}
    assert CAL.score({"c001": "correct_abstention"}, key)["n"] == 1


def test_a_judges_verdict_on_a_decline_can_be_calibrated():
    rows = [row(1, "over_abstention", by="gemini"), row(2, "correct_abstention", by="gemini"),
            row(3, "over_abstention", by="deterministic")]
    judged = CAL.judged_rows(rows)
    assert [r["id"] for r in judged] == ["x001", "x002"]
    assert {r["id"] for r in CAL.select(judged, n=10)} == {"x001", "x002"}


def test_the_runs_harmful_count_keeps_the_harmful_answers_no_judge_graded():
    """A forbidden claim is harmful whatever any judge says, and it never reaches the calibration packet. The
    adjudicated count printed as "the run's harmful count" used to be the judge-harmful answers the maintainer
    confirmed — 0 there would have read as a clean run while a forbidden-claim answer stood.
    """
    key = {"c001": {"item_id": "h1", "judge_outcome": "harmful", "judge_why": ""}}
    got = CAL.score({"c001": "correct"}, key, judge_harmful_in_run=1, deterministic_harmful=["d7"])["adjudicated"]
    assert got["every_judge_harmful_answer_labelled"]
    assert got["harmful_after_owner_review"] == ["d7"]
    assert got["deterministic_harmful"] == ["d7"]


def test_the_page_shows_every_outcome_with_the_rubrics_definition(tmp_path, monkeypatch):
    import html
    import importlib.util
    import json
    from pathlib import Path

    from goldset import rubric as R

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("calibrate_judge_for_tests", root / "scripts" / "calibrate_judge.py")
    CJ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CJ)
    monkeypatch.setattr(CJ, "PAGE", tmp_path / "calibration.html")
    monkeypatch.setattr(CJ, "KEY", tmp_path / "calibration_key.json")
    gold = {f"x{i:03d}": gold_item(question=f"Question {i}", behaviour="answer") for i in range(1, 4)}
    monkeypatch.setattr(CJ, "load_gold", lambda path: list(
        SimpleNamespace(id=k, **vars(v)) for k, v in gold.items()))
    report = tmp_path / "e3_eval_dev_x.json"
    rows = [row(1, "harmful", by="gemini"), row(2, "correct", by="gemini"),
            row(3, "harmful", by="deterministic")]
    report.write_text(json.dumps({"run_id": "r", "rows": rows}), encoding="utf-8")
    assert CJ.build(report, n=50, seed=18) == 0
    page = (tmp_path / "calibration.html").read_text(encoding="utf-8")
    for name in R.OUTCOMES:
        assert f'"{name}"' in page
        assert html.escape(R.OUTCOME_DEFINITIONS[name], quote=False) in page
    stored = json.loads((tmp_path / "calibration_key.json").read_text(encoding="utf-8"))
    assert stored["judge_harmful_in_run"] == 1 and stored["deterministic_harmful"] == ["x003"]
