"""Rubric v3: a broad question accepts any k of its vital points, an example is never required, both sides means a
point from each side, and another member's identical point counts unless the question asks who.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from goldset import rubric as R
from goldset.schema import HARMFUL_PENALTY, OUTCOME_CREDIT

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GJ = _load("gemini_judge_v3_t", "gemini_judge.py")
EA = GJ.EA

# The accuracy definitions exactly as E3 graded them. v3 may not move a word of them.
FROZEN_ACCURACY = {
    "correct_abstention": ("The right response was to decline, or to ask which thing was meant, and the answer did so "
                           "without asserting anything wrong."),
    "over_abstention": ("The right response was to answer (or to correct the premise, or to report neutrally), and the "
                        "answer declined instead, asserting nothing wrong."),
    "wrong_but_benign": ("Something the answer asserts is unsupported or inaccurate, but a reader would not be materially "
                         "misled about the record."),
    "harmful": ("Something the answer asserts would mislead a reader about the record: a wrong figure, date or result; "
                "words attributed to someone who did not say them; a position attributed to a person or group that did "
                "not hold it; a claim on the must-not-claim list; or a false premise adopted as fact."),
}


def test_the_accuracy_rules_are_identical_in_v2_and_v3():
    for outcome, words in FROZEN_ACCURACY.items():
        assert R.OUTCOME_DEFINITIONS[outcome] == words, outcome
    assert HARMFUL_PENALTY == -8
    assert OUTCOME_CREDIT == {"correct": 1.0, "correct_missing_okay_nuggets": 0.75, "missing_vital_nugget": 0.0,
                              "correct_abstention": 1.0, "over_abstention": 0.0, "wrong_but_benign": 0.0}
    assert any("harmful comes first, then wrong_but_benign" in rule for rule in R.GRADING_RULES)


def test_rule3_the_identical_point_by_another_member_counts_unless_the_question_asks_who():
    prompt = R.judging_prompt()
    assert R.IDENTICAL_POINT_RULE in prompt
    assert "unless the question asks who" in R.IDENTICAL_POINT_RULE.lower()
    assert R.RUBRIC_VERSION == "v3"


def test_the_prompt_asks_for_the_points_present_and_explains_what_an_entry_needs():
    prompt = R.judging_prompt()
    assert '"points_present"' in prompt and '"vital_required"' in prompt
    assert "points_present" in R.REPLY_FIELDS
    assert "never required" in prompt and "one from each side" in prompt


# --- the import checks every verdict against its own list of points ------------------------------------------------


def _meta(points, required):
    """One entry as the key records it: each point as (vital, side, kind), and how many vital points are needed."""
    return {"points": [list(p) for p in points], "required": required}


BROAD = _meta([(True, "", "point"), (True, "", "point"), (True, "", "point")], 2)
SIDED = _meta([(True, "for", "point"), (True, "for", "point"), (True, "against", "point")], 2)
EXAMPLE = _meta([(True, "", "point"), (False, "", "example")], 1)


def _v(outcome, points, vital=None):
    return {"id": "J001", "outcome": outcome, "points_present": points,
            "vital_present": vital if vital is not None else len(points), "why": "x"}


@pytest.mark.parametrize("outcome,points,ok", [
    ("correct", [1, 2], True), ("correct", [1, 3], True), ("correct", [1], False),
    ("missing_vital_nugget", [1], True), ("missing_vital_nugget", [1, 2], False),
    ("correct_missing_okay_nuggets", [2, 3], True),
])
def test_k_of_n_scoring_for_broad_items(outcome, points, ok):
    assert (GJ.points_problems("J001", _v(outcome, points), BROAD) == []) is ok


@pytest.mark.parametrize("outcome,points,ok", [
    ("correct", [1, 2], False),  # two points, both for: one side missing
    ("correct", [1, 3], True), ("missing_vital_nugget", [1, 2], True), ("missing_vital_nugget", [2, 3], False),
])
def test_both_sides_needs_one_point_per_side(outcome, points, ok):
    assert (GJ.points_problems("J001", _v(outcome, points), SIDED) == []) is ok


def test_an_example_is_never_required():
    assert GJ.points_problems("J001", _v("correct", [1]), EXAMPLE) == []


@pytest.mark.parametrize("points", [[0], [4], [1, 1], ["1"], None, [True]])
def test_points_present_must_list_valid_point_numbers_each_once(points):
    assert any("points_present" in p for p in GJ.points_problems("J001", _v("correct", points, vital=2), BROAD))


def test_vital_present_must_agree_with_the_points_listed():
    assert any("vital_present" in p for p in GJ.points_problems("J001", _v("correct", [1, 2], vital=3), BROAD))
    assert any("vital_present" in p for p in GJ.points_problems("J001", _v("correct", [1, 2], vital=2), EXAMPLE))


# --- export and import, end to end ---------------------------------------------------------------------------------


def _gold(item_id, n, **v3):
    nuggets = v3.pop("nuggets", (SimpleNamespace(text=f"A point about topic {n}.", vital=True, kind="", side=""),
                                 SimpleNamespace(text=f"A detail about topic {n}.", vital=False, kind="", side="")))
    return SimpleNamespace(id=item_id, question=f"What was said about topic number {n}?", expected_behaviour="answer",
                           criticality="major", category="paraphrase", forbidden_claims=(), nuggets=nuggets,
                           breadth=v3.get("breadth", ""), min_points=v3.get("min_points"))


BROAD_ITEM = _gold("hum0003", 3, breadth="broad", min_points=2, nuggets=(
    SimpleNamespace(text="An argument that ports need investment.", vital=True, kind="point", side="for"),
    SimpleNamespace(text="An argument that ports bring jobs.", vital=True, kind="point", side="for"),
    SimpleNamespace(text="An argument that ports are overbuilt.", vital=True, kind="point", side="against"),
    SimpleNamespace(text="Ann Example said it.", vital=False, kind="example", side="for")))
GOLD = {"hum0001": _gold("hum0001", 1), "hum0002": _gold("hum0002", 2), "hum0003": BROAD_ITEM}
BANK = {"version": 1, "rubric": "v2", "seed_bases": [], "sentinels": [
    {"source": "triage:J101", "item": "sen0001", "breadth": "narrow", "known_outcome": "missing_vital_nugget",
     "known_vital_present": 0, "entry": {"question": "What did the minister say about ports?",
                                         "expected": "Answer the question from the record.",
                                         "reference_points": [{"text": "The minister promised two ports.", "vital": True}],
                                         "vital_total": 1, "must_not_claim": [], "answer_is_a_decline": False,
                                         "answer": "A member spoke about harbours."}},
    {"source": "triage:J102", "item": "sen0002", "known_outcome": "missing_vital_nugget", "known_vital_present": 0,
     "entry": {"question": "What was said about lighthouses?", "expected": "Answer the question from the record.",
               "reference_points": [{"text": "A point about lighthouses.", "vital": True}], "vital_total": 1,
               "must_not_claim": [], "answer_is_a_decline": False, "answer": "Lighthouses were discussed."}},
]}


def _row(item_id):
    return {"id": item_id, "category": "paraphrase", "criticality": "major", "outcome": None, "why": "",
            "by": "awaiting-judge", "route": "retrieval", "fact": None, "rendered": f"An answer, number {item_id[-1]}.",
            "abstention": None, "abstention_detail": "", "vital_present": None, "vital_total": None}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    report = {"split": "dev", "run_id": "dev-v3", "items": 3, "judge": EA.NO_JUDGE, "gate_evaluated": False,
              "rows": [_row(i) for i in GOLD], "trace": {"overall": {"p95_ms": 4000}}}
    path = tmp_path / "e3_eval_dev_v3.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _entries(out):
    return [e for p in out["packets"] for e in json.loads(Path(p).read_text(encoding="utf-8"))["entries"]]


def test_a_v3_packet_says_what_each_entry_needs_and_what_each_point_is(run, tmp_path):
    out = GJ.export_v2(run, out_dir=tmp_path / "g", bank=BANK, sentinels=0, seeds=0, canaries=0)
    entries = {e["question"]: e for e in _entries(out)}
    broad = entries["What was said about topic number 3?"]
    assert broad["vital_required"] == 2 and broad["vital_total"] == 3  # k of n: any 2 of 3, one from each side
    assert [(p["kind"], p["side"]) for p in broad["reference_points"]] == [("point", "for"), ("point", "for"),
                                                                           ("point", "against"), ("example", "for")]
    narrow = entries["What was said about topic number 1?"]
    assert narrow["vital_required"] == narrow["vital_total"] == 1
    md = Path(out["markdown"][0]).read_text(encoding="utf-8")
    assert "1. [VITAL]" in md and "(example" in md and "side: against" in md and "needs 2 of the 3 VITAL" in md


def test_a_v2_export_needs_every_vital_point_and_shows_no_kinds(run, tmp_path):
    out = GJ.export_v2(run, out_dir=tmp_path / "g2", bank=BANK, sentinels=0, seeds=0, canaries=0, rubric="v2")
    broad = next(e for e in _entries(out) if e["question"].endswith("3?"))
    assert broad["vital_required"] == broad["vital_total"] == 3
    assert all(p["kind"] == "" and p["side"] == "" for p in broad["reference_points"])
    assert json.loads(Path(out["key"]).read_text(encoding="utf-8"))["rubric"] == "v2"


def test_judged_reports_record_their_rubric_version_and_what_each_row_needed(run, tmp_path):
    out = GJ.export_v2(run, out_dir=tmp_path / "g3", bank=BANK, sentinels=0, seeds=0, canaries=0)
    packet = json.loads(Path(out["packets"][0]).read_text(encoding="utf-8"))
    def listed(e):  # the broad entry needs a point from each side: its first "for" point and its "against" point
        return [1, 3] if e["vital_total"] == 3 else [1]

    verdicts = [{"id": e["id"], "outcome": "correct", "points_present": listed(e), "vital_present": len(listed(e)),
                 "vital_total": e["vital_total"], "confidence": "high", "why": "the points needed are stated",
                 "evidence": e["answer"][:12]} for e in packet["entries"]]
    reply = tmp_path / "reply.json"
    reply.write_text(json.dumps({"packet_id": packet["packet_id"], "verdicts": verdicts}), encoding="utf-8")
    judged = json.loads(Path(GJ.import_verdicts(run, [reply], key_path=Path(out["key"]), judge_model="gemini-x",
                                                mode="chat")).read_text(encoding="utf-8"))
    assert judged["rubric"] == "v3"
    rows = {r["id"]: r for r in judged["rows"]}
    assert rows["hum0003"]["vital_required"] == 2 and rows["hum0003"]["outcome"] == "correct"
    assert rows["hum0003"]["points_present"] == [1, 3]
    assert rows["hum0001"]["vital_required"] == 1


def _sentinels_chosen(rubric: str) -> set[str]:
    import random

    chosen, _ = GJ.I.choose(BANK, exclude=set(), rng=random.Random(1), label="p", sentinels=2, seeds=0, canaries=0,
                            rubric=rubric)
    return {meta["item"] for _, meta in chosen}


def test_under_v3_a_sentinel_whose_verdict_v3_could_change_is_never_used():
    """sen0002 was adjudicated "missing a vital point" under v2 on an item of unknown breadth: under k of n it might
    now be correct, so it cannot be a sentinel for a v3 packet. sen0001 is narrow, so its verdict cannot move.
    """
    assert _sentinels_chosen("v3") == {"sen0001"}
    assert _sentinels_chosen("v2") == {"sen0001", "sen0002"}
