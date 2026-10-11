"""The maintainer's Gemini as E3's answer judge, offline: export blind, import strictly, score identically."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GJ = _load("gemini_judge_t", "gemini_judge.py")
EA = GJ.EA

Nugget = SimpleNamespace


# Question texts that do not contain the item id, so the leak check below tests the CODE, not the fixture.
WORDS = {"hum0001": "housing", "hum0002": "schools", "db0003": "a division", "adv0004": "the moon",
         "old0005": "a premise"}


def gold_item(item_id, behaviour="answer", vital=(True, False), forbidden=()):
    topic = WORDS[item_id]
    return SimpleNamespace(id=item_id, question=f"What was said about {topic}?", expected_behaviour=behaviour,
                           criticality="major", category="paraphrase", forbidden_claims=tuple(forbidden),
                           nuggets=tuple(Nugget(text=f"point {k} about {topic}", vital=v) for k, v in enumerate(vital)))


GOLD = {g.id: g for g in [gold_item("hum0001"), gold_item("hum0002", vital=(True, True)),
                          gold_item("db0003", vital=(True,)), gold_item("adv0004", behaviour="abstain_not_in_record",
                                                                      vital=(True,), forbidden=["any answer"]),
                          gold_item("old0005", behaviour="correct_false_premise", vital=(True,))]}


def row(item_id, outcome=None, by="awaiting-judge", rendered="An answer from the record.", abstention=None,
        detail=""):
    return {"id": item_id, "category": "paraphrase", "criticality": "major", "outcome": outcome, "why": "",
            "by": by, "route": "retrieval", "fact": None, "rendered": rendered, "abstention": abstention,
            "abstention_detail": detail, "vital_present": None, "vital_total": None}


def run_report(tmp_path):
    rows = [row("hum0001"), row("hum0002"), row("db0003", outcome="correct", by="deterministic-structured"),
            row("adv0004", outcome="correct_abstention", by="deterministic", rendered="", abstention="not_in_record",
                detail="The passages found in the parliamentary record do not answer this question."),
            row("old0005", rendered="", abstention="false_premise", detail="No: it was said in the Lords.")]
    report = {"split": "dev", "run_id": "dev-20261005-120000", "items": len(rows), "judge": EA.NO_JUDGE,
              "gate_evaluated": False, "rows": rows, "trace": {"overall": {"p95_ms": 4000}}}
    path = tmp_path / "e3_eval_dev_final.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    report = run_report(tmp_path)
    out = GJ.export(report, out_dir=tmp_path / "gemini")
    return report, out


def packet_of(out):
    return json.loads(Path(out["packet"]).read_text(encoding="utf-8"))


def key_of(out):
    return json.loads(Path(out["key"]).read_text(encoding="utf-8"))


# --- export ---------------------------------------------------------------------------------------------------------


def test_the_packet_holds_exactly_the_answers_no_judge_has_read(exported):
    _, out = exported
    packet, key = packet_of(out), key_of(out)
    assert len(packet["entries"]) == 3
    assert sorted(key["map"].values()) == ["hum0001", "hum0002", "old0005"]


def test_the_packet_is_blind_by_construction(exported):
    _, out = exported
    text = Path(out["packet"]).read_text(encoding="utf-8") + Path(out["markdown"]).read_text(encoding="utf-8")
    for leak in ("hum0001", "hum0002", "old0005", "db0003", "adv0004", "paraphrase", "retrieval", "major",
                 "deterministic", "awaiting-judge", "correct_abstention"):
        assert leak not in text, leak
    ids = [e["id"] for e in packet_of(out)["entries"]]
    assert ids == sorted(ids) and all(i.startswith("J") for i in ids), "opaque ids, in packet order"


def test_each_entry_carries_what_a_grader_needs_and_the_answer_exactly_as_shown(exported):
    _, out = exported
    packet, key = packet_of(out), key_of(out)
    by_item = {key["map"][e["id"]]: e for e in packet["entries"]}
    two = by_item["hum0002"]
    assert two["question"] == "What was said about schools?" and two["vital_total"] == 2
    assert [p["vital"] for p in two["reference_points"]] == [True, True]
    assert two["answer"] == "An answer from the record."
    premise = by_item["old0005"]
    assert premise["answer_is_a_decline"] is True
    assert "No: it was said in the Lords." in premise["answer"]
    assert premise["expected"] == GJ.EXPECTED_IN_WORDS["correct_false_premise"]


def test_the_key_binds_the_verdicts_to_this_packet(exported):
    _, out = exported
    packet, key = packet_of(out), key_of(out)
    assert packet["packet_id"] == key["packet_id"] and len(key["packet_id"]) >= 12
    assert "map" not in packet


def test_all_mode_also_exports_the_deterministic_items_for_a_cross_check(tmp_path, monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    out = GJ.export(run_report(tmp_path), out_dir=tmp_path / "g", include_graded=True)
    assert len(packet_of(out)["entries"]) == 5


# --- import ---------------------------------------------------------------------------------------------------------


def verdicts_for(out, outcomes: dict[str, str], vital: dict[str, int] | None = None, **extra):
    key = key_of(out)
    reverse = {item: jid for jid, item in key["map"].items()}
    vital = vital or {}
    items = []
    for item_id, outcome in outcomes.items():
        total = key["vital_totals"][reverse[item_id]]
        items.append({"id": reverse[item_id], "outcome": outcome, "vital_total": total,
                      "vital_present": vital.get(item_id, total if outcome.startswith("correct") else 0),
                      "confidence": "high", "why": f"graded {outcome}", "evidence": ""})
    return {"packet_id": key["packet_id"], "judge_model": "gemini-test", "verdicts": items, **extra}


GOOD = {"hum0001": "correct", "hum0002": "missing_vital_nugget", "old0005": "correct"}


def test_imported_verdicts_are_scored_by_the_same_function_as_a_live_judge(exported, tmp_path):
    report, out = exported
    reply = tmp_path / "reply.json"
    reply.write_text(json.dumps(verdicts_for(out, GOOD, vital={"hum0002": 1})), encoding="utf-8")
    judged_path = GJ.import_verdicts(report, reply, key_path=Path(out["key"]))
    judged = json.loads(Path(judged_path).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in judged["rows"]}
    assert rows["hum0001"]["outcome"] == "correct" and rows["hum0001"]["by"] == "gemini"
    assert rows["hum0002"]["vital_present"] == 1 and rows["hum0002"]["vital_total"] == 2
    assert rows["db0003"]["by"] == "deterministic-structured", "deterministic grades are never overwritten"
    assert judged["gate_evaluated"] is True and judged["judge"] == "gemini/gemini-test"
    expected = EA.outcome_fields(judged["rows"])
    assert judged["scored"] == expected["scored"] and judged["by_outcome"] == expected["by_outcome"]
    assert judged["ungraded"] == 0
    assert judged_path != report and json.loads(report.read_text(encoding="utf-8"))["judge"] == EA.NO_JUDGE


@pytest.mark.parametrize("mutate, problem", [
    (lambda v: v["verdicts"].pop(), "missing"),
    (lambda v: v["verdicts"].append(dict(v["verdicts"][0])), "more than once"),
    (lambda v: v["verdicts"][0].update(id="J999"), "unknown"),
    (lambda v: v["verdicts"][0].update(outcome="mostly_fine"), "outcome"),
    (lambda v: v["verdicts"][0].update(vital_total=9), "vital_total"),
    (lambda v: v["verdicts"][0].update(vital_present=-1), "vital_present"),
    (lambda v: v["verdicts"][0].update(why=""), "why"),
    (lambda v: v.update(packet_id="another-packet"), "packet"),
])
def test_an_incomplete_or_incoherent_reply_is_refused_and_nothing_is_written(exported, tmp_path, mutate, problem):
    report, out = exported
    reply = verdicts_for(out, GOOD, vital={"hum0002": 1})
    mutate(reply)
    path = tmp_path / "reply.json"
    path.write_text(json.dumps(reply), encoding="utf-8")
    before = set(tmp_path.rglob("*judged*"))
    with pytest.raises(GJ.ImportRefused, match=problem):
        GJ.import_verdicts(report, path, key_path=Path(out["key"]))
    assert set(tmp_path.rglob("*judged*")) == before


@pytest.mark.parametrize("outcome, present", [("correct", 1), ("missing_vital_nugget", 2)])
def test_a_verdict_that_contradicts_its_own_vital_count_is_refused(exported, tmp_path, outcome, present):
    report, out = exported
    reply = verdicts_for(out, {**GOOD, "hum0002": outcome}, vital={"hum0002": present})
    path = tmp_path / "reply.json"
    path.write_text(json.dumps(reply), encoding="utf-8")
    with pytest.raises(GJ.ImportRefused, match="contradicts"):
        GJ.import_verdicts(report, path, key_path=Path(out["key"]))


def test_a_reply_wrapped_in_prose_or_a_code_fence_is_still_read(exported, tmp_path):
    report, out = exported
    body = json.dumps(verdicts_for(out, GOOD, vital={"hum0002": 1}))
    path = tmp_path / "reply.txt"
    path.write_text(f"Here are the grades.\n```json\n{body}\n```\nDone.", encoding="utf-8")
    assert Path(GJ.import_verdicts(report, path, key_path=Path(out["key"]))).exists()


def test_a_partial_import_is_possible_only_on_purpose_and_the_gate_then_says_incomplete(exported, tmp_path):
    report, out = exported
    reply = verdicts_for(out, {"hum0001": "correct", "hum0002": "correct"})
    path = tmp_path / "reply.json"
    path.write_text(json.dumps(reply), encoding="utf-8")
    with pytest.raises(GJ.ImportRefused, match="missing"):
        GJ.import_verdicts(report, path, key_path=Path(out["key"]))
    judged = json.loads(Path(GJ.import_verdicts(report, path, key_path=Path(out["key"]), partial=True)).read_text(
        encoding="utf-8"))
    assert judged["ungraded"] == 1 and EA.exit_code(judged) == 1


def test_a_cross_check_reports_disagreements_with_deterministic_grades_without_overwriting_them(tmp_path,
                                                                                             monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    report = run_report(tmp_path)
    out = GJ.export(report, out_dir=tmp_path / "g", include_graded=True)
    outcomes = {**GOOD, "db0003": "harmful", "adv0004": "correct_abstention"}
    path = tmp_path / "reply.json"
    path.write_text(json.dumps(verdicts_for(out, outcomes, vital={"hum0002": 1})), encoding="utf-8")
    judged = json.loads(Path(GJ.import_verdicts(report, path, key_path=Path(out["key"]))).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in judged["rows"]}
    assert rows["db0003"]["outcome"] == "correct", "a deterministic grade stands"
    assert judged["judged_offline"]["disagreements_with_deterministic"] == [
        {"id": "db0003", "deterministic": "correct", "gemini": "harmful", "why": "graded harmful"}]


# --- compare --------------------------------------------------------------------------------------------------------


def test_two_passes_are_compared_verdict_by_verdict(exported, tmp_path):
    _, out = exported
    a = verdicts_for(out, GOOD, vital={"hum0002": 1})
    b = verdicts_for(out, {**GOOD, "hum0001": "harmful"}, vital={"hum0002": 1})
    result = GJ.compare(a, b)
    assert result["compared"] == 3 and result["flips"] == 1
    assert result["harmful_disagreements"] == 1 and result["flip_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_a_reply_in_two_parts_is_imported_as_one(exported, tmp_path):
    """The parts are read together and checked as one reply — every id once across them — so nothing has to be stitched
    by hand.
    """
    report, out = exported
    whole = verdicts_for(out, GOOD, vital={"hum0002": 1})
    first, second = dict(whole, verdicts=whole["verdicts"][:2]), dict(whole, verdicts=whole["verdicts"][2:])
    a, b = tmp_path / "part1.json", tmp_path / "part2.json"
    a.write_text(json.dumps(first), encoding="utf-8")
    b.write_text("Continuing:\n```json\n" + json.dumps(second) + "\n```", encoding="utf-8")
    judged = json.loads(Path(GJ.import_verdicts(report, [a, b], key_path=Path(out["key"]))).read_text(encoding="utf-8"))
    assert judged["ungraded"] == 0 and judged["judged_offline"]["verdicts"] == ["part1.json", "part2.json"]


def test_parts_of_different_packets_are_refused(exported, tmp_path):
    report, out = exported
    whole = verdicts_for(out, GOOD, vital={"hum0002": 1})
    a, b = tmp_path / "part1.json", tmp_path / "part2.json"
    a.write_text(json.dumps(dict(whole, verdicts=whole["verdicts"][:2])), encoding="utf-8")
    b.write_text(json.dumps(dict(whole, packet_id="0000000000000000", verdicts=whole["verdicts"][2:])),
                 encoding="utf-8")
    with pytest.raises(GJ.ImportRefused, match="different packets"):
        GJ.import_verdicts(report, [a, b], key_path=Path(out["key"]))


def test_the_prompt_is_written_once_and_replaced_in_place(tmp_path):
    log = tmp_path / "GEMINI_LOG.md"
    head = "# Audit\n\nThe owner's notes.\n"
    log.write_text(head, encoding="utf-8")
    assert GJ.write_prompt(log) is True
    first = log.read_text(encoding="utf-8")
    assert first.startswith(head) and first.count(GJ.PROMPT_START) == 1
    assert GJ.write_prompt(log) is False and log.read_text(encoding="utf-8") == first
    stale = first.replace("THE RULES:", "THE OLD RULES:")
    log.write_text(stale + "\nA later note.\n", encoding="utf-8")
    assert GJ.write_prompt(log) is True
    again = log.read_text(encoding="utf-8")
    assert again.count(GJ.PROMPT_START) == 1 and "THE OLD RULES" not in again
    assert again.endswith("A later note.\n")
