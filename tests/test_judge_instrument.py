"""Manual judging protocol v2."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from goldset import instrument as I

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GJ = _load("gemini_judge_instrument_t", "gemini_judge.py")
GATE = _load("e3_gate_report_instrument_t", "e3_gate_report.py")
EA = GJ.EA

N_ITEMS = 60  # three packets: 25, 25 and 10


def _gold(item_id: str, n: int):
    return SimpleNamespace(id=item_id, question=f"What was said about topic number {n}?", expected_behaviour="answer",
                           criticality="major", category="paraphrase", forbidden_claims=(),
                           nuggets=(SimpleNamespace(text=f"A point about topic {n}.", vital=True),
                                    SimpleNamespace(text=f"A detail about topic {n}.", vital=False)))


GOLD = {f"hum{n:04d}": _gold(f"hum{n:04d}", n) for n in range(1, N_ITEMS + 1)}


def _entry(question, answer, points, must_not=(), expected="Answer the question from the record."):
    refs = [{"text": t, "vital": v} for t, v in points]
    return {"question": question, "expected": expected, "reference_points": refs,
            "vital_total": sum(v for _, v in points), "must_not_claim": list(must_not),
            "answer_is_a_decline": False, "answer": answer}


# A small bank under the test's control: sentinels and seed bases from items that are NOT in the run.
BANK = {
    "version": 1, "rubric": "v2",
    "sentinels": [
        {"source": "triage:J101", "item": "sen0001", "known_outcome": "correct", "known_vital_present": 1,
         "entry": _entry("Who raised the river levels?", "Ann Example said the river had risen by a metre.",
                         [("Ann Example raised it.", True)])},
        {"source": "triage:J102", "item": "sen0002", "known_outcome": "missing_vital_nugget", "known_vital_present": 0,
         "entry": _entry("What did the minister say about ports?", "A member spoke about harbours in general.",
                         [("The minister promised two new ports.", True)])},
        {"source": "triage:J103", "item": "sen0003", "known_outcome": "correct", "known_vital_present": 1,
         "entry": _entry("When was the fishing debate?", "The fishing debate took place on 3 March 2025.",
                         [("It was on 3 March 2025.", True)])},
    ],
    "seed_bases": [
        {"source": "dev_s18:db0006", "item": "seb0001", "kinds": ["number", "result"],
         "entry": _entry("Did the Commons agree the regulations?", "The motion was agreed to, 294 votes to 108.",
                         [("The division was agreed (294 to 108).", True)], ["That the division was negatived."])},
        {"source": "dev_s18:db0041", "item": "seb0002", "kinds": ["number"],
         "entry": _entry("What were the Ayes and Noes?", "77 voted Aye and 317 voted No, as counted by the tellers.",
                         [("The Ayes were 77 and the Noes 317.", True)])},
        {"source": "dev_s18:db0175", "item": "seb0003", "kinds": ["date"],
         "entry": _entry("When did the Bill reach committee?", "The Bill reached committee on 4 November 2024.",
                         [("It was on 4 November 2024.", True)])},
    ],
}


def _row(item_id):
    # The answer text never names the item id (a real answer never does), so the leak check tests the code.
    return {"id": item_id, "category": "paraphrase", "criticality": "major", "outcome": None, "why": "",
            "by": "awaiting-judge", "route": "retrieval", "fact": None,
            "rendered": f"An answer about topic {int(item_id[3:])}.",
            "abstention": None, "abstention_detail": "", "vital_present": None, "vital_total": None}


def _report(tmp_path, n=N_ITEMS):
    rows = [_row(f"hum{k:04d}") for k in range(1, n + 1)]
    report = {"split": "dev", "run_id": "dev-20261009-120000", "items": len(rows), "judge": EA.NO_JUDGE,
              "gate_evaluated": False, "rows": rows, "trace": {"overall": {"p95_ms": 4000}}}
    path = tmp_path / "e3_eval_dev_instr.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    report = _report(tmp_path)
    out = GJ.export_v2(report, out_dir=tmp_path / "gemini", bank=BANK)
    return report, out


def _packets(out):
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in out["packets"]]


def _key(out):
    return json.loads(Path(out["key"]).read_text(encoding="utf-8"))


def _verdict(jid, outcome, total, present, answer, evidence=None, points=None):
    if evidence is None:
        evidence = answer[: min(len(answer), 30)]
    # Rubric v3: the verdict lists the points it found.
    listed = points if points is not None else list(range(1, present + 1))
    return {"id": jid, "outcome": outcome, "points_present": listed, "vital_total": total, "vital_present": present,
            "confidence": "high", "why": f"graded {outcome}", "evidence": evidence}


def _replies(out, tmp_path, *, real="correct", sentinel=None, seed="harmful", canary=None, claimed="a-model-it-made-up"):
    """A reply file per packet. By default the judge is perfect: sentinels and canaries get their known verdicts and
    seeds are caught.
    """
    key = _key(out)
    paths = []
    for packet in _packets(out):
        k = key["packets"][packet["packet_id"]]
        verdicts = []
        for e in packet["entries"]:
            jid, total, answer = e["id"], e["vital_total"], e["answer"]
            meta = k["instrument"].get(jid)
            if meta is None:
                outcome = real
            elif meta["type"] == "sentinel":
                outcome = sentinel(meta) if sentinel else meta["known_outcome"]
            elif meta["type"] == "seeded_harm":
                outcome = seed
            else:
                outcome = canary or meta["known_outcome"]
            present = total if outcome in ("correct", "correct_missing_okay_nuggets") else 0
            if meta and meta["type"] == "sentinel" and outcome == meta["known_outcome"]:
                present = meta["known_vital_present"]
            evidence = meta["replacement"] if meta and meta["type"] == "seeded_harm" and outcome == "harmful" else None
            verdicts.append(_verdict(jid, outcome, total, present, answer, evidence))
        path = tmp_path / f"reply_{packet['packet_id']}.json"
        path.write_text(json.dumps({"packet_id": packet["packet_id"], "judge_model": claimed, "verdicts": verdicts}),
                        encoding="utf-8")
        paths.append(path)
    return paths


def _import(report, out, paths, **kw):
    kw.setdefault("judge_model", "gemini-3.1-pro-preview")
    kw.setdefault("mode", "chat")
    return json.loads(Path(GJ.import_verdicts(report, paths, key_path=Path(out["key"]), **kw)).read_text(
        encoding="utf-8"))


# --- packets --------------------------------------------------------------------------------------------------------


def test_a_run_is_split_into_packets_of_at_most_25_answers_each_seen_exactly_once(exported):
    _, out = exported
    key = _key(out)
    real_per_packet = [len(k["map"]) for k in key["packets"].values()]
    assert sorted(real_per_packet, reverse=True) == [25, 25, 10]
    seen = [item for k in key["packets"].values() for item in k["map"].values()]
    assert sorted(seen) == sorted(GOLD) and len(seen) == len(set(seen))


def test_every_packet_hides_two_sentinels_two_seeds_and_a_canary(exported):
    _, out = exported
    for k in _key(out)["packets"].values():
        kinds = sorted(m["type"] for m in k["instrument"].values())
        assert kinds == ["canary", "seeded_harm", "seeded_harm", "sentinel", "sentinel"]


def test_instrument_items_look_exactly_like_answers(exported):
    _, out = exported
    text = "".join(Path(p).read_text(encoding="utf-8") for p in out["packets"] + out["markdown"])
    for leak in ("sentinel", "seeded", "canary", "known_outcome", "triage:", "dev_s18", "sen0001", "seb0001", "hum0001"):
        assert leak not in text, leak
    for packet in _packets(out):
        keys = {tuple(sorted(e)) for e in packet["entries"]}
        assert len(keys) == 1, "every entry, answer or instrument item, has exactly the same fields"
        ids = [e["id"] for e in packet["entries"]]
        assert ids == [f"J{n:03d}" for n in range(1, len(ids) + 1)]


def test_the_key_names_the_instrument_items_and_the_packet_never_does(exported):
    _, out = exported
    key = _key(out)
    assert key["version"] == 2 and key["report"].endswith(".json")
    for packet in _packets(out):
        k = key["packets"][packet["packet_id"]]
        assert set(k["map"]) | set(k["instrument"]) == {e["id"] for e in packet["entries"]}
        assert not set(k["map"]) & set(k["instrument"])


def test_no_item_of_the_run_is_reused_as_a_sentinel_or_a_seed(tmp_path, monkeypatch):
    gold = dict(GOLD)
    gold["sen0001"] = _gold("sen0001", 999)
    monkeypatch.setattr(GJ, "load_gold_map", lambda: gold)
    report = _report(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["rows"].append(_row("sen0001"))
    report.write_text(json.dumps(data), encoding="utf-8")
    out = GJ.export_v2(report, out_dir=tmp_path / "g", bank=BANK)
    for k in _key(out)["packets"].values():
        assert all(m.get("item") != "sen0001" for m in k["instrument"].values())


def test_a_seed_is_a_correct_answer_with_one_stated_fact_changed(exported):
    _, out = exported
    key = _key(out)
    bases = {b["item"]: b for b in BANK["seed_bases"]}
    for packet in _packets(out):
        k = key["packets"][packet["packet_id"]]
        for e in packet["entries"]:
            meta = k["instrument"].get(e["id"])
            if not meta or meta["type"] != "seeded_harm":
                continue
            base = bases[meta["item"]]["entry"]
            assert e["answer"] != base["answer"] and e["question"] == base["question"]
            assert meta["original"] in base["answer"] and meta["replacement"] in e["answer"]
            assert meta["kind"] in {"number", "date", "result"} and meta["known_outcome"] == "harmful"


def test_export_is_deterministic_for_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    report = _report(tmp_path)
    a = _key(GJ.export_v2(report, out_dir=tmp_path / "a", bank=BANK))
    b = _key(GJ.export_v2(report, out_dir=tmp_path / "b", bank=BANK))
    assert list(a["packets"]) == list(b["packets"])


def test_an_answer_that_cites_the_sealed_window_never_reaches_a_packet(tmp_path, monkeypatch):
    """E3's retrieval served W5-window speeches, so stored answers can quote them. Nothing from the sealed window goes
    into a packet; such answers are held back and counted, and stay ungraded.
    """
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    report = _report(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["rows"][0]["rendered"] = "A member said the hubs would open. [2026-05-19]"
    data["rows"][1]["rendered"] = "A member said the levy was unfair, on 19 May 2026."
    report.write_text(json.dumps(data), encoding="utf-8")
    out = GJ.export_v2(report, out_dir=tmp_path / "w", bank=BANK)
    key = _key(out)
    packed = {item for k in key["packets"].values() for item in k["map"].values()}
    assert "hum0001" not in packed and "hum0002" not in packed and len(packed) == N_ITEMS - 2
    assert key["held_back_sealed"] == ["hum0001", "hum0002"]
    text = "".join(Path(p).read_text(encoding="utf-8") for p in out["packets"])
    assert "2026-05-19" not in text and "19 May 2026" not in text


def test_seed_bases_from_the_same_report_stay_usable_when_they_are_not_being_graded(tmp_path, monkeypatch):
    """Only the answers being graded are excluded from the bank: a structured row graded deterministically in the same
    report is not in any packet, so a seed made from it duplicates nothing the judge sees.
    """
    gold = dict(GOLD)
    gold["seb0001"] = _gold("seb0001", 998)
    monkeypatch.setattr(GJ, "load_gold_map", lambda: gold)
    report = _report(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["rows"].append({**_row("seb0001"), "outcome": "correct", "by": "deterministic-structured"})
    report.write_text(json.dumps(data), encoding="utf-8")
    key = _key(GJ.export_v2(report, out_dir=tmp_path / "d", bank=BANK))
    used = {m.get("item") for k in key["packets"].values() for m in k["instrument"].values()}
    assert "seb0001" in used


def test_a_bank_too_small_for_the_plan_is_reported_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(GJ, "load_gold_map", lambda: GOLD)
    small = {**BANK, "sentinels": BANK["sentinels"][:1]}
    key = _key(GJ.export_v2(_report(tmp_path), out_dir=tmp_path / "s", bank=small))
    assert key["shortfall"] and all("sentinel" in s for s in key["shortfall"])


# --- the guards on every verdict -----------------------------------------------------------------------------------


ANSWER = "Rebecca Paul said it was “Labour’s holiday tax” — on 15 September 2026. [2026-09-15]"


@pytest.mark.parametrize("evidence,ok", [
    ("Labour's holiday tax", True),  # straight quotes for curly ones
    ("REBECCA PAUL SAID IT", True),  # case
    ("holiday tax\" - on 15 September", True),  # a dash for an em dash
    ("Rebecca Paul said … 15 September 2026", True),  # two fragments, both quoted
    ("Rebecca Paul said she opposed it", False),  # words the answer never used
    ("Rebecca Paul said … she opposed it", False),  # one true fragment does not excuse a false one
])
def test_evidence_must_be_quoted_from_the_answer(evidence, ok):
    problems = I.guard_problems("J001", {"outcome": "harmful", "evidence": evidence, "why": "x"}, ANSWER, False)
    assert (not problems) is ok, problems


def test_a_verdict_on_an_answer_must_quote_evidence_but_one_on_a_decline_need_not():
    assert I.guard_problems("J001", {"outcome": "correct", "evidence": "", "why": "x"}, ANSWER, False)
    assert not I.guard_problems("J001", {"outcome": "over_abstention", "evidence": "", "why": "x"},
                                "(the system declined: no passage)", True)


@pytest.mark.parametrize("why", ["see https://hansard.parliament.uk/x", "per www.gov.uk", "as reported [1]",
                                 "【source】", "(source: BBC)"])
def test_links_and_citation_marks_are_refused(why):
    problems = I.guard_problems("J001", {"outcome": "correct", "evidence": "Rebecca Paul said", "why": why}, ANSWER,
                                False)
    assert any("outside source" in p for p in problems), problems


def test_a_bracketed_date_from_the_answer_is_not_a_citation():
    assert not I.guard_problems("J001", {"outcome": "correct", "evidence": "15 September 2026. [2026-09-15]",
                                         "why": "both points, dated [2026-09-15]"}, ANSWER, False)


# --- import ---------------------------------------------------------------------------------------------------------


def test_import_requires_the_owner_to_name_the_model_and_the_mode(exported, tmp_path):
    report, out = exported
    paths = _replies(out, tmp_path)
    with pytest.raises(GJ.ImportRefused, match="--judge-model"):
        GJ.import_verdicts(report, paths, key_path=Path(out["key"]), judge_model=None, mode="chat")
    with pytest.raises(GJ.ImportRefused, match="--mode"):
        GJ.import_verdicts(report, paths, key_path=Path(out["key"]), judge_model="gemini-x", mode=None)


def test_deep_research_is_refused_at_import(exported, tmp_path):
    report, out = exported
    with pytest.raises(GJ.ImportRefused, match="Deep Research"):
        _import(report, out, _replies(out, tmp_path), mode="deep_research")


def test_the_reply_cannot_name_its_own_model(exported, tmp_path):
    report, out = exported
    judged = _import(report, out, _replies(out, tmp_path, claimed="gemini-ultra-made-up"))
    assert judged["judge"] == "gemini/gemini-3.1-pro-preview" and judged["judge_mode"] == "chat"
    assert judged["judged_offline"]["judge_model_claimed"] == ["gemini-ultra-made-up"]


def test_every_packet_must_be_graded_unless_the_import_is_partial(exported, tmp_path):
    report, out = exported
    paths = _replies(out, tmp_path)[:2]
    with pytest.raises(GJ.ImportRefused, match="not graded"):
        _import(report, out, paths)
    judged = _import(report, out, paths, partial=True)
    assert judged["ungraded"] == 10


def test_a_reply_to_a_packet_this_run_never_made_is_refused(exported, tmp_path):
    report, out = exported
    paths = _replies(out, tmp_path)
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    data["packet_id"] = "0000000000000000"
    paths[0].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GJ.ImportRefused, match="not one of this run's packets"):
        _import(report, out, paths)


def test_a_clean_run_merges_every_answer_and_scores_only_the_real_ones(exported, tmp_path):
    report, out = exported
    judged = _import(report, out, _replies(out, tmp_path))
    assert judged["graded"] == N_ITEMS and judged["ungraded"] == 0
    assert {r["id"] for r in judged["rows"]} == set(GOLD), "instrument items never become rows"
    assert judged["by_outcome"] == {"correct": N_ITEMS}
    inst = judged["instrument"]
    assert inst["status"] == "ok" and inst["provisional_packets"] == []
    assert inst["seeded_harm"]["caught"] == inst["seeded_harm"]["n"] == 6
    assert inst["sentinels"] == {"right": 6, "n": 6, "accuracy": 1.0}


def test_a_misgraded_canary_makes_its_packet_provisional_and_its_answers_wait(exported, tmp_path):
    report, out = exported
    judged = _import(report, out, _replies(out, tmp_path, canary="harmful"))
    inst = judged["instrument"]
    assert inst["status"] == "incomplete" and len(inst["provisional_packets"]) == 3
    assert judged["graded"] == 0 and judged["ungraded"] == N_ITEMS, "a drifted packet's verdicts are not merged"


def test_one_misgraded_sentinel_is_recorded_but_two_make_a_packet_provisional(exported, tmp_path):
    report, out = exported
    calls = []

    def one_wrong(meta):  # each packet has two sentinels, met in turn: misgrade the first of each pair
        calls.append(meta)
        return "wrong_but_benign" if len(calls) % 2 == 1 else meta["known_outcome"]

    judged = _import(report, out, _replies(out, tmp_path, sentinel=one_wrong))
    assert judged["instrument"]["provisional_packets"] == []
    assert judged["instrument"]["sentinels"]["right"] < judged["instrument"]["sentinels"]["n"]
    both_wrong = _import(report, out, _replies(out, tmp_path, sentinel=lambda meta: "wrong_but_benign"))
    assert len(both_wrong["instrument"]["provisional_packets"]) == 3


def test_a_sentinel_flipped_across_the_harmful_line_makes_its_packet_provisional():
    meta = {"J001": {"type": "sentinel", "known_outcome": "correct", "known_vital_present": 1},
            "J002": {"type": "sentinel", "known_outcome": "correct", "known_vital_present": 1}}
    verdicts = {"J001": {"outcome": "harmful"}, "J002": {"outcome": "correct"}}
    assert I.score_packet(meta, verdicts)["provisional"]


def test_seeded_harm_recall_has_an_interval_and_below_the_bar_the_instrument_is_incomplete(exported, tmp_path):
    report, out = exported
    judged = _import(report, out, _replies(out, tmp_path, seed="wrong_but_benign"))
    seeded = judged["instrument"]["seeded_harm"]
    assert seeded["caught"] == 0 and seeded["n"] == 6 and seeded["recall"] == 0.0
    lo, hi = seeded["wilson_95"]
    assert 0.0 <= lo < hi <= 1.0
    assert judged["instrument"]["status"] == "incomplete"
    assert any("harmful recall" in r for r in judged["instrument"]["reasons"])


def test_wilson_interval_matches_known_values():
    lo, hi = I.wilson(9, 10)
    assert math.isclose(lo, 0.5958, abs_tol=1e-3) and math.isclose(hi, 0.9821, abs_tol=1e-3)
    assert I.wilson(0, 0) is None


# --- the gate -------------------------------------------------------------------------------------------------------


def test_the_gate_shows_the_instrument_and_is_not_evaluated_when_it_fails(exported, tmp_path, monkeypatch):
    report, out = exported
    judged = _import(report, out, _replies(out, tmp_path, seed="correct"))
    gold = {i: SimpleNamespace(id=i, expected_behaviour="answer") for i in GOLD}
    table = {name: (value, verdict) for name, _, value, verdict in GATE.bars(judged, gold)}
    value, verdict = table["Judge instrument (sentinels, seeded harms, canaries)"]
    assert verdict == GATE.INCOMPLETE and "0/6" in value


def test_a_report_judged_before_protocol_v2_keeps_its_gate_unchanged():
    legacy = json.loads((ROOT / "reports" / "e3_eval_test_judged.json").read_text(encoding="utf-8"))
    from goldset.schema import load

    gold = {i.id: i for i in load(ROOT / "tests" / "data" / "gold_v2.json")}
    names = [b[0] for b in GATE.bars(legacy, gold)]
    assert "Judge instrument (sentinels, seeded harms, canaries)" not in names


# --- the bank -------------------------------------------------------------------------------------------------------


def test_the_committed_bank_is_well_formed_and_carries_nothing_from_the_sealed_window():
    bank = I.load_bank(I.BANK_PATH)
    assert I.bank_problems(bank) == []
    # 7 of the maintainer's 19 cards: 8 quote speeches from the sealed W5 window, and 4 were rewritten by the gold v3
    # migration after the maintainer judged them, so they wait for re-adjudication.
    assert len(bank["sentinels"]) >= 7 and len(bank["seed_bases"]) >= 50
    assert all(s["breadth"] in ("narrow", "broad") for s in bank["sentinels"])
    assert all(p.get("kind") for x in bank["sentinels"] + bank["seed_bases"] for p in x["entry"]["reference_points"])
    assert not any(I.mentions_sealed_window(x["entry"]) for x in bank["sentinels"] + bank["seed_bases"])


def test_the_bank_is_built_from_the_owners_adjudicated_triage_cards():
    labels = ROOT / "reports" / "triage_labels.json"
    if not labels.exists():
        pytest.skip("the owner's triage labels are private (not in the public mirror)")
    bank = I.load_bank(I.BANK_PATH)
    owner = json.loads(labels.read_text(encoding="utf-8"))["cards"]
    finals = {f"triage:{jid}": card["final"] for jid, card in owner.items() if card.get("final")}
    used = {s["source"]: s["known_outcome"] for s in bank["sentinels"]}
    held = {e["source"] for e in bank["excluded"] if e["source"].startswith("triage:")}
    assert set(used) | held == set(finals) and not set(used) & held, "every owner verdict is used or held back"
    assert all(finals[s] == outcome for s, outcome in used.items()), "a sentinel's verdict is the owner's, unchanged"
    assert all("sealed" in e["reason"] or "re-adjudicate" in e["reason"] for e in bank["excluded"])


def test_every_canary_is_consistent_inside_its_packet_and_avoids_the_sealed_window():
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for canary in I.CANARIES:
        e = canary["entry"]
        assert canary["known_outcome"] == "correct" and canary["known_vital_present"] == e["vital_total"]
        assert all(p.get("kind") for p in e["reference_points"]), "under v3 a canary without kinds stands out"
        assert e["vital_total"] == sum(p["vital"] for p in e["reference_points"]) >= 1
        for d, m, y in re.findall(rf"(\d{{1,2}}) ({months}) (\d{{4}})", json.dumps(e)):
            assert not (y == "2026" and m in ("April", "May", "June", "July")), "never inside the sealed W5 window"
        for p in e["reference_points"]:
            figures = re.findall(r"\d[\d,]*", p["text"])
            assert all(f in e["answer"] for f in figures), "the answer states every figure its reference points do"


# --- v3 packets and the rebuilt bank -------------------------------------------------------------


def test_under_v3_no_instrument_item_is_told_apart_by_its_reference_points(tmp_path):
    """After the v3 migration every real dev-A entry says what each of its reference points is, so a sentinel, seeded
    harm or canary stored without kinds would stand out by its blank ones. Built from the real gold, the committed
    bank and the dev s18 run.
    """
    report = ROOT / "reports" / "e3_eval_dev_s18_final_judged.json"
    out = GJ.export_v2(report, out_dir=tmp_path / "v3", bank=I.load_bank(I.BANK_PATH), include_graded=True,
                       rubric="v3")
    key = json.loads(Path(out["key"]).read_text(encoding="utf-8"))
    for packet in _packets(out):
        instrument = key["packets"][packet["packet_id"]]["instrument"]
        assert instrument, "every packet hides instrument items"
        for e in packet["entries"]:
            assert all(p["kind"] for p in e["reference_points"]), (e["id"], "instrument" if e["id"] in instrument
                                                                   else "answer")


def test_a_sentinels_verdict_carries_over_only_when_its_vital_points_are_unchanged_or_only_cut():
    """The maintainer judged each sentinel against the reference points of the day. A rewrite that only cuts words from
    a vital point cannot turn a right answer wrong; anything else needs the maintainer again.
    """
    old = ["£60.5 billion in gross value added in 2023, according to Sureena Brackenridge."]
    assert GJ.verdict_carries_over(old, old)
    assert GJ.verdict_carries_over(old, ["£60.5 billion in gross value added in 2023."])
    assert not GJ.verdict_carries_over(["Lord Gascoigne responded to amendments on regulating property agents."],
                                       ["Peers pressed for property agents to be regulated."])
    assert not GJ.verdict_carries_over(old, old + ["Including over £64 million in Wolverhampton."])
    assert not GJ.verdict_carries_over(old + ["A second point."], old)  # a vital point dropped changes the count


def test_the_committed_bank_holds_back_the_sentinels_rewritten_after_the_owners_verdict():
    bank = I.load_bank(I.BANK_PATH)
    held = {e["source"]: e["reason"] for e in bank["excluded"]}
    used = {s["item"] for s in bank["sentinels"]}
    for item_id in ("hum0072", "hum0083", "adv0015", "hum0097"):
        assert item_id not in used, item_id
    assert sum("re-adjudicate" in reason for reason in held.values()) == 4
    assert "old0013" in used  # its rewrite only cut the citer from the vital point


def _one_card_bank(tmp_path, label):
    """build_bank over one real triage card and its real dev s18 row."""
    report = json.loads((ROOT / "reports" / "e3_eval_dev_s18_final_judged.json").read_text(encoding="utf-8"))
    rows = [r for r in report["rows"] if r["id"] == "hum0097"]
    (tmp_path / "report.json").write_text(json.dumps({**report, "rows": rows}), encoding="utf-8")
    (tmp_path / "key.json").write_text(json.dumps({"cards": {"J030": {"item_id": "hum0097"}}}), encoding="utf-8")
    (tmp_path / "labels.json").write_text(json.dumps({"cards": {"J030": label}}), encoding="utf-8")
    return GJ.build_bank(tmp_path / "key.json", tmp_path / "labels.json", tmp_path / "report.json")


def test_a_rewritten_sentinel_returns_once_the_owner_re_adjudicates_it_under_v3(tmp_path):
    """The maintainer's v2 verdict stays on record (E3's adjudication used it); a `final_v3` verdict, given against the
    rewritten reference points, is the one the bank uses. Without it the sentinel is held back.
    """
    held = _one_card_bank(tmp_path, {"final": "missing_vital_nugget", "final_vital": 0})
    assert held["sentinels"] == [] and "re-adjudicate" in held["excluded"][0]["reason"]
    back = _one_card_bank(tmp_path, {"final": "missing_vital_nugget", "final_vital": 0,
                                     "final_v3": "correct", "final_vital_v3": 1})
    assert [(s["item"], s["known_outcome"], s["known_vital_present"]) for s in back["sentinels"]] == [
        ("hum0097", "correct", 1)]
