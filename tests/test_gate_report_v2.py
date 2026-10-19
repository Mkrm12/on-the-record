"""Gate report v2: every bar split by how it was graded, never blended, never PASS on a run that could not be
evaluated, and no stale placeholders for work the maintainer has done.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from goldset.schema import load

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate_report_v2_t", ROOT / "scripts" / "gate_report_v2.py")
G2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G2)

GOLD = {i.id: i for i in load()}
TEST = ROOT / "reports" / "e3_eval_test_judged.json"
DEV = ROOT / "reports" / "e3_eval_dev_s18_final_judged.json"


def _report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_v2_reproduces_the_e3_test_result():
    m = G2.measure(_report(TEST), GOLD)
    assert m["harmful"] == {"deterministic": (0, 119), "judged": (1, 80), "probe": (0, 0), "all": (1, 199)}
    assert m["over_abstention"]["all"] == (36, 199)
    assert m["over_abstention"]["deterministic"] == (22, 119) and m["over_abstention"]["judged"] == (14, 80)
    assert m["correct_abstention"]["all"] == (9, 11)
    assert (m["router"]["right_fact"], m["router"]["fact_questions"], m["router"]["sent_to_a_fact"]) == (92, 95, 1)
    assert m["vital"] == {"deterministic": (95, 95), "judged": (54, 87), "probe": (0, 0), "all": (149, 182)}
    assert m["p95_ms"] == 5970 and m["degraded"] == 0 and m["awaiting"] == 0


def test_gate_v2_reproduces_the_e3_dev_result():
    m = G2.measure(_report(DEV), GOLD)
    assert m["harmful"]["all"] == (0, 223) and m["over_abstention"]["all"] == (32, 223)
    assert m["vital"]["deterministic"] == (116, 116) and m["vital"]["judged"] == (52, 82)
    assert m["correct_abstention"]["all"] == (15, 15)


def test_the_vital_bar_is_split_and_the_judged_tier_is_held_to_its_own_bar():
    bars = {b.name: b for b in G2.bars(_report(TEST), GOLD)}
    deterministic, judged = bars["Vital coverage, deterministic answers"], bars["Vital coverage, judged free text"]
    assert deterministic.verdict == "PASS" and judged.verdict == "FAIL"  # 62% against >= 75%: no longer hidden
    assert "54/87" in judged.cells["judged"]


def test_every_bar_reports_each_grading_method_with_its_denominator_and_interval():
    for bar in G2.bars(_report(TEST), GOLD):
        if not bar.cells:
            continue
        assert set(bar.cells) == {"deterministic", "judged", "probe", "all"}, bar.name
        for method, cell in bar.cells.items():
            if cell != G2.NONE:
                assert "/" in cell and "95%" in cell, (bar.name, method, cell)
    harmful = next(b for b in G2.bars(_report(TEST), GOLD) if b.name == "Harmful answers")
    assert harmful.cells["probe"] == G2.NONE  # no probe has run yet: said so, never a zero


def _small(tmp_path, rows):
    report = {"split": "dev", "items": len(rows), "gate_evaluated": True, "rows": rows,
              "trace": {"overall": {"p95_ms": 3000}}, "router": {"right_fact": 1, "fact_questions": 1,
                                                                  "sent_to_a_fact": 0, "accuracy": 1.0}}
    path = tmp_path / "e3_eval_dev_small_judged.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _row(item_id, outcome, by="gemini", **kw):
    return {"id": item_id, "outcome": outcome, "by": by, "vital_present": 1, "vital_total": 1, **kw}


def test_the_gate_never_says_passed_when_anything_is_ungraded_or_failed(tmp_path, capsys):
    dev_ids = [i.id for i in GOLD.values() if i.split == "dev" and i.expected_behaviour == "answer"][:3]
    rows = [_row(dev_ids[0], "correct"), _row(dev_ids[1], None, by="awaiting-judge"),
            _row(dev_ids[2], None, by=G2.EA.FAILED)]  # the harness's own marker for a request that failed
    code = G2.main(["gate", str(_small(tmp_path, rows))])
    out = capsys.readouterr().out
    assert code == 2 and "PASSED" not in out and "INCOMPLETE" in out
    assert "1 awaits the judge" in out and "1 FAILED" in out


def test_the_gate_has_no_stale_owner_rows_and_reads_the_owners_evidence_from_files(capsys):
    G2.main(["gate", str(TEST)])
    out = capsys.readouterr().out
    assert "needs the owner's labels" not in out and "not run" not in out
    assert "kappa 0.75" in out and "triage_result.json" in out  # the maintainer's calibration, from its file
    assert "e3_eval_test_judged.json" in out  # the test confirmation, from its file


def test_the_gate_names_the_rubric_it_graded_under(tmp_path, capsys):
    G2.main(["gate", str(TEST)])
    assert "rubric v2" in capsys.readouterr().out  # E3 predates recorded rubric versions: it was v2
    dev_id = next(i.id for i in GOLD.values() if i.split == "dev" and i.expected_behaviour == "answer")
    path = _small(tmp_path, [_row(dev_id, "correct")])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["rubric"] = "v3"
    path.write_text(json.dumps(data), encoding="utf-8")
    G2.main(["gate", str(path)])
    assert "rubric v3" in capsys.readouterr().out


def test_e3s_test_result_still_fails_under_gate_v2():
    assert G2.main(["gate", str(TEST)]) == 1


def test_a_v3_row_is_held_to_what_it_needed_not_to_every_vital_point(tmp_path):
    dev_id = next(i.id for i in GOLD.values() if i.split == "dev" and i.expected_behaviour == "answer")
    rows = [_row(dev_id, "correct", vital_present=2, vital_total=3, vital_required=2)]
    m = G2.measure(json.loads(_small(tmp_path, rows).read_text(encoding="utf-8")), GOLD)
    assert m["vital"]["judged"] == (2, 2)


def test_protocol_v2_runs_carry_the_instrument_bar(tmp_path):
    dev_id = next(i.id for i in GOLD.values() if i.split == "dev" and i.expected_behaviour == "answer")
    data = json.loads(_small(tmp_path, [_row(dev_id, "correct")]).read_text(encoding="utf-8"))
    data["judged_offline"] = {"protocol": 2}
    names = [b.name for b in G2.bars(data, GOLD)]
    assert "Judge instrument (sentinels, seeded harms, canaries)" in names


@pytest.mark.parametrize("k,n", [(1, 199), (36, 199), (0, 0)])
def test_cells_show_the_count_and_its_wilson_interval(k, n):
    cell = G2.cell(k, n)
    assert cell == G2.NONE if n == 0 else (cell.startswith(f"{k}/{n}") and "95%" in cell)
