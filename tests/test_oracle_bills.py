"""Each stage's sittings, re-derived from the Bills API payloads as fetched."""

from __future__ import annotations

import ast
import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from eval.oracles import bills as OB

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)


def stage(sid, name, house, days, order):
    return {"id": sid, "stageId": 1, "description": name, "abbreviation": "X", "house": house, "sortOrder": order,
            "stageSittings": [{"id": sid * 10 + i, "billStageId": sid, "date": f"{d}T00:00:00"} for i, d in enumerate(days)]}


@pytest.fixture
def raw(tmp_path):
    base = tmp_path / "bills"
    _write(base / "bill" / "7" / "detail.json.gz", {"billId": 7, "shortTitle": "Railways Bill", "originatingHouse": "Commons"})
    _write(base / "bill" / "7" / "stages.json.gz", {"items": [
        stage(70, "1st reading", "Commons", ["2025-10-14"], 1),
        stage(71, "2nd reading", "Commons", ["2025-12-09"], 2),
        stage(72, "Committee stage", "Commons", ["2026-01-13", "2026-01-15", "2026-01-20"], 3),
        stage(73, "Report stage", "Commons", ["2026-05-20"], 4),  # inside the window: left out
        stage(74, "1st reading", "Lords", ["2026-03-25", "2026-04-02"], 5),  # sittings touch it: left out
        stage(75, "2nd reading", "Lords", [], 6),  # no sitting: nothing to check
    ]})
    return base


def test_each_stage_is_its_sittings_in_order_and_the_first_is_its_date(raw):
    stages = {s.stage_id: s for s in OB.truths(raw)}
    assert set(stages) == {70, 71, 72}
    s = stages[72]
    assert (s.bill_id, s.bill, s.stage, s.house) == (7, "Railways Bill", "Committee stage", "Commons")
    assert s.sittings == (date(2026, 1, 13), date(2026, 1, 15), date(2026, 1, 20)) and s.first_sitting == date(2026, 1, 13)


def test_a_stage_whose_sittings_touch_the_sealed_window_is_never_part_of_the_truth(raw):
    assert not {73, 74} & {s.stage_id for s in OB.truths(raw)}


def test_the_bills_oracle_shares_no_code_with_ingest_or_the_answer_layer():
    tree = ast.parse((ROOT / "src" / "eval" / "oracles" / "bills.py").read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("ingest", "answer", "retrieval", "goldset")) for m in modules), modules


def test_a_disagreement_with_the_database_is_named():
    truth = OB.StageTruth(7, "Railways Bill", 72, "Committee stage", "Commons",
                          (date(2026, 1, 13), date(2026, 1, 15)))  # fmt: skip
    record = {"id": 72, "bill_id": 7, "stage_name": "Committee stage", "house": "Commons",
              "first_sitting": date(2026, 1, 14), "sittings": 2}  # fmt: skip
    assert [(d.check, d.oracle, d.database) for d in OB.compare(truth, record)] == [
        ("stage.first_sitting", date(2026, 1, 13), date(2026, 1, 14))]
    assert OB.compare(truth, None)[0].check == "stage.missing"
