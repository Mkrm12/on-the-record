"""Who opened each debate, re-derived from the Hansard contributions as fetched."""

from __future__ import annotations

import ast
import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from eval.oracles import openers as OO

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("printed", [
    "Mr Speaker", "Madam Speaker", "Madam Deputy Speaker (Caroline Nokes)", "Mr Deputy Speaker (Mr Nigel Evans)",
    "The Deputy Speaker (Lord Faulkner of Worcester)", "The Senior Deputy Speaker", "The Lord Speaker (Lord Forsyth of Drumlean)",
    "The Deputy Chairman of Committees (Baroness Fookes)", "The Chairman of Ways and Means (Dame Eleanor Laing)",
    "The First Deputy Chairman of Ways and Means", "The Chair", "The Chairman", "The Temporary Chair (Sir Mark Hendrick)",
    "Sir Roger Gale (in the Chair)",
    # the short repeat forms, printed after the first mention
    "The First Deputy Chairman", "The Second Deputy Chairman", "The Deputy Chairman",
    # the Lords' pre-2016 title, inside the ten-year span, and the temporary Ways and Means chair (QA pass 3)
    "The Principal Deputy Chairman of Committees (Lord Brabazon of Tara)", "The Temporary Chair of Ways and Means",
])  # fmt: skip
def test_the_chair_as_hansard_prints_it_presides(printed):
    assert OO.presiding(printed)


@pytest.mark.parametrize("printed", [
    "The Chair of the Treasury Committee (Dame Meg Hillier)",  # a member speaking, not presiding
    "Captain of the Honourable Corps of Gentlemen-at-Arms (Lord Kennedy of Southwark)",  # a government whip
    "The Leader of the House of Commons (Sir Alan Campbell)", "Baroness Smith of Basildon", "Jim Shannon (Strangford) (DUP)",
    "The Parliamentary Under-Secretary of State for Health and Social Care (Ashley Dalton)", "",
    "The Speaker's Chaplain", "Mr Speaker's Secretary",  # the Speaker's staff, who do not preside (QA pass 3)
])  # fmt: skip
def test_a_member_or_minister_does_not_preside(printed):
    assert not OO.presiding(printed)


def contribution(order, mid, printed, text="Words.", ext=None, day="2026-09-08", debate="D1", heading="Housing"):
    return {"OrderInDebateSection": order, "MemberId": mid, "AttributedTo": printed, "MemberName": printed,
            "ContributionTextFull": text, "ContributionExtId": ext or f"{debate}-{order}", "DebateSectionExtId": debate,
            "DebateSection": f" {heading}", "SittingDate": f"{day}T00:00:00", "House": "Commons",
            "Section": "Commons Chamber"}  # fmt: skip


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"Results": rows, "TotalResultCount": len(rows)}, fh)


@pytest.fixture
def raw(tmp_path):
    base = tmp_path / "contributions"
    _write(base / "W0" / "2026-09-08" / "asc_000000.json.gz", [
        contribution(2, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order. I call the Minister."),
        contribution(1, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order."),
        contribution(3, 5001, "The Minister for Housing (Ann Minister)", "I beg to move..."),
        contribution(4, 5002, "Jim Shannon (Strangford) (DUP)"),
    ])
    # the same contribution on a second page: counted once
    _write(base / "W0" / "2026-09-08" / "asc_000100.json.gz", [contribution(4, 5002, "Jim Shannon (Strangford) (DUP)")])
    _write(base / "W5" / "2026-05-20" / "asc_000000.json.gz", [contribution(1, 1, "Mr Speaker", debate="S1", day="2026-05-20")])
    _write(base / "W0" / "2026-06-10" / "asc_000000.json.gz", [contribution(1, 1, "Mr Speaker", debate="S2", day="2026-06-10")])
    return base


def test_a_debate_opened_by_the_chair_has_its_first_member_as_opener(raw):
    (t,) = list(OO.truths(raw))
    assert (t.debate_id, t.house, t.date, t.heading) == ("D1", "Commons", date(2026, 9, 8), "Housing")
    assert [c.order for c in t.contributions] == [1, 2, 3, 4], "in the record's order, each contribution once"
    assert t.presiding_first and t.opener.mnis_id == 5001 and t.presiding_ids == {4048}


def test_the_oracle_never_opens_the_sealed_window(raw, monkeypatch):
    opened = []
    real = gzip.open

    def spy(path, *a, **k):
        opened.append(Path(path))
        return real(path, *a, **k)

    monkeypatch.setattr(OO.gzip, "open", spy)
    assert {t.debate_id for t in OO.truths(raw)} == {"D1"}  # S1 in W5, S2 dated inside it though filed in W0
    assert not any("W5" in p.parts for p in opened)


def test_the_opener_oracle_shares_no_code_with_ingest_or_the_answer_layer():
    tree = ast.parse((ROOT / "src" / "eval" / "oracles" / "openers.py").read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("ingest", "answer", "retrieval", "goldset")) for m in modules), modules
