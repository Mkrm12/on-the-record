"""The bill-stage oracle (ST-bill_stage_date): each stage's sittings, from the Bills API payloads as fetched."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app import sealed

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "corpus_v2" / "bills"


@dataclass(frozen=True)
class StageTruth:
    bill_id: int
    bill: str
    stage_id: int
    stage: str
    house: str
    sittings: tuple[date, ...]  # held: on or before the payload's as-of date
    scheduled: tuple[date, ...] = ()  # after it

    @property
    def reached(self) -> bool:
        return bool(self.sittings)

    @property
    def first_sitting(self) -> date | None:
        return self.sittings[0] if self.sittings else None


@dataclass(frozen=True)
class Disagreement:
    stage_id: int
    check: str
    oracle: Any
    database: Any


def _read(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _touches_window(days: tuple[date, ...]) -> bool:
    w = sealed.window()
    return any(w.contains(d) for d in days) or (days[0] <= w.end and days[-1] >= w.start)


def truths(raw: Path = RAW) -> Iterator[StageTruth]:
    folder = raw / "bill"
    if not folder.exists():
        return
    for bill_dir in sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        detail, stages = bill_dir / "detail.json.gz", bill_dir / "stages.json.gz"
        if not detail.exists() or not stages.exists():
            continue
        bill = _read(detail)
        title = str(bill.get("shortTitle") or "").strip()
        as_of = (date.fromisoformat(str(bill["lastUpdate"])[:10]) if bill.get("lastUpdate")
                 else date.fromtimestamp(detail.stat().st_mtime))  # fmt: skip
        for st in sorted(_read(stages).get("items") or [], key=lambda x: (x.get("sortOrder") or 0, x.get("id") or 0)):
            days = tuple(sorted({date.fromisoformat(str(s["date"])[:10]) for s in st.get("stageSittings") or []
                                 if s.get("date")}))  # fmt: skip
            if not days or (sealed.masked() and _touches_window(days)):
                continue
            house = str(st.get("house") or "").strip()
            # "Unassigned" is the API's word for a stage of no House (Royal Assent): no House, as the record has it.
            yield StageTruth(int(bill["billId"]), title, int(st["id"]), str(st.get("description") or "").strip(),
                             "" if house.lower() == "unassigned" else house,
                             tuple(d for d in days if d <= as_of), tuple(d for d in days if d > as_of))  # fmt: skip


def compare(truth: StageTruth, record: Mapping[str, Any] | None) -> list[Disagreement]:
    """Every way the database's row for a stage disagrees with the payload."""
    if not truth.reached:
        # only scheduled: the route must hold no sitting for it
        return [] if record is None else [Disagreement(truth.stage_id, "stage.scheduled", "not yet held", record)]
    if record is None:
        return [Disagreement(truth.stage_id, "stage.missing", "in the payload", "not in the database")]
    out = []
    for name, mine, theirs in (("bill_id", truth.bill_id, record.get("bill_id")),
                               ("stage_name", truth.stage, (record.get("stage_name") or "").strip()),
                               ("house", truth.house, (record.get("house") or "").strip()),
                               ("first_sitting", truth.first_sitting, record.get("first_sitting")),
                               ("sittings", len(truth.sittings), record.get("sittings"))):  # fmt: skip
        if mine != theirs:
            out.append(Disagreement(truth.stage_id, f"stage.{name}", mine, theirs))
    return out
