"""The opener oracle (MR-08): who opened each debate, from the Hansard contributions as fetched."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app import sealed

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "corpus_v2" / "contributions"

# How Hansard prints the Chair.
PRESIDING = re.compile(
    r"^(?:Mr|Madam)\s+(?:Deputy\s+)?Speaker\b(?!['’]s)"
    r"|^The\s+(?:Senior\s+)?(?:Deputy\s+)?Speaker\b(?!['’]s)"
    r"|^The\s+Lord\s+Speaker\b"
    r"|^The\s+(?:Principal\s+)?(?:Deputy\s+)?Chairman\s+of\s+Committees\b"
    r"|^(?:The\s+)?(?:(?:First|Second|Temporary)\s+)?(?:Deputy\s+)?Chair(?:man)?\s+of\s+Ways\s+and\s+Means\b"
    r"|^The\s+(?:Temporary\s+|(?:(?:First|Second)\s+)?Deputy\s+)?Chair(?:man)?\s*(?:\(|$)"
    r"|\(in\s+the\s+Chair\)", re.I)


def presiding(attributed: str | None) -> bool:
    return bool(PRESIDING.search((attributed or "").strip()))


@dataclass(frozen=True)
class Contribution:
    order: int
    mnis_id: int | None
    attributed: str
    text: str


@dataclass(frozen=True)
class DebateTruth:
    debate_id: str
    house: str
    date: date
    heading: str
    section: str
    contributions: tuple[Contribution, ...]

    @property
    def first(self) -> Contribution | None:
        return self.contributions[0] if self.contributions else None

    @property
    def presiding_first(self) -> bool:
        return self.first is not None and presiding(self.first.attributed)

    @property
    def opener(self) -> Contribution | None:
        return next((c for c in self.contributions if c.mnis_id and not presiding(c.attributed)), None)

    @property
    def presiding_ids(self) -> set[int]:
        return {c.mnis_id for c in self.contributions if c.mnis_id and presiding(c.attributed)}


def _read(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def truths(raw: Path = RAW) -> Iterator[DebateTruth]:
    """Every debate the raw cache holds outside the sealed window, its contributions in the record's own order."""
    if not raw.exists():
        return
    sealed_id = sealed.window().window_id
    rows: dict[str, dict[str, tuple[int, Contribution]]] = {}
    meta: dict[str, tuple[str, date, str, str]] = {}
    for window in sorted(p for p in raw.iterdir() if p.is_dir() and not (sealed.masked() and p.name == sealed_id)):
        for page in sorted(window.rglob("*.json.gz")):
            for r in _read(page).get("Results") or []:
                if sealed.hidden(r.get("SittingDate")) or not r.get("DebateSectionExtId"):
                    continue
                did = str(r["DebateSectionExtId"])
                key = str(r.get("ContributionExtId") or (r.get("OrderInDebateSection"), r.get("ItemId")))
                order = int(r.get("OrderInDebateSection") or 0)
                mid = int(r["MemberId"]) if r.get("MemberId") else None
                text = str(r.get("ContributionTextFull") or r.get("ContributionText") or "").strip()
                rows.setdefault(did, {})[key] = (order, Contribution(order, mid, str(r.get("AttributedTo") or "").strip(),
                                                                     text))  # fmt: skip
                meta.setdefault(did, (str(r.get("House") or ""), date.fromisoformat(str(r["SittingDate"])[:10]),
                                      str(r.get("DebateSection") or "").strip(), str(r.get("Section") or "")))  # fmt: skip
    for did in sorted(rows):
        house, day, heading, section = meta[did]
        ordered = tuple(c for _, c in sorted(rows[did].values(), key=lambda pair: pair[0]))
        yield DebateTruth(did, house, day, heading, section, ordered)
