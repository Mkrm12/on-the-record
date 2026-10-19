"""Procedural eras across ten years: the table, its checks and its notes."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG = Path(__file__).resolve().parents[2] / "config" / "eras.toml"
HOUSES = ("Commons", "Lords", "both")
STATUSES = ("attested", "cited")


@dataclass(frozen=True)
class Era:
    id: str
    name: str
    house: str
    start: date
    end: date | None
    payload: tuple[str, ...]
    status: str
    source: str
    note: str = ""
    seen_first: date | None = None
    seen_last: date | None = None

    def covers(self, day: date, house: str | None = None) -> bool:
        """Whether the era applies on `day` (inclusive), in `house` when one is given."""
        if house is not None and self.house not in ("both", house):
            return False
        return self.start <= day and (self.end is None or day <= self.end)


def _date(value: Any, era: str, key: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{CONFIG.name}: era {era!r} {key} must be a date")
    return value


@lru_cache(maxsize=4)
def load(path: Path = CONFIG) -> tuple[Era, ...]:
    """The table, validated: unique ids, known Houses and statuses, from <= to, a source and a payload for each."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: list[Era] = []
    for raw in data.get("era") or []:
        era = Era(id=str(raw["id"]), name=str(raw["name"]), house=str(raw["house"]),
                  start=_date(raw["from"], raw["id"], "from"), end=_date(raw.get("to"), raw["id"], "to"),
                  payload=tuple(raw.get("payload") or ()), status=str(raw["status"]), source=str(raw["source"]),
                  note=str(raw.get("note") or ""), seen_first=_date(raw.get("seen_first"), raw["id"], "seen_first"),
                  seen_last=_date(raw.get("seen_last"), raw["id"], "seen_last"))  # fmt: skip
        if era.house not in HOUSES or era.status not in STATUSES or (era.end and era.end < era.start):
            raise ValueError(f"{path.name}: era {era.id!r} is malformed")
        out.append(era)
    if len({e.id for e in out}) != len(out):
        raise ValueError(f"{path.name}: era ids must be unique")
    return tuple(out)


def by_id(era_id: str) -> Era:
    return next(e for e in load() if e.id == era_id)


def on(day: date, house: str | None = None) -> tuple[Era, ...]:
    """The eras in force on `day` in `house`."""
    return tuple(e for e in load() if e.covers(day, house))


# --- the gate's checks ----------------------------------------------------------------------------------------------

# Each check: the eras that allow a flag, and whether a division row carries it.
_FLAGS = {
    "E01": (("commons_remote_divisions", "lords_remote_divisions"), lambda r: bool(r.get("is_remote"))),
    "E02": (("evel",), lambda r: r.get("double_majority_ayes") is not None or r.get("double_majority_noes") is not None
            or bool((r.get("evel_type") or "").strip())),
    "E03": (("commons_proxy_voting",), lambda r: bool(r.get("proxy_votes"))),
    "E04": (("commons_deferred_divisions",), lambda r: bool(r.get("is_deferred"))),
}
CHECKS = {
    "E01": "a division flagged remote outside its House's remote-voting era (config/eras.toml)",
    "E02": "a division with a double-majority (EVEL) count outside the EVEL era or outside the Commons",
    "E03": "a division with votes cast by proxy outside its House's proxy-voting era",
    "E04": "a deferred division outside the Commons' deferred-division era",
}


def violations(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[list]]:
    """Per check, the divisions whose flag no era of their House allows on their date: [id, house, date]."""
    out: dict[str, list[list]] = {check: [] for check in _FLAGS}
    for row in rows:
        day = row["division_date"]
        for check, (allowed, flagged) in _FLAGS.items():
            if flagged(row) and not any(by_id(e).covers(day, row["house"]) for e in allowed):
                out[check].append([row["id"], row["house"], day.isoformat()])
    return out


# --- what the reader is told ------------------------------------------------------------------------------------------


def _long(day: date) -> str:
    return f"{day.day} {day:%B %Y}"


def division_notes(row: Mapping[str, Any], put_on: date | None = None) -> list[str]:
    """What an era changes about this division, in a sentence each: a remote vote, a deferred ballot (with the day its
    question was put, when known), both counts of a double majority.
    """
    notes: list[str] = []
    if row.get("is_remote"):
        notes.append("This was a remote division: Members voted through the House's remote voting system, not in the "
                     "division lobbies.")  # fmt: skip
    if row.get("is_deferred"):
        when = row.get("division_date")
        put = f" put on {_long(put_on)}" if put_on and put_on != when else ""
        notes.append(f"This was a deferred division: the question{put} was decided by ballot paper"
                     f"{f' on {_long(when)}' if when else ''}.")
    ayes, noes = row.get("double_majority_ayes"), row.get("double_majority_noes")
    if ayes is not None or noes is not None:
        whose = f"MPs for constituencies in {row['evel_country']}" if row.get("evel_country") else "the MPs concerned"
        second = (f"{whose} voted {ayes} to {noes}" if ayes is not None and noes is not None
                  else f"only part of the second count ({whose}) is recorded")  # fmt: skip
        notes.append(f"This was a double-majority division under English Votes for English Laws: it needed a majority "
                     f"of the whole House and of {whose}; {second}.")
    late = sitting_note(row.get("division_date"), row.get("held_at"))
    if late:
        notes.append(late)
    return notes


# --- the calendar ---------------------------------------------------------------------------------------------------


def calendar_days(sitting: date, held_at: datetime | None) -> set[date]:
    """The calendar days a division can be asked about by: its sitting's date, and the next day when it was held after
    midnight in that sitting. A timestamp of exactly midnight is a date with no time (a deferred ballot's), not a
    division held at 00:00.
    """
    days = {sitting}
    if held_at is not None and held_at.time() != datetime.min.time() and held_at.date() == sitting + timedelta(days=1):
        days.add(held_at.date())
    return days


def _as_date(value: Any) -> date | None:
    """A day from a date, a timestamp or an ISO string (a fact's value carries ISO strings)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def sitting_note(sitting: Any, held_at: datetime | None) -> str:
    """The sentence a division held after midnight carries wherever it is shown: Hansard dates it by its sitting, a
    question may date it by the clock, and a list of one calendar day's divisions read it as that day's own when
    nothing said otherwise. Empty for any other division (`calendar_days` is the rule).
    """
    day = _as_date(sitting)
    if day is None or held_at is None or len(calendar_days(day, held_at)) < 2:
        return ""
    return (f"This division was held after midnight, at {held_at:%H:%M} on {_long(held_at.date())}, in the sitting "
            f"of {_long(day)}; Hansard dates it to that sitting.")


def other_sitting(sitting: Any, day: Any) -> str:
    """Where a division stands against the calendar day a question named: empty when it is that day's own sitting's (or
    either is unknown), else the words that place it, "in the sitting of 18 March 2026". Every list of a day's
    divisions labels its rows with this, so a division of the sitting before is never counted as the day's own.
    """
    own, asked = _as_date(sitting), _as_date(day)
    if own is None or asked is None or own == asked:
        return ""
    return f"in the sitting of {_long(own)}"
