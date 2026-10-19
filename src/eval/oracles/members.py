"""The members oracle: names and party history from the Members API payloads as fetched."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "corpus_v2" / "members"
HOUSE_CODES = {"Commons": 1, "Lords": 2}


@dataclass(frozen=True)
class Profile:
    mnis_id: int
    display: str | None  # the name the member is known by now ("Lord Vaizey of Didcot")
    address: str | None  # how they are addressed ("Mr Alan Mak", "The Lord Hamilton of Epsom")
    full_title: str | None


@dataclass(frozen=True)
class Affiliation:
    party: str
    start: date | None
    end: date | None
    house: int | None = None


def _read(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("value", data) if isinstance(data, dict) else data


def _day(value: Any) -> date | None:
    return date.fromisoformat(str(value)[:10]) if value else None


def _clean(value: Any) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None


def profiles(raw: Path = RAW) -> dict[int, Profile]:
    out: dict[int, Profile] = {}
    if not raw.exists():
        return out
    for path in sorted(raw.glob("*/profile.json.gz")):
        p = _read(path)
        if not isinstance(p, dict) or not str(p.get("id") or path.parent.name).isdigit():
            continue
        mid = int(p.get("id") or path.parent.name)
        out[mid] = Profile(mid, _clean(p.get("nameDisplayAs")), _clean(p.get("nameAddressAs")),
                           _clean(p.get("nameFullTitle")))  # fmt: skip
    return out


def history(raw: Path = RAW) -> dict[int, list[Affiliation]]:
    out: dict[int, list[Affiliation]] = {}
    if not raw.exists():
        return out
    for path in sorted(raw.glob("*/Biography.json.gz")):
        if not path.parent.name.isdigit():
            continue
        bio = _read(path)
        affs = [Affiliation(str(a["name"]), _day(a.get("startDate")), _day(a.get("endDate")), a.get("house"))
                for a in (bio.get("partyAffiliations") or []) if isinstance(a, dict) and a.get("name")]  # fmt: skip
        out[int(path.parent.name)] = affs
    return out


def party_on(affiliations: list[Affiliation] | None, day: date, house: str | None = None) -> str | None:
    """The one party the history gives on `day` (in `house`, when the history says which), strictly inside its
    interval; None when it gives none, more than one, or the day is a start or an end.
    """
    code = HOUSE_CODES.get(house or "")
    found = set()
    for a in affiliations or ():
        if code is not None and a.house is not None and a.house != code:
            continue
        if day in (a.start, a.end):
            return None  # a boundary: whether it counts is not in the payload
        if (a.start is None or a.start < day) and (a.end is None or day < a.end):
            found.add(a.party)
    return next(iter(found)) if len(found) == 1 else None


@dataclass(frozen=True)
class Post:
    mnis_id: int
    name: str
    start: date | None
    end: date | None


# The posts that name an office a person holds (not committee or other roles).
_POST_LISTS = ("governmentPosts", "oppositionPosts")


def posts(raw: Path = RAW) -> list[Post]:
    """Every government and opposition post the biographies record, with its dates (MR-07's truth)."""
    out: list[Post] = []
    if not raw.exists():
        return out
    for path in sorted(raw.glob("*/Biography.json.gz")):
        if not path.parent.name.isdigit():
            continue
        bio = _read(path)
        for name in _POST_LISTS:
            for p in bio.get(name) or []:
                if isinstance(p, dict) and _clean(p.get("name")):
                    out.append(Post(int(path.parent.name), _clean(p["name"]), _day(p.get("startDate")),
                                    _day(p.get("endDate"))))  # fmt: skip
    return out


@dataclass(frozen=True)
class Seat:
    house: int  # 1 Commons, 2 Lords
    start: date | None
    end: date | None


def seats(raw: Path = RAW) -> dict[int, list[Seat]]:
    """Every member's dated seats: Commons seats from `representations`, which show the gaps a dissolution makes, and
    Lords seats from `houseMemberships`; a Commons membership stands in only where a biography has no
    representations.
    """
    out: dict[int, list[Seat]] = {}
    if not raw.exists():
        return out
    for path in sorted(raw.glob("*/Biography.json.gz")):
        if not path.parent.name.isdigit():
            continue
        bio = _read(path)
        reps = [r for r in bio.get("representations") or [] if isinstance(r, dict) and r.get("house") == 1]
        mems = [m for m in bio.get("houseMemberships") or [] if isinstance(m, dict)]
        rows = reps + [m for m in mems if m.get("house") == 2] + ([] if reps else [m for m in mems if m.get("house") == 1])
        out[int(path.parent.name)] = sorted((Seat(int(r["house"]), _day(r.get("startDate")), _day(r.get("endDate")))
                                             for r in rows), key=lambda x: (x.start or date.min, x.house))  # fmt: skip
    return out


def seat_on(member_seats: list[Seat] | None, day: date) -> int | None:
    """The one House the member sat in on `day`, strictly inside a seat's dates; None on a boundary, for none or two.
    """
    houses = set()
    for st in member_seats or ():
        if day in (st.start, st.end):
            return None
        if (st.start is None or st.start < day) and (st.end is None or day < st.end):
            houses.add(st.house)
    return houses.pop() if len(houses) == 1 else None
