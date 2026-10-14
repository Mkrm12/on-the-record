"""Who held a named office on a given date."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from psycopg import AsyncConnection

# Offices too generic to resolve to one person even with a date: several people hold them at once, so a query naming one
# is a query about a group.
PLURAL_OFFICES = frozenset({
    "minister", "minister of state", "parliamentary under-secretary of state",
    "parliamentary under secretary of state", "shadow minister", "spokesperson",
    "the parliamentary under-secretary of state", "a minister",
})  # fmt: skip

_LEADING_THE = re.compile(r"^the\s+", re.I)
_WHITESPACE = re.compile(r"\s+")


def office_key(office: str | None) -> str | None:
    """A normalised lookup key: lower case, one space, no leading 'the', no trailing punctuation."""
    if not office or not office.strip():
        return None
    key = _WHITESPACE.sub(" ", office.strip().strip(".,;:")).casefold()
    key = _LEADING_THE.sub("", key).strip()
    return key or None


@dataclass(frozen=True)
class OfficeHolder:
    mnis_id: int
    office: str
    start_date: date | None
    end_date: date | None
    source: str
    evidence: int = 0


@dataclass(frozen=True)
class OfficeResolution:
    """What an office name resolves to. `ambiguous` is an answer, not a failure."""

    key: str | None = None
    holders: tuple[OfficeHolder, ...] = ()
    on_date: date | None = None
    ambiguous: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        return bool(self.holders)

    @property
    def member_ids(self) -> tuple[int, ...]:
        seen: list[int] = []
        for h in self.holders:
            if h.mnis_id not in seen:
                seen.append(h.mnis_id)
        return tuple(seen)


class OfficeIndex:
    """Every (office, holder, dates) row, in memory. Small: a few thousand rows at most."""

    def __init__(self, rows: list[OfficeHolder]) -> None:
        self._by_key: dict[str, list[OfficeHolder]] = {}
        for row in rows:
            key = office_key(row.office)
            if key:
                self._by_key.setdefault(key, []).append(row)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    @property
    def offices(self) -> list[str]:
        return sorted(self._by_key)

    def holders(self, office: str, on_date: date | None = None) -> list[OfficeHolder]:
        key = office_key(office)
        rows = list(self._by_key.get(key or "", ()))
        if on_date is None:
            return rows
        return [r for r in rows
                if (r.start_date is None or r.start_date <= on_date)
                and (r.end_date is None or on_date <= r.end_date)]

    def resolve(self, office: str, on_date: date | None = None) -> OfficeResolution:
        """The holder of an office at a date."""
        key = office_key(office)
        if not key:
            return OfficeResolution(reason="no office named")
        if key in PLURAL_OFFICES:
            return OfficeResolution(key=key, on_date=on_date, ambiguous=True,
                                    reason="several people hold this office at once")
        rows = self.holders(key, on_date)
        if not rows:
            known = self._by_key.get(key)
            return OfficeResolution(key=key, on_date=on_date,
                                    reason="office not held on that date" if known else "office not in the corpus")
        # Prefer the source with dated evidence when both agree on one person.
        holders = tuple(sorted(rows, key=lambda r: (-r.evidence, r.mnis_id)))
        distinct = {h.mnis_id for h in holders}
        return OfficeResolution(key=key, holders=holders, on_date=on_date,
                                ambiguous=len(distinct) > 1,
                                reason="one holder" if len(distinct) == 1 else
                                       f"{len(distinct)} people held this office" +
                                       ("" if on_date else " across the corpus; no date was given"))


SQL_READ = """
    SELECT mnis_id, office, start_date, end_date, source, evidence FROM member_office
"""

# Hansard's own attributions, aggregated.
SQL_FROM_HANSARD = """
    INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence)
    SELECT s.mnis_id, s.speaker_office, %(normalise)s, min(d.date), max(d.date),
           'hansard_attribution', count(*)
    FROM speech s
    JOIN debate d ON d.id = s.debate_id
    WHERE s.mnis_id IS NOT NULL AND s.speaker_office IS NOT NULL AND btrim(s.speaker_office) <> ''
    GROUP BY s.mnis_id, s.speaker_office
    ON CONFLICT (mnis_id, office_key, start_date, source) DO UPDATE SET
        office = excluded.office, end_date = excluded.end_date, evidence = excluded.evidence
"""


async def build_offices(conn: AsyncConnection) -> dict[str, int]:
    """Rebuild `member_office` from what is in the database. Idempotent."""
    async with conn.transaction():
        await conn.execute("DELETE FROM member_office WHERE source = 'hansard_attribution'")
        cur = await conn.execute(
            SQL_FROM_HANSARD.replace("%(normalise)s",
                                     "regexp_replace(lower(btrim(s.speaker_office)), '^the\\s+', '')"))
        from_hansard = cur.rowcount
        from_mnis, rejected = await _build_from_mnis(conn)
    return {"hansard_attribution": from_hansard, "mnis_biography": from_mnis,
            "rejected_reversed": len(rejected), "rejected_examples": rejected[:5]}


_POST_LISTS = ("governmentPosts", "oppositionPosts", "otherPosts")


async def _build_from_mnis(conn: AsyncConnection) -> tuple[int, list[tuple[int, str]]]:
    """Government and opposition posts out of the stored Members API biography."""
    rows = await (await conn.execute(
        "SELECT mnis_id, raw -> 'biography' FROM member_id_map WHERE raw ? 'biography'")).fetchall()
    values: list[tuple[Any, ...]] = []
    rejected: list[tuple[int, str]] = []
    for mnis_id, biography in rows:
        for list_name in _POST_LISTS:
            for post in (biography or {}).get(list_name) or []:
                name = post.get("name")
                key = office_key(name)
                if not key:
                    continue
                start, end = _date(post.get("startDate")), _date(post.get("endDate"))
                if start and end and end < start:
                    # The source contradicts itself.
                    rejected.append((mnis_id, name))
                    continue
                values.append((mnis_id, name, key, start, end, "mnis_biography", 0))
    if not values:
        return 0, rejected
    await conn.execute("DELETE FROM member_office WHERE source = 'mnis_biography'")
    cur = await conn.execute(
        "INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence)"
        " SELECT * FROM unnest(%s::int[], %s::text[], %s::text[], %s::date[], %s::date[], %s::text[], %s::int[])"
        " ON CONFLICT (mnis_id, office_key, start_date, source) DO NOTHING",
        [[v[i] for v in values] for i in range(7)])
    return cur.rowcount, rejected


def _date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


async def load_offices(conn: AsyncConnection) -> OfficeIndex:
    rows = await (await conn.execute(SQL_READ)).fetchall()
    return OfficeIndex([OfficeHolder(mnis_id=r[0], office=r[1], start_date=r[2], end_date=r[3],
                                     source=r[4], evidence=r[5]) for r in rows])
