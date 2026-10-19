"""Debate threads: one logical debate across the records Hansard splits a sitting's business into."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date as Date


# A heading that is a bill's name and nothing else ("Crime and Policing Bill", "… Bill [HL]", "… Bill [Lords]"), never a
# motion about one ("… Bill (Money)", "… Bill: Programme").
_BILL_NAME = re.compile(r"\bbill(?:\s*\[(?:hl|lords)\])?$")


def is_bill(heading: str | None) -> bool:
    return bool(_BILL_NAME.search(key(heading)))


def key(heading: str | None) -> str:
    """A heading as a reader compares it: case, apostrophes and spacing folded."""
    folded = (heading or "").casefold().replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", folded).strip()


@dataclass(frozen=True)
class Meta:
    """What threading reads of a record."""

    debate_id: str
    house: str
    date: Date
    section: str | None
    section_order: int | None
    heading: str | None


@dataclass(frozen=True)
class Thread:
    debate_ids: tuple[str, ...]  # in the record's order
    house: str
    date: Date
    section: str | None
    heading: str | None  # as the first record prints it

    @property
    def thread_id(self) -> str:
        return self.debate_ids[0]

    @property
    def key(self) -> str:
        return key(self.heading)


def _place(m: Meta) -> tuple:
    return (m.house, m.date, m.section or "")


def build(metas: Iterable[Meta]) -> list[Thread]:
    """Every record in exactly one thread."""
    by_place: dict[tuple, list[Meta]] = {}
    for m in metas:
        by_place.setdefault(_place(m), []).append(m)
    out: list[Thread] = []
    for place in sorted(by_place, key=lambda p: (p[0], p[1], p[2])):
        ordered = sorted(by_place[place], key=lambda m: (m.section_order is None, m.section_order or 0, m.debate_id))
        run: list[Meta] = []
        for m in ordered:
            joins = (run and m.section_order is not None and run[-1].section_order is not None and key(m.heading)
                     and key(m.heading) == key(run[-1].heading))  # fmt: skip
            if run and not joins:
                out.append(_thread(run))
                run = []
            run.append(m)
        if run:
            out.append(_thread(run))
    return _join_bills(out)


def _join_bills(threads: list[Thread]) -> list[Thread]:
    """Threads under one bill's heading in one House-day-section become one, in the day's order."""
    out: list[Thread] = []
    seen: dict[tuple, int] = {}
    for t in threads:
        place = (t.house, t.date, t.section or "", t.key)
        if is_bill(t.heading) and place in seen:
            first = out[seen[place]]
            out[seen[place]] = Thread(first.debate_ids + t.debate_ids, first.house, first.date, first.section,
                                      first.heading)  # fmt: skip
            continue
        if is_bill(t.heading):
            seen[place] = len(out)
        out.append(t)
    return out


def _thread(run: list[Meta]) -> Thread:
    first = run[0]
    return Thread(tuple(m.debate_id for m in run), first.house, first.date, first.section, first.heading)


def index(threads: Iterable[Thread]) -> dict[str, Thread]:
    return {did: t for t in threads for did in t.debate_ids}


def violations(threads: list[Thread], metas: list[Meta]) -> list[tuple[str, str]]:
    """Pairs of threads that share a House, day, section and heading with no other recorded business between them, as
    (first thread id, second thread id).
    """
    order = {m.debate_id: m.section_order for m in metas}
    by_place: dict[tuple, list[Meta]] = {}
    for m in metas:
        by_place.setdefault(_place(m), []).append(m)
    found = []
    groups: dict[tuple, list[Thread]] = {}
    for t in threads:
        if t.key:
            groups.setdefault((t.house, t.date, t.section or "", t.key), []).append(t)
    for (house, day, section, k), same in groups.items():
        placed = sorted((t for t in same if order.get(t.debate_ids[0]) is not None), key=lambda t: order[t.debate_ids[0]])
        for a, b in zip(placed, placed[1:]):
            lo, hi = order[a.debate_ids[-1]], order[b.debate_ids[0]]
            between = [m for m in by_place.get((house, day, section), ()) if m.section_order is not None
                       and lo < m.section_order < hi and key(m.heading) != k]  # fmt: skip
            if not between:
                found.append((a.thread_id, b.thread_id))
    return found


# --- the database --------------------------------------------------------------------------------------------------

_SQL = """
    SELECT d.id, d.house, d.date, d.section, d.section_order, d.heading
    FROM debate d
    WHERE %(all)s OR (d.house, d.date) IN (SELECT house, date FROM debate WHERE id = ANY(%(ids)s))
"""


async def threads_of(conn, debate_ids: list[str] | tuple[str, ...]) -> dict[str, Thread]:
    """Each asked record's thread, built from every record of its House and day (on a served connection, the masked
    record).
    """
    if not debate_ids:
        return {}
    cur = await conn.execute(_SQL, {"all": False, "ids": list(debate_ids)})
    built = index(build(Meta(*row) for row in await cur.fetchall()))
    return {did: built[did] for did in debate_ids if did in built}


def all_metas_sync(conn) -> list[Meta]:
    """Every record the connection can see (the invariant gate)."""
    return [Meta(*row) for row in conn.execute(_SQL, {"all": True, "ids": []}).fetchall()]


async def day_threads(conn, house: str, day: Date) -> list[Thread]:
    """Every thread of one House on one day, in every section."""
    cur = await conn.execute("SELECT id, house, date, section, section_order, heading FROM debate "
                             "WHERE house = %s AND date = %s", (house, day))
    return build(Meta(*row) for row in await cur.fetchall())
