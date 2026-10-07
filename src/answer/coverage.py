"""What the corpus holds, and what follows from that."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from enum import Enum
from typing import Any

# Windows whose numbers must never appear in a report.
HELDOUT_ROLES = frozenset({"heldout"})


class Standing(str, Enum):
    """What the corpus can say about one date."""

    SITTING = "sitting"
    NO_SITTING = "no_sitting"  # inside a window, but the House did not sit
    OUTSIDE_CORPUS = "outside_corpus"  # inside no window at all


@dataclass(frozen=True)
class Row:
    """One coverage row: window x house x section, as measured."""

    window_id: str
    house: str
    section: str
    role: str | None = None
    corpus_version: str | None = None
    declared_start: Date | None = None
    declared_end: Date | None = None
    first_sitting: Date | None = None
    last_sitting: Date | None = None
    sitting_days: int = 0
    debates: int = 0
    speeches: int = 0
    contributions: int = 0
    boilerplate: int = 0
    speakers: int = 0
    unattributed: int = 0
    chunks: int = 0
    embedded_chunks: int = 0
    divisions: int = 0
    division_votes: int = 0
    unreconciled_divisions: int = 0
    max_gap_days: int | None = None
    max_gap_start: Date | None = None
    max_gap_end: Date | None = None

    @property
    def heldout(self) -> bool:
        return (self.role or "") in HELDOUT_ROLES

    def covers(self, when: Date) -> bool:
        """Whether this row's window contains the date."""
        start = self.declared_start or self.first_sitting
        end = self.declared_end or self.last_sitting
        if start is None or end is None:
            return False
        return start <= when <= end


# --- reading ---------------------------------------------------------------------------------------------


async def _rows(conn, where: str = "", params: tuple = ()) -> list[Row]:
    cur = await conn.execute(f"""
        SELECT window_id, house, section, role, corpus_version, declared_start, declared_end,
               first_sitting, last_sitting, sitting_days, debates, speeches, contributions, boilerplate,
               speakers, unattributed, chunks, embedded_chunks, divisions, division_votes,
               unreconciled_divisions, max_gap_days, max_gap_start, max_gap_end
        FROM coverage {where}
        ORDER BY window_id, house, section
    """, params)
    columns = [c.name for c in cur.description or []]
    return [Row(**dict(zip(columns, row, strict=True))) for row in await cur.fetchall()]


async def load(conn, house: str | None = None, include_heldout: bool = True) -> list[Row]:
    """The coverage rows, optionally for one House."""
    clauses, params = [], []
    if house:
        clauses.append("house = %s")
        params.append(house)
    if not include_heldout:
        clauses.append("(role IS NULL OR role <> ALL(%s))")
        params.append(sorted(HELDOUT_ROLES))
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return await _rows(conn, where, tuple(params))


async def standing(conn, when: Date, house: str | None = None, section: str | None = None) -> dict[str, Any]:
    """What the corpus can say about a date: sitting, inside a window but not sitting, or outside every window."""
    rows = [r for r in await load(conn, house)
            if (section is None or r.section == section) and r.covers(when)]  # fmt: skip
    if not rows:
        return {
            "standing": Standing.OUTSIDE_CORPUS,
            "date": when,
            "house": house,
            "windows": [],
            "covered": await spans(conn, house),
        }
    cur = await conn.execute("""
        SELECT count(*) FROM debate
        WHERE date = %s AND (%s::text IS NULL OR house = %s) AND (%s::text IS NULL OR section = %s)
    """, (when, house, house, section, section))
    sat = ((await cur.fetchall())[0][0] or 0) > 0
    return {
        "standing": Standing.SITTING if sat else Standing.NO_SITTING,
        "date": when,
        "house": house,
        # Deduplicated: there is one row per section, so a date in the Commons matched four rows and the list read
        # ['W0', 'W0', 'W0', 'W0'].
        "windows": sorted({r.window_id for r in rows}),
        "sections": sorted({r.section for r in rows}),
    }


async def spans(conn, house: str | None = None) -> list[dict[str, Any]]:
    """The date ranges the corpus covers, one per window, for telling a user what they could ask about instead."""
    seen: dict[str, dict[str, Any]] = {}
    for row in await load(conn, house):
        entry = seen.setdefault(row.window_id, {
            "window_id": row.window_id,
            "start": row.declared_start or row.first_sitting,
            "end": row.declared_end or row.last_sitting,
            "houses": set(),
        })
        entry["houses"].add(row.house)
    out = []
    for entry in seen.values():
        out.append({**entry, "houses": sorted(entry["houses"])})
    return sorted(out, key=lambda e: (e["start"] or Date.min))


async def note(conn, house: str | None = None, section: str | None = None,
               from_date: Date | None = None, to_date: Date | None = None) -> str:  # fmt: skip
    """The `coverage_note`: one sentence a reader can use to place an answer."""
    rows = await load(conn, house)
    if section:
        rows = [r for r in rows if r.section == section]
    if from_date or to_date:
        rows = [r for r in rows
                if (not to_date or (r.first_sitting and r.first_sitting <= to_date))
                and (not from_date or (r.last_sitting and r.last_sitting >= from_date))]  # fmt: skip
    rows = [r for r in rows if r.sitting_days]
    if not rows:
        return "The corpus holds nothing for that House, section and period."

    days = sum(r.sitting_days for r in rows)
    contributions = sum(r.contributions for r in rows)
    houses = sorted({r.house for r in rows})
    sections = sorted({r.section for r in rows})
    where = " and ".join(sections) if len(sections) <= 2 else f"{len(sections)} sections"

    # Per window, never as one span.
    periods: dict[str, tuple[Date | None, Date | None]] = {}
    for row in rows:
        low, high = periods.get(row.window_id, (None, None))
        if row.first_sitting:
            low = row.first_sitting if low is None else min(low, row.first_sitting)
        if row.last_sitting:
            high = row.last_sitting if high is None else max(high, row.last_sitting)
        periods[row.window_id] = (low, high)
    spelled = ", ".join(f"{low} to {high}" for low, high in sorted(periods.values(), key=lambda p: p[0] or Date.min))

    if len(periods) == 1:
        scope = spelled
    elif len(periods) <= 3:
        scope = f"{len(periods)} separate periods ({spelled})"
    else:
        earliest = min(low for low, _ in periods.values() if low)
        latest = max(high for _, high in periods.values() if high)
        scope = (f"{len(periods)} separate periods between {earliest} and {latest}, "
                 f"not the whole of that range")  # fmt: skip

    parts = [f"Drawn from {where} in the {' and '.join(houses)}, {scope}: "
             f"{days:,} sitting day{'s' if days != 1 else ''}, "
             f"{contributions:,} contribution{'s' if contributions != 1 else ''}."]  # fmt: skip

    # The gap is named when it is long enough that a reader would otherwise wonder.
    worst = max((r for r in rows if r.max_gap_days), key=lambda r: r.max_gap_days or 0, default=None)
    if worst and (worst.max_gap_days or 0) >= 14:
        parts.append(f"The longest break in sittings runs {worst.max_gap_start} to {worst.max_gap_end} "
                     f"({worst.max_gap_days} days).")  # fmt: skip
    unattributed = sum(r.unattributed for r in rows)
    if unattributed and contributions:
        share = unattributed / (contributions + unattributed)
        if share >= 0.01:
            parts.append(f"{unattributed:,} contribution{'s' if unattributed != 1 else ''} ({share:.0%}) could "
                         "not be matched to a member, so a roster for this period may be incomplete.")  # fmt: skip
    return " ".join(parts)


async def gaps(conn, minimum_days: int = 14, house: str | None = None) -> list[dict[str, Any]]:
    """Runs of days with no sitting, longest first. A silence reported as a silence."""
    return sorted(
        ({"window_id": r.window_id, "house": r.house, "section": r.section,
          "days": r.max_gap_days, "start": r.max_gap_start, "end": r.max_gap_end}  # fmt: skip
         for r in await load(conn, house) if (r.max_gap_days or 0) >= minimum_days),
        key=lambda g: -(g["days"] or 0),
    )


# --- refreshing ------------------------------------------------------------------------------------------

# Every aggregate in one pass per window.
_REFRESH = """
WITH sittings AS (
    SELECT d.window_id, d.house, d.section, d.date,
           -- DISTINCT d.id, not count(*): the LEFT JOIN to speech multiplies each debate by its speeches, so
           -- count(*) here reported the speech count as the debate count and `stale()` then flagged every
           -- window as out of date immediately after a successful refresh.
           count(DISTINCT d.id) AS debates,
           count(sp.id) AS speeches,
           count(sp.id) FILTER (WHERE NOT sp.is_boilerplate) AS contributions,
           count(sp.id) FILTER (WHERE sp.is_boilerplate) AS boilerplate,
           count(sp.id) FILTER (WHERE NOT sp.is_boilerplate AND sp.mnis_id IS NULL) AS unattributed
    FROM debate d
    LEFT JOIN speech sp ON sp.debate_id = d.id
    WHERE (%(window)s::text IS NULL OR d.window_id = %(window)s)
    GROUP BY d.window_id, d.house, d.section, d.date
),
-- The gap is computed over DISTINCT sitting dates per group, so two debates on one day are one sitting day.
ordered AS (
    SELECT window_id, house, section, date,
           lag(date) OVER (PARTITION BY window_id, house, section ORDER BY date) AS previous
    FROM (SELECT DISTINCT window_id, house, section, date FROM sittings) s
),
biggest AS (
    SELECT DISTINCT ON (window_id, house, section)
           window_id, house, section,
           (date - previous) AS gap_days, previous AS gap_start, date AS gap_end
    FROM ordered WHERE previous IS NOT NULL
    ORDER BY window_id, house, section, (date - previous) DESC, date
),
totals AS (
    SELECT window_id, house, section,
           min(date) AS first_sitting, max(date) AS last_sitting,
           count(DISTINCT date) AS sitting_days,
           sum(debates)::int AS debates, sum(speeches)::int AS speeches,
           sum(contributions)::int AS contributions, sum(boilerplate)::int AS boilerplate,
           sum(unattributed)::int AS unattributed
    FROM sittings GROUP BY window_id, house, section
),
speakers AS (
    SELECT d.window_id, d.house, d.section, count(DISTINCT sp.mnis_id)::int AS speakers
    FROM debate d JOIN speech sp ON sp.debate_id = d.id
    WHERE sp.mnis_id IS NOT NULL AND NOT sp.is_boilerplate
      AND (%(window)s::text IS NULL OR d.window_id = %(window)s)
    GROUP BY d.window_id, d.house, d.section
),
chunks AS (
    SELECT c.window_id, c.house, c.section, count(*)::int AS chunks,
           count(*) FILTER (WHERE c.embedding IS NOT NULL)::int AS embedded_chunks
    FROM chunk c
    WHERE NOT c.is_boilerplate AND (%(window)s::text IS NULL OR c.window_id = %(window)s)
    GROUP BY c.window_id, c.house, c.section
),
divisions AS (
    SELECT d.window_id, d.house, d.section,
           count(DISTINCT v.id)::int AS divisions,
           count(dv.mnis_id)::int AS division_votes,
           count(DISTINCT v.id) FILTER (WHERE v.tally_reconciled IS FALSE)::int AS unreconciled_divisions
    FROM division v
    JOIN debate d ON d.id = v.debate_id
    LEFT JOIN division_vote dv ON dv.division_id = v.id
    WHERE (%(window)s::text IS NULL OR d.window_id = %(window)s)
    GROUP BY d.window_id, d.house, d.section
)
INSERT INTO coverage (
    window_id, house, section, corpus_version, role, declared_start, declared_end,
    first_sitting, last_sitting, sitting_days, debates, speeches, contributions, boilerplate,
    speakers, unattributed, chunks, embedded_chunks, divisions, division_votes, unreconciled_divisions,
    max_gap_days, max_gap_start, max_gap_end, refreshed_at
)
SELECT t.window_id, t.house, t.section, w.corpus_version, w.role, w.start_date, w.end_date,
       t.first_sitting, t.last_sitting, t.sitting_days, t.debates, t.speeches, t.contributions, t.boilerplate,
       COALESCE(s.speakers, 0), t.unattributed,
       COALESCE(k.chunks, 0), COALESCE(k.embedded_chunks, 0),
       COALESCE(v.divisions, 0), COALESCE(v.division_votes, 0), COALESCE(v.unreconciled_divisions, 0),
       b.gap_days, b.gap_start, b.gap_end, now()
FROM totals t
LEFT JOIN corpus_window w ON w.window_id = t.window_id
LEFT JOIN speakers  s ON (s.window_id, s.house, s.section) = (t.window_id, t.house, t.section)
LEFT JOIN chunks    k ON (k.window_id, k.house, k.section) = (t.window_id, t.house, t.section)
LEFT JOIN divisions v ON (v.window_id, v.house, v.section) = (t.window_id, t.house, t.section)
LEFT JOIN biggest   b ON (b.window_id, b.house, b.section) = (t.window_id, t.house, t.section)
WHERE t.window_id IS NOT NULL
ON CONFLICT (window_id, house, section) DO UPDATE SET
    corpus_version = EXCLUDED.corpus_version, role = EXCLUDED.role,
    declared_start = EXCLUDED.declared_start, declared_end = EXCLUDED.declared_end,
    first_sitting = EXCLUDED.first_sitting, last_sitting = EXCLUDED.last_sitting,
    sitting_days = EXCLUDED.sitting_days, debates = EXCLUDED.debates, speeches = EXCLUDED.speeches,
    contributions = EXCLUDED.contributions, boilerplate = EXCLUDED.boilerplate,
    speakers = EXCLUDED.speakers, unattributed = EXCLUDED.unattributed,
    chunks = EXCLUDED.chunks, embedded_chunks = EXCLUDED.embedded_chunks,
    divisions = EXCLUDED.divisions, division_votes = EXCLUDED.division_votes,
    unreconciled_divisions = EXCLUDED.unreconciled_divisions,
    max_gap_days = EXCLUDED.max_gap_days, max_gap_start = EXCLUDED.max_gap_start,
    max_gap_end = EXCLUDED.max_gap_end, refreshed_at = now()
"""


async def refresh(conn, window_id: str | None = None) -> int:
    """Recompute coverage for one window, or for all of them. Returns the number of rows written."""
    cur = await conn.execute(_REFRESH, {"window": window_id})
    return cur.rowcount or 0


async def stale(conn) -> list[str]:
    """Windows present in the corpus with no coverage row, or whose debate count no longer matches."""
    cur = await conn.execute("""
        SELECT d.window_id
        FROM debate d
        LEFT JOIN coverage c ON c.window_id = d.window_id AND c.house = d.house AND c.section = d.section
        GROUP BY d.window_id, d.house, d.section, c.debates
        HAVING c.debates IS NULL OR c.debates <> count(*)
    """)
    return sorted({row[0] for row in await cur.fetchall() if row[0]})
