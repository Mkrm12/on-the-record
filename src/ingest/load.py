"""The offline loader: build `hansard_v2` from the Corpus v2 cache and nothing else."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from ingest import bills as bills_mod
from ingest import corrections as corrections_mod
from ingest import divisions as div_mod
from ingest.corpus_cache import CorpusCache
from ingest.corpus_fetch import WindowSpec
from ingest.hansard_meta import parse_item
from ingest.members import COLUMNS as MEMBER_COLUMNS
from ingest.members import parse_member

# Text fields that live in their own columns and are not repeated inside `meta`.
_TEXT_KEYS = ("ContributionText", "ContributionTextFull")


def clean(value: str | None) -> str | None:
    """Exactly what the original ingest did to a text before storing it: strip NUL, nothing else. HTML cleaning happens
    in the chunker.
    """
    return value.replace("\x00", "") if value else value


# --------------------------------------------------------------------------- Row shaping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebateRow:
    id: str
    date: date
    house: str | None
    heading: str | None
    section: str | None
    section_id: int | None
    section_order: int | None
    window_id: str
    corpus_version: str


@dataclass(frozen=True)
class SpeechRow:
    id: str
    debate_id: str
    seq: int | None
    speaker_name: str | None
    mnis_id: int | None
    text: str | None
    timecode: str | None
    item_id: int | None
    hrs_tag: str | None
    hansard_section: str | None
    api_rank: int | None
    api_member_name: str | None
    meta: str
    window_id: str


def _debate_date(value: object) -> date | None:
    """The sitting date an item carries, or None when it is missing or malformed. Never raises: one bad item must not
    abort a window, it must be counted.
    """
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def shape_day(items: Iterable[dict], window: WindowSpec, corpus_version: str,
              ) -> tuple[list[DebateRow], list[SpeechRow], list[str]]:
    """(debates, speeches, ids that could not be placed) for one day's merged items."""
    debates: dict[str, DebateRow] = {}
    speeches: list[SpeechRow] = []
    unplaceable: list[str] = []
    for item in items:
        parsed = parse_item(item)
        if parsed is None:
            unplaceable.append(str(item.get("ContributionExtId") or item.get("ItemId") or "?"))
            continue
        speech, debate = parsed
        debate_id = debate["id"]
        if debate_id not in debates:
            # The first item of a debate must carry what the debate row needs: a sitting date and a House.
            placed = _debate_date(item.get("SittingDate"))
            if placed is None or not item.get("House"):
                unplaceable.append(str(speech["id"]))
                continue
            debates[debate_id] = DebateRow(
                id=debate_id, date=placed, house=item.get("House"),
                heading=(item.get("DebateSection") or "").strip() or None,
                section=item.get("Section"), section_id=debate["section_id"],
                section_order=debate["section_order"], window_id=window.id,
                corpus_version=corpus_version,
            )  # fmt: skip
        speeches.append(SpeechRow(
            id=speech["id"], debate_id=debate_id, seq=item.get("OrderInDebateSection"),
            speaker_name=item.get("AttributedTo"), mnis_id=item.get("MemberId"),
            # '' not NULL for a contribution with no text: Hansard's "tabled by" markers are empty contributions, and
            # NULL would claim a missing value where there is an empty one.
            text=clean(item.get("ContributionTextFull") or item.get("ContributionText")) or "",
            timecode=item.get("Timecode"), item_id=speech["item_id"], hrs_tag=speech["hrs_tag"],
            hansard_section=speech["hansard_section"], api_rank=speech["api_rank"],
            api_member_name=speech["api_member_name"],
            meta=json.dumps({k: v for k, v in item.items() if k not in _TEXT_KEYS}, ensure_ascii=False),
            window_id=window.id,
        ))  # fmt: skip
    return list(debates.values()), speeches, unplaceable


def shape_correction(item: dict, day: date, window: WindowSpec) -> dict[str, Any] | None:
    """One Written or Ministerial correction, stored raw. Phase 3 links it to the statement it corrects; this session
    only has to make sure the record is not lost.
    """
    cid = item.get("ContributionExtId")
    if not cid:
        return None
    sitting = item.get("SittingDate")
    return {
        "id": cid, "window_id": window.id, "debate_id": item.get("DebateSectionExtId"),
        "date": date.fromisoformat(sitting[:10]) if isinstance(sitting, str) else day,
        "house": item.get("House"), "section": item.get("Section"),
        "speaker_name": item.get("AttributedTo"), "mnis_id": item.get("MemberId"),
        "body": clean(item.get("ContributionTextFull") or item.get("ContributionText")),
        "raw": {k: v for k, v in item.items() if k not in _TEXT_KEYS},
    }  # fmt: skip


# --------------------------------------------------------------------------- SQL
# ---------------------------------------------------------------------------

_DEBATE_COLS = ("id", "date", "house", "heading", "section", "section_id", "section_order",
                "window_id", "corpus_version")
_SPEECH_COLS = ("id", "debate_id", "seq", "speaker_name", "mnis_id", "text", "timecode",
                "item_id", "hrs_tag", "hansard_section", "api_rank", "api_member_name",
                "meta", "window_id")
_CORRECTION_COLS = ("id", "window_id", "debate_id", "date", "house", "section", "speaker_name",
                    "mnis_id", "body", "raw")

_DEBATE_TYPES = ("text", "date", "text", "text", "text", "int", "int", "text", "text")
_SPEECH_TYPES = ("text", "text", "int", "text", "int", "text", "text", "int", "text", "text",
                 "int", "text", "jsonb", "text")
_CORRECTION_TYPES = ("text", "text", "text", "date", "text", "text", "text", "int", "text", "jsonb")


def _upsert(table: str, columns: Sequence[str], types: Sequence[str], key: str = "id") -> str:
    """An idempotent bulk upsert from parallel arrays."""
    others = [c for c in columns if c != key]
    cast = ", ".join(f"%s::{t}[]" for t in types)
    names = ", ".join(columns)
    return f"""
        INSERT INTO {table} ({names})
        SELECT {names} FROM unnest({cast}) AS u({names})
        ON CONFLICT ({key}) DO UPDATE SET {", ".join(f"{c} = excluded.{c}" for c in others)}
        WHERE ({", ".join(f"{table}.{c}" for c in others)})
              IS DISTINCT FROM ({", ".join(f"excluded.{c}" for c in others)})
    """


SQL_UPSERT_DEBATE = _upsert("debate", _DEBATE_COLS, _DEBATE_TYPES)
SQL_UPSERT_SPEECH = _upsert("speech", _SPEECH_COLS, _SPEECH_TYPES)
SQL_UPSERT_CORRECTION = _upsert("hansard_correction", _CORRECTION_COLS, _CORRECTION_TYPES)

SQL_UPSERT_WINDOW = """
    INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role, note, loaded_at)
    VALUES (%s, %s, %s, %s, %s, %s, NULL)
    ON CONFLICT (window_id) DO UPDATE SET
        corpus_version = excluded.corpus_version, start_date = excluded.start_date,
        end_date = excluded.end_date, role = excluded.role, note = excluded.note
"""

SQL_FINISH_WINDOW = """
    UPDATE corpus_window SET loaded_at = now(),
        speeches = (SELECT count(*) FROM speech WHERE window_id = %(w)s),
        debates  = (SELECT count(*) FROM debate WHERE window_id = %(w)s),
        chunks   = (SELECT count(*) FROM chunk  WHERE window_id = %(w)s)
    WHERE window_id = %(w)s
"""

# Every window's counts, recomputed.
SQL_REFRESH_WINDOW_COUNTS = """
    UPDATE corpus_window w SET
        speeches = (SELECT count(*) FROM speech s WHERE s.window_id = w.window_id),
        debates  = (SELECT count(*) FROM debate d WHERE d.window_id = w.window_id),
        chunks   = (SELECT count(*) FROM chunk  k WHERE k.window_id = w.window_id)
"""


async def refresh_window_counts(conn: AsyncConnection) -> int:
    async with conn.transaction():
        return (await conn.execute(SQL_REFRESH_WINDOW_COUNTS)).rowcount


SQL_UPSERT_MEMBER = f"""
    INSERT INTO member_id_map (mnis_id, {", ".join(MEMBER_COLUMNS)}, fetched_at)
    VALUES (%(mnis_id)s, {", ".join(f"%({c})s" for c in MEMBER_COLUMNS)}, now())
    ON CONFLICT (mnis_id) DO UPDATE SET
        {", ".join(f"{c} = excluded.{c}" for c in MEMBER_COLUMNS)}, fetched_at = now()
    WHERE ({", ".join(f"member_id_map.{c}" for c in MEMBER_COLUMNS)})
          IS DISTINCT FROM ({", ".join(f"excluded.{c}" for c in MEMBER_COLUMNS)})
"""

# Denormalise the filter columns from debate/speech onto chunk.
SQL_SYNC_CHUNK_FILTERS = """
    UPDATE chunk c
    SET window_id = d.window_id, debate_date = d.date, house = d.house,
        section = d.section, mnis_id = s.mnis_id
    FROM debate d, speech s
    WHERE d.id = c.debate_id AND s.id = c.speech_id
      AND (c.window_id, c.debate_date, c.house, c.section, c.mnis_id)
          IS DISTINCT FROM (d.window_id, d.date, d.house, d.section, s.mnis_id)
"""


# --------------------------------------------------------------------------- Loading
# ---------------------------------------------------------------------------

# Measured on W0-W5 — 154,882 items, 586 divisions, 492 bills — every one of these was 0.
LOAD_CEILINGS: dict[str, tuple[int, float]] = {
    "unplaceable": (25, 0.005),
    "division_without_date": (3, 0.02),
    "bill_without_title": (3, 0.02),
}


@dataclass
class WindowLoad:
    """What one window's load did. Everything here is checkable against the fetch report."""

    window_id: str
    days: int = 0
    debates: int = 0
    speeches: int = 0
    listed: int = 0
    served: int = 0
    unplaceable: list[str] = field(default_factory=list)
    unserved_by_day: dict[str, int] = field(default_factory=dict)
    # Contributions the API served under more than one debate id on the same day.
    debate_conflicts: dict[str, list[str]] = field(default_factory=dict)
    corrections: int = 0
    divisions: int = 0
    division_votes: int = 0
    division_links: Counter = field(default_factory=Counter)
    bills: int = 0
    bill_stages: int = 0
    bill_sittings: int = 0
    rows_changed: dict[str, int] = field(default_factory=dict)
    # What `bills.link_all` did after this window: sittings -> debates, stages -> debates, divisions -> stages.
    links: dict[str, Any] = field(default_factory=dict)
    # Source records skipped because they break a rule the schema enforces, by reason.
    quarantined: Counter = field(default_factory=Counter)

    @property
    def unserved(self) -> int:
        return sum(self.unserved_by_day.values())

    def problems(self) -> list[str]:
        """Losses too large to be stray records — each "<what>: <count> of <total> (ceiling <n>)"."""
        found = []
        for what, count, total in (
                ("unplaceable", len(self.unplaceable), self.served),
                ("division_without_date", self.quarantined["division_without_date"],
                 self.divisions + self.quarantined["division_without_date"]),
                ("bill_without_title", self.quarantined["bill_without_title"],
                 self.bills + self.quarantined["bill_without_title"])):
            floor, share = LOAD_CEILINGS[what]
            ceiling = max(floor, int(share * total))
            if count > ceiling:
                found.append(f"{what}: {count} of {total} (ceiling {ceiling})")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id, "days": self.days, "debates": self.debates,
            "speeches": self.speeches, "listed": self.listed, "served": self.served,
            "unserved": self.unserved, "unserved_by_day": self.unserved_by_day,
            "unplaceable": self.unplaceable[:50], "corrections": self.corrections,
            "debate_conflicts": len(self.debate_conflicts),
            "debate_conflict_examples": dict(list(self.debate_conflicts.items())[:5]),
            "divisions": self.divisions, "division_votes": self.division_votes,
            "division_links": dict(self.division_links), "bills": self.bills,
            "bill_stages": self.bill_stages, "bill_sittings": self.bill_sittings,
            "rows_changed": self.rows_changed, "links": self.links, "quarantined": dict(self.quarantined),
            "problems": self.problems(),
        }  # fmt: skip


def _columns(rows: Sequence[Any], fields: Sequence[str]) -> list[list[Any]]:
    return [[getattr(r, f) for r in rows] for f in fields]


async def _bulk(conn: AsyncConnection, sql: str, rows: Sequence[Any], fields: Sequence[str]) -> int:
    if not rows:
        return 0
    cur = await conn.execute(sql, _columns(rows, fields))
    return cur.rowcount


async def load_window_text(conn: AsyncConnection, cache: CorpusCache, window: WindowSpec,
                           corpus_version: str, load: WindowLoad, *, batch_days: int = 1) -> None:
    """Debates and speeches for one window, a day at a time."""
    debates: list[DebateRow] = []
    speeches: list[SpeechRow] = []
    pending = 0
    for day_items in cache.contributions(window):
        load.days += 1
        load.served += len(day_items.items)
        if day_items.listed:
            load.listed += day_items.listed
        if day_items.unserved:
            load.unserved_by_day[day_items.day.isoformat()] = day_items.unserved
        for cid, ids in day_items.debate_conflicts.items():
            load.debate_conflicts[cid] = list(ids)
        day_debates, day_speeches, unplaceable = shape_day(day_items.items.values(), window, corpus_version)
        debates += day_debates
        speeches += day_speeches
        load.unplaceable += unplaceable
        pending += 1
        if pending >= batch_days:
            await _flush_text(conn, debates, speeches, load)
            debates, speeches, pending = [], [], 0
    await _flush_text(conn, debates, speeches, load)


async def _flush_text(conn: AsyncConnection, debates: list[DebateRow], speeches: list[SpeechRow],
                      load: WindowLoad) -> None:
    if not debates and not speeches:
        return
    # One id can only be written once per statement: ON CONFLICT DO UPDATE refuses to touch the same row twice, and a
    # day read in two sort orders is exactly how that happens.
    unique_debates = list({d.id: d for d in debates}.values())
    unique_speeches = list({s.id: s for s in speeches}.values())
    async with conn.transaction():
        changed_d = await _bulk(conn, SQL_UPSERT_DEBATE, unique_debates, _DEBATE_COLS)
        changed_s = await _bulk(conn, SQL_UPSERT_SPEECH, unique_speeches, _SPEECH_COLS)
    load.debates += len(unique_debates)
    load.speeches += len(unique_speeches)
    load.rows_changed["debate"] = load.rows_changed.get("debate", 0) + changed_d
    load.rows_changed["speech"] = load.rows_changed.get("speech", 0) + changed_s


async def load_corrections(conn: AsyncConnection, cache: CorpusCache, window: WindowSpec,
                           load: WindowLoad) -> None:
    rows = []
    seen: set[str] = set()
    for day, item in cache.corrections(window):
        shaped = shape_correction(item, day, window)
        if shaped and shaped["id"] not in seen:
            seen.add(shaped["id"])
            rows.append(shaped)
    if not rows:
        return
    arrays = [[r[c] if c != "raw" else Jsonb(r[c]) for r in rows] for c in _CORRECTION_COLS]
    # Jsonb adapts a single value, not an array of them; the array goes as JSON text.
    arrays[_CORRECTION_COLS.index("raw")] = [json.dumps(r["raw"], ensure_ascii=False) for r in rows]
    async with conn.transaction():
        cur = await conn.execute(SQL_UPSERT_CORRECTION, arrays)
    load.corrections = len(rows)
    load.rows_changed["hansard_correction"] = cur.rowcount


def _hansard_links(cache: CorpusCache, window: WindowSpec) -> dict[tuple[str, str, int], str]:
    """(house, day, division number) -> DebateSectionExtId, straight from Hansard."""
    out: dict[tuple[str, str, int], str] = {}
    for row in cache.hansard_divisions(window):
        ext, house, day, number = (row.get("DebateSectionExtId"), row.get("House"),
                                   row.get("Date"), row.get("Number"))
        if not (ext and house and isinstance(day, str)):
            continue
        try:
            n = int(str(number).strip())
        except (TypeError, ValueError):
            continue
        out[(house, day[:10], n)] = ext
    return out


async def load_divisions(conn: AsyncConnection, cache: CorpusCache, window: WindowSpec,
                         load: WindowLoad) -> None:
    """Commons and Lords divisions, linked to their debate by Hansard's own statement of the link where there is one,
    and by the old inference only where there is not.
    """
    links = _hansard_links(cache, window)
    known_debates = {r[0] for r in await (await conn.execute(
        "SELECT id FROM debate WHERE window_id = %s", (window.id,))).fetchall()}
    context: tuple[list, list] | None = None

    parsed_rows: list[div_mod.Parsed] = []
    for row in cache.commons_divisions(window):
        detail = cache.commons_detail(row.get("DivisionId")) if isinstance(row.get("DivisionId"), int) else None
        if detail:
            parsed_rows.append(div_mod.parse_commons(detail))
    for row in cache.lords_divisions(window):
        parsed_rows.append(div_mod.parse_lords(row))

    for parsed in parsed_rows:
        division = parsed.division
        if division.get("division_date") is None:
            # division_date is NOT NULL, and the key below read `.isoformat` of it: one undated record crashed the whole
            # window's load.
            load.quarantined["division_without_date"] += 1
            continue
        key = (division["house"], division["division_date"].isoformat(), division.get("number"))
        debate_id = links.get(key)
        if debate_id and debate_id in known_debates:
            division["debate_id"], division["debate_link"] = debate_id, "hansard_division_list"
        elif debate_id:
            # Hansard names a debate that is not in this window: a division taken on a day the window does not cover, or
            # a section the load did not keep.
            division["debate_id"], division["debate_link"] = None, "hansard_link_outside_window"
            if division.get("is_deferred"):
                # A deferred division's named section is the empty "Deferred Division" list, which never becomes a
                # debate row.
                if context is None:
                    context = await div_mod.load_link_context(conn)
                found, basis = div_mod.link_debate(division, *context)
                if found is not None:
                    division["debate_id"], division["debate_link"] = found, basis
        else:
            if context is None:
                context = await div_mod.load_link_context(conn)
            division["debate_id"], division["debate_link"] = div_mod.link_debate(division, *context)
        load.division_links[division["debate_link"]] += 1
        await div_mod.store(conn, parsed)
        load.divisions += 1
        load.division_votes += len(parsed.votes)


async def load_bills(conn: AsyncConnection, cache: CorpusCache, window: WindowSpec,
                     load: WindowLoad) -> None:
    """Bills whose sittings fall inside the window, with their stages."""
    sittings = cache.bill_sittings(window)
    by_bill: dict[int, list[dict]] = {}
    for row in sittings:
        if isinstance(row.get("billId"), int):
            by_bill.setdefault(row["billId"], []).append(row)
    debates = [div_mod.DebateInfo(i, d, house, heading, section)
               for i, d, house, heading, section in await (await conn.execute(
                   "SELECT id, date, house, heading, section FROM debate WHERE window_id = %s",
                   (window.id,))).fetchall()]
    for bill_id in sorted(by_bill):
        detail, stages_payload = cache.bill(bill_id)
        if detail is None:
            continue
        bill = bills_mod.parse_bill({"billId": bill_id, **(detail or {})}, detail)
        if not bill.get("title"):
            # bill.title is NOT NULL and trimmed (migration 017): a bill with no short title is a broken record, and one
            # of them must not abort the window's other bills.
            load.quarantined["bill_without_title"] += 1
            continue
        stages, stage_sittings = bills_mod.parse_stages(bill_id, stages_payload)
        for sitting in stage_sittings:
            day = sitting.get("date")
            if isinstance(day, date):
                # Every name the bill has had: it is stored under its Royal Assent title and debated under its Bill
                # title (1,052 of 1,247 in-window sittings went unlinked on the first alone).
                debate_id, basis = bills_mod.link_sitting(bills_mod.bill_name_forms(bill), sitting.get("house"),
                                                          day, debates)  # fmt: skip
                sitting["debate_id"], sitting["debate_link"] = debate_id, basis
        await bills_mod.store_bill(conn, bill, stages, stage_sittings)
        load.bills += 1
        load.bill_stages += len(stages)
        load.bill_sittings += len(stage_sittings)


async def load_members(conn: AsyncConnection, cache: CorpusCache, *,
                       only: Iterable[int] | None = None) -> dict[str, int]:
    """Every member in the cache, with the name Hansard attributed to them as the fallback."""
    fallbacks = dict(await (await conn.execute(
        "SELECT mnis_id, mode() WITHIN GROUP (ORDER BY speaker_name) FROM speech "
        "WHERE mnis_id IS NOT NULL GROUP BY mnis_id")).fetchall())
    ids = sorted(set(only) if only is not None else set(cache.member_ids()))
    stats = {"seen": len(ids), "stubs": 0, "changed": 0}
    for mnis_id in ids:
        payloads, missing = cache.member(mnis_id)
        if not payloads:
            continue
        row = parse_member(mnis_id, payloads, missing, fallback_name=fallbacks.get(mnis_id))
        if row.get("name") is None or payloads.get("profile") is None:
            stats["stubs"] += 1
        row = {**row, "raw": Jsonb(row["raw"] or {})}
        async with conn.transaction():
            cur = await conn.execute(SQL_UPSERT_MEMBER, row)
        stats["changed"] += cur.rowcount
    return stats


async def register_window(conn: AsyncConnection, window: WindowSpec, corpus_version: str) -> None:
    async with conn.transaction():
        await conn.execute(SQL_UPSERT_WINDOW, (window.id, corpus_version, window.start,
                                               window.end, window.role, window.note or None))


async def finish_window(conn: AsyncConnection, window: WindowSpec) -> None:
    async with conn.transaction():
        await conn.execute(SQL_FINISH_WINDOW, {"w": window.id})


async def sync_chunk_filters(conn: AsyncConnection) -> int:
    async with conn.transaction():
        cur = await conn.execute(SQL_SYNC_CHUNK_FILTERS)
    return cur.rowcount


async def load_window(conn: AsyncConnection, cache: CorpusCache, window: WindowSpec,
                      corpus_version: str, *, with_divisions: bool = True,
                      with_bills: bool = True) -> WindowLoad:
    """One window, end to end, from the cache. Idempotent: a second call changes nothing."""
    load = WindowLoad(window_id=window.id)
    await register_window(conn, window, corpus_version)
    await load_window_text(conn, cache, window, corpus_version, load)
    await load_corrections(conn, cache, window, load)
    if with_divisions:
        await load_divisions(conn, cache, window, load)
    if with_bills:
        await load_bills(conn, cache, window, load)
    if with_divisions and with_bills:
        # The step the v2 loader never ran (gotcha 9): division -> bill stage, and a stage's debate.
        load.links = await bills_mod.link_all(conn)
    # Corrections -> the speech they correct.
    load.links["corrections"] = await corrections_mod.link_all(conn)
    load.links["deferred"] = await div_mod.link_deferred(conn)
    if with_divisions:
        load.links["source_gaps"] = await record_source_gaps(conn, cache, window)
    return load


SQL_UPSERT_GAP = """
    INSERT INTO source_gap (window_id, source, house, kind, count, detail)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (window_id, source, house, kind) DO UPDATE SET count = excluded.count, detail = excluded.detail,
        recorded_at = now()
    WHERE (source_gap.count, source_gap.detail) IS DISTINCT FROM (excluded.count, excluded.detail)
"""
SQL_CLEAR_GAPS = "DELETE FROM source_gap WHERE window_id = %s AND kind = %s AND house <> ALL(%s::text[])"


def voter_list_gaps(tallies: dict[str, Any]) -> list[tuple[str, int]]:
    """(house, divisions Hansard lists) for every House whose Votes API holds no division in the window at all."""
    return [(house, len(t.hansard)) for house, t in sorted(tallies.items()) if t.hansard and not t.votes]


async def record_source_gaps(conn: AsyncConnection, cache: CorpusCache, window: WindowSpec) -> dict[str, int]:
    """Write what the sources lack for this window into `source_gap` (migration 020), and clear what they now hold.
    """
    from ingest.corpus_fetch import Cache, reconcile_divisions

    gaps = voter_list_gaps(reconcile_divisions(Cache(cache.root), window))
    async with conn.transaction():
        for house, listed in gaps:
            await conn.execute(SQL_UPSERT_GAP, (
                window.id, f"{house.lower()}_votes_api", house, "no_voter_lists", listed,
                f"Hansard lists {listed} division(s) in {window.id}; the {house} Votes API holds no record of any, "
                "so who voted is not available for them"))  # fmt: skip
        cleared = (await conn.execute(SQL_CLEAR_GAPS, (window.id, "no_voter_lists", [h for h, _ in gaps]))).rowcount
    return {"no_voter_lists": len(gaps), "cleared": cleared}


def parse_timecode(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None
