"""Commons and Lords divisions, with every vote."""

import asyncio
import re
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, NamedTuple

from psycopg.types.json import Jsonb

from ingest.text import clean_text
from ingest.http import FetchError, PoliteClient

LORDS_ID_OFFSET = 1_000_000
COMMONS_SEARCH = "https://commonsvotes-api.parliament.uk/data/divisions.json/search"
COMMONS_DETAIL = "https://commonsvotes-api.parliament.uk/data/division/{id}.json"
LORDS_SEARCH = "https://lordsvotes-api.parliament.uk/data/Divisions/search"

DIVISION_COLUMNS = (
    "id", "house", "source_id", "number", "division_date", "held_at", "title", "motion_text", "motion_type",
    "ayes", "noes", "result", "is_deferred", "ayes_listed", "noes_listed", "no_vote_count",
    "ayes_named", "noes_named", "tally_reconciled",
    "government_won", "debate_id", "debate_link", "anomalies", "raw",
    "is_remote", "remote_start", "remote_end",
    "evel_type", "evel_country", "double_majority_ayes", "double_majority_noes",
)  # fmt: skip
VOTE_COLUMNS = (
    "division_id", "mnis_id", "lobby", "member_name", "party", "party_abbreviation", "sub_party", "member_from",
    "is_teller", "proxy_name", "flags",
)  # fmt: skip

_COMMONS_LISTS = ("Ayes", "Noes", "NoVoteRecorded", "AyeTellers", "NoTellers")
_LORDS_LISTS = ("contents", "notContents", "contentTellers", "notContentTellers")


class Parsed(NamedTuple):
    division: dict[str, Any]
    votes: list[dict[str, Any]]


# --------------------------------------------------------------------------- Parsing
# ---------------------------------------------------------------------------


def _get(member: dict, *keys: str) -> Any:
    for key in keys:
        if member.get(key) is not None:
            return member[key]
    return None


def _fields(member: dict) -> dict[str, Any]:
    """Commons uses PascalCase keys, Lords camelCase; normalise to one shape."""
    return {
        "id": _get(member, "MemberId", "memberId"),
        "name": _get(member, "Name", "name"),
        "party": _get(member, "Party", "party"),
        "abbr": _get(member, "PartyAbbreviation", "partyAbbreviation"),
        "sub": _get(member, "SubParty"),
        "from": _get(member, "MemberFrom", "memberFrom"),
        "proxy": _get(member, "ProxyName"),
    }


def _resolve_votes(division_id: int, sources: list[tuple[str, str, bool, list | None]]) -> tuple[list[dict], Counter]:
    """Collapse every listing of every member into exactly one row per member."""
    grouped: dict[int, list[tuple[str, str, bool, dict]]] = {}
    counts: Counter = Counter()
    for source, lobby, teller, members in sources:
        for m in members or []:
            f = _fields(m)
            if f["id"] is None:
                counts["vote_without_member_id"] += 1
                continue
            grouped.setdefault(f["id"], []).append((source, lobby, teller, f))

    rows = []
    for mnis_id, items in grouped.items():
        flags: list[str] = []
        if any(n > 1 for n in Counter(src for src, _, _, _ in items).values()):
            flags.append("duplicate_listing")
            counts["duplicate_listing"] += 1
        lobbies = {lobby for _, lobby, _, _ in items if lobby in ("aye", "no")}
        if lobbies == {"aye", "no"}:
            lobby = "conflict"
            flags += ["listed_aye", "listed_no"]
            counts["conflicting_votes"] += 1
        elif lobbies:
            lobby = next(iter(lobbies))
            if any(l == "no_vote" for _, l, _, _ in items):
                flags.append("also_no_vote")
                counts["also_no_vote"] += 1
        else:
            lobby = "no_vote"
        # Prefer a non-teller listing of the winning lobby: those carry proxy and sub-party details.
        primary = next((i for i in items if i[1] == lobby and not i[2]), None) or next((i for i in items if i[1] == lobby), items[0])
        f = primary[3]
        rows.append({
            "division_id": division_id, "mnis_id": mnis_id, "lobby": lobby, "member_name": f["name"],
            "party": f["party"], "party_abbreviation": f["abbr"], "sub_party": f["sub"], "member_from": f["from"],
            "is_teller": any(t for _, _, t, _ in items), "proxy_name": f["proxy"], "flags": flags,
        })  # fmt: skip
    return rows, counts


def _named(sources: list[tuple[str, str, bool, list | None]], *, tellers_count: bool) -> tuple[int, int]:
    """The members each lobby's list names, as that House's figures count them: the Commons figures leave the tellers
    out (they are listed separately and do not count themselves), the Lords' authoritative counts put them in. Each
    member once; a member named in both lobbies counts in each, as the tellers counted them; an entry with no member
    id is not a member anyone can count (it is reported as `vote_without_member_id`).
    """
    out = []
    for lobby in ("aye", "no"):
        voters, tellers = set(), set()
        for _, listed, teller, members in sources:
            if listed == lobby:
                (tellers if teller else voters).update(m for m in (_fields(x)["id"] for x in members or []) if m is not None)
        out.append(len(voters | tellers) if tellers_count else len(voters - tellers))
    return out[0], out[1]


def _finish(division: dict, votes: list[dict], counts: Counter, named: tuple[int, int]) -> Parsed:
    anomalies: list[str] = []
    division["ayes_named"], division["noes_named"] = named
    for side, official, n in (("aye", division["ayes"], named[0]), ("no", division["noes"], named[1])):
        if official is not None and official != n:
            anomalies.append(f"{side}_count_mismatch:official={official},named={n}")
    for key in ("conflicting_votes", "also_no_vote", "duplicate_listing", "vote_without_member_id"):
        if counts[key]:
            anomalies.append(f"{key}:{counts[key]}")
    division["ayes_listed"] = sum(1 for v in votes if v["lobby"] == "aye")
    division["noes_listed"] = sum(1 for v in votes if v["lobby"] == "no")
    division["no_vote_count"] = sum(1 for v in votes if v["lobby"] == "no_vote")
    division["tally_reconciled"] = (None if division["ayes"] is None or division["noes"] is None
                                    else named == (division["ayes"], division["noes"]))  # fmt: skip
    division["anomalies"] = anomalies
    return Parsed(division, votes)


def _result(ayes: int | None, noes: int | None) -> str | None:
    if ayes is None or noes is None:
        return None
    return "agreed" if ayes > noes else "negatived" if noes > ayes else "tied"


def _timestamp(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _base_division(house: str, source_id: int, **fields) -> dict[str, Any]:
    division = dict.fromkeys(DIVISION_COLUMNS)
    division.update(
        id=source_id + (LORDS_ID_OFFSET if house == "Lords" else 0), house=house, source_id=source_id,
        motion_type="division", is_deferred=False, debate_id=None, debate_link=None, **fields,
    )  # fmt: skip
    return division


def parse_commons(detail: dict) -> Parsed:
    held_at = datetime.fromisoformat(detail["Date"]) if detail.get("Date") else None
    title = (detail.get("Title") or "").strip() or None
    ayes, noes = detail.get("AyeCount"), detail.get("NoCount")
    division = _base_division(
        "Commons", detail["DivisionId"], number=detail.get("Number"), division_date=held_at.date() if held_at else None,
        held_at=held_at, title=title, motion_text=(detail.get("FriendlyDescription") or "").strip() or title,
        ayes=ayes, noes=noes, result=_result(ayes, noes),
        raw={k: v for k, v in detail.items() if k not in _COMMONS_LISTS},
    )  # fmt: skip
    division["is_deferred"] = bool(detail.get("IsDeferred"))
    # The Commons API sets these two fields only on a remote division (measured: 10 of W3's 14, 0 of W4's 71).
    division["remote_start"] = _timestamp(detail.get("RemoteVotingStart"))
    division["remote_end"] = _timestamp(detail.get("RemoteVotingEnd"))
    division["is_remote"] = division["remote_start"] is not None
    sources = [
        ("ayes", "aye", False, detail.get("Ayes")),
        ("noes", "no", False, detail.get("Noes")),
        ("no_vote", "no_vote", False, detail.get("NoVoteRecorded")),
        ("aye_tellers", "aye", True, detail.get("AyeTellers")),
        ("no_tellers", "no", True, detail.get("NoTellers")),
    ]  # fmt: skip
    votes, counts = _resolve_votes(division["id"], sources)
    parsed = _finish(division, votes, counts, _named(sources, tellers_count=False))
    _double_majority(parsed.division, detail)
    return parsed


def _double_majority(division: dict[str, Any], detail: dict) -> None:
    """English Votes for English Laws (2015-2021), as the Commons Votes API recorded it."""
    division["evel_type"] = (detail.get("EVELType") or "").strip() or None
    division["evel_country"] = (detail.get("EVELCountry") or "").strip() or None
    dm_ayes, dm_noes = detail.get("DoubleMajorityAyeCount"), detail.get("DoubleMajorityNoCount")
    division["double_majority_ayes"], division["double_majority_noes"] = dm_ayes, dm_noes
    if dm_ayes is None and dm_noes is None:
        return
    if dm_ayes is None or dm_noes is None:
        division["result"] = None
        division["anomalies"].append("double_majority_incomplete")
        return
    whole, subset = division["result"], _result(dm_ayes, dm_noes)
    if whole != subset:
        division["result"] = None
        division["anomalies"].append(f"double_majority_disagrees:whole={whole},subset={subset}")


def parse_lords(detail: dict) -> Parsed:
    day = datetime.fromisoformat(detail["date"]) if detail.get("date") else None
    # The Lords API gives a date only; remote voting start is the nearest thing to a time.
    held_at = datetime.fromisoformat(detail["remoteVotingStart"]) if detail.get("remoteVotingStart") else day
    title = (detail.get("title") or "").strip() or None
    ayes, noes = detail.get("authoritativeContentCount"), detail.get("authoritativeNotContentCount")
    division = _base_division(
        "Lords", detail["divisionId"], number=detail.get("number"), division_date=day.date() if day else None,
        held_at=held_at, title=title,
        motion_text=clean_text(detail.get("amendmentMotionNotes")) or clean_text(detail.get("notes")) or title,
        ayes=ayes, noes=noes, result=_result(ayes, noes), government_won=detail.get("isGovernmentWin"),
        raw={k: v for k, v in detail.items() if k not in _LORDS_LISTS},
    )  # fmt: skip
    # NOT `remoteVotingStart`: the Lords API sets that on every division (all 117 in W4), so it is a timestamp, not a
    # signal.
    division["remote_start"] = _timestamp(detail.get("remoteVotingStart"))
    division["remote_end"] = _timestamp(detail.get("remoteVotingEnd"))
    division["is_remote"] = bool(detail.get("divisionWasExclusivelyRemote"))
    sources = [
        ("contents", "aye", False, detail.get("contents")),
        ("not_contents", "no", False, detail.get("notContents")),
        ("content_tellers", "aye", True, detail.get("contentTellers")),
        ("not_content_tellers", "no", True, detail.get("notContentTellers")),
    ]  # fmt: skip
    votes, counts = _resolve_votes(division["id"], sources)
    return _finish(division, votes, counts, _named(sources, tellers_count=True))


# --------------------------------------------------------------------------- Linking a division to its Hansard debate
# ---------------------------------------------------------------------------


class DebateInfo(NamedTuple):
    id: str
    date: date
    house: str
    heading: str | None
    section: str | None
    times: tuple[datetime, ...] = ()  # timecodes of the debate's speeches, where present


class TallyEvent(NamedTuple):
    debate_id: str
    date: date
    house: str
    ayes: int
    noes: int


_TALLY = re.compile(r"Ayes\s+(\d+),?\s+Noes\s+(\d+)|Contents?\s+(\d+),?\s+Not[- ]Contents?\s+(\d+)", re.I)
DEFERRED_WINDOW = timedelta(days=60)
TIME_WINDOW = timedelta(minutes=30)
MIN_PREFIX_LENGTH = 12


def parse_tally(description: str | None) -> tuple[int, int] | None:
    m = _TALLY.search(description or "")
    if not m:
        return None
    a, n = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return int(a), int(n)


def normalise_title(value: str | None) -> str:
    s = re.sub(r"\[[^\]]*\]", " ", (value or "").casefold())  # [HL], [Lords]
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(?:(?:draft|the)\s+)+", "", s)


def title_match(title: str | None, heading: str | None) -> str | None:
    t, h = normalise_title(title), normalise_title(heading)
    if not t or not h:
        return None
    first_segment = normalise_title((title or "").split(":")[0])
    if h in (first_segment, t):
        return "title_exact"
    shorter, longer = sorted((t, h), key=len)
    if len(shorter) >= MIN_PREFIX_LENGTH and longer.startswith(shorter + " "):
        return "title_prefix"
    return None


def _distance(at: datetime, times: tuple[datetime, ...]) -> timedelta | None:
    """Time since the latest speech at or before `at`; None if no speech precedes it."""
    before = [t for t in times if t <= at]
    return at - max(before) if before else None


def link_debate(division: dict, debates: list[DebateInfo], tallies: list[TallyEvent]) -> tuple[str | None, str]:
    """(debate_id, method). Strongest signal first: a tally event in the Hansard text, then the title, then the time.
    """
    house, day, held_at = division["house"], division["division_date"], division.get("held_at")
    deferred = bool(division.get("is_deferred"))

    hits = {t.debate_id for t in tallies if (t.house, t.date, t.ayes, t.noes) == (house, day, division["ayes"], division["noes"])}
    if len(hits) == 1:
        return hits.pop(), "tally_event"

    same_house = [d for d in debates if d.house == house]
    matches = [(d, m) for d in same_house if (m := title_match(division.get("title"), d.heading))]
    pool = [(d, m) for d, m in matches if d.date == day]
    label = None
    if not pool and deferred:
        pool = [(d, m) for d, m in matches if timedelta(0) < day - d.date <= DEFERRED_WINDOW]
        label = "title_deferred"
    if pool:
        best = "title_exact" if any(m == "title_exact" for _, m in pool) else "title_prefix"
        pool = [(d, m) for d, m in pool if m == best]
        label = label or best
        if len(pool) > 1 and held_at and held_at.time() != datetime.min.time():
            timed = [(dist, d.id) for d, _ in pool if (dist := _distance(held_at, d.times)) is not None]
            if timed:
                return min(timed)[1], f"{label}+time"
        return min(d.id for d, _ in pool), label

    if not deferred and held_at and held_at.time() != datetime.min.time():
        timed = [(dist, d.id) for d in same_house if d.date == day and (dist := _distance(held_at, d.times)) is not None and dist <= TIME_WINDOW]
        if timed:
            return min(timed)[1], "time_window"
    return None, "none"


SQL_UNPLACED_DEFERRED = """
    SELECT id, house, division_date, title, ayes, noes, held_at, is_deferred, debate_link
      FROM division WHERE is_deferred AND debate_id IS NULL
"""
SQL_PLACE = """
    UPDATE division v SET debate_id = u.debate_id, debate_link = u.basis
      FROM unnest(%s::int[], %s::text[], %s::text[]) AS u(id, debate_id, basis)
     WHERE v.id = u.id AND (v.debate_id, v.debate_link) IS DISTINCT FROM (u.debate_id, u.basis)
"""


async def link_deferred(conn) -> dict[str, int]:
    """Place every deferred division that has no debate, by its instrument's committee debate."""
    from collections import Counter

    rows = await (await conn.execute(SQL_UNPLACED_DEFERRED)).fetchall()
    if not rows:
        return {"unplaced": 0, "placed": 0, "changed": 0}
    debates, tallies = await load_link_context(conn)
    ids, found_ids, bases, counts = [], [], [], Counter()
    for vid, house, day, title, ayes, noes, held_at, deferred, old_basis in rows:
        division = {"house": house, "division_date": day, "title": title, "ayes": ayes, "noes": noes,
                    "held_at": held_at, "is_deferred": deferred}
        found, basis = link_debate(division, debates, tallies)
        if found is None:
            continue  # keep what was recorded: the reason it is unplaced has not changed
        ids.append(vid)
        found_ids.append(found)
        bases.append(basis)
        counts[basis] += 1
    changed = 0
    if ids:
        async with conn.transaction():
            changed = (await conn.execute(SQL_PLACE, (ids, found_ids, bases))).rowcount
    return {"unplaced": len(rows), "placed": len(ids), "methods": dict(counts), "changed": changed}


# --------------------------------------------------------------------------- Fetching
# ---------------------------------------------------------------------------


async def _pages(client: PoliteClient, url: str, params_for, page_size: int, max_pages: int = 200) -> list[dict]:
    items: list[dict] = []
    for page in range(max_pages):
        batch = await client.get_json(url, params_for(page * page_size)) or []
        items.extend(batch)
        if len(batch) < page_size:
            break
    return items


async def fetch_commons(client: PoliteClient, start: date, end: date, page_size: int = 100) -> tuple[list[dict], list[int]]:
    """Every Commons division in the window, in full. Returns (details, ids that could not be fetched)."""
    summaries = await _pages(client, COMMONS_SEARCH, lambda skip: {
        "queryParameters.startDate": start.isoformat(), "queryParameters.endDate": end.isoformat(),
        "queryParameters.take": page_size, "queryParameters.skip": skip,
    }, page_size)  # fmt: skip
    ids = sorted({s["DivisionId"] for s in summaries})

    async def one(division_id: int) -> tuple[int, dict | None]:
        try:
            return division_id, await client.get_json(COMMONS_DETAIL.format(id=division_id))
        except FetchError:
            return division_id, None

    results = await asyncio.gather(*(one(i) for i in ids))
    return [d for _, d in results if d is not None], [i for i, d in results if d is None]


async def fetch_lords(client: PoliteClient, start: date, end: date, page_size: int = 100) -> list[dict]:
    """Every Lords division in the window; the search response already carries the voter lists."""
    return await _pages(client, LORDS_SEARCH, lambda skip: {
        "startDate": start.isoformat(), "endDate": end.isoformat(), "take": page_size, "skip": skip,
    }, page_size)  # fmt: skip


# --------------------------------------------------------------------------- Database
# ---------------------------------------------------------------------------

SQL_DEBATE_INFO = """
    SELECT d.id, d.date, d.house, d.heading, d.section,
           coalesce(array_agg(s.timecode ORDER BY s.timecode) FILTER (WHERE s.timecode IS NOT NULL), '{}')
    FROM debate d LEFT JOIN speech s ON s.debate_id = d.id GROUP BY d.id
"""
SQL_TALLY_EVENTS = """
    SELECT pe.debate_id, d.date, d.house, pe.description
    FROM procedure_event pe JOIN debate d ON d.id = pe.debate_id WHERE pe.event_type = 'division_result'
"""


def _placeholder(column: str) -> str:
    return f"%({column})s::motion_type" if column == "motion_type" else f"%({column})s"


SQL_UPSERT_DIVISION = f"""
    INSERT INTO division ({", ".join(DIVISION_COLUMNS)}, fetched_at)
    VALUES ({", ".join(_placeholder(c) for c in DIVISION_COLUMNS)}, now())
    ON CONFLICT (id) DO UPDATE SET {", ".join(f"{c} = EXCLUDED.{c}" for c in DIVISION_COLUMNS if c != "id")}, fetched_at = now()
"""
# (division_id, mnis_id) is the primary key. parse_* already collapses a member listed twice into one row, so a conflict
# here means "we have seen this vote before": update it if the API corrected it, skip it if nothing changed.
_VOTE_DATA = [c for c in VOTE_COLUMNS if c not in ("division_id", "mnis_id")]
SQL_UPSERT_VOTE = f"""
    INSERT INTO division_vote ({", ".join(VOTE_COLUMNS)}) VALUES ({", ".join(f"%({c})s" for c in VOTE_COLUMNS)})
    ON CONFLICT (division_id, mnis_id) DO UPDATE SET {", ".join(f"{c} = EXCLUDED.{c}" for c in _VOTE_DATA)}
    WHERE ({", ".join(f"division_vote.{c}" for c in _VOTE_DATA)}) IS DISTINCT FROM ({", ".join(f"EXCLUDED.{c}" for c in _VOTE_DATA)})
"""
SQL_DELETE_STALE_VOTES = "DELETE FROM division_vote WHERE division_id = %s AND mnis_id <> ALL(%s::int[])"


def _parse_timecode(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


async def load_link_context(conn) -> tuple[list[DebateInfo], list[TallyEvent]]:
    debates = [
        DebateInfo(i, d, house, heading, section, tuple(t for t in map(_parse_timecode, codes) if t))
        for i, d, house, heading, section, codes in await (await conn.execute(SQL_DEBATE_INFO)).fetchall()
    ]
    tallies = []
    for debate_id, day, house, description in await (await conn.execute(SQL_TALLY_EVENTS)).fetchall():
        if tally := parse_tally(description):
            tallies.append(TallyEvent(debate_id, day, house, *tally))
    return debates, tallies


async def store(conn, parsed: Parsed) -> None:
    """Upsert one division and reconcile its votes, atomically."""
    row = {**parsed.division, "raw": Jsonb(parsed.division["raw"])}
    async with conn.transaction():
        await conn.execute(SQL_UPSERT_DIVISION, row)
        async with conn.cursor() as cur:
            await cur.executemany(SQL_UPSERT_VOTE, parsed.votes)
        # An empty list means the API sent no member lists (e.g. a partial response), not that every stored vote was
        # withdrawn: "<> ALL('{}')" is true for every row and would delete them all.
        if parsed.votes:
            await conn.execute(SQL_DELETE_STALE_VOTES, (parsed.division["id"], [v["mnis_id"] for v in parsed.votes]))
