"""Bills, their stages and per-date sittings, and the two links that make them useful: sitting -> Hansard debate, and
division -> bill stage.
"""

import asyncio
import re
from datetime import date, datetime
from collections.abc import Sequence
from typing import Any, NamedTuple

from ingest.divisions import DebateInfo, normalise_title, title_match
from ingest.http import FetchError, PoliteClient

BASE = "https://bills-api.parliament.uk/api/v1"
WEAK_DEBATE_LINKS = frozenset({"time_window"})  # debate links that are a guess from the time of day


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None


def parse_bill(item: dict, detail: dict | None) -> dict[str, Any]:
    detail = detail or {}
    return {
        "id": item["billId"],
        # Stripped: "Finance Act 2020 " arrived with a trailing space from the source and was stored with it.
        "title": (item.get("shortTitle") or "").strip() or None,
        "long_title": detail.get("longTitle"),
        "former_short_title": (item.get("formerShortTitle") or "").strip() or None,
        "bill_type_id": item.get("billTypeId"),
        "originating_house": item.get("originatingHouse"),
        "current_house": item.get("currentHouse"),
        "introduced_session_id": item.get("introducedSessionId"),
        "is_act": bool(item.get("isAct")),
        "is_defeated": bool(item.get("isDefeated")),
        "is_withdrawn": item.get("billWithdrawn") is not None,
        "last_update": datetime.fromisoformat(item["lastUpdate"]) if item.get("lastUpdate") else None,
        "raw": {"list": item, "detail": detail or None},  # sponsors, agent, promoters live in detail
    }  # fmt: skip


def parse_stages(bill_id: int, payload: dict | None) -> tuple[list[dict], list[dict]]:
    """(stage rows, sitting rows). A stage with no sittings has no dates; house 'Unassigned' becomes NULL."""
    stages, sittings = [], []
    for st in (payload or {}).get("items") or []:
        # Commons, Lords, or unknown: the CHECK (migration 014) allows nothing else, and a value the Bills API adds
        # later ("Unassigned" today) must not abort a load — it is stored as not known.
        house = st.get("house") if st.get("house") in ("Commons", "Lords") else None
        dates = sorted(d for d in (_date(s.get("date")) for s in st.get("stageSittings") or []) if d)
        stages.append({
            "id": st["id"], "bill_id": bill_id, "stage_name": st.get("description"), "house": house,
            "date": dates[0] if dates else None, "debate_id": None, "stage_type_id": st.get("stageId"),
            "abbreviation": st.get("abbreviation"), "session_id": st.get("sessionId"), "sort_order": st.get("sortOrder"),
            "first_sitting": dates[0] if dates else None, "last_sitting": dates[-1] if dates else None,
        })  # fmt: skip
        for s in st.get("stageSittings") or []:
            if s.get("date"):
                sittings.append({
                    "id": s["id"], "bill_stage_id": st["id"], "bill_id": bill_id, "date": _date(s["date"]),
                    "house": house, "debate_id": None, "debate_link": None,
                })  # fmt: skip
    return stages, sittings


# The Royal Assent rename.
_ACT_TITLE = re.compile(r"^(?P<stem>.+?)\s+Act(?:\s+(?P<number>\(No\.?\s*\d+\)))?\s+(?P<year>\d{4})\s*$")


def bill_name_forms(bill: dict[str, Any]) -> tuple[str, ...]:
    """Every title this bill has been known by, stripped and deduplicated, current title first."""
    forms: list[str] = []

    def add(value: object) -> None:
        text = " ".join(str(value).split()) if value else ""
        if text and text not in forms:
            forms.append(text)

    add(bill.get("title"))
    add(bill.get("former_short_title"))
    for existing in list(forms):
        match = _ACT_TITLE.match(existing)
        if match:
            number = f" {match.group('number')}" if match.group("number") else ""
            add(f"{match.group('stem')}{number} Bill")
    return tuple(forms)


# Stages that are recorded as sittings but have no Hansard debate of their own: a first reading is the Clerk reading the
# title, Royal Assent is notified, a reintroduction or carry-over is a formal motion.
FORMAL_STAGES = frozenset({"1st reading", "first reading", "royal assent", "bill reintroduced",
                           "order of commitment discharged"})  # fmt: skip


def is_formal_stage(stage_name: str | None) -> bool:
    return " ".join((stage_name or "").split()).casefold() in FORMAL_STAGES


def bill_relevant(sittings: list[dict], start: date, end: date) -> bool:
    return any(s.get("date") and start <= s["date"] <= end for s in sittings)


def link_sitting(bill_titles: str | Sequence[str] | None, house: str | None, day: date,
                 debates: list[DebateInfo]) -> tuple[str | None, str]:  # fmt: skip
    """The Hansard debate for a bill sitting: same day, same House, heading = ANY name the bill has had."""
    titles = (bill_titles,) if isinstance(bill_titles, str) or bill_titles is None else tuple(bill_titles)

    def best_match(heading: str | None) -> str | None:
        found = [m for t in titles if (m := title_match(t, heading))]
        return "title_exact" if "title_exact" in found else (found[0] if found else None)

    same_day = [d for d in debates if d.date == day and (house is None or d.house == house)]
    hits = [(d, m) for d in same_day if (m := best_match(d.heading))]
    suffix = ""
    if not hits:
        # A "(No. 2)" bill renamed at Royal Assent loses its number: "Finance (No. 2) Bill" became "Finance Act 2026"
        # and the API's former title is empty.
        plain = {_unnumbered(normalise_title(t)) for t in titles if t}
        plain.discard("")
        hits = [(d, "title_exact") for d in same_day if _unnumbered(normalise_title(d.heading)) in plain]
        suffix = "+numbered"
    if not hits:
        return None, "none"
    label = "title_exact" if any(m == "title_exact" for _, m in hits) else "title_prefix"
    best = min(d.id for d, _ in hits)
    return best, label + suffix + ("+multi" if len(hits) > 1 else "")


_KEYWORDS = (
    (re.compile(r"second reading", re.I), {"2R"}),
    (re.compile(r"third reading", re.I), {"3R"}),
    (re.compile(r"report stage", re.I), {"RS"}),
    (re.compile(r"committee (?:stage|of the whole house)", re.I), {"CS", "COTWH"}),
    (re.compile(r"lords amendments?|consideration of lords", re.I), {"COLA"}),
    (re.compile(r"programme", re.I), {"PM"}),
    (re.compile(r"money resolution", re.I), {"MR"}),
    (re.compile(r"ways and means", re.I), {"WAMR"}),
    (re.compile(r"carry-over", re.I), {"CM"}),
)  # fmt: skip


def stage_keywords(title: str | None) -> set[str]:
    """Stage abbreviations a division title names. Only the part after the first colon counts, so a bill called "Second
    Reading Reform Bill" does not read as a second reading.
    """
    if not title or ":" not in title:
        return set()
    rest = title.split(":", 1)[1]
    found: set[str] = set()
    for pattern, abbreviations in _KEYWORDS:
        if pattern.search(rest):
            found |= abbreviations
    return found


class StageRef(NamedTuple):
    stage_id: int
    bill_id: int
    bill_title: str
    abbreviation: str | None
    house: str | None
    dates: frozenset[date]  # every sitting date
    debate_ids: frozenset[str]  # debates its sittings were linked to
    # Every name the bill has had, normalised.
    bill_titles: frozenset[str] = frozenset()
    # The stage's name, so a formal stage (a first reading, Royal Assent) can be excluded: no division is ever taken on
    # one, and including it made a bill-only Lords title ambiguous whenever a first reading shared the day.
    stage_name: str = ""

    def names(self) -> frozenset[str]:
        if self.bill_titles:
            return self.bill_titles
        return frozenset(normalise_title(t) for t in bill_name_forms({"title": self.bill_title}))


# The stage a division title names, read from the text AFTER the bill's name (so "Second Reading Reform Bill" is not a
# second reading) and only from the FIRST phrase there: that is the stage designation, and what follows is the question
# — "Report Stage: New Clause 5 (Committee on Standards)" is report stage, not committee.
_REMAINING_STAGES = frozenset({"CS", "COTWH", "RS", "3R"})  # "remaining stages" is an umbrella: title_stage_phrase
_STAGE_PHRASES: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (re.compile(r"\bsecond reading\b"), frozenset({"2R"})),
    (re.compile(r"\bthird reading\b"), frozenset({"3R"})),
    (re.compile(r"\breport stage\b"), frozenset({"RS"})),
    (re.compile(r"\bremaining stages\b"), _REMAINING_STAGES),
    (re.compile(r"\bcommittee\b"), frozenset({"CS", "COTWH"})),
    (re.compile(r"\blords? (?:amendments?|reasons?|messages?)\b|\bconsideration of lords\b|\binsist\b|\bin lieu\b"),
     frozenset({"COLA", "COLM"})),
    (re.compile(r"\breasoned amendment\b"), frozenset({"2R", "3R"})),  # moved at either reading, nowhere else
    (re.compile(r"\bprogramme\b"), frozenset({"PM"})),
    (re.compile(r"\bmoney resolution\b"), frozenset({"MR"})),
    (re.compile(r"\bways and means\b"), frozenset({"WAMR"})),
    (re.compile(r"\bcarry over\b"), frozenset({"CM"})),
    (re.compile(r"\ballocation of time\b"), frozenset({"AOTM"})),
    (re.compile(r"\binstruction\b"), frozenset({"I"})),
)
# A question on an amendment, a clause or a schedule is only ever put at a stage that considers them.
_AMENDING_WORDS = re.compile(r"\b(?:new clauses?|new schedules?|amendments?|clauses?|schedules?)\b")
AMENDING_STAGES = frozenset({"CS", "COTWH", "RS", "COLA", "COLM", "SCS", "LGC", "COCAA/OR"})
_NUMBER = re.compile(r"\bno \d+\b")


def _unnumbered(normalised: str) -> str:
    """A normalised name without its "(No. N)": "finance no 2 bill" -> "finance bill"."""
    return " ".join(_NUMBER.sub(" ", normalised).split())


def _word_set(normalised: str) -> frozenset[str]:
    """The words of a name, order- and plural-insensitive, for Hansard's own typos in division titles: "Pensions Scheme
    Bill", "Children's School and Wellbeing Bill", "Post Office (Horizon Systems) Offences Bill".
    """
    words = set()
    for word in normalised.split():
        if len(word) <= 1 or word.isdigit() or word in ("bill", "no"):
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(word)
    return frozenset(words)


def title_stage_phrase(remainder: str) -> frozenset[str]:
    """The stages the FIRST stage phrase in a (normalised) remainder names; empty when it names none."""
    found = sorted((m.start(), stages) for pattern, stages in _STAGE_PHRASES
                   for m in [pattern.search(remainder)] if m)  # fmt: skip
    if not found:
        return frozenset()
    first = found[0][1]
    if first == _REMAINING_STAGES and len(found) > 1 and found[1][1] <= first:
        return found[1][1]
    return first


class StageIndex:
    """The stage refs, indexed once for a whole linking pass: by (House, sitting day) without formal stages, and by
    (House, debate). A scan of every stage for every division is 600 x 2,300 today and 6,000 x 25,000 at ten years.
    """

    def __init__(self, stages: list[StageRef]):
        from collections import defaultdict

        self.by_day: dict[tuple[str | None, date], list[StageRef]] = defaultdict(list)
        self.by_debate: dict[tuple[str | None, str], list[StageRef]] = defaultdict(list)
        for ref in stages:
            for debate_id in ref.debate_ids:
                self.by_debate[(ref.house, debate_id)].append(ref)
            if is_formal_stage(ref.stage_name):  # nobody divides on a first reading
                continue
            for day in ref.dates:
                self.by_day[(ref.house, day)].append(ref)

    def on_the_day(self, house: str | None, day: date) -> list[StageRef]:
        return self.by_day.get((house, day), [])

    def via_debate(self, house: str | None, debate_id: str | None) -> list[StageRef]:
        return self.by_debate.get((house, debate_id), []) if debate_id else []


def _split_title(title: str | None) -> tuple[str, str]:
    """(the part before the first colon, the rest), each normalised."""
    head, _, tail = (title or "").partition(":")
    return normalise_title(head), normalise_title(tail)


def _bill_boundary(rest: str) -> bool:
    """What may follow a bill's name before the colon: nothing, or the stage. "Armed Forces Bill Committee" is the
    Armed Forces Bill; "Health Bill Reform" is not the Health Bill.
    """
    return not rest or any(pattern.match(rest) for pattern, _ in _STAGE_PHRASES)


def _prefix_hits(title: str | None, candidates: list[StageRef], transform=None) -> list[tuple[StageRef, str]]:
    """(stage, remainder) for every candidate one of whose names is the title's bill part; the longest name wins."""
    head, tail = _split_title(title)
    if transform:
        head = transform(head)
    hits: list[tuple[StageRef, str, int]] = []
    for ref in candidates:
        best: tuple[str, int] | None = None
        for name in ref.names():
            name = transform(name) if transform else name
            if not name:
                continue
            if head == name:
                rest = ""
            elif head.startswith(name + " "):
                rest = head[len(name) + 1:]
            else:
                continue
            if _bill_boundary(rest) and (best is None or len(name) > best[1]):
                best = (rest, len(name))
        if best:
            hits.append((ref, " ".join(f"{best[0]} {tail}".split()), best[1]))
    if not hits:
        return []
    longest = max(h[2] for h in hits)
    return [(ref, rest) for ref, rest, length in hits if length == longest]


def _fuzzy_hits(title: str | None, candidates: list[StageRef]) -> list[tuple[StageRef, str]]:
    """Same words as a name, in any order, singular or plural. The bill part is everything up to the LAST "bill" before
    the colon ("Passenger Railway Services Bill (Public Ownership) Bill: Second Reading" has two), and what follows
    it there must still be nothing or the stage.
    """
    head, tail = _split_title(title)
    words = head.split()
    if "bill" not in words:
        return []
    cut = len(words) - 1 - words[::-1].index("bill")
    rest = " ".join(words[cut + 1:])
    wanted = _word_set(" ".join(words[: cut + 1]))
    if not wanted or not _bill_boundary(rest):
        return []
    remainder = " ".join(f"{rest} {tail}".split())
    return [(ref, remainder) for ref in candidates if any(_word_set(n) == wanted for n in ref.names())]


def _choose(hits: list[tuple[StageRef, str]]) -> tuple[int | None, str]:
    """Pick the stage among the title's hits: the stage phrase decides, then the amendment rule, then the date alone.
    """
    remainder = hits[0][1]  # the same bill part for every hit, so the same remainder
    stages = [ref for ref, _ in hits]
    phrase = title_stage_phrase(remainder)
    amending = bool(_AMENDING_WORDS.search(remainder))
    if phrase:
        stages = [ref for ref in stages if ref.abbreviation in phrase]
        if len(stages) > 1 and amending:
            stages = [ref for ref in stages if ref.abbreviation in AMENDING_STAGES]
        method = "title_stage"
    elif amending:
        stages = [ref for ref in stages if ref.abbreviation in AMENDING_STAGES]
        method = "title_amendment"
    else:
        method = "title_date"
    if len(stages) == 1:
        return stages[0].stage_id, method
    return (None, "ambiguous") if stages else (None, "none")


def link_division_stage(division: dict, stages: list[StageRef], *,
                        index: StageIndex | None = None) -> tuple[int | None, str]:  # fmt: skip
    """(bill_stage_id, method). The bill named at the start of the title, sitting that day in that House, at the stage
    the title names; else the division's own debate.
    """
    if division.get("is_deferred"):  # deferred divisions are statutory instruments, not bill stages
        return None, "not_a_bill"
    index = index or StageIndex(stages)
    house, day, title = division["house"], division["division_date"], division.get("title")
    title_norm = normalise_title(title)
    candidates = index.on_the_day(house, day)
    if title_norm and candidates:
        tiers = (("", lambda: _prefix_hits(title, candidates)),
                 ("+numbered", lambda: _prefix_hits(title, candidates, _unnumbered)),
                 ("+fuzzy", lambda: _fuzzy_hits(title, candidates)))
        for suffix, find in tiers:
            hits = find()
            if hits:
                stage_id, method = _choose(hits)
                return stage_id, method + suffix if stage_id is not None else method

    via_debate = index.via_debate(house, division.get("debate_id"))
    if len(via_debate) == 1:
        # time -> debate -> stage is two inferences deep: keep the doubt visible in the label
        weak = division.get("debate_link") in WEAK_DEBATE_LINKS
        return via_debate[0].stage_id, "debate_weak" if weak else "debate"
    if via_debate:
        return None, "ambiguous"
    return None, "none" if "bill" in title_norm.split() else "not_a_bill"


def bill_mentioned(title: str | None, headings: set[str]) -> bool:
    """A bill with no sitting in the window still matters if a debate heading or division title names it."""
    return normalise_title(title) in headings


# --------------------------------------------------------------------------- Fetching
# ---------------------------------------------------------------------------


async def fetch_recent_bills(client: PoliteClient, since: date, page_size: int = 50) -> list[dict]:
    """List items updated on or after `since`, newest first, stopping once a page crosses the cutoff."""
    cutoff = since.isoformat()
    items: list[dict] = []
    skip = 0
    while True:
        payload = await client.get_json(f"{BASE}/Bills", {"SortOrder": "DateUpdatedDescending", "Take": page_size, "Skip": skip})
        batch = (payload or {}).get("items") or []
        if not batch:
            break
        items.extend(b for b in batch if (b.get("lastUpdate") or "")[:10] >= cutoff)
        if any((b.get("lastUpdate") or "")[:10] < cutoff for b in batch) or len(batch) < page_size:
            break
        skip += page_size
    return items


async def fetch_bill(client: PoliteClient, bill_id: int) -> tuple[dict | None, dict | None]:
    """(detail, stages), or (None, None) when the bill is not found."""
    detail = await client.get_json(f"{BASE}/Bills/{bill_id}")
    if detail is None:
        return None, None
    return detail, await client.get_json(f"{BASE}/Bills/{bill_id}/Stages")


async def fetch_bills(client: PoliteClient, items: list[dict]) -> tuple[list[tuple[dict, dict | None, dict | None]], list[int]]:
    """Detail and stages for every listed bill. Returns ((item, detail, stages) rows, ids that failed)."""

    async def one(item: dict):
        try:
            detail, stages = await fetch_bill(client, item["billId"])
            return item, detail, stages, None
        except FetchError:
            return item, None, None, item["billId"]

    results = await asyncio.gather(*(one(i) for i in items))
    return [(i, d, s) for i, d, s, failed in results if failed is None and d is not None], [f for *_, f in results if f is not None]


# --------------------------------------------------------------------------- Database
# ---------------------------------------------------------------------------

BILL_COLUMNS = ("id", "title", "long_title", "former_short_title", "bill_type_id", "originating_house", "current_house",
                "introduced_session_id", "is_act", "is_defeated", "is_withdrawn", "last_update", "raw")  # fmt: skip
STAGE_COLUMNS = ("id", "bill_id", "stage_name", "house", "date", "debate_id", "stage_type_id", "abbreviation", "session_id",
                 "sort_order", "first_sitting", "last_sitting")  # fmt: skip
SITTING_COLUMNS = ("id", "bill_stage_id", "bill_id", "date", "house", "debate_id", "debate_link")


def _upsert(table: str, columns: tuple[str, ...], stamp: bool = False) -> str:
    stamp_col, stamp_val = (", fetched_at", ", now()") if stamp else ("", "")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != "id") + (", fetched_at = now()" if stamp else "")
    return (f"INSERT INTO {table} ({', '.join(columns)}{stamp_col}) VALUES ({', '.join(f'%({c})s' for c in columns)}{stamp_val}) "
            f"ON CONFLICT (id) DO UPDATE SET {updates}")  # fmt: skip


SQL_UPSERT_BILL = _upsert("bill", BILL_COLUMNS, stamp=True)
SQL_UPSERT_STAGE = _upsert("bill_stage", STAGE_COLUMNS)
SQL_UPSERT_SITTING = _upsert("bill_stage_sitting", SITTING_COLUMNS)
# A re-fetched bill replaces its stage/sitting list: drop rows the API no longer lists.
SQL_DELETE_STALE_STAGES = "DELETE FROM bill_stage WHERE bill_id = %s AND id <> ALL(%s::int[])"
SQL_DELETE_STALE_SITTINGS = "DELETE FROM bill_stage_sitting WHERE bill_id = %s AND id <> ALL(%s::int[])"
SQL_STAGE_REFS = """
    SELECT st.id, st.bill_id, b.title, st.abbreviation, st.house,
           coalesce(array_agg(DISTINCT t.date) FILTER (WHERE t.date IS NOT NULL), '{}'),
           coalesce(array_agg(DISTINCT t.debate_id) FILTER (WHERE t.debate_id IS NOT NULL), '{}'),
           b.former_short_title, st.stage_name
    FROM bill_stage st JOIN bill b ON b.id = st.bill_id LEFT JOIN bill_stage_sitting t ON t.bill_stage_id = st.id
    GROUP BY st.id, b.title, b.former_short_title, st.stage_name
"""
SQL_BILL_NAMES = "SELECT id, title, former_short_title FROM bill"
SQL_SITTINGS_IN_CORPUS = """
    SELECT t.id, t.bill_id, t.date, t.house, t.debate_id, t.debate_link, st.stage_name FROM bill_stage_sitting t
      JOIN bill_stage st ON st.id = t.bill_stage_id
     WHERE EXISTS (SELECT 1 FROM corpus_window w WHERE t.date BETWEEN w.start_date AND w.end_date)
"""
SQL_ALL_DEBATES = "SELECT id, date, house, heading, section FROM debate"
SQL_LINK_SITTINGS = """
    UPDATE bill_stage_sitting t SET debate_id = u.debate_id, debate_link = u.basis
    FROM unnest(%s::int[], %s::text[], %s::text[]) AS u(id, debate_id, basis)
    WHERE t.id = u.id AND (t.debate_id, t.debate_link) IS DISTINCT FROM (u.debate_id, u.basis)
"""
SQL_DIVISIONS = "SELECT id, house, division_date, title, debate_id, is_deferred, debate_link FROM division"
SQL_LINK_DIVISIONS = """
    UPDATE division d SET bill_stage_id = u.stage_id, bill_link = u.method
    FROM unnest(%s::int[], %s::int[], %s::text[]) AS u(id, stage_id, method)
    WHERE d.id = u.id AND (d.bill_stage_id, d.bill_link) IS DISTINCT FROM (u.stage_id, u.method)
"""
# A stage's debate is DERIVED — its earliest linked sitting's — so it is set AND cleared here.
SQL_STAGE_DEBATES = """
    UPDATE bill_stage st SET debate_id = u.debate_id
    FROM (SELECT s.id, (SELECT t.debate_id FROM bill_stage_sitting t
                         WHERE t.bill_stage_id = s.id AND t.debate_id IS NOT NULL
                         ORDER BY t.date, t.debate_id LIMIT 1) AS debate_id
            FROM bill_stage s) u
    WHERE st.id = u.id AND st.debate_id IS DISTINCT FROM u.debate_id
"""


async def store_bill(conn, bill: dict, stages: list[dict], sittings: list[dict]) -> None:
    from psycopg.types.json import Jsonb

    async with conn.transaction():
        await conn.execute(SQL_UPSERT_BILL, {**bill, "raw": Jsonb(bill["raw"], dumps=_dumps)})
        async with conn.cursor() as cur:
            await cur.executemany(SQL_UPSERT_STAGE, stages)
            await cur.executemany(SQL_UPSERT_SITTING, sittings)
        await conn.execute(SQL_DELETE_STALE_SITTINGS, (bill["id"], [s["id"] for s in sittings]))
        await conn.execute(SQL_DELETE_STALE_STAGES, (bill["id"], [s["id"] for s in stages]))


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, default=str, ensure_ascii=False)


async def load_stage_refs(conn) -> list[StageRef]:
    rows = await (await conn.execute(SQL_STAGE_REFS)).fetchall()
    return [StageRef(i, b, title or "", abbr, house, frozenset(dates), frozenset(debates),
                     bill_titles=frozenset(normalise_title(t) for t in
                                           bill_name_forms({"title": title, "former_short_title": former})),
                     stage_name=stage_name or "")
            for i, b, title, abbr, house, dates, debates, former, stage_name in rows]  # fmt: skip


async def link_all(conn) -> dict[str, Any]:
    """Every bill link, recomputed from what is in the database. Offline, idempotent, and safe to run at any time.
    """
    from collections import Counter

    names = {bill_id: bill_name_forms({"title": title, "former_short_title": former})
             for bill_id, title, former in await (await conn.execute(SQL_BILL_NAMES)).fetchall()}  # fmt: skip
    debates = [DebateInfo(i, d, house, heading, section)
               for i, d, house, heading, section in await (await conn.execute(SQL_ALL_DEBATES)).fetchall()]
    by_day: dict[tuple[date, str | None], list[DebateInfo]] = {}
    for d in debates:
        by_day.setdefault((d.date, d.house), []).append(d)

    sittings = await (await conn.execute(SQL_SITTINGS_IN_CORPUS)).fetchall()
    ids, debate_ids, bases, sitting_counts = [], [], [], Counter()
    for sitting_id, bill_id, day, house, _old_debate, _old_basis, stage_name in sittings:
        if is_formal_stage(stage_name):
            # Never matched by title: a first reading or Royal Assent is not a contribution to whatever debate on the
            # same bill happens to share its day, and linking it there was a misattribution the tests caught.
            found, basis = None, "formal_no_debate"
        else:
            pool = (by_day.get((day, house), []) if house
                    else [d for (dd, _h), ds in by_day.items() if dd == day for d in ds])  # fmt: skip
            found, basis = link_sitting(names.get(bill_id, ()), house, day, pool)
        ids.append(sitting_id)
        debate_ids.append(found)
        bases.append(basis)
        sitting_counts["linked" if found else ("formal_no_debate" if basis == "formal_no_debate" else "unlinked")] += 1
    async with conn.transaction():
        changed_sittings = (await conn.execute(SQL_LINK_SITTINGS, (ids, debate_ids, bases))).rowcount
        changed_stages = (await conn.execute(SQL_STAGE_DEBATES)).rowcount

    refs = await load_stage_refs(conn)
    index = StageIndex(refs)
    rows = await (await conn.execute(SQL_DIVISIONS)).fetchall()
    columns = ("id", "house", "division_date", "title", "debate_id", "is_deferred", "debate_link")
    div_ids, stage_ids, methods, methods_count = [], [], [], Counter()
    for row in rows:
        division = dict(zip(columns, row, strict=True))
        stage_id, method = link_division_stage(division, refs, index=index)
        div_ids.append(division["id"])
        stage_ids.append(stage_id)
        methods.append(method)
        methods_count[method] += 1
    async with conn.transaction():
        changed_divisions = (await conn.execute(SQL_LINK_DIVISIONS, (div_ids, stage_ids, methods))).rowcount

    return {"sittings_in_corpus": len(sittings), "sittings": dict(sitting_counts),
            "sittings_changed": changed_sittings, "stages_debate_set": changed_stages,
            "divisions": len(rows), "division_methods": dict(methods_count),
            "divisions_changed": changed_divisions}  # fmt: skip
