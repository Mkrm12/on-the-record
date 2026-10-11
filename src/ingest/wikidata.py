"""Member dates of birth (and other identity facts) from Wikidata, matched on the MNIS id."""

import calendar
import re
from collections.abc import Iterable
from datetime import date
from typing import Any, NamedTuple

SPARQL_URL = "https://query.wikidata.org/sparql"
SOURCE = "wikidata P10428"
MIN_BIRTH_YEAR = 1880
MIN_AGE = 18  # nobody sits in Parliament younger than this
MAX_AGE = 105  # every member here is serving; a birth that makes them older is a wrong or mismatched Wikidata entry

_TIME = re.compile(r"^([+-]?)(\d+)-(\d{2})-(\d{2})T")
_PRECISION = {9: "year", 10: "month"}  # 11 and above (day, hour,...) is "day"; below 9 is too coarse to use
_ORDER = {"day": 3, "month": 2, "year": 1}


def parse_time(value: str | None) -> date | None:
    """A Wikidata timestamp as a date. Month or day 00 (imprecise) becomes 1; BCE, year 0 and invalid dates are None.
    """
    m = _TIME.match(value or "")
    if not m or m.group(1) == "-":
        return None
    year, month, day = int(m.group(2)), int(m.group(3)) or 1, int(m.group(4)) or 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def dob_bounds(value: date, precision: str) -> tuple[date, date]:
    """The earliest and latest dates a value of this precision can stand for."""
    if precision == "year":
        return date(value.year, 1, 1), date(value.year, 12, 31)
    if precision == "month":
        return date(value.year, value.month, 1), date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])
    return value, value


class DobCandidate(NamedTuple):
    value: date
    precision: str  # day | month | year | coarse
    rank: str  # PreferredRank | NormalRank | DeprecatedRank


def parse_dob_statements(field: str | None) -> list[DobCandidate]:
    """Parse "value~precision~rank|value~precision~rank" as produced by the query's GROUP_CONCAT."""
    out = []
    for piece in (field or "").split("|"):
        parts = piece.split("~")
        if len(parts) != 3:
            continue
        value = parse_time(parts[0])
        try:
            code = int(parts[1])
        except ValueError:
            continue
        if value is None:
            continue
        precision = "day" if code >= 11 else _PRECISION.get(code, "coarse")
        out.append(DobCandidate(value, precision, parts[2]))
    return out


class Dob(NamedTuple):
    date_of_birth: date | None  # only for day precision
    precision: str | None
    earliest: date | None
    latest: date | None
    year: int | None
    note: str | None


_NONE = Dob(None, None, None, None, None, None)


def _years_before(today: date, years: int) -> date:
    try:
        return today.replace(year=today.year - years)
    except ValueError:  # 29 February
        return today.replace(year=today.year - years, day=28)


def choose_dob(candidates: list[DobCandidate], today: date) -> Dob:
    note = None
    usable = []
    for c in candidates:
        if c.rank == "DeprecatedRank":
            continue
        if c.precision == "coarse":
            note = note or "coarse_dob_ignored"
            continue
        earliest, latest = dob_bounds(c.value, c.precision)
        # Rejected only when NO date the value could stand for is plausible: a year straddling a boundary stays.
        if earliest.year < MIN_BIRTH_YEAR or earliest > _years_before(today, MIN_AGE) or latest < _years_before(today, MAX_AGE):
            note = note or "implausible_dob"
            continue
        usable.append(c)
    if not usable:
        return _NONE._replace(note=note)

    pool = [c for c in usable if c.rank == "PreferredRank"] or usable
    best = max(_ORDER[c.precision] for c in pool)
    top = {(c.value, c.precision) for c in pool if _ORDER[c.precision] == best}
    if len(top) > 1:
        return _NONE._replace(note="conflicting_dob")  # never guess between two full dates
    (value, precision), = top
    earliest, latest = dob_bounds(value, precision)
    for c in usable:
        lo, hi = dob_bounds(c.value, c.precision)
        if hi < earliest or lo > latest:
            note = note or "inconsistent_candidates"
    return Dob(value if precision == "day" else None, precision, earliest, latest, value.year, note)


class CoreResult(NamedTuple):
    rows: dict[int, dict[str, Any]]
    problems: list[str]


def _val(binding: dict, key: str) -> str | None:
    return (binding.get(key) or {}).get("value") or None


def parse_core(bindings: Iterable[dict], today: date | None = None) -> CoreResult:
    """SPARQL result rows -> one member_id_map update per MNIS id. Anything that cannot be trusted is left out and
    named in `problems` rather than guessed.
    """
    today = today or date.today()
    problems: list[str] = []
    grouped: dict[int, list[dict]] = {}
    for b in bindings:
        mnis = _val(b, "mnis") or ""
        if not mnis.isdigit():
            problems.append(f"non_numeric_mnis:{mnis}")
            continue
        grouped.setdefault(int(mnis), []).append(b)

    rows: dict[int, dict[str, Any]] = {}
    for mnis, items in grouped.items():
        if len(items) > 1:
            problems.append(f"duplicate_mnis:{mnis}")
            continue
        b = items[0]
        candidates = parse_dob_statements(_val(b, "dobs"))
        dob = choose_dob(candidates, today)
        deaths = [(parse_time(p.split("~")[0]), p.split("~")[1] if "~" in p else "") for p in (_val(b, "dods") or "").split("|") if p]
        death_days = {d for d, prec in deaths if d and prec.isdigit() and int(prec) >= 11}
        death_years = {d.year for d, prec in deaths if d and prec.isdigit() and int(prec) >= 9}
        twfy = _val(b, "twfy_id")
        twfy = twfy if twfy and twfy.isdigit() else None
        wikidata: dict[str, Any] = {
            "qid": (_val(b, "item") or "").rsplit("/", 1)[-1] or None,
            "source": SOURCE,
            "dob": {
                "candidates": [{"value": c.value.isoformat(), "precision": c.precision, "rank": c.rank} for c in candidates],
                "note": dob.note,
            },
        }
        if death_years:
            wikidata["death_year"] = min(death_years)
        rows[mnis] = {
            "wikidata_id": wikidata["qid"],
            "date_of_birth": dob.date_of_birth, "dob_precision": dob.precision, "dob_earliest": dob.earliest,
            "dob_latest": dob.latest, "birth_year": dob.year,
            "date_of_death": next(iter(death_days)) if len(death_days) == 1 else None,
            "twfy_id": twfy, "person_id": f"uk.org.publicwhip/person/{twfy}" if twfy else None,
            "democracy_club_id": _val(b, "dc_id"), "wikidata": wikidata,
        }  # fmt: skip

    by_twfy: dict[str, list[int]] = {}
    for mnis, r in rows.items():
        if r["twfy_id"]:
            by_twfy.setdefault(r["twfy_id"], []).append(mnis)
    for twfy, owners in by_twfy.items():
        if len(owners) > 1:  # person_id is UNIQUE: withhold it from both rather than pick one
            problems.append(f"duplicate_twfy:{twfy}")
            for mnis in owners:
                rows[mnis]["person_id"] = None
    return CoreResult(rows, problems)


def parse_extras(bindings: Iterable[dict]) -> dict[int, dict[str, str | None]]:
    out = {}
    for b in bindings:
        mnis = _val(b, "mnis")
        if mnis and mnis.isdigit():
            out[int(mnis)] = {"place_of_birth": _val(b, "pob"), "wikipedia_url": _val(b, "wikipedia")}
    return out


CORE_COLUMNS = (
    ("wikidata_id", "text"), ("date_of_birth", "date"), ("dob_precision", "text"), ("dob_earliest", "date"),
    ("dob_latest", "date"), ("birth_year", "int"), ("date_of_death", "date"), ("twfy_id", "text"),
    ("democracy_club_id", "text"), ("wikidata", "jsonb"),
)  # fmt: skip
EXTRA_COLUMNS = (("place_of_birth", "text"), ("wikipedia_url", "text"))


def build_update(columns: tuple[tuple[str, str], ...]) -> str:
    """UPDATE member_id_map FROM unnest(arrays): one round trip for every member. The arrays are, in order: mnis_id,
    one per column, then person_id.
    """
    arrays = ", ".join(["%s::int[]"] + [f"%s::{'text' if t == 'jsonb' else t}[]" for _, t in columns] + ["%s::text[]"])
    names = ", ".join(["mnis_id"] + [c for c, _ in columns] + ["person_id"])
    sets = ", ".join(f"{c} = u.{c}" + ("::jsonb" if t == "jsonb" else "") for c, t in columns)
    return (f"UPDATE member_id_map m SET {sets}, person_id = u.person_id, wikidata_fetched_at = now() "
            f"FROM unnest({arrays}) AS u({names}) WHERE m.mnis_id = u.mnis_id")  # fmt: skip


def _values(mnis_ids: Iterable[int]) -> str:
    ids = sorted(set(mnis_ids))
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise ValueError("MNIS ids must be integers")
    return " ".join(f'"{i}"' for i in ids)


def core_query(mnis_ids: Iterable[int]) -> str:
    """ONE query for everything that is cheap: birth statements with precision and rank, death, TheyWorkForYou and
    Democracy Club ids. Restricted to our ids: over all ~6,000 P10428 items the join approaches Wikidata's 60 second
    limit; over ours it takes about 3 seconds.
    """
    return f"""
SELECT ?item ?mnis
  (GROUP_CONCAT(DISTINCT CONCAT(STR(?dobv), "~", STR(?dobp), "~", REPLACE(STR(?rank), ".*#", "")); separator="|") AS ?dobs)
  (GROUP_CONCAT(DISTINCT CONCAT(STR(?dodv), "~", STR(?dodp)); separator="|") AS ?dods)
  (SAMPLE(?twfy) AS ?twfy_id) (SAMPLE(?dc) AS ?dc_id)
WHERE {{
  VALUES ?mnis {{ {_values(mnis_ids)} }}
  ?item wdt:P10428 ?mnis .
  OPTIONAL {{ ?item p:P569 ?ds . ?ds wikibase:rank ?rank ; psv:P569 [ wikibase:timeValue ?dobv ; wikibase:timePrecision ?dobp ] . }}
  OPTIONAL {{ ?item p:P570 ?dd . ?dd psv:P570 [ wikibase:timeValue ?dodv ; wikibase:timePrecision ?dodp ] . }}
  OPTIONAL {{ ?item wdt:P2171 ?twfy . }}
  OPTIONAL {{ ?item wdt:P6465 ?dc . }}
}}
GROUP BY ?item ?mnis
"""


def extras_query(mnis_ids: Iterable[int]) -> str:
    """Place of birth and the English Wikipedia article. Separate because the label and sitelink joins make it about
    ten times slower (~27 s) than the core query.
    """
    return f"""
SELECT ?mnis (SAMPLE(?pobl) AS ?pob) (SAMPLE(?wp) AS ?wikipedia)
WHERE {{
  VALUES ?mnis {{ {_values(mnis_ids)} }}
  ?item wdt:P10428 ?mnis .
  OPTIONAL {{ ?item wdt:P19 ?pobi . ?pobi rdfs:label ?pobl FILTER(lang(?pobl) = "en") }}
  OPTIONAL {{ ?wp schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}
}}
GROUP BY ?mnis
"""
