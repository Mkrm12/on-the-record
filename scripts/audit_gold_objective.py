"""The objective-correction audit: every checkable fact in the gold, re-derived from the primary sources by its own
SQL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.sealed import window as _sealed_window  # noqa: E402

SEALED = (_sealed_window().start, _sealed_window().end)  # the one definition
from goldset.lint import member_named  # noqa: E402
from goldset.schema import GoldItem, Nugget  # noqa: E402

# The Bills API's own bill types.
BILL_TYPES = {1: "Government Bill", 2: "Private Members' Bill (Starting in the House of Lords)", 3: "Consolidation Bill",
              4: "Hybrid Bill", 5: "Private Members' Bill (under the Ten Minute Rule)", 6: "Private Bill",
              7: "Private Members' Bill (Ballot Bill)", 8: "Private Members' Bill (Presentation Bill)", 9: "Draft Bill",
              10: "Supply and Appropriation Bill"}  # fmt: skip
# The bill types that make a claim of each kind true.
KIND_TYPES = {"private member's bill": frozenset({2, 5, 7, 8}), "government bill": frozenset({1, 10})}
_BILL_KIND = re.compile(r"\b(?P<pmb>private members?['’]?s?\s+bill)\b|\b(?P<gov>government\s+bill|the government"
                        r"['’]s\s+(?:own\s+)?bill)\b", re.I)
_DENIED = re.compile(r"(?:\b(?:not|never|neither|nor|isn['’]t|wasn['’]t)\s+(?:(?:a|an|the|one)\s+)?|\bnon-)$",
                     re.I)
# Roles that preside rather than speak to the business.
PRESIDING = frozenset({"speaker", "deputy_speaker", "chair", "lord_speaker", "lords_deputy"})
# ...and the Chair's printed forms, because the role data tags a presiding Chairman of Ways and Means `office_holder`.
_CHAIR_PRINTED = re.compile(r"^The (?:(?:First |Second )?Deputy )?Chairman(?: of Ways and Means)?(?: \(|$)|^The (?:Temporary )?Chair(?: \(|$)")
# The gold's key -> the record's own column.
DIVISION_FIELDS = {"result": "result", "ayes": "ayes", "noes": "noes", "ayes_listed": "ayes_listed",
                   "noes_listed": "noes_listed", "tally_reconciled": "tally_reconciled", "house": "house",
                   "division_number": "number", "date": "date"}
VOTE_FIELDS = {"lobby": "lobby", "party_at_vote": "party", "is_teller": "is_teller", "proxy_name": "proxy_name"}
DATED_FACTS = ("date", "first_sitting", "last_sitting", "month", "from", "to")
# Titles that change what a member is called (a peerage); Sir, Dame and Dr come and go in print and prove nothing.
_PEERAGE = frozenset({"lord", "baroness", "viscount", "viscountess", "earl", "countess", "marquess", "duke", "bishop"})
_NOT_A_NAME_PART = _PEERAGE | {"lady", "sir", "dame", "dr", "mr", "mrs", "ms", "miss", "the", "rt", "hon", "right",
                               "honourable", "mp", "kc", "qc"}
_NEGATED = re.compile(r"^\W*(?:did not|didn['’]t|does not|doesn['’]t|never|was not|wasn['’]t|is not|"
                      r"isn['’]t|has not|hasn['’]t)\b", re.I)
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
SHOWN_SPLITS = frozenset({"dev"})  # every other split is a confirmation split: ids, checks and sources only
CORRECTION_FIELDS = ("id", "field", "new", "reason", "source", "approved_by", "date")
PRIMARY_SOURCES = ("sql", "api", "record")


class Finding(NamedTuple):
    item: str
    check: str
    source: str  # the primary-source key: "division.id=421", "bill.id=4059", "speech.id=..."
    expected: Any = ""  # what the gold says (shown on dev only)
    found: Any = ""  # what the record says (shown on dev only)


class Sources(Protocol):
    """The primary sources the audit asks. A row dated inside the sealed window comes back as its date alone."""

    def division(self, division_id: int) -> dict[str, Any] | None: ...

    def vote(self, division_id: int, mnis_id: int) -> dict[str, Any] | None: ...

    def party_on(self, mnis_id: int, day: date) -> set[str]: ...

    def office_holders(self, office_key: str, day: date) -> set[int]: ...

    def stage_sittings(self, bill_id: int, stage: str, house: str | None) -> tuple[date, int] | None: ...

    def division_count(self, house: str, month: date) -> int: ...

    def debate_count(self, house: str, day: date) -> int: ...

    def contributions(self, mnis_id: int, start: date, end: date) -> int: ...

    def debate(self, debate_id: str) -> dict[str, Any] | None: ...

    def debates_like(self, house: str, day: date, heading: str) -> int: ...

    def first_speeches(self, debate_id: str) -> list[dict[str, Any]]: ...

    def speech(self, speech_id: str) -> dict[str, Any] | None: ...

    def member_names(self, mnis_id: int | None) -> set[str]: ...

    def bills_of_speech(self, speech_id: str) -> list[dict[str, Any]]: ...


class _Sealed(Exception):
    """Raised the moment anything about an item turns out to be dated inside the sealed window."""


def is_sealed(day: date | None) -> bool:
    return day is not None and SEALED[0] <= day <= SEALED[1]


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO.match(value):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _guard(*values: Any) -> None:
    if any(is_sealed(_day(v)) for v in values):
        raise _Sealed


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("’", "'").replace("‘", "'")).strip().casefold()


def _same(gold: Any, record: Any) -> bool:
    if gold in (None, "") and record in (None, ""):
        return True
    g, r = _day(gold), _day(record)
    if g is not None or r is not None:
        return g == r
    if isinstance(gold, bool) or isinstance(record, bool):
        return isinstance(gold, bool) and isinstance(record, bool) and gold == record
    if isinstance(gold, (int, float)) and isinstance(record, (int, float)):
        return gold == record
    return _norm(gold) == _norm(record)


def _differences(item_id: str, prefix: str, source: str, facts: dict[str, Any], record: dict[str, Any],
                 fields: dict[str, str]) -> list[Finding]:
    return [Finding(item_id, f"{prefix}.{g}", source, facts[g], record[r]) for g, r in fields.items()
            if g in facts and r in record and not _same(facts[g], record[r])]


# --- names ---------------------------------------------------------------------------------------------------------


def printed_name(speaker_name: str, known: Iterable[str] = ()) -> str:
    """The name as Hansard printed it: without the constituency and party, and from the brackets when the record
    printed an office first ("The Minister for Housing (Matthew Pennycook)"; the Lords also print "Captain of the
    Honourable Corps... (Lord Kennedy of Southwark)", without "The", which only the member's own names can tell).
    """
    text = (speaker_name or "").strip()
    before = re.split(r"\s*\(", text, maxsplit=1)[0].strip()
    inside = [g.strip() for g in re.findall(r"\(([^()]+)\)", text)]
    if text.startswith("The ") and inside:
        return inside[0]
    names = [k for k in known if k]
    if names and not _is(before, names):
        return next((g for g in inside if _is(g, names)), before)
    return before


def _name_parts(name: str) -> list[str]:
    """A name's own words: no titles or honorifics, and no Lords territorial designation ("of Flint")."""
    words = re.split(r"[\s,.]+", re.split(r"\s+of\s+", name, maxsplit=1)[0])
    return [w.casefold() for w in words if w and w.casefold() not in _NOT_A_NAME_PART]


def _peerage(name: str) -> frozenset[str]:
    return frozenset(w.casefold() for w in re.split(r"[\s,.]+", name) if w.casefold() in _PEERAGE)


def _is(name: str, known: Iterable[str]) -> bool:
    """Whether `name` can be this member: the surname matches one of the member's names and, when both give a first
    name, so does that.
    """
    parts = _name_parts(name)
    if not parts:
        return False
    for other in known:
        theirs = _name_parts(other)
        if theirs and theirs[-1] == parts[-1] and (len(parts) < 2 or len(theirs) < 2 or theirs[0] == parts[0]):
            return True
    return False


# --- structured facts ----------------------------------------------------------------------------------------------


def _division(item: GoldItem, facts: dict[str, Any], src: Sources) -> list[Finding]:
    key = f"division.id={facts['division_id']}"
    record = src.division(int(facts["division_id"]))
    if record is None:
        return [Finding(item.id, "division.missing", key, "a division", "none")]
    _guard(record.get("date"))
    return _differences(item.id, "division", key, facts, record, DIVISION_FIELDS)


def _vote(item: GoldItem, facts: dict[str, Any], src: Sources) -> list[Finding]:
    key = f"division_vote.division_id={facts['division_id']},mnis_id={facts['mnis_id']}"
    record = src.vote(int(facts["division_id"]), int(facts["mnis_id"]))
    if record is None:
        return [Finding(item.id, "division_vote.missing", key, "a vote", "none")]
    _guard(record.get("date"))
    return _differences(item.id, "division_vote", key, facts, record, VOTE_FIELDS)


def _party(item: GoldItem, facts: dict[str, Any], day: date, src: Sources) -> list[Finding]:
    """The gold's own basis: the party recorded against the member in a division list on that date."""
    key = f"division_vote.mnis_id={facts['mnis_id']}"
    parties = src.party_on(int(facts["mnis_id"]), day)
    if not parties:
        return [Finding(item.id, "party_at_date.no_division_list", key, facts.get("party"), "none")]
    if not any(_same(facts.get("party"), p) for p in parties):
        return [Finding(item.id, "party_at_date", key, facts.get("party"), sorted(parties))]
    return []


def _office(item: GoldItem, facts: dict[str, Any], day: date, src: Sources) -> list[Finding]:
    key = f"member_office.mnis_id={facts.get('mnis_id')}"
    holders = src.office_holders(facts["office_key"], day)
    if not holders:
        return [Finding(item.id, "office_held_at_date.no_holder", key, facts.get("mnis_id"), "none")]
    if facts.get("mnis_id") not in holders:
        return [Finding(item.id, "office_held_at_date", key, facts.get("mnis_id"), sorted(holders))]
    return []


def _stage(item: GoldItem, facts: dict[str, Any], src: Sources) -> list[Finding]:
    key = f"bill.id={facts['bill_id']}"
    record = src.stage_sittings(int(facts["bill_id"]), facts["stage"], facts.get("house"))
    if record is None:
        return [Finding(item.id, "bill_stage.missing", key, facts["stage"], "none")]
    first, sittings = record
    _guard(first)
    out = []
    if "first_sitting" in facts and not _same(facts["first_sitting"], first):
        out.append(Finding(item.id, "bill_stage.first_sitting", key, facts["first_sitting"], first.isoformat()))
    if "sittings" in facts and facts["sittings"] != sittings:
        out.append(Finding(item.id, "bill_stage.sittings", key, facts["sittings"], sittings))
    return out


def _count(item: GoldItem, facts: dict[str, Any], days: dict[str, date], src: Sources) -> list[Finding]:
    if "divisions" in facts and "month" in days:
        check, source, want = "count.divisions", "division", facts["divisions"]
        got = src.division_count(facts["house"], days["month"])
    elif "debates" in facts and "date" in days:
        check, source, want = "count.debates", "debate", facts["debates"]
        got = src.debate_count(facts["house"], days["date"])
    elif "contributions" in facts and {"from", "to"} <= days.keys():
        check, source, want = "count.contributions", f"speech.mnis_id={facts['mnis_id']}", facts["contributions"]
        got = src.contributions(int(facts["mnis_id"]), days["from"], days["to"])
    else:
        return []
    return [] if got == want else [Finding(item.id, check, source, want, got)]


def _debate(item: GoldItem, facts: dict[str, Any], src: Sources) -> list[Finding]:
    key = f"debate.id={facts['debate_id']}"
    record = src.debate(facts["debate_id"])
    if record is None:
        return [Finding(item.id, "debate.missing", key, "a debate", "none")]
    _guard(record.get("date"))
    out = []
    if "heading" in facts and not _same(facts["heading"], record.get("heading")):
        out.append(Finding(item.id, "debate.heading", key, facts["heading"], record.get("heading")))
    day, house = _day(item.date), getattr(item, "house", "")
    if item.category == "who_spoke" and "heading" in facts and day and house:
        # The question names its debate by heading, House and date.
        if (same := src.debates_like(house, day, facts["heading"])) > 1:
            out.append(Finding(item.id, "who_spoke.ambiguous_debate", key, 1, same))
    if item.category == "who_spoke" and "mnis_id" in facts:
        first = next((s for s in src.first_speeches(facts["debate_id"]) if not s["boilerplate"]
                      and s["role"] not in PRESIDING and not _CHAIR_PRINTED.search(s.get("speaker_name") or "")), None)
        if first is not None and first["mnis_id"] != facts["mnis_id"]:
            out.append(Finding(item.id, "who_spoke.opener", key, facts["mnis_id"], first["mnis_id"]))
        elif first is not None and facts.get("opened_by") and first.get("speaker_name"):
            out += _renamed(item, "who_spoke.printed_name", key, facts["opened_by"], *_names(first, src))
    return out


def _names(speech: dict[str, Any], src: Sources) -> tuple[str, set[str]]:
    """The name the record printed for a speech, and every name its member is known by."""
    base = {speech.get("api_member_name") or "", *src.member_names(speech.get("mnis_id"))} - {""}
    printed = printed_name(speech.get("speaker_name") or speech.get("api_member_name") or "", base)
    return printed, ({printed} | base) - {""}


def _renamed(item: GoldItem, check: str, key: str, name: str, printed: str, known: set[str]) -> list[Finding]:
    """The right member under a name the record did not print (a later peerage, or a changed surname)."""
    if printed and _is(name, known) and (not _is(name, {printed}) or _peerage(name) != _peerage(printed)):
        return [Finding(item.id, check, key, name, printed)]
    return []


def _structured(item: GoldItem, src: Sources) -> list[Finding]:
    facts = item.expected_structured_facts or {}
    _guard(*(facts.get(k) for k in DATED_FACTS))
    days = {k: d for k in DATED_FACTS if k in facts and (d := _day(facts[k])) is not None}
    out = [Finding(item.id, f"facts.{k}", "gold", facts[k], "not a YYYY-MM-DD date") for k in DATED_FACTS
           if facts.get(k) and k not in days]
    if "debate_id" in facts:
        out += _debate(item, facts, src)
    category = item.category
    if "division_id" in facts and category in ("division_result", "division_tally", "multi_speech"):
        out += _division(item, facts, src)
    elif category == "division_vote" and {"division_id", "mnis_id"} <= facts.keys():
        out += _vote(item, facts, src)
    elif category == "party_at_date" and "mnis_id" in facts and "date" in days:
        out += _party(item, facts, days["date"], src)
    elif category == "office_held_at_date" and "office_key" in facts and "date" in days:
        out += _office(item, facts, days["date"], src)
    elif category == "bill_stage_date" and {"bill_id", "stage"} <= facts.keys():
        out += _stage(item, facts, src)
    elif category == "count":
        out += _count(item, facts, days, src)
    return out


# --- free text -----------------------------------------------------------------------------------------------------


def _attribution(item: GoldItem, n: Nugget, src: Sources) -> list[Finding]:
    """A point that leads with a member's name: did that member speak the evidence, under that name?"""
    name = member_named(n.text)
    if not name or _NEGATED.match(n.text[n.text.find(name) + len(name):]):
        return []
    ids = list(dict.fromkeys([*n.speech_ids, *(e.speech_id for e in item.evidence)]))
    speeches = [(sid, s) for sid, s in ((sid, src.speech(sid)) for sid in ids) if s is not None]
    _guard(*(s.get("date") for _, s in speeches))
    speeches = [(sid, s) for sid, s in speeches if s.get("speaker_name") or s.get("api_member_name")]
    if not speeches:
        return []
    for sid, s in speeches:
        printed, known = _names(s, src)
        if _is(name, known):  # this member: under the name the record printed, or a later one
            return _renamed(item, "attribution.printed_name", f"speech.id={sid}", name, printed, known)
    sid, s = speeches[0]
    return [Finding(item.id, "attribution.speaker", f"speech.id={sid}", name, _names(s, src)[0])]


def _kind_claims(text: str) -> list[tuple[str, bool]]:
    """Each bill-kind claim in `text`, with whether it is affirmed (False for "not a private Member's Bill")."""
    return [("private member's bill" if m.group("pmb") else "government bill", not _DENIED.search(text[: m.start()]))
            for m in _BILL_KIND.finditer(text)]


def _bill_types(item: GoldItem, src: Sources) -> list[Finding]:
    texts = [(n.text, False) for n in item.nuggets] + [(f, True) for f in item.forbidden_claims]
    claimed = [(claims, forbidden) for text, forbidden in texts if (claims := _kind_claims(text))]
    if not claimed:
        return []
    bills = list({b["id"]: b for e in item.evidence for b in src.bills_of_speech(e.speech_id)}.values())
    if not bills:
        return []
    out = []
    for claims, forbidden in claimed:
        truths = []
        for kind, affirmed in claims:
            holds = {b["type"] in KIND_TYPES[kind] for b in bills}
            if len(holds) != 1:  # bills of both kinds on these debates: ambiguous, so not objective
                break
            truths.append(holds.pop() == affirmed)
        else:
            if all(truths) == forbidden:  # a point that is false, or a forbidden claim that is true
                out.append(Finding(item.id, "bill.type.forbidden" if forbidden else "bill.type",
                                   f"bill.id={bills[0]['id']}", claims[0][0],
                                   BILL_TYPES.get(bills[0]["type"], "unknown")))
    return out


def audit_item(item: GoldItem, src: Sources) -> list[Finding]:
    """Every contradiction between this gold item and the primary sources; one `sealed_window` finding, and nothing
    else, when anything about the item is dated inside the sealed window.
    """
    try:
        _guard(item.date)
        out = _structured(item, src)
        for n in item.nuggets:
            out += _attribution(item, n, src)
        return out + _bill_types(item, src)
    except _Sealed:
        return [Finding(item.id, "sealed_window", "")]


# --- output and the ledger -----------------------------------------------------------------------------------------


def render(findings: list[Finding], items: dict[str, Any]) -> str:
    """The report. Off the dev split, ids, checks and source keys only: never an item's text or values."""
    lines, sealed = [], set()
    for f in findings:
        if f.check == "sealed_window":
            sealed.add(f.item)
        elif getattr(items.get(f.item), "split", "") in SHOWN_SPLITS:
            lines.append(f"{f.item} | {f.check} | {f.source} | gold {f.expected!r} | record {f.found!r}")
        else:
            lines.append(f"{f.item} | {f.check} | {f.source}")
    counts = Counter(f.check for f in findings if f.check != "sealed_window")
    flagged = len({f.item for f in findings if f.check != "sealed_window"})
    summary = f"{sum(counts.values())} contradiction(s) in {flagged} item(s)"
    if counts:
        summary += ": " + ", ".join(f"{check} {n}" for check, n in sorted(counts.items()))
    if sealed:
        summary += f"; {len(sealed)} item(s) inside the sealed window, counted and not checked"
    return "\n".join([*lines, summary])


def correction_problems(record: dict[str, Any]) -> list[str]:
    """Why a proposed correction cannot be applied; empty when it can. A primary source and the maintainer's approval.
    """
    problems = [f"no {f}" for f in CORRECTION_FIELDS if not record.get(f)]
    source = record.get("source")
    if source and (not isinstance(source, dict) or source.get("kind") not in PRIMARY_SOURCES
                   or not str(source.get("ref") or "").strip()):
        problems.append(f"the source must be a primary one ({', '.join(PRIMARY_SOURCES)}) with a reference")
    if record.get("approved_by") and record["approved_by"] != "owner":
        problems.append("only the owner approves a correction")
    if record.get("date") and _day(record["date"]) is None:
        problems.append("the date must be YYYY-MM-DD")
    return problems


class SqlSources:
    """The primary sources as hansard_v2 holds them: independent SQL, a read-only session, and a row dated inside the
    sealed window returned as its date alone (the database never hands this script W5's content).
    """

    def __init__(self, conn: Any):
        self.conn = conn
        self.window = {"lo": SEALED[0], "hi": SEALED[1]}

    @classmethod
    def connect(cls) -> SqlSources:
        import psycopg

        from app import safety
        from app.config import settings

        conn = psycopg.connect(safety.working_dsn(settings.pg_dsn), connect_timeout=3, autocommit=True)
        conn.execute("SET default_transaction_read_only = on")
        return cls(conn)

    def close(self) -> None:
        self.conn.close()

    def _rows(self, sql: str, **args: Any) -> list[tuple[Any, ...]]:
        return self.conn.execute(sql, {**self.window, **args}).fetchall()

    def _dated(self, sql: str, **args: Any) -> dict[str, Any] | None:
        """For `SELECT <date>, <jsonb, or NULL when sealed>`: the row, or its date alone when sealed."""
        rows = self._rows(sql, **args)
        if not rows:
            return None
        day, body = rows[0]
        if is_sealed(_day(day)):
            return {"date": day}
        return None if body is None else {**body, "date": day}

    def division(self, division_id: int) -> dict[str, Any] | None:
        return self._dated(
            "SELECT division_date, CASE WHEN division_date BETWEEN %(lo)s AND %(hi)s THEN NULL ELSE jsonb_build_object("
            "'house', house, 'number', number, 'result', result, 'ayes', ayes, 'noes', noes, 'ayes_listed', "
            "ayes_listed, 'noes_listed', noes_listed, 'tally_reconciled', tally_reconciled) END FROM division "
            "WHERE id = %(id)s", id=division_id)

    def vote(self, division_id: int, mnis_id: int) -> dict[str, Any] | None:
        return self._dated(
            "SELECT d.division_date, CASE WHEN d.division_date BETWEEN %(lo)s AND %(hi)s OR v.division_id IS NULL THEN "
            "NULL ELSE jsonb_build_object('lobby', v.lobby, 'party', v.party, 'is_teller', v.is_teller, 'proxy_name', "
            "v.proxy_name) END FROM division d LEFT JOIN division_vote v ON v.division_id = d.id AND v.mnis_id = "
            "%(mnis)s WHERE d.id = %(id)s", id=division_id, mnis=mnis_id)

    def party_on(self, mnis_id: int, day: date) -> set[str]:
        rows = self._rows("SELECT DISTINCT v.party FROM division_vote v JOIN division d ON d.id = v.division_id WHERE "
                          "v.mnis_id = %(mnis)s AND d.division_date = %(day)s AND NOT (d.division_date BETWEEN %(lo)s "
                          "AND %(hi)s)", mnis=mnis_id, day=day)
        return {r[0] for r in rows if r[0]}

    def office_holders(self, office_key: str, day: date) -> set[int]:
        rows = self._rows("SELECT DISTINCT mnis_id FROM member_office WHERE office_key = %(key)s AND start_date <= "
                          "%(day)s AND (end_date IS NULL OR end_date >= %(day)s)", key=office_key, day=day)
        return {r[0] for r in rows}

    def stage_sittings(self, bill_id: int, stage: str, house: str | None) -> tuple[date, int] | None:
        stages, first, sittings, stage_date = self._rows(
            "SELECT count(DISTINCT bs.id), min(bss.date), count(bss.id), min(bs.date) FROM bill_stage bs LEFT JOIN "
            "bill_stage_sitting bss ON bss.bill_stage_id = bs.id WHERE bs.bill_id = %(bill)s AND bs.stage_name = "
            "%(stage)s AND bs.house IS NOT DISTINCT FROM %(house)s", bill=bill_id, stage=stage, house=house)[0]
        day = first or stage_date
        return None if not stages or day is None else (day, sittings)

    def division_count(self, house: str, month: date) -> int:
        return self._rows("SELECT count(*) FROM division WHERE house = %(house)s AND division_date >= %(month)s AND "
                          "division_date < (%(month)s::date + interval '1 month')", house=house, month=month)[0][0]

    def debate_count(self, house: str, day: date) -> int:
        return self._rows("SELECT count(*) FROM debate WHERE house = %(house)s AND date = %(day)s", house=house,
                          day=day)[0][0]

    def contributions(self, mnis_id: int, start: date, end: date) -> int:
        return self._rows("SELECT count(*) FROM speech s JOIN debate d ON d.id = s.debate_id WHERE s.mnis_id = "
                          "%(mnis)s AND d.date BETWEEN %(start)s AND %(end)s AND NOT s.is_boilerplate", mnis=mnis_id,
                          start=start, end=end)[0][0]

    def debate(self, debate_id: str) -> dict[str, Any] | None:
        return self._dated("SELECT date, CASE WHEN date BETWEEN %(lo)s AND %(hi)s THEN NULL ELSE jsonb_build_object("
                           "'heading', heading, 'house', house) END FROM debate WHERE id = %(id)s", id=debate_id)

    def debates_like(self, house: str, day: date, heading: str) -> int:
        return self._rows("SELECT count(*) FROM debate WHERE house = %(house)s AND date = %(day)s AND heading = "
                          "%(heading)s", house=house, day=day, heading=heading)[0][0]

    def first_speeches(self, debate_id: str) -> list[dict[str, Any]]:
        rows = self._rows("SELECT s.mnis_id, s.speaker_role, s.is_boilerplate, s.speaker_name, s.api_member_name FROM "
                          "speech s JOIN debate d ON d.id = s.debate_id WHERE s.debate_id = %(id)s AND NOT (d.date "
                          "BETWEEN %(lo)s AND %(hi)s) ORDER BY s.seq LIMIT 25", id=debate_id)
        return [{"mnis_id": r[0], "role": r[1], "boilerplate": bool(r[2]), "speaker_name": r[3] or "",
                 "api_member_name": r[4] or ""} for r in rows]

    def speech(self, speech_id: str) -> dict[str, Any] | None:
        return self._dated("SELECT d.date, CASE WHEN d.date BETWEEN %(lo)s AND %(hi)s THEN NULL ELSE jsonb_build_object("
                           "'speaker_name', s.speaker_name, 'mnis_id', s.mnis_id, 'api_member_name', s.api_member_name) "
                           "END FROM speech s JOIN debate d ON d.id = s.debate_id WHERE s.id = %(id)s", id=speech_id)

    def member_names(self, mnis_id: int | None) -> set[str]:
        if mnis_id is None:
            return set()
        rows = self._rows("SELECT name, name_full_title, name_address_as FROM member_id_map WHERE mnis_id = %(mnis)s",
                          mnis=mnis_id)
        return {x for row in rows for x in row if x}

    def bills_of_speech(self, speech_id: str) -> list[dict[str, Any]]:
        """The bills a speech's debate belongs to: by the Bills API's own sitting links, or by the debate's heading
        being the bill's title (most sittings carry no debate link).
        """
        rows = self._rows(
            "SELECT DISTINCT b.id, b.bill_type_id FROM speech s JOIN debate d ON d.id = s.debate_id JOIN bill b ON "
            "b.id IN (SELECT bss.bill_id FROM bill_stage_sitting bss WHERE bss.debate_id = s.debate_id UNION SELECT "
            "bs.bill_id FROM bill_stage bs WHERE bs.debate_id = s.debate_id) OR d.heading IN (b.title, "
            "b.former_short_title) WHERE s.id = %(id)s AND NOT (d.date BETWEEN %(lo)s AND %(hi)s)", id=speech_id)
        return [{"id": r[0], "type": r[1]} for r in rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--split", choices=("dev", "test", "heldout", "all"), default="dev")
    p.add_argument("--json", type=Path, default=None, help="also write the findings (values on dev only)")
    p.add_argument("--check-ledger", type=Path, default=None, help="validate a corrections ledger (one JSON per line)")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if args.check_ledger:
        refused = 0
        for n, line in enumerate(args.check_ledger.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip() and (problems := correction_problems(json.loads(line))):
                refused += 1
                print(f"line {n}: " + "; ".join(problems))
        print(f"{refused} refused" if refused else "every correction has a primary source and the owner's approval")
        return 1 if refused else 0
    from goldset.schema import load

    items = {i.id: i for i in load() if args.split == "all" or i.split == args.split}
    src = SqlSources.connect()
    try:
        findings = [f for item in items.values() for f in audit_item(item, src)]
    finally:
        src.close()
    print(render(findings, items))
    if args.json:
        rows = [{"item": f.item, "check": f.check, "source": f.source,
                 **({"gold": f.expected, "record": f.found} if items[f.item].split in SHOWN_SPLITS else {})}
                for f in findings if f.check != "sealed_window"]
        sealed = len({f.item for f in findings if f.check == "sealed_window"})
        args.json.write_text(json.dumps({"split": args.split, "items": len(items), "sealed_items": sealed,
                                         "findings": rows}, indent=1, default=str) + "\n", encoding="utf-8")
    return 1 if any(f.check != "sealed_window" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
