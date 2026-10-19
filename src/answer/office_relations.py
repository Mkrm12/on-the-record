"""An office qualified by a party or a relation, answered from the record."""

from __future__ import annotations

import re
import weakref
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from typing import Any

from answer import facts as F
from answer import resolve as RS
from answer import router as RO
from answer.contract import Abstention, Answer
from app import sealed

RELATIONS = RO.OFFICE_RELATIONS

# Labels the Votes API puts in a list's party column that are a role in the House or a grouping of its members, not a
# party.
NOT_PARTIES = frozenset({"speaker", "deputy speaker", "lord speaker", "bishops", "non-affiliated", "crossbench",
                         "independent"})  # fmt: skip


@dataclass(frozen=True)
class Qualified:
    kind: str  # "party", or one of RELATIONS
    office_key: str
    office: str
    party: str | None = None  # the record's label, for kind == "party"


def _long(day: Any) -> str:
    d = day if isinstance(day, Date) else Date.fromisoformat(str(day)[:10])
    return f"{d.day} {d:%B %Y}"


def _same(a: str | None, b: str | None) -> bool:
    return bool(a and b) and " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


_PARTIES_BY_CONN: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def parties(conn) -> list[tuple[str, str | None]]:
    """The record's party labels and abbreviations, from its division lists: read once per connection (a scan of every
    vote, 240 ms a question before; S06's timing), as `resolve.with_bills` reads the bills.
    """
    found = _PARTIES_BY_CONN.get(conn)
    if found is None:
        rows = await RS._all(conn, "SELECT DISTINCT party, party_abbreviation FROM division_vote WHERE party IS NOT NULL")
        found = sorted(((r["party"], r["party_abbreviation"]) for r in rows
                        if r["party"].casefold() not in NOT_PARTIES), key=lambda p: -len(p[0]))  # fmt: skip
        _PARTIES_BY_CONN[conn] = found
    return found


def party_forms(party: str, abbreviation: str | None) -> list[str]:
    """How a question may name a party: its label, its members ("Liberal Democrats"), "the X Party" where the label
    lacks it, and an all-capitals abbreviation of two or more letters ("SNP"; not "Con" or "Lab", which are words).
    """
    forms = {party, f"{party}s"}
    if not party.casefold().endswith("party"):
        forms.add(f"{party} Party")
    if abbreviation and len(abbreviation) >= 2 and abbreviation.isupper():
        forms.add(abbreviation)
    return sorted(forms, key=len, reverse=True)


def strip_party(question: str, forms: list[str]) -> str | None:
    """The question with one naming of the party taken out: after the office ("in the SNP", "for the Liberal
    Democrats") or before it ("the Green Party Chancellor"). None when it does not name the party.
    """
    for form in forms:
        pattern = re.escape(form)
        out, n = re.subn(rf"[\s,]*\b(?:in|for|from|of|with|at)\s+the\s+{pattern}(?![\w-])", "", question, count=1,
                         flags=re.I)  # fmt: skip
        if not n:
            # case-sensitive for an abbreviation, so "snp" in running text is not taken for one
            flags = 0 if form.isupper() else re.I
            out, n = re.subn(rf"(?<![\w-]){pattern}\s+(?=\S)", "", question, count=1, flags=flags)
        if n:
            return out
    return None


def strip_relation(question: str, relation: str) -> str | None:
    """The question with the relation taken out: "the X's deputy", "the predecessor of the X", "deputy to the X"."""
    out, n = re.subn(rf"['’]s\s+{relation}\b", "", question, count=1, flags=re.I)
    if not n:
        out, n = re.subn(rf"\b(?:the\s+)?{relation}\s+(?:of|to)\s+(?=the\b)", "", question, count=1, flags=re.I)
    return out if n else None


async def qualified(conn, question: str) -> Qualified | None:
    """The office the question names and how it qualifies it, or None (not qualified, or by something else)."""
    rows = await RS.office_candidates(conn, question)
    if not rows:
        return None
    # Each qualifier is taken out of the question once, and a party only where the question names it at all; then every
    # office is tested against those readings, relations before parties, longest office first (as `resolve.office_key`)
    readings: list[tuple[str, str | None, str]] = []
    for relation in RELATIONS:
        stripped = strip_relation(question, relation)
        if stripped:
            readings.append((relation, None, stripped))
    folded = question.casefold()
    for party, abbreviation in await parties(conn):
        forms = party_forms(party, abbreviation)
        if any(form.casefold() in folded for form in forms):
            stripped = strip_party(question, forms)
            if stripped:
                readings.append(("party", party, stripped))
    for row in rows:
        for kind, party, stripped in readings:
            if RS._office_named_in_full(stripped, row["office"]):
                return Qualified(kind, row["office_key"], F.office_display(row["office"]), party)
    return None


def _decline(question: str, provenance: Any, why: Abstention, detail: str) -> Answer:
    return Answer(question=question, provenance=provenance, abstention=why, abstention_detail=detail,
                  rendered=detail)  # fmt: skip


async def answer(conn, question: str, when: Date | None, provenance: Any) -> tuple[Answer | None, str]:
    """(answer, note) for a qualified office on a date; (None, why) when the question is not one this module reads.
    """
    found = await qualified(conn, question)
    if found is None:
        return None, "no party or relation qualifies an office the record holds"
    if when is None:
        return None, f"a qualified office ({found.kind}) needs a date"
    if found.kind == "party":
        return await _by_party(conn, question, when, provenance, found), f"the office qualified by a party: {found.party}"
    if found.kind == "deputy":
        return await _deputy(conn, question, when, provenance, found), "the office's deputy"
    return await _neighbour(conn, question, when, provenance, found), f"the office's {found.kind}"


async def _by_party(conn, question: str, when: Date, provenance: Any, q: Qualified) -> Answer:
    holder = await F.office_held_at_date(conn, q.office_key, when)
    if not holder.ok:
        return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                        f"The record does not name one holder of the office of {q.office} on {_long(when)}, so it "
                        f"cannot say who the {q.party} one was.")  # fmt: skip
    who = holder.value
    party = await F.party_at_date(conn, who["mnis_id"], when)
    if party.ok and _same(party.value["party"], q.party):
        return Answer(question=question, provenance=provenance, structured_facts=(holder.to_contract(),),
                      coverage_note=holder.caveat,
                      rendered=f"{holder.sentence()} They sat for {q.party} that day.")  # fmt: skip
    if party.ok:
        return _decline(question, provenance, Abstention.FALSE_PREMISE,
                        f"The record names no {q.party} holder of the office of {q.office} on {_long(when)}: it was "
                        f"held by {who['holder']}, who sat for {party.value['party']} that day. The record's posts name "
                        f"one holder of an office at a time.")  # fmt: skip
    return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                    f"The record names {who['holder']} as {q.office} on {_long(when)}, but does not settle which party "
                    f"they sat for that day, so it cannot say whether they were the {q.party} one.")  # fmt: skip


async def _deputy(conn, question: str, when: Date, provenance: Any, q: Qualified) -> Answer:
    # every office as the record prints it, from the connection's catalogue (read once, as the parties are)
    printed = [office for office in (await RS.offices(conn)).printed if "deputy" in office.lower()]
    target = " ".join(q.office.split()).casefold()
    deputies = sorted({office for office in printed
                       if " ".join(re.sub(r"\bdeputy\s+", "", office, count=1, flags=re.I).split()).casefold()
                       == target})  # fmt: skip
    if deputies:
        named = " and ".join(deputies)
        return Answer(question=question, provenance=provenance,
                      facets={"clarify": [{"office": o} for o in deputies],
                              "reason": "a deputy is an office of its own"},
                      rendered=(f"The record's posts name the holder of each office, not who deputised for one. It holds "
                                f"the office of {named}: ask who held that on {_long(when)}."))  # fmt: skip
    return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                    f"The record's posts name the holder of each office, not who deputised for one, and none of its "
                    f"offices is named as the {q.office}'s deputy. Ask about the deputy's own office.")  # fmt: skip


_RECORDED_SQL = """
SELECT o.mnis_id, o.start_date, o.end_date, m.name,
       coalesce((to_jsonb(o) ->> 'start_withheld')::boolean, false) AS start_withheld,
       coalesce((to_jsonb(o) ->> 'end_withheld')::boolean, false) AS end_withheld
  FROM member_office o LEFT JOIN member_id_map m USING (mnis_id)
 WHERE o.office_key = %s AND o.source = 'mnis_biography'
 ORDER BY o.start_date NULLS FIRST
"""


def _holds(t: dict, day: Date) -> bool:
    return (t["start_date"] is None or t["start_date"] <= day) and (t["end_date"] is None or t["end_date"] >= day)


def _overlap(a: dict, b: dict) -> bool:
    """Two tenures share a day beyond a change day (one's end is the other's start)."""
    a0, a1 = a["start_date"] or Date.min, a["end_date"] or Date.max
    b0, b1 = b["start_date"] or Date.min, b["end_date"] or Date.max
    return a0 < b1 and b0 < a1 and a1 != b0 and b1 != a0


def _across_the_window(ended: Date | None, began: Date | None) -> bool:
    """Whether a change of holder, from a tenure that `ended` to one that `began`, touches the sealed window (QA pass
    9, High 2). Migration 027 drops a post wholly inside the window and clips one that meets it to its edge, so
    between two tenures whose gap touches it someone the record cannot name may have held the office.
    """
    if not sealed.masked() or ended is None or began is None:
        return False
    window = sealed.window()
    return ended <= window.end and began >= window.start


def _inside(t: dict) -> Date | None:
    """A day of a tenure outside the sealed window, for asking the fact engine to name its holder: one strictly inside
    it first (its ends are where another tenure may meet it), else its own start or end (a post held for a day or
    two has no inside; the fact engine must still name this holder alone on it). None when there is none.
    """
    start, end = t["start_date"], t["end_date"]
    inner = ((end - timedelta(days=1)) if end else None, (start + timedelta(days=1)) if start else None)
    strict = [d for d in inner if d is not None and d not in (start, end)]
    for day in (*strict, start, end):
        if day is not None and _holds(t, day) and not sealed.hidden(day):
            return day
    return None


async def _neighbour(conn, question: str, when: Date, provenance: Any, q: Qualified) -> Answer:
    tenures = await RS._all(conn, _RECORDED_SQL, (q.office_key,))
    on = [t for t in tenures if _holds(t, when)]
    if len({t["mnis_id"] for t in on}) != 1:
        return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                        f"The record's recorded appointments do not name one holder of the office of {q.office} on "
                        f"{_long(when)}, so it cannot say who held it {'before' if q.kind == 'predecessor' else 'after'} "
                        f"them.")  # fmt: skip
    held = on[0]
    if any(_overlap(a, b) for i, a in enumerate(tenures) for b in tenures[i + 1:] if a["mnis_id"] != b["mnis_id"]):
        return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                        f"The office of {q.office} has been held by more than one person at a time, so no one holder "
                        f"came {'before' if q.kind == 'predecessor' else 'after'} another.")  # fmt: skip
    if q.kind == "predecessor":
        side = [t for t in tenures if t["mnis_id"] != held["mnis_id"] and t["end_date"] is not None
                and held["start_date"] is not None and t["end_date"] <= held["start_date"]]
        pick = max(side, key=lambda t: t["end_date"], default=None)
        missing = (f"The record's recorded appointments to the office of {q.office} begin with {held['name']}'s, so "
                   f"it does not say who held it before them.")  # fmt: skip
    else:
        side = [t for t in tenures if t["mnis_id"] != held["mnis_id"] and t["start_date"] is not None
                and held["end_date"] is not None and t["start_date"] >= held["end_date"]]
        pick = min(side, key=lambda t: t["start_date"], default=None)
        missing = (f"The record's recorded appointments to the office of {q.office} end with {held['name']}'s, so it "
                   f"does not say who held it after them.")  # fmt: skip
    if pick is None:
        return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE, missing)
    earlier, later = (pick, held) if q.kind == "predecessor" else (held, pick)
    if _across_the_window(earlier["end_date"], later["start_date"]):
        return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                        f"The change of holder of the office of {q.office} after {earlier['name']} falls in a period "
                        f"of the record that is not yet available, so it cannot say who held it "
                        f"{'before' if q.kind == 'predecessor' else 'after'} {held['name']}.")  # fmt: skip
    ties = [t for t in side if (t["end_date"] if q.kind == "predecessor" else t["start_date"])
            == (pick["end_date"] if q.kind == "predecessor" else pick["start_date"]) and t["mnis_id"] != pick["mnis_id"]]
    day = _inside(pick)
    fact = await F.office_held_at_date(conn, q.office_key, day) if day and not ties else None
    if fact is None or not fact.ok or fact.value.get("mnis_id") != pick["mnis_id"]:
        return _decline(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                        f"The record's recorded appointments do not settle who held the office of {q.office} "
                        f"{'before' if q.kind == 'predecessor' else 'after'} {held['name']}.")  # fmt: skip
    if q.kind == "predecessor":
        until = "" if pick["end_withheld"] else f", until {_long(pick['end_date'])}"
        text = f"{pick['name']} held the office of {q.office} before {held['name']}{until}."
    else:
        since = "" if pick["start_withheld"] else f", from {_long(pick['start_date'])}"
        text = f"{pick['name']} held the office of {q.office} after {held['name']}{since}."
    return Answer(question=question, provenance=provenance, structured_facts=(fact.to_contract(),),
                  coverage_note=(f"{held['name']} held it on {_long(when)}. From the recorded appointments; periods "
                                 f"inferred from Hansard's labels are not used for who came before or after."),
                  rendered=text)  # fmt: skip
