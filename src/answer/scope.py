"""The scope policy: a question about a time the record does not cover is told what it does cover."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from answer.contract import Abstention, Answer
from app import sealed

SITTINGS, MEMBERS, BILLS = "sittings", "members", "bills"

# Where each structured category's answer comes from.
SOURCE: dict[str, str] = {
    "division_result": SITTINGS,
    "division_tally": SITTINGS,
    "division_vote": SITTINGS,
    "division_debate": SITTINGS,
    "count": SITTINGS,
    # a party is a member's, held by the Members API's party history; the division lists are its best evidence inside
    # the windows, and outside them the fact says what it lacks
    "party_at_date": MEMBERS,
    "office_held_at_date": MEMBERS,
    "bill_stage_date": BILLS,
}

_SOURCE_NAMES = {
    MEMBERS: "the Members API's record of members, their parties and their posts",
    BILLS: "the Bills API's record of each bill's stages",
}

# What the record is made of, and what it leaves out.
KIND = "spoken proceedings"
LEFT_OUT = "Written questions and answers, written statements and select committee evidence are not part of it."

_FAR_PAST, _FAR_FUTURE = date(1, 1, 1), date(9999, 12, 31)


@dataclass(frozen=True)
class Period:
    """One stretch of time the record holds: a window's declared dates. `available` is False for the sealed one."""

    start: date
    end: date
    available: bool = True


@dataclass(frozen=True)
class Scope:
    """How the question's time stands against the record's periods."""

    asked: tuple[date, date]
    relation: str  # "inside" | "partial" | "outside"
    periods: tuple[Period, ...]
    houses: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()

    @property
    def outside(self) -> bool:
        return self.relation == "outside"

    def statement(self) -> str:
        """The scope, in one paragraph a reader can act on."""
        held = sorted((p for p in self.periods if p.available), key=lambda p: p.start)
        where = f"the {_and(self.sections)}" if self.sections else f"the {_and(self.houses) or 'Commons and Lords'}"
        if len(held) == 1:
            spans = f"from {_period(held[0])}"
        else:
            spans = f"for {len(held)} periods: {_and([_period(p) for p in held])}"
        parts = [f"The record does not cover {_asked(self.asked)}.", f"It holds the {KIND} of {where} {spans}."]
        if any(not p.available for p in self.periods):
            parts.append(sealed.SCOPE_STATEMENT)
        parts.append(LEFT_OUT)
        return " ".join(parts)

    def holdings(self) -> str:
        """What the record holds, without reference to any question's time."""
        return self.statement().split(". ", 1)[1]

    def remarks_label(self) -> str:
        """What text found anyway is: remarks made at another time about the time asked."""
        held = [p for p in self.periods if p.available]
        if held and all(p.start > self.asked[1] for p in held):
            return "What follows are later remarks about it, each with the date it was made."
        if held and all(p.end < self.asked[0] for p in held):
            return "What follows are earlier remarks about it, each with the date it was made."
        return "What follows are remarks from other dates that refer to it, each with the date it was made."

    def source_note(self, category: str) -> str:
        name = _SOURCE_NAMES.get(SOURCE.get(category, SITTINGS))
        if not name:
            return ""
        return (f"The record's sittings do not cover {_asked(self.asked)}; this answer comes from {name}, which "
                f"reaches beyond them.")  # fmt: skip


# --- the question's time ----------------------------------------------------------------------------------------------


def _date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def asked_span(filters: Any) -> tuple[date, date] | None:
    """The question's own time, as a closed span: a day, a range (an open side runs on forever), or a year."""
    if not filters:
        return None
    get = (lambda k: filters.get(k)) if isinstance(filters, Mapping) else (lambda k: getattr(filters, k, None))
    on = _date(get("on_date"))
    if on:
        return on, on
    start, end = _date(get("from_date")), _date(get("to_date"))
    if start or end:
        return start or _FAR_PAST, end or _FAR_FUTURE
    year = get("year")
    if year:
        return date(int(year), 1, 1), date(int(year), 12, 31)
    return None


def relate(span: tuple[date, date], periods: Iterable[Period]) -> str:
    """"inside" when the periods together hold every day of the span, "partial" when they hold some, else "outside".
    """
    start, end = span
    meeting = sorted((p for p in periods if p.start <= end and start <= p.end), key=lambda p: p.start)
    if not meeting:
        return "outside"
    reached = start
    for p in meeting:  # walk the union: contiguous windows (W4 then W5) hold a span neither holds alone
        if p.start > reached:
            return "partial"
        if p.end >= end:
            return "inside"
        reached = max(reached, date.fromordinal(p.end.toordinal() + 1))
    return "partial"


def bounded_by_windows(category: str | None) -> bool:
    """Whether a category's answers exist only inside the windows. Unknown is the sittings' (the safe direction)."""
    return SOURCE.get(category or "", SITTINGS) == SITTINGS


async def as_of(conn: Any, house: str | None = None) -> date | None:
    """The record's present: its most recent sitting (in `house`, when given), from the coverage rows the mask serves,
    so a sealed window's sittings are never "the latest". None when coverage cannot be read.
    """
    from answer import coverage as CV

    try:
        rows = await CV.load(conn, house)
    except Exception:  # noqa: BLE001 - no coverage, no present
        return None
    days = [r.last_sitting for r in rows if r.last_sitting and not sealed.hidden(r.last_sitting)]
    return max(days) if days else None


async def of_question(conn: Any, filters: Any) -> Scope | None:
    """The question's scope, or None when it names no time or the record's coverage cannot be read."""
    span = asked_span(filters)
    if span is None:
        return None
    from answer import coverage as CV

    house = filters.get("house") if isinstance(filters, Mapping) else getattr(filters, "house", None)
    try:
        rows = await CV.load(conn, house)
    except Exception:  # noqa: BLE001 - no coverage, no claim about it
        return None
    periods: dict[str, Period] = {}
    for row in rows:
        start, end = row.declared_start or row.first_sitting, row.declared_end or row.last_sitting
        if start is None or end is None:
            continue
        held = periods.get(row.window_id)
        periods[row.window_id] = Period(min(start, held.start) if held else start, max(end, held.end) if held else end)
    if not periods:
        return None
    window = sealed.window()
    if sealed.masked() and not any(p.start <= window.start and window.end <= p.end for p in periods.values()):
        periods[window.window_id] = Period(window.start, window.end, available=False)
    ordered = tuple(sorted(periods.values(), key=lambda p: p.start))
    return Scope(asked=span, relation=relate(span, ordered), periods=ordered,
                 houses=tuple(sorted({r.house for r in rows})),
                 sections=tuple(sorted({r.section for r in rows}, key=_section_order)))  # fmt: skip


# --- questions about the record itself -------------------------------------------------------------------------

# The record, the corpus or the database as the subject of a question about what it covers, holds or reaches.
_RECORD = r"(?:this|the|your)\s+(?:record|corpus|database|archive|dataset)"
_REACH = r"(?:cover|include|hold|contain|span|go|start|begin|end|run|reach)\w*"
# A question about the record in general: a wh-word, the record, its verb, and nothing after ("which periods does the
# record cover?", "how far back does the database go?").
_WHOLE = re.compile(rf"\b(?:what|which|when|how\s+far)\b[^?]{{0,60}}?\b(?:does|do|is|are)\s+{_RECORD}\s+{_REACH}"
                    rf"(?:\s+back)?\s*\??\s*$|\bwhat\s+is\s+in\s+{_RECORD}\s*\??\s*$", re.I)  # fmt: skip
# A yes-or-no question about a time or a kind of record ("does the record cover 4 March 2012?", "… include written
# questions?"): its object, read as words, holds nothing else.
_YES_NO = re.compile(rf"^\s*(?:does|do|is|are)\s+{_RECORD}\s+{_REACH}\s+(?P<object>[^?]*)\??\s*$", re.I)
_KINDS = frozenset("""
written oral questions question answers answer statements statement ministerial select committee committees evidence
reports report debates debate divisions division votes vote speeches speech proceedings lords commons westminster hall
grand general public bill bills hansard sittings sitting period periods dates date years year months month days day
sections section chamber chambers house houses the a an any all of and or in from to on between before after since until
""".split())


def asks_about_the_record(question: str) -> bool:
    """Whether the question asks what the record itself covers, rather than anything the record holds."""
    from answer import router as RO

    text = " ".join((question or "").split())
    if _WHOLE.search(text):
        return True
    yes_no = _YES_NO.match(text)
    if not yes_no:
        return False
    words = re.findall(r"[a-z]+|\d+", yes_no.group("object").lower())
    return bool(words) and all(w in _KINDS or w in RO.MONTHS or w.isdigit() or w in ("st", "nd", "rd", "th")
                               for w in words)  # fmt: skip


def span_named(question: str) -> tuple[date, date] | None:
    """The span from the first to the last of the days a question names, when it names two or more; else None (one day,
    or none, is the router's to read). A question about the record itself asks about the whole of it: "does the
    record cover 1 January 2026 to 31 July 2026?" was read as its first day (QA pass 9, Low 9).
    """
    from answer import router as RO

    days = []
    for pattern, order in ((RO._ISO, "ymd"), (RO._DMY, "dmy"), (RO._MDY, "mdy")):
        for m in pattern.finditer(question or ""):
            try:
                if order == "ymd":
                    days.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
                elif order == "dmy":
                    days.append(date(int(m.group(3)), RO.MONTHS[m.group(2).lower()], int(m.group(1))))
                else:
                    days.append(date(int(m.group(3)), RO.MONTHS[m.group(1).lower()], int(m.group(2))))
            except (ValueError, KeyError):
                continue
    return (min(days), max(days)) if len(set(days)) >= 2 else None


async def meta(conn: Any, question: str, filters: Any, provenance: Any) -> Answer | None:
    """The answer to a question about the record itself, from its coverage rows alone: the periods, the sections, the
    sealed window as not yet available, what it leaves out; for a time asked about, whether it is covered. None when
    the question is not one, or coverage cannot be read (then nothing is claimed).
    """
    if not asks_about_the_record(question):
        return None
    named = span_named(question)
    span = named or asked_span(filters)
    found = await of_question(conn, {"from_date": named[0], "to_date": named[1]} if named
                              else filters if span else {"on_date": date(1, 1, 1)})  # fmt: skip
    if found is None:
        return None
    # only what a reader may be given makes a span covered: the window not yet available holds nothing for them (QA pass
    # 9, Low 9: a span across it was "Yes")
    held = relate(span, [p for p in found.periods if p.available]) if span else None
    if span is None:
        text = found.holdings()
    elif held == "outside":
        return answer(question, provenance, found)
    elif held == "inside":
        text = f"Yes: the record covers {_asked(span)}. {found.holdings()}"
    else:
        text = f"The record covers part of {_asked(span)}. {found.holdings()}"
    # only the periods a reader may see: the sealed one is named by its statement, never by its dates (the exit guard
    # withholds an answer that carries them, and caught the first version of this)
    return Answer(question=question, provenance=provenance, rendered=text, coverage_note=text,
                  facets={"coverage": [{"from": p.start.isoformat(), "to": p.end.isoformat()}
                                       for p in found.periods if p.available]})  # fmt: skip


# --- what the reader gets ---------------------------------------------------------------------------------------------


def answer(question: str, provenance: Any, scope: Scope) -> Answer:
    """The answer to a question the windows cannot hold: the scope, as the reason."""
    text = scope.statement()
    # no facets: an abstention carries nothing else (the contract), and OUTSIDE_WINDOW itself marks it as placed
    return Answer(question=question, provenance=provenance, abstention=Abstention.OUTSIDE_WINDOW,
                  abstention_detail=text, coverage_note=text, rendered=text)  # fmt: skip


def after_present(question: str, provenance: Any, scope: Scope, present: date) -> Answer:
    """The answer to a question about a day after the record's present: nothing is stated, whatever the source. A post
    or a party the Members API holds open is open as of the day its data was read, not for ever; stating it for a
    later day asserted "John Healey held the office of Chancellor of the Exchequer on 1 January 2030" (QA pass 9,
    High 1).
    """
    text = (f"The record runs to {_day(present)}, its most recent sitting, and states nothing about a later day such as "
            f"{_asked(scope.asked)}: what its sources hold open on that day is open only as far as they were read. "
            f"{scope.holdings()}")
    return Answer(question=question, provenance=provenance, abstention=Abstention.OUTSIDE_WINDOW,
                  abstention_detail=text, coverage_note=text, rendered=text)  # fmt: skip


def apply(found: Answer | None, scope: Scope | None, category: str | None = None) -> Answer | None:
    """Place an answer to a question about a time the record does not cover. Idempotent; anything else unchanged."""
    if found is None or scope is None or not scope.outside or found.facets.get("scope") == "outside":
        return found
    if found.abstention == Abstention.OUTSIDE_WINDOW:
        return found
    facets = {**found.facets, "scope": "outside"}
    if found.abstention is None and (found.structured_facts or found.facets.get("clarify")):
        templates = [sf.template for sf in found.structured_facts] or [category or ""]
        note = " ".join(dict.fromkeys(n for n in (scope.source_note(t) for t in templates) if n))
        return found.model_copy(update={"facets": facets,
                                        "coverage_note": " ".join(x for x in (found.coverage_note, note) if x)})
    if found.abstention is None and (found.claims or found.roster):
        lead = f"{scope.statement()} {scope.remarks_label()}"
        return found.model_copy(update={
            "facets": facets, "rendered": f"{lead}\n\n{found.rendered}".strip(),
            "coverage_note": " ".join(x for x in (lead, found.coverage_note) if x),
        })  # fmt: skip
    return answer(found.question, found.provenance, scope)


# --- wording ----------------------------------------------------------------------------------------------------------


def _day(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def _asked(span: tuple[date, date]) -> str:
    start, end = span
    if start == end:
        return _day(start)
    if start == _FAR_PAST:
        return f"dates up to {_day(end)}"
    if end == _FAR_FUTURE:
        return f"dates from {_day(start)}"
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31) and start.year == end.year:
        return str(start.year)
    next_day = date.fromordinal(end.toordinal() + 1)
    if start.day == 1 and next_day.day == 1 and (start.year, start.month) == (end.year, end.month):
        return f"{start:%B %Y}"
    return f"{_day(start)} to {_day(end)}"


def _period(p: Period) -> str:
    if (p.start.year, p.start.month) == (p.end.year, p.end.month):
        return f"{p.start.day} to {_day(p.end)}"
    if p.start.year == p.end.year:
        return f"{p.start.day} {p.start:%B} to {_day(p.end)}"
    return f"{_day(p.start)} to {_day(p.end)}"


def _and(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


_SECTION_ORDER = ("Commons Chamber", "Westminster Hall", "Public Bill Committees", "General Committees",
                  "Lords Chamber", "Grand Committee")  # fmt: skip


def _section_order(section: str) -> tuple[int, str]:
    """The chambers first, each House's own; a section added later sorts after them by name."""
    return (_SECTION_ORDER.index(section) if section in _SECTION_ORDER else len(_SECTION_ORDER), section)
