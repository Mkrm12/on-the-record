"""The structured-fact engine: every fact the system states comes from SQL, never from a model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from typing import Any, ClassVar

from answer import tally as T
from procedure import eras as ERAS
from procedure import threads as TH

# Every template this engine can answer.
TEMPLATES = (
    "division_result",
    "division_tally",
    "division_vote",
    "division_debate",
    "party_at_date",
    "office_held_at_date",
    "bill_stage_date",
    "count",
)


@dataclass
class Fact:
    """One fact, its provenance, and anything that must be said alongside it."""

    template: str
    params: dict[str, Any]
    value: dict[str, Any] | None = None
    # Always "sql" when a value is present.
    source: str = "sql"
    # Something a correct answer MUST carry.
    caveat: str = ""
    # Candidates when the parameters do not identify one answer.
    clarify: list[dict[str, Any]] = field(default_factory=list)
    # Why there is no value, when there is none and no ambiguity either.
    missing: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.clarify

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarify)

    def render(self) -> str:
        """A sentence a user can read, with the caveat attached. Never drops the caveat."""
        if self.needs_clarification:
            # each option in words, never a raw mapping
            def option(c: object) -> str:
                if not isinstance(c, dict):
                    return str(c)
                parts = [f"{k.replace('_', ' ')} {_day(v) if k in ('first_sitting', 'last_sitting', 'date', 'from', 'to') and v else v}"
                         for k, v in c.items() if v is not None and k not in self.INTERNAL_KEYS]  # fmt: skip
                return ", ".join(parts)

            # the caveat too: it says why there is more than one (S06: a clarification lost it, and with it the sources
            # a reader needs to choose between two parties)
            return (f"That could mean {len(self.clarify)} different things: "
                    + "; ".join(option(c) for c in self.clarify[:3]) + "." + (f" {self.caveat}" if self.caveat else ""))
        if not self.ok:
            return self.missing or "The record does not hold that."
        body = ", ".join(f"{k} {v}" for k, v in self.value.items() if v is not None)
        return f"{body}. {self.caveat}".strip()


    # Keys that exist for the scorer, the verifier or the query's author and never for a reader.
    INTERNAL_KEYS: ClassVar[frozenset[str]] = frozenset({
        "division_id", "must_cite", "tally_reconciled", "motion", "ayes_listed", "noes_listed", "ayes_named", "noes_named",
        "no_vote_count",
        "source_id", "bill_stage_id", "office_key", "mnis_id", "bill_id", "stage_id", "thread_id", "thread",
        "printed_ayes", "printed_noes",
    })

    # The fields that ARE the answer, per template.
    ANSWER_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "division_result": ("result", "ayes", "noes", "printed_ayes", "printed_noes"),
        "division_tally": ("ayes", "noes", "tally_reconciled", "ayes_named", "noes_named", "printed_ayes", "printed_noes"),
        "division_vote": ("lobby", "is_teller", "proxy_name"),
        # the debate, not the record: two records of one debate (a bill's stage resumed after a statement) are one
        # answer
        "division_debate": ("thread_id",),
        "party_at_date": ("party",),
        "office_held_at_date": ("holder",),
        "bill_stage_date": ("first_sitting",),
        "count": ("divisions", "debates", "contributions"),
    }

    # Plain English for the record's verdict words.
    VERDICT: ClassVar[dict[str, tuple[str, str]]] = {"agreed": ("agreed to", "agreed"), "negatived": ("rejected", "negatived"),
               "tied": ("tied", "tied")}  # fmt: skip

    def answer_fields(self) -> tuple | None:
        """What this fact answers, stripped of everything that differs between candidates for other reasons."""
        if not self.ok:
            return None
        keys = self.ANSWER_FIELDS.get(self.template)
        if not keys:
            return tuple(sorted((k, str(v)) for k, v in self.value.items() if k not in self.INTERNAL_KEYS))
        return tuple((k, str(self.value.get(k))) for k in keys if k in self.value)

    def sentence(self) -> str:
        """What a reader is shown: English, per template, with the caveat attached."""
        if self.needs_clarification or not self.ok:
            return self.render()
        body = _SENTENCES.get(self.template, _fallback_sentence)(self.value)
        return f"{body} {self.caveat}".strip()


    def to_contract(self):
        """This fact as the answer contract's `StructuredFact`."""
        from answer.contract import StructuredFact

        return StructuredFact(
            template=self.template,
            params={k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in (self.params or {}).items()},
            value={k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in (self.value or {}).items()}
            if self.value is not None else None,
            caveat=self.caveat or "",
        )  # fmt: skip


class FactError(RuntimeError):
    """A template called with parameters it cannot use. Raised, never guessed around."""


# --- how a structured fact reads to a person ------------------------------------------------------------ One function
# per template rather than a format string per template, because three of these have to make a decision (a tally that
# reconciles says less than one that does not; a bill stage sat once or many times; a `count` carries whichever of three
# shapes its parameters chose) and a format string cannot.


def _day(value) -> str:
    """The maintainer reads dates, not ISO."""
    from datetime import date as _date

    try:
        when = value if isinstance(value, _date) else _date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return str(value)
    return f"{when.day} {when.strftime('%B')} {when.year}"


def _month(value) -> str:
    from datetime import date as _date

    try:
        when = value if isinstance(value, _date) else _date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return str(value)
    return f"{when.strftime('%B')} {when.year}"


def _verdict(result: str) -> str:
    plain, recorded = Fact.VERDICT.get((result or "").lower(), (result, result))
    # The record's own word is worth adding only when the plain one does not already contain it: "agreed to" needs no
    # gloss, "rejected" does, because a reader looking the division up will find "negatived".
    if not recorded or recorded.lower() in (plain or "").lower():
        return plain
    return f"{plain} \u2014 \"{recorded}\", in the record's own word"


def _fallback_sentence(value: dict) -> str:
    """For a template with no sentence of its own: the readable keys, joined. Never the internal ones."""
    parts = [f"{k.replace('_', ' ')} {v}" for k, v in value.items()
             if v is not None and k not in Fact.INTERNAL_KEYS]  # fmt: skip
    return ", ".join(parts) + "." if parts else "The record holds no readable value for this."


def _which(v: dict) -> str:
    """" (division 159: Finance (No. 2) Bill: Third Reading)": which division a sentence is about, so a reader can see
    it is the one they asked about (QA pass 3: a different division answered as the one asked was invisible).
    """
    number, title = v.get("division_number"), v.get("title")
    if number is None:
        return f" ({title})" if title else ""
    return f" (division {number}: {title})" if title else f" (division {number})"


def _described(v: dict) -> str:
    """What the division was on, where its title does not say (a Lords division's title is the bill's name; its motion
    says which amendment): the record's own words, quoted. A sentence of its own after the answer, so the answer's
    own wording is unchanged.
    """
    from answer import resolve as RS

    title = v.get("title")
    desc = RS.describe(title, v.get("motion"))
    if not desc or desc == title:
        return ""
    return f" The record describes the division as: \u201c{desc}\u201d."


def _short_lists(v: dict) -> int:
    """How many fewer members the published lists name than the tellers counted, or 0 when unknown or none."""
    return sum(max(0, figure - named) for figure, named in ((v.get("ayes"), v.get("ayes_named")),
                                                          (v.get("noes"), v.get("noes_named")))
               if figure is not None and named is not None)  # fmt: skip


def _s_division_result(v: dict) -> str:
    where = f"The {v['house']}" if v.get("house") else "The House"
    number = _which(v)
    tally = (f", {v['ayes']} votes to {v['noes']}"
             if v.get("ayes") is not None and v.get("noes") is not None else "")  # fmt: skip
    if not v.get("result"):
        # Withheld or never recorded.
        return (f"{where} divided on {_day(v.get('date'))}{number}{tally}, and the record does not give a single result."
                f"{_described(v)}")
    if (v.get("result") or "").lower() == "tied":
        # A tie is resolved by a rule of the House (the Speaker's casting vote; the Lords' standing orders), and only
        # what the record says of it is stated: its motion text, quoted below.
        return f"{where} divided on {_day(v.get('date'))}{number} and the vote was tied{tally}.{_described(v)}"
    return (f"{where} divided on {_day(v.get('date'))}{number} and the motion was {_verdict(v.get('result'))}{tally}."
            f"{_described(v)}")


def _era_caveat(row: dict) -> str:
    """What the division's procedural era changes, in the reader's words: a remote vote, a deferred ballot and the day
    its question was put, both counts of a double majority, a division held after midnight. One source for the
    result and the tally, so a tally under EVEL states both counts as the result did.
    """
    return " ".join(ERAS.division_notes(row, put_on=row.get("debate_date")))


def passage_sentence(house: str | None, day, rows: list[dict]) -> str:
    """A passage question on a day whose divisions on the business were all on amendments: each named by what it was
    on, and no result asserted, since none of them was a vote on passing it. A division of the sitting before, held
    after midnight on `day`, is placed in its sitting.
    """
    from answer import resolve as RS

    def one(r: dict) -> str:
        described = str(RS.describe(r.get("title"), r.get("motion_text")))
        sitting = ERAS.other_sitting(r.get("division_date"), day)
        if r.get("number") is None:
            return f"{described} ({sitting})" if sitting else described
        return f"division {r['number']} ({described}{f'; {sitting}' if sitting else ''})"

    where = f"The {house}" if house else "The House"
    listed = "; ".join(one(r) for r in rows)
    times = "once" if len(rows) == 1 else f"{len(rows)} times"
    which = "it" if len(rows) == 1 else "any of them"
    return (f"{where} divided {times} on {_day(day)} on this business, and the record does not show {which} as a vote "
            f"on passing it: {listed}. Say which division you mean for its result.")


def several_results_sentence(facts: list[Fact], asked=None) -> str:
    """Several divisions on one day matching one description, sharing ONE result: the result once, with every tally
    beside it, so no single division's numbers stand for the others. `asked` is the calendar day the question named:
    it heads the sentence, and a division of another sitting is placed in it.
    """
    from answer import resolve as RS

    first = facts[0].value or {}
    where = f"The {first['house']}" if first.get("house") else "The House"

    # Each division is labelled with what it was on.
    labels = {id(f): RS.describe((f.value or {}).get("title"), (f.value or {}).get("motion")) for f in facts}
    distinct = len(set(labels.values())) > 1

    def one(f: Fact) -> str:
        v = f.value or {}
        label = f"division {v['division_number']}" if v.get("division_number") is not None else "a division"
        placed = [x for x in (labels[id(f)] if distinct else "", ERAS.other_sitting(v.get("date"), asked)) if x]
        if placed:
            label += f" ({'; '.join(placed)})"
        figures = f", {v['ayes']} to {v['noes']}" if v.get("ayes") is not None and v.get("noes") is not None else ""
        return label + figures

    # what they share is said once, after the words that claim no more than "these matched"
    shared = "" if distinct or not next(iter(labels.values()), None) else f" ({next(iter(labels.values()))})"
    return (f"{where} divided {len(facts)} times on {_day(asked or first.get('date'))} on business matching this "
            f"description{shared}, and the motion was {_verdict(first.get('result'))} each time "
            f"({'; '.join(one(f) for f in facts)}).")


# Templates whose differing answers are stated side by side, in a table of every matching division.
TABLED = frozenset({"division_result", "division_tally", "division_vote"})


def _cell(text: object) -> str:
    """A table cell: one line, no column separator."""
    return " ".join(str(text if text is not None else "").replace("|", "/").split())


def summary_table(category: str, facts: list[Fact], meta: dict[int, tuple[int | None, str | None]], *,
                  house: str | None = None, date=None, named_business: bool = True,
                  unlisted: tuple[str, ...] = (), divided: int | None = None,
                  houses: tuple[str, ...] = (), asked=None) -> str:  # fmt: skip
    """Several divisions answer the question: every one, its answer stated against its own row. `meta` is {division_id:
    (number, description)}; `unlisted` holds sentences for divisions a vote question's member is not recorded in,
    `divided` how many divisions matched in all (the rows are those with an answer) and `houses` the House of each
    of them, since the lead speaks for all of them, not only the rows.
    """
    values = [dict(f.value or {}) for f in facts]
    houses_rows = {v.get("house") or house for v in values} - {None}
    lords = houses_rows == {"Lords"}
    ayes, noes = ("Contents", "Not-Contents") if lords else ("Ayes", "Noes")
    shown = asked or values[0].get("date") or date
    day = _day(shown) if shown else "that day"
    matched = (set(houses) - {None}) or houses_rows
    where = f"The {next(iter(matched))}" if len(matched) == 1 else "The two Houses"
    about = " on business matching this description" if named_business else ""
    elsewhere = {v.get("division_id"): ERAS.other_sitting(v.get("date"), asked) for v in values}
    placed = [words for words in elsewhere.values() if words]
    # how many rows belong to another sitting is said where the count is; why (held after midnight) is each row's caveat
    late = f", {len(placed)} of them {'; '.join(sorted(set(placed)))}" if placed else ""
    lead = f"{where} divided {divided or len(facts)} times on {day}{about}{late}:"
    by_house = len(houses_rows) > 1

    def ordered():
        return sorted(zip(facts, values), key=lambda fv: (meta.get(fv[1].get("division_id"), (None,))[0] is None,
                                                          meta.get(fv[1].get("division_id"), (0,))[0] or 0))

    lines = []
    for fact, v in ordered():
        number, desc = meta.get(v.get("division_id"), (v.get("division_number"), v.get("title")))
        cell = str(number) if number is not None else "-"
        if elsewhere.get(v.get("division_id")):
            cell += f" ({elsewhere[v.get('division_id')].removeprefix('in the ')})"
        first = [_cell(cell)] + ([_cell(v.get("house"))] if by_house else [])
        on = _cell(desc or v.get("title") or "")
        if category == "division_result":
            plain = Fact.VERDICT.get((v.get("result") or "").lower(), (None,))[0] or "no single result recorded"
            lines.append([*first, on, plain, _cell(v.get("ayes")), _cell(v.get("noes"))])
        elif category == "division_tally":
            lines.append([*first, on, _cell(v.get("ayes")), _cell(v.get("noes"))])
        else:
            lobby = (v.get("lobby") or "").lower()
            word = {"aye": "Content" if lords else "Aye", "no": "Not Content" if lords else "No",
                    "no_vote": "listed, no vote recorded"}.get(lobby, "on both sides: cannot be stated")
            if v.get("is_teller"):
                word += " (a teller)"
            if v.get("proxy_name"):
                word += f" (cast by proxy by {v['proxy_name']})"
            lines.append([*first, on, _cell(word)])
    house_col = ["House"] if by_house else []
    if category == "division_result":
        header = ["Division", *house_col, "On", "Result", ayes, noes]
    elif category == "division_tally":
        header = ["Division", *house_col, "On", ayes, noes]
    else:
        who = values[0].get("member") or "The member"
        header = ["Division", *house_col, "On", _cell(who)]
    table = "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                      + ["| " + " | ".join(row) + " |" for row in lines])
    notes = []
    for fact, v in ordered():
        if fact.caveat:
            number = meta.get(v.get("division_id"), (v.get("division_number"),))[0]
            notes.append(f"Division {number}: {fact.caveat}" if number is not None else fact.caveat)
    tail = " ".join([*unlisted, *dict.fromkeys(notes)])
    return f"{lead}\n\n{table}" + (f"\n\n{tail}" if tail else "")


def candidates_sentence(category: str, facts: list[Fact], meta: dict[int, tuple[int | None, str | None]],
                        house: str | None = None, date=None, asked=None) -> str:  # fmt: skip
    """Several divisions on one day match the question and give DIFFERENT answers: which one is meant is asked, and the
    reader is never shown an empty answer (six dev clarifications rendered as nothing). The templates in `TABLED`
    are answered by `summary_table` instead.
    """
    first = dict(facts[0].value or {})
    first.setdefault("house", house)
    if first.get("date") is None and date is not None:
        first["date"] = _iso(date)
    where = f"The {first['house']}" if first.get("house") else "The House"
    parts = []
    for fact in facts:
        v = fact.value or {}
        number, title = meta.get(v.get("division_id"), (v.get("division_number"), None))
        name = f"division {number}" if number is not None else "one division"
        placed = [x for x in (title, ERAS.other_sitting(v.get("date"), asked)) if x]
        if placed:
            name += f" ({'; '.join(placed)})"
        if category == "division_result":
            tally = f", {v['ayes']} to {v['noes']}" if v.get("ayes") is not None and v.get("noes") is not None else ""
            parts.append(f"{name}: {_verdict(v.get('result')) if v.get('result') else 'no single result recorded'}"
                         f"{tally}")
        elif category == "division_tally":
            parts.append(f"{name}: {v.get('ayes')} Ayes to {v.get('noes')} Noes")
        elif category == "division_vote":
            parts.append(f"{name}: {_s_division_vote(v, named=False).rstrip('.')}")
        elif category == "division_debate":
            parts.append(f"{name}: taken during the {v.get('debate')!r} debate")
        else:
            parts.append(f"{name}: {_fallback_sentence(v).rstrip('.')}")
    shown = asked or first.get("date")
    day = _day(shown) if shown else "that day"
    return (f"{where} divided {len(facts)} times on {day} on business matching this description, and the answers "
            f"differ — say which division you mean. " + "; ".join(parts) + ".")


def _s_division_tally(v: dict) -> str:
    if v.get("ayes") is None or v.get("noes") is None:
        return "The tellers' figures are not recorded for this division."
    figures = f"{v['ayes']} voted Aye and {v['noes']} voted No, as counted by the tellers."
    if not (v.get("division_number") is not None or v.get("title")):
        return figures
    where = f"the {v['house']} division" if v.get("house") else "the division"
    when = f" on {_day(v['date'])}" if v.get("date") else ""
    return f"In {where}{when}{_which(v)}, {figures}{_described(v)}"


# What a division list can never prove of a member it does not show voting: that they were absent, abstained, or did not
# vote.
UNPROVABLE_VOTE_WORDS = re.compile(r"\babsent\b|\babstain\w*|\bdid(?:n['\u2019]t| not) (?:vote|take part)\b|\bcast no vote\b",
                                   re.I)


def _s_division_vote(v: dict, *, named: bool = True) -> str:
    """How the member is recorded. `named=False` leaves the division out, for a sentence that has already named it.
    """
    who = v.get("member") or "That member"
    party = f" ({v['party_at_vote']})" if v.get("party_at_vote") else ""
    lobby = (v.get("lobby") or "").lower()
    where = ""
    if named and (v.get("division_number") is not None or v.get("title")):
        where = f" in division {v['division_number']}" if v.get("division_number") is not None else " in the division"
        where += f" ({v['title']})" if v.get("title") else ""
        where += f" on {_day(v['date'])}" if v.get("date") else ""
    if lobby == "no_vote":
        listed = f" in division {v['division_number']}" if named and v.get("division_number") is not None else ""
        listed += f" ({v['title']})" if named and v.get("title") else ""
        listed += f" on {_day(v['date'])}" if named and v.get("date") else ""
        short = _short_lists(v)
        caveat = (f" The published lists name {short} fewer member{'s' if short != 1 else ''} than the tellers counted, "
                  f"so a vote may have gone unrecorded." if short else "")  # fmt: skip
        return (f"{who}{party} is listed{listed or ' in the division'} with no vote recorded.{caveat}"
                f"{_described(v) if named else ''}")
    how = {"aye": "voted Aye", "no": "voted No"}.get(lobby)
    if how is None:
        # "conflict" is 62 votes corpus-wide where the member appears in both lobbies.
        return (f"{who}{party} appears on both sides of this division in the record, so how they voted "
                f"cannot be stated.")
    teller = " and acted as a teller" if v.get("is_teller") else ""
    return f"{who}{party} {how}{where}{teller}.{_described(v) if named else ''}"


def _s_division_debate(v: dict) -> str:
    where = f" in the {v['section']}" if v.get("section") else ""
    number = f" (division {v['division_number']})" if v.get("division_number") is not None else ""
    speeches = f", a debate of {v['speeches']} contributions" if v.get("speeches") else ""
    return (f"That vote{number} was taken during the {v.get('debate')!r} debate{where} on "
            f"{_day(v.get('date'))}{speeches}.")


def _s_party_at_date(v: dict) -> str:
    who = v.get("member") or "That member"
    text = v.get("basis") or ""
    # the list day's basis names the list ("That is the party recorded …"); the bracket's is a sentence of its own (L2)
    basis = (f" That is {text}." if text.startswith("the party") else f" {text[0].upper()}{text[1:]}.") if text else ""
    return f"{who} sat as {v.get('party')} on {_day(v.get('date'))}.{basis}"


def office_display(office: str | None) -> str:
    """An office as a reader should see it: whitespace normalised, and without the leading "The" Hansard prints before
    one.
    """
    return re.sub(r"^the\s+", "", " ".join((office or "").split()), flags=re.I)


def _s_office_held_at_date(v: dict) -> str:
    return f"{v.get('holder')} held the office of {office_display(v.get('office'))} on {_day(v.get('date'))}."


def _s_bill_stage_date(v: dict) -> str:
    where = f" in the {v['house']}" if v.get("house") else ""
    first = _day(v.get("first_sitting"))
    sittings = v.get("sittings") or 1
    if sittings > 1:
        return (f"{v.get('bill')} reached {v.get('stage')}{where} on {first}, the first of {sittings} sittings; "
                f"the last was {_day(v.get('last_sitting'))}.")
    return f"{v.get('bill')} reached {v.get('stage')}{where} on {first}."


def _s_count(v: dict) -> str:
    where = f"The {v['house']}" if v.get("house") else "The two Houses"
    if v.get("divisions") == 0 and v.get("date"):
        # reached only for a day the windows cover (the scope answers any other), so the record's own silence
        house = f"the {v['house']}" if v.get("house") else "either House"
        return f"The record holds no division in {house} on {_day(v['date'])}."
    if v.get("divisions") is not None and v.get("date"):
        late = v.get("of_the_sitting_before")
        before = ERAS.other_sitting(Date.fromisoformat(v["date"]) - timedelta(days=1), v["date"]) if late else ""
        times = "once" if v["divisions"] == 1 else f"{v['divisions']} times"
        return (f"{where} divided {times} on {_day(v['date'])}"
                f"{f', {late} of them after midnight {before}' if late else ''}.")
    if v.get("divisions") is not None:
        return f"{where} held {v['divisions']} divisions in {_month(v.get('month'))}."
    if v.get("debates") is not None and v.get("month"):
        return f"{where} held {v['debates']} separate debates in {_month(v['month'])}."
    if v.get("debates") is not None:
        return f"{where} held {v['debates']} separate debates on {_day(v.get('date'))}."
    if v.get("contributions") is not None:
        return (f"The record holds {v['contributions']} contributions between {_day(v.get('from'))} and "
                f"{_day(v.get('to'))}, not counting procedural boilerplate.")
    return _fallback_sentence(v)


def template_sentence(category: str, value: dict) -> str | None:
    """The template's sentence for `value`, or None when the category has no template or the value lacks what it needs.
    An evaluation checks a database-generated item by applying the SAME template to the gold values.
    """
    render = _SENTENCES.get(category)
    if render is None:
        return None
    try:
        return render(value)
    except (KeyError, TypeError, ValueError):
        return None


_SENTENCES = {
    "division_result": _s_division_result,
    "division_tally": _s_division_tally,
    "division_vote": _s_division_vote,
    "division_debate": _s_division_debate,
    "party_at_date": _s_party_at_date,
    "office_held_at_date": _s_office_held_at_date,
    "bill_stage_date": _s_bill_stage_date,
    "count": _s_count,
}


# --- divisions ------------------------------------------------------------------------------------------

_DIVISION_SQL = """
SELECT d.id, d.house, d.number, d.division_date, d.title, d.motion_text, d.result,
       d.ayes, d.noes, d.ayes_listed, d.noes_listed, d.ayes_named, d.noes_named, d.tally_reconciled,
       d.no_vote_count, d.is_deferred, d.debate_id, d.evel_country, d.double_majority_ayes, d.double_majority_noes,
       d.printed_ayes, d.printed_noes, d.is_remote, d.held_at, db.date AS debate_date
  FROM division d LEFT JOIN debate db ON db.id = d.debate_id WHERE d.id = %(division_id)s
"""


def sources_caveat(v: dict) -> str:
    """Where Hansard's printed figures and the Votes API's part, both, so the answer never chooses silently. The stated
    figures elsewhere are the Votes service's, on which every list and reconciliation is built.
    """
    printed = (v.get("printed_ayes"), v.get("printed_noes"))
    if None in printed or printed == (v.get("ayes"), v.get("noes")):
        return ""
    ayes, noes = ("Contents", "Not-Contents") if v.get("house") == "Lords" else ("Ayes", "Noes")
    return (f"The record's two sources differ on this division's figures: Hansard prints {printed[0]} {ayes} and "
            f"{printed[1]} {noes}; the Votes service records {v.get('ayes')} and {v.get('noes')}.")


async def division_result(conn, division_id: int) -> Fact:
    """Was it agreed to. The one division fact a person is most likely to ask for."""
    params = {"division_id": int(division_id)}
    row = await _one(conn, _DIVISION_SQL, params)
    if row is None:
        return Fact("division_result", params, missing=f"no division with id {division_id}")
    value = {
        "division_id": row["id"], "house": row["house"], "division_number": row["number"],
        "title": row["title"], "motion": row["motion_text"], "date": _iso(row["division_date"]),
        "result": row["result"], "ayes": row["ayes"], "noes": row["noes"],
        "printed_ayes": row.get("printed_ayes"), "printed_noes": row.get("printed_noes"),
    }  # fmt: skip
    caveat = " ".join(c for c in (_era_caveat(row), _tally_caveat(row), sources_caveat(value)) if c)
    return Fact("division_result", params, value=value, caveat=caveat)


async def division_tally(conn, division_id: int) -> Fact:
    params = {"division_id": int(division_id)}
    row = await _one(conn, _DIVISION_SQL, params)
    if row is None:
        return Fact("division_tally", params, missing=f"no division with id {division_id}")
    value = {
        "division_id": row["id"], "house": row["house"], "division_number": row["number"], "title": row["title"],
        "motion": row["motion_text"], "date": _iso(row["division_date"]), "ayes": row["ayes"], "noes": row["noes"],
        "ayes_listed": row["ayes_listed"], "noes_listed": row["noes_listed"],
        "ayes_named": row.get("ayes_named"), "noes_named": row.get("noes_named"),
        "tally_reconciled": _reconciled(row),
        "must_cite": "official tellers' figures (division.ayes / division.noes)",
        "printed_ayes": row.get("printed_ayes"), "printed_noes": row.get("printed_noes"),
    }  # fmt: skip
    caveat = " ".join(c for c in (_era_caveat(row), _tally_caveat(row), sources_caveat(value)) if c)
    return Fact("division_tally", params, value=value, caveat=caveat)


_DIVISION_DEBATE_SQL = """
SELECT d.id AS division_id, d.number, d.house, d.division_date, d.held_at, d.title AS division_title, d.result,
       b.id AS debate_id, b.heading, b.section, b.debate_type, b.date AS debate_date,
       (SELECT count(*) FROM speech s WHERE s.debate_id = b.id) AS speeches
  FROM division d
  LEFT JOIN debate b ON b.id = d.debate_id
 WHERE d.id = %(division_id)s
"""


async def division_debate(conn, division_id: int) -> Fact:
    """Which debate a division was taken during."""
    params = {"division_id": int(division_id)}
    row = await _one(conn, _DIVISION_DEBATE_SQL, params)
    if row is None:
        return Fact("division_debate", params, missing=f"no division with id {division_id}")
    if row["debate_id"] is None:
        return Fact("division_debate", params,
                    missing=(f"the record does not link division {row['number']} in the {row['house']} on "
                             f"{_iso(row['division_date'])} to a debate"))  # fmt: skip
    # The debate is the thread the record belongs to: a stage resumed after other business is one debate, and its
    # contributions are counted over all of its records.
    thread = (await TH.threads_of(conn, [row["debate_id"]])).get(row["debate_id"])
    ids = list(thread.debate_ids) if thread else [row["debate_id"]]
    speeches = int(row["speeches"] or 0)
    if len(ids) > 1:
        counted = await _one(conn, "SELECT count(*) AS n FROM speech WHERE debate_id = ANY(%(ids)s)", {"ids": ids})
        speeches = int(counted["n"] or 0) if counted else speeches
    return Fact("division_debate", params, value={
        "division_id": row["division_id"], "division_number": row["number"], "house": row["house"],
        "date": _iso(row["division_date"]), "division_title": row["division_title"],
        "debate_id": row["debate_id"], "debate": row["heading"], "section": row["section"],
        "debate_type": row["debate_type"], "speeches": speeches, "thread_id": ids[0], "thread": ids,
    }, caveat=" ".join(c for c in (
        "" if _iso(row["debate_date"]) == _iso(row["division_date"]) else
        f"The debate is recorded on {_iso(row['debate_date'])} and the division on {_iso(row['division_date'])}.",
        # a division held after midnight is placed in its sitting here too
        ERAS.sitting_note(row["division_date"], row.get("held_at"))) if c))  # fmt: skip


def _reconciled(row: dict) -> bool | None:
    """Whether the name lists match the tellers' figures: the loader's stored answer, never re-derived here."""
    return T.reconciled(row)


def _tally_caveat(row: dict) -> str:
    return T.caveat(row)


_VOTE_SQL = """
SELECT v.division_id, v.mnis_id, v.lobby, v.member_name, v.party, v.party_abbreviation,
       v.is_teller, v.proxy_name, v.flags, d.division_date, d.held_at, d.house, d.title, d.motion_text, d.number,
       d.ayes, d.noes, d.ayes_named, d.noes_named
  FROM division_vote v JOIN division d ON d.id = v.division_id
 WHERE v.division_id = %(division_id)s AND v.mnis_id = %(mnis_id)s
"""


async def division_vote(conn, division_id: int, mnis_id: int) -> Fact:
    """How one named member voted, including proxy and remote votes."""
    params = {"division_id": int(division_id), "mnis_id": int(mnis_id)}
    row = await _one(conn, _VOTE_SQL, params)
    if row is None:
        return Fact("division_vote", params,
                    missing=f"member {mnis_id} is not recorded in the lists for division {division_id}")
    return Fact("division_vote", params, value={
        "division_id": row["division_id"], "mnis_id": row["mnis_id"], "member": row["member_name"],
        "lobby": (row["lobby"] or "").lower(), "party_at_vote": row["party"],
        "is_teller": bool(row["is_teller"]), "proxy_name": row["proxy_name"],
        "house": row["house"], "division_number": row["number"], "title": row["title"], "motion": row["motion_text"],
        "date": _iso(row["division_date"]), "ayes": row["ayes"], "noes": row["noes"],
        "ayes_named": row["ayes_named"], "noes_named": row["noes_named"],
    # The proxy's NAME is in the record, and "cast by proxy" without it left six dev answers short of a vital point.
    }, caveat=" ".join(c for c in (f"Cast by proxy by {row['proxy_name']}." if row["proxy_name"] else "",
                                   ERAS.sitting_note(row["division_date"], row.get("held_at"))) if c))  # fmt: skip


# --- party and office at a date -------------------------------------------------------------------------

_PARTY_SQL = """
SELECT DISTINCT v.party, v.party_abbreviation, v.member_name
  FROM division_vote v JOIN division d ON d.id = v.division_id
 WHERE v.mnis_id = %(mnis_id)s AND d.division_date = %(on_date)s AND v.party IS NOT NULL
"""

# The lists that bracket a day: the last before it and the first after it that name the member with a party.
_PARTY_BEFORE_SQL = """
SELECT v.party, v.member_name, d.division_date
  FROM division_vote v JOIN division d ON d.id = v.division_id
 WHERE v.mnis_id = %(mnis_id)s AND v.party IS NOT NULL AND d.division_date < %(on_date)s
 ORDER BY d.division_date DESC LIMIT 1
"""
_PARTY_AFTER_SQL = """
SELECT v.party, v.member_name, d.division_date
  FROM division_vote v JOIN division d ON d.id = v.division_id
 WHERE v.mnis_id = %(mnis_id)s AND v.party IS NOT NULL AND d.division_date > %(on_date)s
 ORDER BY d.division_date ASC LIMIT 1
"""


_BIOGRAPHY_SQL = "SELECT raw -> 'biography' AS bio, name FROM member_id_map WHERE mnis_id = %(mnis_id)s"
_HOUSES = {1: "Commons", 2: "Lords"}


def _sources_differ(params: dict, day: str, before: dict, history: str | None, where: str,
                    after: dict | None = None) -> Fact:  # fmt: skip
    """The division lists and the Members API's party history disagree about a day no list covers. Neither is stated as
    the party: both are offered, each with its source and date, so the reader sees the record's evidence.
    """
    lists = (f"the division lists either side ({_day(before['division_date'])} and {_day(after['division_date'])})"
             if after else f"the latest division list before it ({_day(before['division_date'])})")
    if history is None:
        return Fact("party_at_date", params,
                    missing=(f"no division list names member {params['mnis_id']} on {day}, {where}; {lists} record "
                             f"{before['party']}, but the Members API's party history gives no single party that "
                             f"day, so no party is stated"))  # fmt: skip
    return Fact("party_at_date", params, clarify=[{"party": before["party"]}, {"party": history}],
                caveat=(f"No division list names {before['member_name']} on {_day(day)}, {where}: {lists} "
                        f"record{'' if after else 's'} {before['party']}, and the Members API's party history gives "
                        f"{history} that day."))  # fmt: skip


async def _outside_every_window(conn, day) -> bool:
    """Whether a day lies outside every period the record holds (`answer.scope`); False when coverage cannot be read.
    """
    from answer import scope as SCOPE

    found = await SCOPE.of_question(conn, {"on_date": _as_date(day)})
    return found is not None and found.outside


async def _past_the_last_division(conn, day: str, house: int) -> bool:
    """Whether `day` is on or after the record's latest division in `house`: the edge, where no list can follow. Read
    through the served views, so a sealed window's divisions never count as "later".
    """
    row = await _one(conn, "SELECT max(division_date) AS last FROM division WHERE house = %(house)s",
                     {"house": _HOUSES[house]})  # fmt: skip
    last = row["last"] if row else None
    return last is not None and _as_date(day) >= last


def _when(value) -> Date | None:
    try:
        return Date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _inside(entry: dict, day: Date) -> bool | None:
    """Whether `day` lies strictly inside the entry's dates; None on a start or end day (whether it counts is not in
    the payload), False outside.
    """
    start, end = _when(entry.get("startDate")), _when(entry.get("endDate"))
    if day in (start, end):
        return None
    return (start is None or start < day) and (end is None or day < end)


def _seat_on(bio: dict, day: Date) -> int | None:
    """The House the member sat in on `day` from the Members API's dated seats: Commons `representations` (which show
    the dissolution gaps), Lords `houseMemberships`. None for no seat, for a boundary day, or for two.
    """
    seats = {1 for r in bio.get("representations") or [] if isinstance(r, dict) and r.get("house") == 1
             and _inside(r, day)}
    if not bio.get("representations"):
        seats |= {1 for m in bio.get("houseMemberships") or [] if isinstance(m, dict) and m.get("house") == 1
                  and _inside(m, day)}
    seats |= {2 for m in bio.get("houseMemberships") or [] if isinstance(m, dict) and m.get("house") == 2
              and _inside(m, day)}
    return seats.pop() if len(seats) == 1 else None


async def house_sat(conn, mnis_id: int, on_date: Date | str) -> str | None:
    """The House a member sat in on a day by the Members API's dated seats ("Commons" or "Lords"), or None when the
    biography gives none, two, or a boundary day.
    """
    member = await _one(conn, _BIOGRAPHY_SQL, {"mnis_id": int(mnis_id)})
    bio = member["bio"] if member and isinstance(member.get("bio"), dict) else {}
    return _HOUSES.get(_seat_on(bio, _as_date(on_date)) or 0)


def _history_party(bio: dict, day: Date, house: int) -> str | None:
    """The one party the Members API's history gives for `house` on `day`, strictly inside its interval, else None.
    """
    found = set()
    for a in bio.get("partyAffiliations") or []:
        if not isinstance(a, dict) or not a.get("name") or (a.get("house") not in (None, house)):
            continue
        inside = _inside(a, day)
        if inside is None:
            return None
        if inside:
            found.add(str(a["name"]))
    return found.pop() if len(found) == 1 else None


def _same_party(a: str | None, b: str | None) -> bool:
    return bool(a and b) and " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


async def party_at_date(conn, mnis_id: int, on_date: Date | str) -> Fact:
    """Which party a member sat for on a date, from how they are recorded in a division list that day."""
    params = {"mnis_id": int(mnis_id), "on_date": _as_date(on_date)}
    rows = await _all(conn, _PARTY_SQL, params)
    parties = sorted({r["party"] for r in rows if r["party"]})
    if len(parties) == 1:
        return Fact("party_at_date", params, value={
            "mnis_id": params["mnis_id"], "member": rows[0]["member_name"],
            "date": _iso(params["on_date"]), "party": parties[0],
            "basis": "the party recorded against this member in a division list on that date",
        })  # fmt: skip
    if len(parties) > 1:
        return Fact("party_at_date", params, clarify=[{"party": p} for p in parties],
                    caveat="More than one party is recorded against this member on that date.")  # fmt: skip

    # No list names them that day.
    before = await _one(conn, _PARTY_BEFORE_SQL, params)
    after = await _one(conn, _PARTY_AFTER_SQL, params)
    day = _iso(params["on_date"])
    member = await _one(conn, _BIOGRAPHY_SQL, params)
    bio = member["bio"] if member and isinstance(member.get("bio"), dict) else {}
    house = _seat_on(bio, params["on_date"])
    if (before is None or after is None) and await _outside_every_window(conn, params["on_date"]):
        # No division list can name anyone on a day outside every window, so the Members API's party history is the
        # record's only statement of it.
        if house is None:
            return Fact("party_at_date", params,
                        missing=(f"{day} is outside the periods the record holds, and the Members API records no "
                                 f"single seat for member {mnis_id} that day, so no party is stated"))  # fmt: skip
        history = _history_party(bio, params["on_date"], house)
        if history is None:
            return Fact("party_at_date", params,
                        missing=(f"{day} is outside the periods the record holds, and the Members API's party history "
                                 f"gives no single party for member {mnis_id} that day, so none is stated"))  # fmt: skip
        return Fact("party_at_date", params, value={
            "mnis_id": params["mnis_id"], "member": (before or after or {}).get("member_name") or member.get("name"),
            "date": day, "party": history,
            "basis": (f"the Members API's party history in the {_HOUSES[house]}; no division list in the record covers "
                      f"that day, which is outside the periods it holds"),
        }, caveat="")  # fmt: skip
    if before is None and after is None:
        return Fact("party_at_date", params,
                    missing=f"member {mnis_id} does not appear in any division list, so the record carries no "
                            "party for them on any date")  # fmt: skip
    if before is not None and after is None and house is not None and await _past_the_last_division(conn, day, house):
        # The record's edge, not a gap in it.
        history = _history_party(bio, params["on_date"], house)
        if not _same_party(history, before["party"]):
            return _sources_differ(params, day, before, history,
                                   f"which is after the record's latest division in the {_HOUSES[house]}")
        return Fact("party_at_date", params, value={
            "mnis_id": params["mnis_id"], "member": before["member_name"], "date": day, "party": before["party"],
            "basis": (f"no division list names them that day, which is after the record's latest division in the "
                      f"{_HOUSES[house]}; the latest division list before it ({_iso(before['division_date'])}) and the "
                      f"Members API's party history agree on {before['party']}"),
        }, caveat="")  # fmt: skip
    if before is None or after is None:
        side = before or after
        return Fact("party_at_date", params,
                    missing=(f"no division list names member {mnis_id} on {day}, and only one side of it holds one "
                             f"({_iso(side['division_date'])}, {side['party']}); whether that still held on {day} is "
                             f"not in the lists"))  # fmt: skip
    if before["party"] != after["party"]:
        return Fact("party_at_date", params,
                    missing=(f"no division list names member {mnis_id} on {day}; they are listed as {before['party']} "
                             f"on {_iso(before['division_date'])} and as {after['party']} on "
                             f"{_iso(after['division_date'])}, so which applied on {day} is not in the lists"))  # fmt: skip
    if house is None:
        return Fact("party_at_date", params,
                    missing=(f"no division list names member {mnis_id} on {day}, and the Members API records no single "
                             f"seat for them that day, so no party is stated"))  # fmt: skip
    history = _history_party(bio, params["on_date"], house)
    if not _same_party(history, before["party"]):
        return _sources_differ(params, day, before, history, "between two lists that agree", after=after)
    # Reported, not silently substituted: the honest answer names what it comes from, once.
    return Fact("party_at_date", params, value={
        "mnis_id": params["mnis_id"], "member": before["member_name"], "date": day, "party": before["party"],
        "basis": (f"no division list names them that day; the Members API's party history gives {history} in the "
                  f"{_HOUSES[house]}, as do the lists either side ({_iso(before['division_date'])} and "
                  f"{_iso(after['division_date'])})"),
    }, caveat="")  # fmt: skip


_OFFICE_SQL = """
SELECT o.mnis_id, o.office, o.office_key, o.start_date, o.end_date, o.source, o.evidence, m.name,
       coalesce((to_jsonb(o) ->> 'start_withheld')::boolean, false) AS start_withheld,
       coalesce((to_jsonb(o) ->> 'end_withheld')::boolean, false) AS end_withheld
  FROM member_office o LEFT JOIN member_id_map m USING (mnis_id)
 WHERE o.office_key = ANY(%(keys)s)
   AND (o.start_date IS NULL OR o.start_date <= %(on_date)s)
   AND (o.end_date IS NULL OR o.end_date >= %(on_date)s)
 -- a recorded appointment before a period inferred from Hansard's labels, for one holder (QA pass 4)
 ORDER BY (o.source = 'mnis_biography') DESC, o.evidence DESC, o.start_date DESC NULLS LAST
"""

# `start_withheld` and `end_withheld` exist only in the served view (migration 027), so they are read through jsonb: the
# same query runs on a connection that reads the whole record, where they are absent and so false.
_OFFICE_ALL_SQL = """
SELECT o.mnis_id, o.office, o.start_date, o.end_date, m.name,
       coalesce((to_jsonb(o) ->> 'start_withheld')::boolean, false) AS start_withheld,
       coalesce((to_jsonb(o) ->> 'end_withheld')::boolean, false) AS end_withheld
  FROM member_office o LEFT JOIN member_id_map m USING (mnis_id)
 WHERE o.office_key = %(office_key)s ORDER BY o.start_date NULLS FIRST
"""
_WITHHELD = "a date inside a period that is not yet available is withheld"


def _period(row: dict) -> dict:
    """A holder's period for a list: a side the sealed window withheld is None and says so, never the clipped date.
    """
    out = {"from": None if row.get("start_withheld") else _iso(row["start_date"]),
           "to": None if row.get("end_withheld") else _iso(row["end_date"])}
    if row.get("start_withheld") or row.get("end_withheld"):
        out["note"] = _WITHHELD
    return out


async def _joint_titles(conn, key: str) -> list[str]:
    """The record's office keys that begin with `key` as the first part of a joint title (`key` and then ", " or " and
    "): "prime minister and first lord of the treasury" for "prime minister". Only from the start: the parts of a
    department's name ("secretary of state for education and skills") are not offices of their own, and the first
    part is the office the title is held as.
    """
    from answer import resolve as RS

    joints = (key + ", ", key + " and ")
    return sorted({r["office_key"] for r in (await RS.offices(conn)).ranked if r["office_key"].startswith(joints)})


async def office_held_at_date(conn, office_key: str, on_date: Date | str | None) -> Fact:
    """Who held an office on a date."""
    key = (office_key or "").strip().lower()
    if not key:
        raise FactError("office_held_at_date needs an office key")
    if on_date is None:
        holders = await _all(conn, _OFFICE_ALL_SQL, {"office_key": key})
        if not holders:
            return Fact("office_held_at_date", {"office_key": key},
                        missing=f"the record holds no office matching {office_key!r}")  # fmt: skip
        return Fact("office_held_at_date", {"office_key": key}, clarify=[
            {"mnis_id": h["mnis_id"], "holder": h["name"], "office": h["office"], **_period(h)} for h in holders
        ], caveat="No date was given, and this office has been held by more than one person.")  # fmt: skip

    params = {"office_key": key, "on_date": _as_date(on_date)}
    joint = await _joint_titles(conn, key)
    rows = await _all(conn, _OFFICE_SQL, {**params, "keys": [key, *joint]})
    own = [r for r in rows if r["office_key"] == key]
    # A recorded appointment outranks a period inferred from how Hansard printed a member: where the Members API records
    # a holder that day, an inferred period for anyone else is not a second holder (S06: Housing, 13 October 2025, asked
    # back between the recorded Secretary of State and a printed label that ran on).
    recorded = [r for r in own if r["source"] != "hansard_attribution"]
    # Where nothing is recorded under the office's own name that day, a recorded joint title that begins with it is its
    # holder too ("Prime Minister and First Lord of the Treasury"; QA pass 9, Med 7), beside the inferred periods and
    # never over them: a joint title may be one of several holders ("Deputy Speaker and Chairman of Ways and Means"), so
    # where they disagree the reader is asked.
    sharing = [r for r in rows if r["office_key"] != key and r["source"] != "hansard_attribution"]
    rows = recorded or sorted(own + sharing, key=lambda r: r["source"] == "hansard_attribution")
    if not rows:
        return Fact("office_held_at_date", params,
                    missing=f"no holder of {office_key!r} is recorded on {_iso(params['on_date'])}")  # fmt: skip
    distinct = {r["mnis_id"] for r in rows}
    if len(distinct) > 1:
        # Joint and acting appointments are real; so are overlapping records.
        return Fact("office_held_at_date", params, clarify=[
            {"mnis_id": r["mnis_id"], "holder": r["name"], "office": r["office"], **_period(r)} for r in rows
        ], caveat=f"{len(distinct)} people are recorded in this office on that date.")  # fmt: skip

    row = rows[0]
    return Fact("office_held_at_date", params, value={
        "office": row["office"], "office_key": row["office_key"], "date": _iso(params["on_date"]),
        "mnis_id": row["mnis_id"], "holder": row["name"],
    # Inferred where it comes from Hansard's labels, recorded where it comes from the Members API (migration 005).
    }, caveat=("This period is inferred from how Hansard attributes the member's speeches, not from a recorded "
               "appointment." if row["source"] == "hansard_attribution" else ""))  # fmt: skip


# --- bills ----------------------------------------------------------------------------------------------

_STAGE_SQL = """
SELECT s.id, s.bill_id, s.stage_name, s.house, b.title,
       -- sitting DAYS (D-088): one day can hold two rows, for two linked debates, and is still one sitting
       min(t.date) AS first_sitting, max(t.date) AS last_sitting, count(DISTINCT t.date) AS sittings
  FROM bill_stage s
  JOIN bill b ON b.id = s.bill_id
  -- A sitting counts only on or before the Bills API's own as-of (the bill's last update; the load time when it gives
  -- none): the API lists sittings scheduled ahead, and "reached 2nd reading on 15 October 2026" was said on the 14th
  -- (QA pass 4, D-112). The 2 sittings between a bill's last update and the load are not yet known to have happened.
  JOIN bill_stage_sitting t ON t.bill_stage_id = s.id AND t.date <= coalesce(b.last_update, b.fetched_at)::date
 WHERE s.bill_id = %(bill_id)s AND s.stage_name = %(stage)s
   -- A stage whose House the Bills API does not give (a private bill's "Commons Examiners") is compatible with any
   -- House the question names: unknown is not the other House (db0188 abstained on exactly this).
   AND (%(house)s::text IS NULL OR s.house IS NULL OR s.house = %(house)s)
 GROUP BY s.id, s.bill_id, s.stage_name, s.house, b.title
"""


async def bill_stage_date(conn, bill_id: int, stage: str, house: str | None = None) -> Fact:
    """When a bill reached a stage, from the sittings actually held."""
    params = {"bill_id": int(bill_id), "stage": stage, "house": house}
    rows = await _all(conn, _STAGE_SQL, params)
    if not rows:
        return Fact("bill_stage_date", params,
                    missing=f"no sitting is recorded for {stage!r} on bill {bill_id}")  # fmt: skip
    if len(rows) > 1:
        return Fact("bill_stage_date", params, clarify=[
            {"stage_id": r["id"], "house": r["house"], "first_sitting": _iso(r["first_sitting"]),
             "sittings": r["sittings"]} for r in rows
        ], caveat=f"{stage!r} appears {len(rows)} times for this bill.")  # fmt: skip

    row = rows[0]
    value = {"bill_id": row["bill_id"], "bill": row["title"], "stage": row["stage_name"],
             "house": row["house"], "first_sitting": _iso(row["first_sitting"]),
             "sittings": int(row["sittings"])}  # fmt: skip
    if row["sittings"] > 1:
        value["last_sitting"] = _iso(row["last_sitting"])
    return Fact("bill_stage_date", params, value=value)


# --- counts ---------------------------------------------------------------------------------------------

_COUNT_DIVISIONS_SQL = """
SELECT count(*) AS divisions FROM division
 WHERE house = %(house)s
   AND division_date >= %(month)s::date
   AND division_date < (%(month)s::date + interval '1 month')
"""


async def count_divisions(conn, house: str, month: Date | str) -> Fact:
    """How many divisions a House held in a month."""
    params = {"house": house, "month": _as_date(month)}
    row = await _one(conn, _COUNT_DIVISIONS_SQL, params)
    return Fact("count", params, value={
        "house": house, "month": _iso(params["month"]), "divisions": int(row["divisions"]),
    })  # fmt: skip


_COUNT_DEBATES_SQL = """
SELECT count(DISTINCT d.id) AS debates
  FROM debate d
 WHERE d.house = %(house)s AND d.date = %(on_date)s
"""


async def count_debates(conn, house: str, on_date: Date | str) -> Fact:
    """How many separate debates a House held on a day."""
    params = {"house": house, "on_date": _as_date(on_date)}
    row = await _one(conn, _COUNT_DEBATES_SQL, params)
    return Fact("count", params, value={
        "house": house, "date": _iso(params["on_date"]), "debates": int(row["debates"]),
    })  # fmt: skip


async def count_divisions_on_day(conn, house: str | None, on_date: Date | str) -> Fact:
    """How many divisions a House (or both) held on a calendar day: its sitting's, and the sitting before's held after
    midnight that day, the late ones counted apart so the count is never read as the day's sitting's alone.
    """
    from answer import resolve as RS

    day = _as_date(on_date)
    params = {"house": house, "on_date": day, "counts": "divisions"}
    row = await _one(conn, "SELECT count(*) AS divisions, count(*) FILTER (WHERE d.division_date <> %(day)s) AS late "
                           "FROM division d WHERE " + RS.ON_DAY.replace("%s", "%(day)s") +
                           " AND (%(house)s::text IS NULL OR d.house = %(house)s)", {"day": day, "house": house})
    value = {"house": house, "date": _iso(day), "divisions": int(row["divisions"])}
    if row["late"]:
        value["of_the_sitting_before"] = int(row["late"])
    return Fact("count", params, value=value)


async def count_debates_in_month(conn, house: str | None, month: Date | str) -> Fact:
    """How many separate debates a House (or both) held in a month (QA pass 9: a month had meant divisions)."""
    start = _as_date(month).replace(day=1)
    params = {"house": house, "month": start, "counts": "debates"}
    row = await _one(conn, "SELECT count(DISTINCT d.id) AS debates FROM debate d WHERE d.date >= %(month)s AND "
                           "d.date < (%(month)s::date + interval '1 month') AND (%(house)s::text IS NULL OR "
                           "d.house = %(house)s)", {"month": start, "house": house})  # fmt: skip
    return Fact("count", params, value={"house": house, "month": _iso(start), "debates": int(row["debates"])})


_COUNT_CONTRIBUTIONS_SQL = """
SELECT count(*) AS contributions
  FROM speech s
 WHERE s.mnis_id = %(mnis_id)s
   AND NOT s.is_boilerplate
   AND s.id IN (SELECT DISTINCT c.speech_id FROM chunk c
                 WHERE c.speech_id = s.id AND c.debate_date BETWEEN %(from_date)s AND %(to_date)s)
"""


async def count_contributions(conn, mnis_id: int, from_date: Date | str, to_date: Date | str) -> Fact:
    """How many times a member spoke between two dates, boilerplate excluded."""
    params = {"mnis_id": int(mnis_id), "from_date": _as_date(from_date), "to_date": _as_date(to_date)}
    row = await _one(conn, _COUNT_CONTRIBUTIONS_SQL, params)
    return Fact("count", params, value={
        "mnis_id": params["mnis_id"], "from": _iso(params["from_date"]), "to": _iso(params["to_date"]),
        "contributions": int(row["contributions"]), "excludes": "boilerplate",
    })  # fmt: skip


async def count_any(conn, house: str | None = None, month=None, on_date=None,
                    mnis_id=None, from_date=None, to_date=None, counts=None, **_ignored) -> Fact:
    """Route a `count` item to the template its own parameters name."""
    if counts == "divisions" and on_date is not None:
        return await count_divisions_on_day(conn, house, on_date)
    if counts == "debates" and month is not None:
        return await count_debates_in_month(conn, house, month)
    if month is not None:
        return await count_divisions(conn, house, month)
    if on_date is not None:
        return await count_debates(conn, house, on_date)
    if mnis_id is not None and from_date is not None and to_date is not None:
        return await count_contributions(conn, mnis_id, from_date, to_date)
    raise FactError("a count item must carry a month (divisions), a date (debates), or an mnis_id with a "
                    "from/to range (contributions)")


# --- dispatch -------------------------------------------------------------------------------------------

# gold category -> the coroutine that answers it, and which of the item's expected facts are its parameters.
DISPATCH: dict[str, tuple[Any, tuple[str, ...]]] = {
    "division_result": (division_result, ("division_id",)),
    "division_tally": (division_tally, ("division_id",)),
    "division_vote": (division_vote, ("division_id", "mnis_id")),
    "division_debate": (division_debate, ("division_id",)),
    "party_at_date": (party_at_date, ("mnis_id", "date")),
    "office_held_at_date": (office_held_at_date, ("office_key", "date")),
    "bill_stage_date": (bill_stage_date, ("bill_id", "stage", "house")),
    "count": (count_any, ("house", "month", "date")),
}


async def answer(conn, category: str, facts: dict[str, Any]) -> Fact:
    """Answer one gold item's question from its own expected parameters."""
    if category not in DISPATCH:
        raise FactError(f"no template answers {category!r}; known: {', '.join(sorted(DISPATCH))}")
    handler, names = DISPATCH[category]
    if category == "count":
        # Alternatives, not a fixed signature: a count item carries a month OR a date.
        return await count_any(conn, facts.get("house"), month=facts.get("month"),
                               on_date=facts.get("date"), mnis_id=facts.get("mnis_id"),
                               from_date=facts.get("from"), to_date=facts.get("to"),
                               counts=facts.get("counts"))  # fmt: skip
    args = []
    for name in names:
        if name not in facts and name != "house":
            raise FactError(f"{category} needs {name!r} and the item does not carry it")
        args.append(facts.get(name))
    return await handler(conn, *args)


# --- small helpers --------------------------------------------------------------------------------------


async def _one(conn, sql: str, params: dict) -> dict | None:
    rows = await _all(conn, sql, params)
    return rows[0] if rows else None


async def _all(conn, sql: str, params: dict) -> list[dict]:
    from psycopg.rows import dict_row

    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, params)
        return list(await cur.fetchall())


def _as_date(value: Date | str) -> Date:
    if isinstance(value, Date):
        return value
    return Date.fromisoformat(str(value)[:10])


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)[:10]
