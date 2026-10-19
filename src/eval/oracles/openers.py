"""The opener oracle (MR-08): who opened each debate, from the Hansard contributions as fetched."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app import sealed

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "corpus_v2" / "contributions"

# How Hansard prints the Chair.
PRESIDING = re.compile(
    r"^(?:Mr|Madam)(?:Deputy)?Speaker(?![\u2019']s)\b"
    r"|^The(?:Senior)?(?:Deputy)?Speaker(?![\u2019']s)(?:\(|$)"
    r"|^TheLordSpeaker(?:\(|$)"
    r"|^The(?:Principal)?(?:Deputy)?ChairmanofCommittees"
    r"|^(?:The)?(?:First|Second|Temporary)?(?:Deputy)?Chair(?:man)?ofWaysandMeans"
    r"|^The(?:Temporary|(?:First|Second)?Deputy)?Chair(?:man)?(?:\(|$)"
    r"|\(intheChair\)", re.I)


def _squeezed(attributed: str | None) -> str:
    return re.sub(r"[\s.]+", "", attributed or "")


def presiding(attributed: str | None) -> bool:
    return bool(PRESIDING.search(_squeezed(attributed)))


# How a point of order is raised, at the start of a contribution; and a heading whose business is one.
POINT_OF_ORDER = re.compile(r"^\W*(?:I\s+rise\s+)?on\s+a\s+(?:\w+\s+)?point\s+of\s+order\b"
                            r"|^\W*further\s+to\s+.{0,80}?\bpoint\s+of\s+order\b", re.I | re.S)
ABOUT_ORDER = re.compile(r"\bpoints?\s+of\s+order\b", re.I)

# A heading that names the presiding Chair's own statement, read with its apostrophes and colons taken out and its
# spacing folded (`_heading_words`): "Speaker's Statement", "Speakers Statement", "Speaker: Statement", "Deputy
# Speaker's Statement", "Chairman of Ways and Means' Statement", "First Deputy Chairman's Statement", "Statement by the
# Speaker".
_SPEAKER = r"(?:the )?(?:(?:mr|madam) )?(?:(?:lord|senior|deputy|first|second|principal) )*speakers?"
_CHAIRMAN_OF = r"(?:the )?(?:(?:first|second|deputy|principal|temporary) )*chair(?:man)?s? of (?:ways and means|committees)s?"
_CHAIRMAN = r"(?:the )?(?:(?:first|second|deputy|principal|temporary) )*chair(?:man)?s?"
CHAIRS_STATEMENT = re.compile(
    rf"^(?:{_SPEAKER}|{_CHAIRMAN_OF}) statements?(?: |$)|^{_CHAIRMAN} statements?$"
    rf"|^statements? (?:by|from) (?:{_SPEAKER}|{_CHAIRMAN_OF}|{_CHAIRMAN})$")


def _heading_words(heading: str | None) -> str:
    return " ".join(re.sub(r"[\u2019'\u2018]", "", (heading or "").replace(":", " ")).casefold().split())


def chairs_statement(heading: str | None) -> bool:
    """Whether a heading names the presiding Chair's own statement."""
    return bool(CHAIRS_STATEMENT.search(_heading_words(heading)))


# The acts that open business, however printed (moving it, asking a question, presenting a bill or a petition), and the
# most words a stage direction runs to (a phrase: "King's consent signified."; a longer run of italics is a passage,
# such as a quotation or words in Welsh).
OPENING_ACTS = re.compile(r"\b(?:move|moved|moves|moving|ask|asks|asked|asking|present|presents|presented|presenting)\b",
                          re.I)
STAGE_WORDS = 12


class _Words(HTMLParser):
    """A contribution's words, split into those Hansard sets in italics (the reporter's descriptions) and the rest.
    Every tag is a word boundary; words after an italic tag never closed are not shown to be italic, so they are the
    rest's.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth, self.italic, self.plain, self.held = 0, [], [], []

    def _gap(self) -> None:
        (self.held if self.depth else self.plain).append(" ")

    def handle_starttag(self, tag, attrs):
        self._gap()
        self.depth += tag in ("em", "i")

    def handle_endtag(self, tag):
        self._gap()
        if tag in ("em", "i") and self.depth:
            self.depth -= 1
            if not self.depth:
                self.italic += self.held
                self.held = []

    def handle_data(self, data):
        (self.held if self.depth else self.plain).append(data)

    def close(self) -> None:
        super().close()
        self.plain += self.held
        self.held = []


def says_nothing(text: str | None) -> bool:
    """Whether a contribution, as fetched, says nothing a member said: no words, or italic stage directions alone.
    """
    words = _Words()
    words.feed(text or "")
    words.close()
    plain, italic = " ".join("".join(words.plain).split()), " ".join("".join(words.italic).split())
    notes = _alone_as_notes(f"{plain} {italic}")
    if notes is not None:
        # A note alone is read for what it says.
        import html

        glued = _alone_as_notes(html.unescape(re.sub(r"<[^>]*>", "", text or ""))) or []
        return not any(GAP_IN_TRANSCRIPT.search(n) and not REPORTED_SPEECH.match(n) for n in [*notes, *glued])
    if plain:
        return False  # words outside italics, or in italics never closed: a speech
    if not italic:
        return True
    return not (len(italic.split()) > STAGE_WORDS or OPENING_ACTS.search(italic) or POINT_OF_ORDER.search(italic))


# The reporter's words for a gap in the transcript, read independently of `procedure.opener`'s rule.
GAP_IN_TRANSCRIPT = re.compile(r"inaudib|indistinct|unclear|no audio|audio (?:lost|unavailable|cut|failed|failure|"
                               r"dropped)|not (?:recorded|audible|heard|caught|transcribed)|connection (?:lost|failed|"
                               r"dropped|broken)|technical (?:difficult|problem|fault|issue)", re.I)  # fmt: skip
# A note reporting what someone said ("Hon.
REPORTED_SPEECH = re.compile(r"[^:\[\]()]{1,40}:\s*[\"“‘']")
_WIDE = {"［": "[", "］": "]", "（": "(", "）": ")"}


def _alone_as_notes(text: str) -> list[str] | None:
    """The notes, when the contribution is nothing but the reporter's notes: square-bracketed (nested or never closed),
    or one note in round or full-width brackets. None when anything else is there.
    """
    flat = " ".join("".join(_WIDE.get(ch, ch) for ch in text).split())
    if len(flat) > 2 and flat[0] == "(" and flat[-1] == ")" and flat.count("(") == flat.count(")") == 1:
        return [flat[1:-1].strip()]
    if flat and _all_in_notes(flat):
        return [piece.strip() for piece in flat.replace("]", "[").split("[") if piece.strip()]
    return None


def _all_in_notes(italic: str) -> bool:
    """Whether every letter of the italics is inside square brackets (the reporter's notes, nested or never closed).
    Alone in a member's place, a note stands where their words were; beside a stage direction ("rose—
    [Interruption.]"), the stage direction is what the member did.
    """
    depth, inside, outside = 0, False, False
    for ch in italic:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(depth - 1, 0)
        elif ch.isalnum():
            if depth:
                inside = inside or ch.isalpha()
            else:
                outside = True
    return inside and not outside


@dataclass(frozen=True)
class Contribution:
    order: int
    mnis_id: int | None
    attributed: str
    text: str


@dataclass(frozen=True)
class DebateTruth:
    debate_id: str
    house: str
    date: date
    heading: str
    section: str
    contributions: tuple[Contribution, ...]
    section_order: int | None = None  # the record's place in its House's day (`DebateSectionOrder`)

    @property
    def first(self) -> Contribution | None:
        return self.contributions[0] if self.contributions else None

    @property
    def presiding_first(self) -> bool:
        return self.first is not None and presiding(self.first.attributed)

    @property
    def chair_business(self) -> bool:
        return chairs_statement(self.heading)

    @property
    def chair_statement(self) -> Contribution | None:
        """The Chair's statement, where the business is one."""
        return next((c for c in self.contributions if presiding(c.attributed)), None) if self.chair_business else None

    @property
    def opener(self) -> Contribution | None:
        if self.chair_business:
            return None
        members = [c for c in self.contributions if c.mnis_id and not presiding(c.attributed) and not says_nothing(c.text)]
        if ABOUT_ORDER.search(self.heading or ""):
            return members[0] if members else None
        return next((c for c in members if not POINT_OF_ORDER.search(c.text or "")), members[0] if members else None)

    @property
    def presiding_ids(self) -> set[int]:
        return {c.mnis_id for c in self.contributions if c.mnis_id and presiding(c.attributed)}

    @property
    def members(self) -> set[int]:
        return {c.mnis_id for c in self.contributions if c.mnis_id and not presiding(c.attributed)}


@dataclass(frozen=True)
class ThreadTruth:
    """One logical debate: a run of records, in the day's order."""

    records: tuple[DebateTruth, ...]

    @property
    def debate_ids(self) -> tuple[str, ...]:
        return tuple(r.debate_id for r in self.records)

    @property
    def house(self) -> str:
        return self.records[0].house

    @property
    def date(self) -> date:
        return self.records[0].date

    @property
    def heading(self) -> str:
        return self.records[0].heading

    @property
    def presiding_first(self) -> bool:
        return self.records[0].presiding_first

    @property
    def chair_business(self) -> bool:
        return self.records[0].chair_business

    @property
    def chair_statement(self) -> Contribution | None:
        return next((r.chair_statement for r in self.records if r.chair_statement is not None), None)

    @property
    def opener(self) -> Contribution | None:
        return next((r.opener for r in self.records if r.opener is not None), None)

    @property
    def opened_in(self) -> str | None:
        return next((r.debate_id for r in self.records if r.opener is not None), None)

    @property
    def presiding_ids(self) -> set[int]:
        return set().union(*(r.presiding_ids for r in self.records))

    @property
    def members(self) -> set[int]:
        return set().union(*(r.members for r in self.records))


def heading_key(heading: str | None) -> str:
    return " ".join((heading or "").replace("\u2019", "'").casefold().split())


BILL_HEADING = re.compile(r"\bBill(?:\s*\[(?:HL|Lords)\])?\s*$", re.I)


def threads(truths: Iterator[DebateTruth] | list[DebateTruth]) -> list[ThreadTruth]:
    """The records grouped into threads: same House, day and section, same heading, next to each other in the day's
    order. A record without a place in the order, or without a heading, stands alone.
    """
    by_day: dict[tuple, list[DebateTruth]] = {}
    for t in truths:
        by_day.setdefault((t.house, t.date, t.section), []).append(t)
    out = []
    for day in sorted(by_day, key=lambda k: (k[0], k[1], k[2] or "")):
        run: list[DebateTruth] = []
        for t in sorted(by_day[day], key=lambda r: (r.section_order is None, r.section_order or 0, r.debate_id)):
            same = (run and t.section_order is not None and run[-1].section_order is not None and heading_key(t.heading)
                    and heading_key(t.heading) == heading_key(run[-1].heading))  # fmt: skip
            if run and not same:
                out.append(ThreadTruth(tuple(run)))
                run = []
            run.append(t)
        if run:
            out.append(ThreadTruth(tuple(run)))
    # a bill's proceedings that day, resumed after other business, are one debate
    joined: list[ThreadTruth] = []
    where: dict[tuple, int] = {}
    for th in out:
        place = (th.house, th.date, th.records[0].section, heading_key(th.heading))
        if BILL_HEADING.search(" ".join((th.heading or "").split())):
            if place in where:
                joined[where[place]] = ThreadTruth(joined[where[place]].records + th.records)
                continue
            where[place] = len(joined)
        joined.append(th)
    return joined


def _read(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def truths(raw: Path = RAW) -> Iterator[DebateTruth]:
    """Every debate the raw cache holds outside the sealed window, its contributions in the record's own order."""
    if not raw.exists():
        return
    sealed_id = sealed.window().window_id
    rows: dict[str, dict[str, tuple[int, Contribution]]] = {}
    meta: dict[str, tuple[str, date, str, str]] = {}
    for window in sorted(p for p in raw.iterdir() if p.is_dir() and not (sealed.masked() and p.name == sealed_id)):
        for page in sorted(window.rglob("*.json.gz")):
            for r in _read(page).get("Results") or []:
                if sealed.hidden(r.get("SittingDate")) or not r.get("DebateSectionExtId"):
                    continue
                did = str(r["DebateSectionExtId"])
                key = str(r.get("ContributionExtId") or (r.get("OrderInDebateSection"), r.get("ItemId")))
                order = int(r.get("OrderInDebateSection") or 0)
                mid = int(r["MemberId"]) if r.get("MemberId") else None
                text = str(r.get("ContributionTextFull") or r.get("ContributionText") or "").strip()
                rows.setdefault(did, {})[key] = (order, Contribution(order, mid, str(r.get("AttributedTo") or "").strip(),
                                                                     text))  # fmt: skip
                placed = r.get("DebateSectionOrder")
                meta.setdefault(did, (str(r.get("House") or ""), date.fromisoformat(str(r["SittingDate"])[:10]),
                                      str(r.get("DebateSection") or "").strip(), str(r.get("Section") or ""),
                                      int(placed) if placed not in (None, "") else None))  # fmt: skip
    for did in sorted(rows):
        house, day, heading, section, placed = meta[did]
        ordered = tuple(c for _, c in sorted(rows[did].values(), key=lambda pair: pair[0]))
        yield DebateTruth(did, house, day, heading, section, ordered, placed)
